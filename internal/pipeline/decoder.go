package pipeline

import (
	"bytes"
	"context"
	"image"
	_ "image/png" // register PNG decoder for image.Decode dispatch
	"log"
	"sync"

	"golang.org/x/image/draw"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/shm"
)

// Decoder is the optional stage between BackgroundLoader and DealerWorker that
// JPEG-decodes + resizes each sample on the daemon side, replacing the slot
// contents with packed RGB uint8 (HWC layout). Per-item size becomes a fixed
// ImageSize*ImageSize*3 bytes regardless of source JPEG size.
//
// Motivation: profiling (workers=0, single-process iteration) showed PIL
// JPEG decode + resize = 83% of per-sample Python time while the daemon was
// 99.4% idle. Moving decode here makes the otherwise-wasted daemon CPU useful
// and lets the Python worker skip PIL entirely.
//
// Parallelism (optimization 02): the decode of each item is independent, and
// the packed output is fixed-size (perItem), so item i always lands at scratch
// offset i*perItem with NO inter-item serialization. We therefore fan items of
// a slot out to a pool of `parallelism` persistent worker goroutines. This
// matters because with a single decode goroutine (the original design) the
// daemon decoded on ONE core while the rest sat idle — so server-side decode
// could not keep even one Python consumer fed (rgb_uint8 num_workers=0 was
// SLOWER than client-side PIL). See docs/optimizations/02-parallel-decode.md.
//
// The JPEG decoder itself is swappable via build tag:
//   - default (cgo on)            → libjpeg-turbo via TurboJPEG API (fast)
//   - `-tags datasetfs_purego`    → stdlib image/jpeg (slow, no cgo)
//
// See decoder_libjpeg.go and decoder_purego.go for the two implementations.
//
// Memory: a per-decoder scratch buffer (sized SlotSize) accumulates decoded
// items before a single copy back into the SHM slot. This keeps SHM contents
// consistent for the consumer — readers never see a half-rewritten slot. Pool
// workers write DISJOINT [i*perItem,(i+1)*perItem) ranges, so sharing the
// scratch backing array is race-free.
type Decoder struct {
	cfg          DecodeConfig
	allocator    *shm.Allocator
	in           <-chan *SlotMeta
	out          chan<- *SlotMeta
	freeSlotChan chan<- int
	scratch      []byte

	// Decode worker pool. Each worker owns a non-thread-safe jpegDecoder (e.g.
	// a TurboJPEG handle) and a private resize buffer; jobs are pulled from the
	// shared jobs channel. parallelism == len(workers) >= 1.
	parallelism int
	workers     []*decodeWorker
	jobs        chan decodeJob
	poolWg      sync.WaitGroup
}

// decodeWorker is one goroutine's private state. jpegDec and resized are NEVER
// shared between workers — a TurboJPEG handle is not concurrency-safe and the
// resize buffer is reused across that worker's jobs (safe: one goroutine, and
// draw.Scale fully overwrites the dst rect).
type decodeWorker struct {
	jpegDec jpegDecoder
	resized *image.RGBA
}

// decodeJob is one item's decode unit of work. raw and dst alias the SHM slot
// and the shared scratch respectively; dst ranges are disjoint across jobs.
type decodeJob struct {
	idx             int
	raw             []byte // source bytes (read-only) in the slot
	dst             []byte // destination region in scratch (perItem bytes)
	src             *Metadata
	out             **Metadata // = &results[idx]; worker stores rewritten meta here
	globalSlotStart int64
	perItem         int64
	w, h            int
	wg              *sync.WaitGroup
}

// jpegDecoder is implemented by both decoder_purego.go (stdlib) and
// decoder_libjpeg.go (cgo + TurboJPEG). One instance per decode worker;
// implementations may hold non-thread-safe state (e.g. a TurboJPEG handle).
type jpegDecoder interface {
	// Decode parses `b` as a JPEG and returns an image.Image backed by the
	// decoded RGB(A) pixels. The returned image MAY be invalidated by the
	// next Decode call on the same instance (impls are free to recycle
	// internal buffers), so callers must consume before the next Decode.
	Decode(b []byte) (image.Image, error)
	// Close releases any resources held by the decoder. Safe to call multiple
	// times.
	Close()
}

func NewDecoder(
	cfg DecodeConfig,
	alloc *shm.Allocator,
	in <-chan *SlotMeta,
	out chan<- *SlotMeta,
	freeSlots chan<- int,
	parallelism int,
) *Decoder {
	if parallelism < 1 {
		parallelism = 1
	}
	w, h := cfg.ImageSize, cfg.ImageSize
	workers := make([]*decodeWorker, parallelism)
	for i := range workers {
		workers[i] = &decodeWorker{
			jpegDec: newJPEGDecoder(),
			resized: image.NewRGBA(image.Rect(0, 0, w, h)),
		}
	}
	d := &Decoder{
		cfg:          cfg,
		allocator:    alloc,
		in:           in,
		out:          out,
		freeSlotChan: freeSlots,
		scratch:      make([]byte, shm.SlotSize),
		parallelism:  parallelism,
		workers:      workers,
		// Buffered so the dispatcher in decodeSlot stays ahead of the workers
		// without a lock-step handshake per item.
		jobs: make(chan decodeJob, parallelism*4),
	}
	return d
}

// startPool launches the persistent decode worker goroutines. Workers live for
// the whole session so TurboJPEG handle init + RGBA-buffer growth amortize
// across all slots. Each worker owns its decoder/resize buffer and pulls from
// the shared jobs channel.
func (d *Decoder) startPool() {
	for _, w := range d.workers {
		d.poolWg.Add(1)
		go func(dw *decodeWorker) {
			defer d.poolWg.Done()
			for j := range d.jobs {
				dw.process(j)
			}
		}(w)
	}
}

// stopPool closes the jobs channel, waits for workers to drain in-flight jobs,
// then releases each decoder. Must be called after the last decodeSlot so no
// handle is closed mid-decode.
func (d *Decoder) stopPool() {
	close(d.jobs)
	d.poolWg.Wait()
	for _, w := range d.workers {
		w.jpegDec.Close()
	}
}

// Launch consumes SlotMetas from the loader, decodes each item across the
// worker pool, and emits new SlotMetas pointing at the packed RGB bytes in the
// same slot. Closes `out` on input-channel close so the dealer correctly
// detects end-of-epoch.
func (d *Decoder) Launch(ctx context.Context) {
	d.startPool()
	// Shutdown order matters: stop feeding the dealer, then drain+stop workers
	// before closing their decoders (closing a handle mid-decode would crash).
	defer func() {
		close(d.out)
		d.stopPool()
	}()

	w := d.cfg.ImageSize
	h := d.cfg.ImageSize
	perItem := int64(w * h * 3)

	for {
		select {
		case <-ctx.Done():
			return
		case meta, ok := <-d.in:
			if !ok {
				return
			}
			slotBuf := d.allocator.GetSlotBuffer(meta.SlotID)
			out, used := d.decodeSlot(slotBuf, meta, perItem, w, h)
			if len(out) == 0 {
				// Every item failed to decode — release the slot so the
				// loader pool stays drained-correct.
				select {
				case d.freeSlotChan <- meta.SlotID:
				case <-ctx.Done():
					return
				}
				continue
			}
			// Replace slot contents with the packed decoded data. Safe to do
			// because the only other reader of this slot — Python — has not
			// yet been told the slot has data (we send to `out` below).
			copy(slotBuf[:used], d.scratch[:used])

			select {
			case <-ctx.Done():
				return
			case d.out <- &SlotMeta{Objects: out, SlotID: meta.SlotID}:
			}
		}
	}
}

// process decodes one item, resizes it, and packs RGB into the job's scratch
// region, recording the rewritten Metadata. On any failure it leaves
// *job.out == nil (a gap) and signals the WaitGroup — runs entirely within one
// worker goroutine, so jpegDec/resized reuse is safe.
func (dw *decodeWorker) process(j decodeJob) {
	defer j.wg.Done()

	var img image.Image
	var err error
	// Fast-path JPEG via the swappable backend; non-JPEG falls through to
	// stdlib image.Decode (PNG etc. via the blank import above).
	if isJPEG(j.raw) {
		img, err = dw.jpegDec.Decode(j.raw)
	} else {
		img, _, err = image.Decode(bytes.NewReader(j.raw))
	}
	if err != nil {
		log.Printf("[Decoder] decode failed for %s: %v", j.src.Path, err)
		return
	}

	draw.BiLinear.Scale(dw.resized, dw.resized.Bounds(), img, img.Bounds(), draw.Src, nil)
	packRGB(j.dst, dw.resized.Pix, j.w, j.h)

	// New Metadata: offset/size rewritten to the packed layout. Other fields
	// (Path, ObjectMetadata, ShardID) preserved so downstream (Dealer →
	// Python) sees the same semantic identifiers.
	m2 := *j.src
	m2.Offset = j.globalSlotStart + int64(j.idx)*j.perItem
	m2.Size = j.perItem
	*j.out = &m2
}

// decodeSlot fans every Metadata in `meta` out to the worker pool, each item
// decoded/resized into a fixed scratch region scratch[i*perItem]. Returns the
// surviving Metadata list (in original slot order, failures dropped) and the
// number of scratch bytes spanned (which is copied back into the slot — gap
// bytes from failed items are harmless, Python reads each item by its own
// offset/size).
func (d *Decoder) decodeSlot(
	slotBuf []byte,
	meta *SlotMeta,
	perItem int64,
	w, h int,
) ([]*Metadata, int64) {
	globalSlotStart := int64(meta.SlotID * shm.SlotSize)

	n := len(meta.Objects)
	maxItems := len(d.scratch) / int(perItem)
	if n > maxItems {
		// Decoded data won't fit. With imagenette-scale (96×96 = 27 KB) and
		// ~1000 items/shard this is ~27 MB into a 110 MB slot, so only triggers
		// for very large image_size or huge shards.
		log.Printf("[Decoder] slot %d: scratch holds %d items, shard has %d — dropping %d",
			meta.SlotID, maxItems, n, n-maxItems)
		n = maxItems
	}

	// results[i] == nil means item i failed/was-skipped (a gap). Each worker
	// writes only its own index, so no synchronization is needed on the slice.
	results := make([]*Metadata, n)
	var wg sync.WaitGroup

	for i := 0; i < n; i++ {
		m := meta.Objects[i]
		localOffset := m.Offset - globalSlotStart
		if localOffset < 0 || localOffset+m.Size > int64(len(slotBuf)) {
			log.Printf("[Decoder] slot %d: malformed offset for %s (offset=%d size=%d slot_len=%d)",
				meta.SlotID, m.Path, localOffset, m.Size, len(slotBuf))
			continue // results[i] stays nil
		}
		dstStart := int64(i) * perItem
		wg.Add(1)
		d.jobs <- decodeJob{
			idx:             i,
			raw:             slotBuf[localOffset : localOffset+m.Size],
			dst:             d.scratch[dstStart : dstStart+perItem],
			src:             m,
			out:             &results[i],
			globalSlotStart: globalSlotStart,
			perItem:         perItem,
			w:               w,
			h:               h,
			wg:              &wg,
		}
	}
	wg.Wait()

	// Compact: keep survivors in original slot order (preserves seed
	// determinism), drop the gaps.
	out := make([]*Metadata, 0, n)
	for _, m := range results {
		if m != nil {
			out = append(out, m)
		}
	}
	// Span copied back into the slot covers every dispatched index's region.
	used := int64(n) * perItem
	return out, used
}

// packRGB writes a w*h*3 RGB tensor (HWC, byte-per-channel) into dst, reading
// from an RGBA buffer (4 bytes/pixel). Used to drop the alpha channel before
// the data goes to Python.
func packRGB(dst []byte, rgba []byte, w, h int) {
	n := w * h
	for i := 0; i < n; i++ {
		s := i * 4
		d := i * 3
		dst[d+0] = rgba[s+0] // R
		dst[d+1] = rgba[s+1] // G
		dst[d+2] = rgba[s+2] // B
	}
}

// isJPEG returns true if `b` starts with the JPEG SOI marker. Lets us skip
// image.Decode's format-sniffing+registry lookup for the common case.
func isJPEG(b []byte) bool {
	return len(b) >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF
}
