package pipeline

import (
	"bytes"
	"context"
	"image"
	_ "image/png" // register PNG decoder for image.Decode dispatch
	"log"

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
// The JPEG decoder itself is swappable via build tag:
//   - default (cgo on)            → libjpeg-turbo via TurboJPEG API (fast)
//   - `-tags datasetfs_purego`    → stdlib image/jpeg (slow, no cgo)
//
// See decoder_libjpeg.go and decoder_purego.go for the two implementations.
//
// Memory: a per-decoder scratch buffer (sized SlotSize) accumulates decoded
// items before a single copy back into the SHM slot. This keeps SHM contents
// consistent for the consumer — readers never see a half-rewritten slot.
type Decoder struct {
	cfg          DecodeConfig
	allocator    *shm.Allocator
	in           <-chan *SlotMeta
	out          chan<- *SlotMeta
	freeSlotChan chan<- int
	scratch      []byte
	jpegDec      jpegDecoder
}

// jpegDecoder is implemented by both decoder_purego.go (stdlib) and
// decoder_libjpeg.go (cgo + TurboJPEG). One instance per Decoder goroutine;
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
) *Decoder {
	return &Decoder{
		cfg:          cfg,
		allocator:    alloc,
		in:           in,
		out:          out,
		freeSlotChan: freeSlots,
		scratch:      make([]byte, shm.SlotSize),
		jpegDec:      newJPEGDecoder(),
	}
}

// Launch consumes SlotMetas from the loader, decodes each item, and emits new
// SlotMetas pointing at the packed RGB bytes in the same slot. Closes `out`
// on input-channel close so the dealer correctly detects end-of-epoch.
func (d *Decoder) Launch(ctx context.Context) {
	defer close(d.out)
	defer d.jpegDec.Close()

	w := d.cfg.ImageSize
	h := d.cfg.ImageSize
	perItem := int64(w * h * 3)

	// One reusable RGBA buffer for resize output — re-zeroed per item by Scale.
	resized := image.NewRGBA(image.Rect(0, 0, w, h))

	for {
		select {
		case <-ctx.Done():
			return
		case meta, ok := <-d.in:
			if !ok {
				return
			}
			out, used := d.decodeSlot(meta, resized, perItem, w, h)
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
			slotBuf := d.allocator.GetSlotBuffer(meta.SlotID)
			copy(slotBuf[:used], d.scratch[:used])

			select {
			case <-ctx.Done():
				return
			case d.out <- &SlotMeta{Objects: out, SlotID: meta.SlotID}:
			}
		}
	}
}

// decodeSlot decodes/resizes every Metadata in `meta` into d.scratch, packed
// contiguously. Returns the resulting Metadata list (offsets/sizes rewritten
// for the packed layout) and the number of bytes used in scratch.
func (d *Decoder) decodeSlot(
	meta *SlotMeta,
	resized *image.RGBA,
	perItem int64,
	w, h int,
) ([]*Metadata, int64) {
	slotBuf := d.allocator.GetSlotBuffer(meta.SlotID)
	globalSlotStart := int64(meta.SlotID * shm.SlotSize)

	cursor := int64(0)
	out := make([]*Metadata, 0, len(meta.Objects))
	for _, m := range meta.Objects {
		if cursor+perItem > int64(len(d.scratch)) {
			// Decoded data won't fit. With imagenette-scale (96×96 = 27 KB)
			// and ~1000 items/shard this is ~27 MB into a 110 MB slot, so
			// only triggers for very large image_size or huge shards.
			log.Printf("[Decoder] slot %d: scratch full at item %s (decoded %d items, dropping %d more)",
				meta.SlotID, m.Path, len(out), len(meta.Objects)-len(out))
			break
		}
		localOffset := m.Offset - globalSlotStart
		if localOffset < 0 || localOffset+m.Size > int64(len(slotBuf)) {
			log.Printf("[Decoder] slot %d: malformed offset for %s (offset=%d size=%d slot_len=%d)",
				meta.SlotID, m.Path, localOffset, m.Size, len(slotBuf))
			continue
		}
		rawBytes := slotBuf[localOffset : localOffset+m.Size]

		// Fast-path JPEG via the swappable backend; non-JPEG falls through to
		// stdlib image.Decode (PNG etc. via the blank import above).
		var img image.Image
		var err error
		if isJPEG(rawBytes) {
			img, err = d.jpegDec.Decode(rawBytes)
		} else {
			img, _, err = image.Decode(bytes.NewReader(rawBytes))
		}
		if err != nil {
			log.Printf("[Decoder] decode failed for %s: %v", m.Path, err)
			continue
		}

		draw.BiLinear.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Src, nil)
		packRGB(d.scratch[cursor:cursor+perItem], resized.Pix, w, h)

		// New Metadata: offset/size rewritten to the packed layout. Other
		// fields (Path, ObjectMetadata, ShardID) preserved so downstream
		// (Dealer → Python) sees the same semantic identifiers.
		m2 := *m
		m2.Offset = globalSlotStart + cursor
		m2.Size = perItem
		out = append(out, &m2)
		cursor += perItem
	}
	return out, cursor
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
