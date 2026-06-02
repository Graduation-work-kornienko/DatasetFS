package pipeline

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"syscall"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/metrics"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/shm"
)

// Binary wire protocol (opt 03). Replaces the legacy newline-delimited JSON
// batches over the named pipe. Motivation: server-side decode (opt 01/02) only
// helps images; for audio / any cheap-decode type the JSON encode (here) +
// json.loads (Python) + per-item dict build became the dominant per-sample cost.
// The frame is a compact length-prefixed binary message parsed in Python with a
// single vectorized numpy.frombuffer over the fixed columnar block.
//
// Layout (little-endian), one Write to the pipe per batch:
//
//	HEADER    magic u32 | total_len u32 | generation u64 | item_count u32 | blob_len u32
//	COLUMNAR  item_count × ItemWireSize, struct-of-arrays per item:
//	            slot_id i32 | offset i64 | size i64 | path_len u32 | meta_len u32
//	BLOBS     per item, in order: path (utf8) bytes, then meta (raw JSON) bytes
//
// total_len counts the bytes AFTER the 8-byte (magic+total_len) prefix, i.e.
// header-rest (16) + columnar + blobs, so Python reads 8 bytes then total_len
// more. The end-of-epoch signal is a frame with item_count==0 (blob_len==0).
// Dead JSON fields (ShardID/c_id, Deleted) are dropped — unused by the client.
const (
	// frameMagic ("DFS1") is a version/sentinel: a mismatch on the Python side
	// means a Go/Python protocol skew and fails loudly rather than misreading.
	frameMagic uint32 = 0x44465331
	// ItemWireSize is the fixed columnar record width. MUST equal the Python
	// ITEM_DTYPE.itemsize in clients/python/dataset_fs.py (asserted there).
	ItemWireSize = 28
	frameHdrLen  = 8  // magic + total_len
	frameRestLen = 16 // generation + item_count + blob_len
)

// shufflePool shuffles in place, using the given rng if non-nil, else the
// package-level global rand.
func shufflePool(rng *rand.Rand, pool []*Metadata) {
	swap := func(i, j int) { pool[i], pool[j] = pool[j], pool[i] }
	if rng != nil {
		rng.Shuffle(len(pool), swap)
	} else {
		rand.Shuffle(len(pool), swap)
	}
}

// Metadata is the pipeline's per-object record. The json tags are vestigial
// (the wire protocol is now binary — see encodeFrame); they remain only because
// index.Metadata is also (de)serialized for the on-disk manifest. The frame's
// `generation` field (constant for a whole epoch across all workers) lets the
// Python client detect a torn read across a concurrent mutation (feature F1).
type Metadata struct {
	index.Metadata

	SlotID int `json:"slot_id"`
}

type SlotMeta struct {
	Objects []*Metadata
	SlotID  int
}

func DealerWorker(
	ctx context.Context,
	metaIn <-chan *SlotMeta,
	allocator *shm.Allocator,
	pipePath string,
	rng *rand.Rand,
	gen uint64,
) {
	// WindowSize bounds how many SlotMetas we *may* merge for one emit cycle
	// (for cross-shard shuffling), but we never BLOCK waiting to reach it —
	// only the FIRST SlotMeta is awaited, then we drain whatever else is
	// already available. This avoids deadlock when a worker has fewer slots
	// than shards (slots can't be freed until Python consumes a batch, which
	// requires the dealer to emit).
	const WindowSize = 3
	if err := ensurePipe(pipePath); err != nil {
		log.Fatalf("[Dealer] Критическая ошибка: %v", err)
	}
	pipeFile, err := os.OpenFile(pipePath, os.O_WRONLY, os.ModeNamedPipe)
	if err != nil {
		log.Printf("[Dealer] Ошибка открытия трубы: %v", err)
		return
	}
	defer pipeFile.Close()
	// Reused across batches: the dealer is single-goroutine per worker, so no
	// locking is needed. Avoids a per-batch allocation on the hot path.
	var frameBuf bytes.Buffer
	emit := func(items []*Metadata) error {
		encodeFrame(&frameBuf, items, gen)
		_, err := pipeFile.Write(frameBuf.Bytes())
		return err
	}

	// On session teardown a consumer that stopped early (e.g. training capped
	// at max_batches) leaves us blocked in encoder.Encode on a full pipe.
	// Closing the pipe on ctx cancellation unblocks that write so this worker
	// returns promptly — required so the allocator is not unmapped while we
	// (or upstream stages joined with us) still touch shared memory.
	stopWatch := make(chan struct{})
	defer close(stopWatch)
	go func() {
		select {
		case <-ctx.Done():
			pipeFile.Close()
		case <-stopWatch:
		}
	}()

	for {
		var shadowPool []*Metadata
		isEOF := false

		// Block on the first SlotMeta of this batch — ctx cancellation also OK.
		select {
		case <-ctx.Done():
			return
		case slotMeta, ok := <-metaIn:
			if !ok {
				log.Printf("[Dealer] Канал закрыт, эпоха завершена")
				emit(nil) // empty frame = end of epoch
				metrics.EpochsCompletedTotal.Add(1)
				return
			}
			log.Printf("[Dealer] Пришел слот %d", slotMeta.SlotID)
			shadowPool = append(shadowPool, slotMeta.Objects...)
			allocator.SetRefCount(slotMeta.SlotID, int32(len(slotMeta.Objects)))
		}

		// Drain whatever else is *immediately* available, up to WindowSize total.
		// Non-blocking: as soon as no SlotMeta is ready, stop and emit.
	drain:
		for i := 1; i < WindowSize; i++ {
			select {
			case slotMeta, ok := <-metaIn:
				if !ok {
					isEOF = true
					break drain
				}
				log.Printf("[Dealer] Дренировали слот %d", slotMeta.SlotID)
				shadowPool = append(shadowPool, slotMeta.Objects...)
				allocator.SetRefCount(slotMeta.SlotID, int32(len(slotMeta.Objects)))
			default:
				break drain
			}
		}

		log.Printf("[Dealer] ✅ Окно размера %d (eof=%v)", len(shadowPool), isEOF)

		shufflePool(rng, shadowPool)

		const BatchSize = 256
		for i := 0; i < len(shadowPool); i += BatchSize {
			end := i + BatchSize
			if end > len(shadowPool) {
				end = len(shadowPool)
			}

			if err := emit(shadowPool[i:end]); err != nil {
				return
			}
			metrics.DealerBatchesSentTotal.Add(1)
			metrics.SamplesEmittedTotal.Add(int64(end - i))
		}

		if isEOF {
			log.Printf("[Dealer] Отправили команду окончания")
			emit(nil) // empty frame = end of epoch
			metrics.EpochsCompletedTotal.Add(1)
			return
		}
	}
}

// encodeFrame serializes one batch of items into buf using the binary wire
// protocol documented above. buf is reset first, so the caller may reuse it.
// An empty items slice produces a valid end-of-epoch frame (item_count==0).
func encodeFrame(buf *bytes.Buffer, items []*Metadata, gen uint64) {
	buf.Reset()

	blobLen := 0
	for _, it := range items {
		blobLen += len(it.Path) + len(it.ObjectMetadata)
	}
	totalLen := frameRestLen + len(items)*ItemWireSize + blobLen

	var hdr [frameHdrLen + frameRestLen]byte
	binary.LittleEndian.PutUint32(hdr[0:4], frameMagic)
	binary.LittleEndian.PutUint32(hdr[4:8], uint32(totalLen))
	binary.LittleEndian.PutUint64(hdr[8:16], gen)
	binary.LittleEndian.PutUint32(hdr[16:20], uint32(len(items)))
	binary.LittleEndian.PutUint32(hdr[20:24], uint32(blobLen))
	buf.Write(hdr[:])

	// Columnar block: one fixed-width record per item (struct-of-arrays so the
	// Python side reads it with a single numpy.frombuffer).
	var rec [ItemWireSize]byte
	for _, it := range items {
		binary.LittleEndian.PutUint32(rec[0:4], uint32(int32(it.SlotID)))
		binary.LittleEndian.PutUint64(rec[4:12], uint64(it.Offset))
		binary.LittleEndian.PutUint64(rec[12:20], uint64(it.Size))
		binary.LittleEndian.PutUint32(rec[20:24], uint32(len(it.Path)))
		binary.LittleEndian.PutUint32(rec[24:28], uint32(len(it.ObjectMetadata)))
		buf.Write(rec[:])
	}

	// Variable block: path then meta for each item, in the same order.
	for _, it := range items {
		buf.WriteString(it.Path)
		buf.Write(it.ObjectMetadata)
	}
}

// ensurePipe created named pipe if not exists
func ensurePipe(pipePath string) error {
	info, err := os.Stat(pipePath)
	if err == nil {
		if info.Mode()&os.ModeNamedPipe == 0 {
			return fmt.Errorf("файл %s существует, но это не Named Pipe", pipePath)
		}
		return nil
	}

	log.Printf("[Dealer] Создаю Named Pipe: %s", pipePath)
	err = syscall.Mkfifo(pipePath, 0666)
	if err != nil {
		return fmt.Errorf("ошибка создания FIFO pipe: %w", err)
	}

	return nil
}
