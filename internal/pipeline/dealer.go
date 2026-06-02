package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"syscall"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/metrics"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/shm"
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

type Metadata struct {
	index.Metadata

	SlotID int `json:"slot_id"`
}

type SlotMeta struct {
	Objects []*Metadata
	SlotID  int
}

type Batch struct {
	Items []*Metadata `json:"items"`
	// Generation is the snapshot generation this batch was served from (feature
	// F1). It is constant for every batch of one epoch (all workers share the
	// session's pinned snapshot); the Python client asserts this to detect any
	// torn read across a concurrent mutation.
	Generation uint64 `json:"generation"`
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
	encoder := json.NewEncoder(pipeFile)

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
				encoder.Encode(Batch{Items: []*Metadata{}, Generation: gen})
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

			batch := Batch{Items: shadowPool[i:end], Generation: gen}
			if err := encoder.Encode(batch); err != nil {
				return
			}
			metrics.DealerBatchesSentTotal.Add(1)
			metrics.SamplesEmittedTotal.Add(int64(end - i))
		}

		if isEOF {
			log.Printf("[Dealer] Отправили команду окончания")
			encoder.Encode(Batch{Items: []*Metadata{}, Generation: gen})
			metrics.EpochsCompletedTotal.Add(1)
			return
		}
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
