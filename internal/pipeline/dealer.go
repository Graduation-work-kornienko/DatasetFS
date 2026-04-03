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
	"github.com/Graduation-work-kornienko/DatasetFS/internal/shm"
)

const (
	pipePath = "/tmp/datasetfs_pipe"
)

type SlotMeta struct {
	Objects []*index.Metadata
	SlotID  int
}

type Batch struct {
	Items []*index.Metadata
}

func DealerWorker(
	ctx context.Context,
	metaIn <-chan *SlotMeta,
	allocator *shm.Allocator,
) {
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

	for {
		var shadowPool []*index.Metadata
		isEOF := false

		for i := 0; i < WindowSize; i++ {
			select {
			case <-ctx.Done():
				return
			case slotMeta, ok := <-metaIn:
				if !ok {
					isEOF = true
					break
				}
				shadowPool = append(shadowPool, slotMeta.Objects...)

				allocator.SetRefCount(slotMeta.SlotID, int32(len(slotMeta.Objects)))
			}
			if isEOF {
				break
			}
		}

		if len(shadowPool) == 0 {
			// signal for dataloader - stop
			encoder.Encode(Batch{Items: []*index.Metadata{}})
			return
		}

		rand.Shuffle(len(shadowPool), func(i, j int) {
			shadowPool[i], shadowPool[j] = shadowPool[j], shadowPool[i]
		})

		const BatchSize = 256
		for i := 0; i < len(shadowPool); i += BatchSize {
			end := i + BatchSize
			if end > len(shadowPool) {
				end = len(shadowPool)
			}

			batch := Batch{
				Items: shadowPool[i:end],
			}

			// TODO: remove named pipe and make ring buffer in shared memory for batches

			if err := encoder.Encode(batch); err != nil {
				return
			}
		}

		if isEOF {
			encoder.Encode(Batch{Items: []*index.Metadata{}})
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
