package pipeline

import (
	"context"
	"io"
	"log"
	"os"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/shm"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
)

type BackgroundLoader struct {
	storage      *storage.Storage
	allocator    *shm.Allocator
	loaderChan   <-chan *LoadJob
	metadataChan chan<- *SlotMeta
	freeSlotChan chan int
}

func NewBackgroundLoader(strg *storage.Storage, alloc *shm.Allocator, req <-chan *LoadJob, res chan<- *SlotMeta, freeSlotChan chan int) *BackgroundLoader {
	return &BackgroundLoader{
		storage:      strg,
		allocator:    alloc,
		loaderChan:   req,
		metadataChan: res,
		freeSlotChan: freeSlotChan,
	}
}

func (b *BackgroundLoader) Launch(ctx context.Context) {
	// Closing metadataChan downstream lets DealerWorker know the epoch is done.
	defer close(b.metadataChan)

	for {
		select {
		case <-ctx.Done():
			return
		case job, ok := <-b.loaderChan:
			if !ok {
				// Planner has scheduled all shards; drain done.
				return
			}

			log.Printf("[Loader] Нужно загрузить Слот %d", job.SlotID)

			shardPath := b.storage.ShardPath(job.ShardID)
			file, err := os.Open(shardPath)
			if err != nil {
				log.Printf("[Loader] ❌ Ошибка открытия шарда %d: %v", job.ShardID, err)
				continue
			}

			targetSlice := b.allocator.GetSlotBuffer(job.SlotID)

			_, err = io.ReadFull(file, targetSlice[:job.Shard.TotalSize])
			file.Close()

			if err != nil {
				log.Printf("[Loader] ❌ Ошибка io.ReadFull для шарда %d: %v", job.ShardID, err)
				continue
			}

			var validMeta []*Metadata
			globalSlotStartOffset := int64(job.SlotID * shm.SlotSize)

			for _, meta := range job.Shard.Objects {
				if !meta.Deleted {
					localMeta := Metadata{
						Metadata: *meta,
						SlotID:   job.SlotID,
					}

					localMeta.Offset = globalSlotStartOffset + meta.Offset
					localMeta.SlotID = job.SlotID

					validMeta = append(validMeta, &localMeta)
				}
			}

			log.Printf("[Loader] Загружен Слот %d , шард %d файлов %d (Валидных файлов: %d). Передаю Dealer.", job.SlotID, job.ShardID, len(job.Shard.Objects), len(validMeta))

			if len(validMeta) == 0 {
				b.freeSlotChan <- job.SlotID
				continue
			}
			log.Printf("[Loader] ✅ Загружен Слот %d (Валидных файлов: %d). Передаю Dealer.", job.SlotID, len(validMeta))

			b.metadataChan <- &SlotMeta{
				Objects: validMeta,
				SlotID:  job.SlotID,
			}
		}
	}
}
