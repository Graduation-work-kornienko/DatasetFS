package pipeline

import (
	"context"
	"io"
	"log"
	"os"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/shm"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
)

type BackgroundLoader struct {
	storage      *storage.Storage
	allocator    *shm.Allocator
	loaderChan   <-chan *LoadJob
	metadataChan chan<- *SlotMeta
}

func NewBackgroundLoader(strg *storage.Storage, alloc *shm.Allocator, req <-chan *LoadJob, res chan<- *SlotMeta) *BackgroundLoader {
	return &BackgroundLoader{
		storage:      strg,
		allocator:    alloc,
		loaderChan:   req,
		metadataChan: res,
	}
}

func (b *BackgroundLoader) Launch(ctx context.Context) {

	for {
		select {
		case <-ctx.Done():
			return
		case job := <-b.loaderChan:

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

			var validMeta []*index.Metadata
			globalSlotStartOffset := int64(job.SlotID * shm.SlotSize)

			for _, meta := range job.Shard.Objects {
				if !meta.Deleted {
					localMeta := *meta

					localMeta.Offset = globalSlotStartOffset + meta.Offset
					localMeta.SlotID = job.SlotID

					validMeta = append(validMeta, &localMeta)
				}
			}

			log.Printf("[Loader] ✅ Загружен Слот %d (Валидных файлов: %d). Передаю ディлеру.", job.SlotID, len(validMeta))

			b.metadataChan <- &SlotMeta{
				Objects: validMeta,
				SlotID:  job.SlotID,
			}
		}
	}
}
