package pipeline

import (
	"context"
	"log"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/shm"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
)

const SocketPort = ":51891"

type Pipeline struct {
	planner *Planner
	loader  *BackgroundLoader

	ctx    context.Context
	cancel context.CancelFunc
}

func NewPipeline(
	coreIdx *index.CoreIndex,
	strg *storage.Storage,
	alloc *shm.Allocator,
) *Pipeline {

	loaderChan := make(chan *LoadJob, 100)
	freeSlotChan := make(chan int, 100)
	metaChan := make(chan *SlotMeta, 100)

	planner := NewPlanner(coreIdx, alloc, loaderChan, freeSlotChan)
	loader := NewBackgroundLoader(strg, alloc, loaderChan, metaChan, freeSlotChan)

	ctx, cancel := context.WithCancel(context.Background())

	p := &Pipeline{
		planner: planner,
		loader:  loader,
		ctx:     ctx,
		cancel:  cancel,
	}

	log.Println("[Pipeline] Запуск фоновых воркеров Data Plane...")

	go planner.WatchRefCounts(ctx)

	go loader.Launch(ctx)

	go DealerWorker(ctx, metaChan, alloc)

	return p
}

func (p *Pipeline) Initiate() error {
	log.Println("[Pipeline] Получен сигнал Initiate! Запуск эпохи...")
	return p.planner.Initiate(p.ctx)
}

func (p *Pipeline) Stop() {
	log.Println("[Pipeline] Остановка конвейера...")
	p.cancel()
}
