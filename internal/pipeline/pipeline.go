package pipeline

import (
	"context"
	"fmt"
	"log"
	"math/rand/v2"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/shm"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
)

const SocketPort = ":51891"

type WorkerConfig struct {
	WorkerID   int
	NumWorkers int
	SlotStart  int
	SlotEnd    int
	PipePath   string
	// Seed: nil = non-deterministic (global rand). Otherwise the master seed
	// from /initialize_loading; each pipeline derives planner+dealer RNGs from
	// it via (Seed, WorkerID) so different workers get independent streams.
	Seed *uint64
}

// PlannerRand returns a seeded RNG for this worker's planner, or nil to use
// the global package-level rand. Deterministic given (Seed, WorkerID).
func (c WorkerConfig) PlannerRand() *rand.Rand {
	if c.Seed == nil {
		return nil
	}
	return rand.New(rand.NewPCG(*c.Seed, uint64(c.WorkerID)*2))
}

// DealerRand uses an offset (WorkerID*2+1) so the dealer's shadowPool shuffle
// is decorrelated from the planner's shard-order shuffle within the same worker.
func (c WorkerConfig) DealerRand() *rand.Rand {
	if c.Seed == nil {
		return nil
	}
	return rand.New(rand.NewPCG(*c.Seed, uint64(c.WorkerID)*2+1))
}

// SlotRange returns [start, end) for worker K out of N, partitioning
// shm.NumSlots into contiguous ranges. First (NumSlots % N) workers get
// one extra slot each.
func SlotRange(workerID, numWorkers int) (start, end int) {
	base := shm.NumSlots / numWorkers
	rem := shm.NumSlots % numWorkers
	if workerID < rem {
		start = workerID * (base + 1)
		end = start + base + 1
	} else {
		start = rem*(base+1) + (workerID-rem)*base
		end = start + base
	}
	return
}

func PipePath(workerID int) string {
	return fmt.Sprintf("/tmp/datasetfs_pipe_%d", workerID)
}

type Pipeline struct {
	cfg     WorkerConfig
	planner *Planner
	loader  *BackgroundLoader

	ctx    context.Context
	cancel context.CancelFunc
}

func NewPipeline(
	coreIdx *index.CoreIndex,
	strg *storage.Storage,
	alloc *shm.Allocator,
	cfg WorkerConfig,
) *Pipeline {

	loaderChan := make(chan *LoadJob, 100)
	freeSlotChan := make(chan int, 100)
	metaChan := make(chan *SlotMeta, 100)

	planner := NewPlanner(coreIdx, alloc, loaderChan, freeSlotChan, cfg)
	loader := NewBackgroundLoader(strg, alloc, loaderChan, metaChan, freeSlotChan)

	ctx, cancel := context.WithCancel(context.Background())

	p := &Pipeline{
		cfg:     cfg,
		planner: planner,
		loader:  loader,
		ctx:     ctx,
		cancel:  cancel,
	}

	log.Printf("[Pipeline w=%d] Запуск фоновых воркеров Data Plane (slots [%d,%d) pipe=%s)",
		cfg.WorkerID, cfg.SlotStart, cfg.SlotEnd, cfg.PipePath)

	go planner.WatchRefCounts(ctx)

	go loader.Launch(ctx)

	go DealerWorker(ctx, metaChan, alloc, cfg.PipePath, cfg.DealerRand())

	return p
}

func (p *Pipeline) Initiate() error {
	log.Printf("[Pipeline w=%d] Получен сигнал Initiate! Запуск эпохи...", p.cfg.WorkerID)
	return p.planner.Initiate(p.ctx)
}

func (p *Pipeline) Stop() {
	log.Printf("[Pipeline w=%d] Остановка конвейера...", p.cfg.WorkerID)
	p.cancel()
}
