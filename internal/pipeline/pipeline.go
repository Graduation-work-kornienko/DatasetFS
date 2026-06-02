package pipeline

import (
	"context"
	"fmt"
	"log"
	"math/rand/v2"
	"runtime"
	"sync"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/shm"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
)

const SocketPort = ":51891"

// DecodeMode controls whether the daemon serves raw shard bytes (legacy path)
// or pre-decoded image tensors. Server-side decode moves JPEG decoding + resize
// off the Python critical path — profiling shows PIL.decode + PIL.resize were
// ~83% of per-sample Python time, while daemon CPU sat at 0.6% utilization.
type DecodeMode string

const (
	// DecodeRaw: slot contains the raw shard bytes as read from disk; Python
	// must decode each sample itself. Backwards-compatible default.
	DecodeRaw DecodeMode = "raw"

	// DecodeRGBUint8: daemon decodes each JPEG and resizes to
	// (image_size, image_size, 3) uint8 HWC; Python receives ready-to-tensor
	// bytes via the same SHM slot.
	DecodeRGBUint8 DecodeMode = "rgb_uint8"
)

// DecodeConfig is the per-session decode policy. ImageSize is only meaningful
// when Mode requires resize (rgb_uint8); ignored for raw.
type DecodeConfig struct {
	Mode      DecodeMode
	ImageSize int
	// Parallelism is the number of decode worker goroutines per pipeline.
	// 0 = auto (resolved in NewPipeline to max(1, NumCPU/NumWorkers) so the
	// total decode goroutines across all pipelines does not oversubscribe
	// cores). Only meaningful for server-side decode modes.
	Parallelism int
}

// IsServerSide reports whether the decoder stage must run between Loader and Dealer.
func (d DecodeConfig) IsServerSide() bool {
	return d.Mode == DecodeRGBUint8
}

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
	// Decode is the per-session decode policy. Zero value (Mode=="") is treated
	// as DecodeRaw by NewPipeline.
	Decode DecodeConfig
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

// resolveParallelism picks the decode worker count per pipeline. An explicit
// requested value (>0) wins. Otherwise auto = NumCPU/numWorkers (floored at 1),
// so the total decode goroutines across all pipelines (numWorkers*K) stays
// near the core count rather than oversubscribing.
func resolveParallelism(requested, numWorkers int) int {
	if requested > 0 {
		return requested
	}
	if numWorkers < 1 {
		numWorkers = 1
	}
	k := runtime.NumCPU() / numWorkers
	if k < 1 {
		k = 1
	}
	return k
}

type Pipeline struct {
	cfg     WorkerConfig
	planner *Planner
	loader  *BackgroundLoader

	ctx    context.Context
	cancel context.CancelFunc
	// wg tracks every goroutine that touches shared memory (loader, decoder,
	// planner, dealer + the planner's scheduling goroutine). Stop() waits on it
	// so the allocator is never unmapped while a stage still references a slot.
	wg sync.WaitGroup
}

func NewPipeline(
	snap *index.Snapshot,
	strg *storage.Storage,
	alloc *shm.Allocator,
	cfg WorkerConfig,
) *Pipeline {

	loaderChan := make(chan *LoadJob, 100)
	freeSlotChan := make(chan int, 100)
	metaChan := make(chan *SlotMeta, 100)

	planner := NewPlanner(snap, alloc, loaderChan, freeSlotChan, cfg)
	loader := NewBackgroundLoader(strg, alloc, loaderChan, metaChan, freeSlotChan)

	ctx, cancel := context.WithCancel(context.Background())

	p := &Pipeline{
		cfg:     cfg,
		planner: planner,
		loader:  loader,
		ctx:     ctx,
		cancel:  cancel,
	}

	log.Printf("[Pipeline w=%d] Запуск фоновых воркеров Data Plane (slots [%d,%d) pipe=%s decode=%s)",
		cfg.WorkerID, cfg.SlotStart, cfg.SlotEnd, cfg.PipePath, cfg.Decode.Mode)

	p.goTracked(func() { planner.WatchRefCounts(ctx) })
	p.goTracked(func() { loader.Launch(ctx) })

	// Wire the dealer's input either directly to the loader (raw mode) or
	// through a decoder stage that rewrites the slot to packed RGB uint8.
	dealerIn := metaChan
	if cfg.Decode.IsServerSide() {
		k := resolveParallelism(cfg.Decode.Parallelism, cfg.NumWorkers)
		log.Printf("[Pipeline w=%d] decode parallelism = %d", cfg.WorkerID, k)
		decodedChan := make(chan *SlotMeta, 100)
		dec := NewDecoder(cfg.Decode, alloc, metaChan, decodedChan, freeSlotChan, k)
		p.goTracked(func() { dec.Launch(ctx) })
		dealerIn = decodedChan
	}
	p.goTracked(func() { DealerWorker(ctx, dealerIn, alloc, cfg.PipePath, cfg.DealerRand(), snap.Gen) })

	return p
}

// goTracked runs fn in a goroutine counted by p.wg so Stop() can join it.
func (p *Pipeline) goTracked(fn func()) {
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		fn()
	}()
}

func (p *Pipeline) Initiate() error {
	log.Printf("[Pipeline w=%d] Получен сигнал Initiate! Запуск эпохи...", p.cfg.WorkerID)
	return p.planner.Initiate(p.ctx, &p.wg)
}

// Stop cancels the pipeline and BLOCKS until all its goroutines have exited.
// Callers (session teardown) rely on this so the shared-memory allocator is
// only unmapped after every stage has stopped touching it — otherwise an
// in-flight decode/load would read or write freed memory (SIGSEGV).
func (p *Pipeline) Stop() {
	log.Printf("[Pipeline w=%d] Остановка конвейера...", p.cfg.WorkerID)
	p.cancel()
	p.wg.Wait()
}
