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

	log.Printf("[Pipeline w=%d] Запуск фоновых воркеров Data Plane (slots [%d,%d) pipe=%s decode=%s)",
		cfg.WorkerID, cfg.SlotStart, cfg.SlotEnd, cfg.PipePath, cfg.Decode.Mode)

	go planner.WatchRefCounts(ctx)
	go loader.Launch(ctx)

	// Wire the dealer's input either directly to the loader (raw mode) or
	// through a decoder stage that rewrites the slot to packed RGB uint8.
	dealerIn := metaChan
	if cfg.Decode.IsServerSide() {
		decodedChan := make(chan *SlotMeta, 100)
		dec := NewDecoder(cfg.Decode, alloc, metaChan, decodedChan, freeSlotChan)
		go dec.Launch(ctx)
		dealerIn = decodedChan
	}
	go DealerWorker(ctx, dealerIn, alloc, cfg.PipePath, cfg.DealerRand())

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
