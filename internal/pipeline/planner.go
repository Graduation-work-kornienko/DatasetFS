package pipeline

import (
	"context"
	"log"
	"math/rand/v2"
	"sort"
	"sync"
	"time"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/metrics"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/shm"
)

// refCountPollInterval is how often WatchRefCounts scans SHM refcounts for
// slots freed by the consumer. Kept tight (was 100 ms in Phase 3): each scan
// is a handful of atomic loads, and the interval is a hard floor on slot-reuse
// latency on the critical path. See opt 02.
const refCountPollInterval = 2 * time.Millisecond

type Planner struct {
	cfg          WorkerConfig
	rng          *rand.Rand // nil = use global rand (non-deterministic)
	snap         *index.Snapshot
	allocator    *shm.Allocator
	loaderChan   chan *LoadJob
	freeSlotChan chan int
}

type LoadJob struct {
	ShardID int
	SlotID  int
	Shard   *index.ShardSnap
}

// NewPlanner builds a planner bound to an immutable Snapshot pinned for this
// session (feature F1). The planner reads only the snapshot, so concurrent
// mutations (which publish a new generation) cannot change what this epoch sees.
func NewPlanner(snap *index.Snapshot, alloc *shm.Allocator, loaderChan chan *LoadJob, freeSlots chan int, cfg WorkerConfig) *Planner {
	return &Planner{
		cfg:          cfg,
		rng:          cfg.PlannerRand(),
		snap:         snap,
		allocator:    alloc,
		loaderChan:   loaderChan,
		freeSlotChan: freeSlots,
	}
}

// shardsForWorker returns the shard IDs assigned to this worker, in shuffled
// order. Sharding is by index in the sorted shard ID list (not by raw ID),
// since ShardMap keys can be sparse.
func (p *Planner) shardsForWorker() []int {
	shards := p.snap.Shards
	allIDs := make([]int, 0, len(shards))
	for id, ss := range shards {
		// Delta shards are served only once they actually hold added files
		// (a pinned generation with no mutations has empty deltas). Base
		// shards are always scheduled, even if fully deleted — the loader frees
		// the slot when no live objects remain.
		if (ss.Type == index.Delta || id < 0) && len(ss.Objects) == 0 {
			continue
		}
		allIDs = append(allIDs, id)
	}
	sort.Ints(allIDs)

	// Owns() generalizes per-worker round-robin to the DDP rank dimension
	// (feature F2). With WorldSize<=1 it is exactly i%NumWorkers==WorkerID, so
	// the non-distributed path is unchanged.
	dist := NewDistributer(p.cfg)
	var mine []int
	for i, id := range allIDs {
		if dist.Owns(i) {
			mine = append(mine, id)
		}
	}
	swap := func(i, j int) { mine[i], mine[j] = mine[j], mine[i] }
	if p.rng != nil {
		p.rng.Shuffle(len(mine), swap)
	} else {
		rand.Shuffle(len(mine), swap)
	}
	return mine
}

func (p *Planner) Initiate(ctx context.Context, wg *sync.WaitGroup) error {
	shards := p.snap.Shards
	myShardIDs := p.shardsForWorker()
	log.Printf("[Planner w=%d] %d shards assigned", p.cfg.WorkerID, len(myShardIDs))

	wg.Add(1)
	go func() {
		defer wg.Done()
		// Closing loaderChan signals BackgroundLoader that no more jobs are
		// coming, which lets it close metadataChan, which lets DealerWorker
		// drain its window and emit the end-of-epoch batch.
		defer close(p.loaderChan)

		for _, sID := range myShardIDs {
			// Try non-blocking first to detect starvation (planner waiting
			// for consumer to free a slot) — informative bottleneck signal.
			var targetSlot int
			select {
			case <-ctx.Done():
				return
			case targetSlot = <-p.freeSlotChan:
			default:
				metrics.SlotStarvationTotal.Add(1)
				select {
				case <-ctx.Done():
					return
				case targetSlot = <-p.freeSlotChan:
				}
			}
			job := &LoadJob{
				ShardID: sID,
				SlotID:  targetSlot,
				Shard:   shards[sID],
			}
			select {
			case <-ctx.Done():
				return
			case p.loaderChan <- job:
			}
		}
	}()

	return nil
}

func (p *Planner) WatchRefCounts(ctx context.Context) {
	slotIsFree := make([]bool, shm.NumSlots)

	// Seed only this worker's slot range as initially free.
	for i := p.cfg.SlotStart; i < p.cfg.SlotEnd; i++ {
		slotIsFree[i] = true
		p.freeSlotChan <- i
	}

	// Poll fast — slot reuse is on the critical path. When Python decrements
	// a refcount to 0, we want to surface a free slot back to the planner
	// quickly so the next shard can load. ReadRefCount is an atomic load over
	// ≤9 SHM ints, so a tight interval costs negligible CPU; the old 100 ms
	// tick added up to 100 ms of slot-recycle latency per cycle, which starved
	// the consumer (especially with few slots/worker). See opt 02.
	ticker := time.NewTicker(refCountPollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			for i := p.cfg.SlotStart; i < p.cfg.SlotEnd; i++ {
				d := p.allocator.ReadRefCount(i)
				if d == 0 && !slotIsFree[i] {
					slotIsFree[i] = true
					p.freeSlotChan <- i
				} else if d != 0 && slotIsFree[i] {
					slotIsFree[i] = false
				}
			}
		}
	}
}
