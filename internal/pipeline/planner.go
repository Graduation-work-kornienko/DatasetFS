package pipeline

import (
	"context"
	"log"
	"math/rand/v2"
	"sort"
	"time"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/shm"
)

type Planner struct {
	cfg          WorkerConfig
	rng          *rand.Rand // nil = use global rand (non-deterministic)
	coreIndex    *index.CoreIndex
	allocator    *shm.Allocator
	loaderChan   chan *LoadJob
	freeSlotChan chan int
}

type LoadJob struct {
	ShardID int
	SlotID  int
	Shard   *index.Shard
}

func NewPlanner(idx *index.CoreIndex, alloc *shm.Allocator, loaderChan chan *LoadJob, freeSlots chan int, cfg WorkerConfig) *Planner {
	return &Planner{
		cfg:          cfg,
		rng:          cfg.PlannerRand(),
		coreIndex:    idx,
		allocator:    alloc,
		loaderChan:   loaderChan,
		freeSlotChan: freeSlots,
	}
}

// shardsForWorker returns the shard IDs assigned to this worker, in shuffled
// order. Sharding is by index in the sorted shard ID list (not by raw ID),
// since ShardMap keys can be sparse.
func (p *Planner) shardsForWorker() []int {
	shards := p.coreIndex.AllShards()
	allIDs := make([]int, 0, len(shards))
	for id := range shards {
		if id == -1 {
			continue
		}
		allIDs = append(allIDs, id)
	}
	sort.Ints(allIDs)

	var mine []int
	for i, id := range allIDs {
		if i%p.cfg.NumWorkers == p.cfg.WorkerID {
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

func (p *Planner) Initiate(ctx context.Context) error {
	shards := p.coreIndex.AllShards()
	myShardIDs := p.shardsForWorker()
	log.Printf("[Planner w=%d] %d shards assigned", p.cfg.WorkerID, len(myShardIDs))

	go func() {
		// Closing loaderChan signals BackgroundLoader that no more jobs are
		// coming, which lets it close metadataChan, which lets DealerWorker
		// drain its window and emit the end-of-epoch batch.
		defer close(p.loaderChan)

		for _, sID := range myShardIDs {
			select {
			case <-ctx.Done():
				return
			case targetSlot := <-p.freeSlotChan:
				log.Printf("[Planner w=%d] got free slot %d for shard %d", p.cfg.WorkerID, targetSlot, sID)
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
	// quickly so the next shard can load.
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			for i := p.cfg.SlotStart; i < p.cfg.SlotEnd; i++ {
				d := p.allocator.ReadRefCount(i)
				if d == 0 && !slotIsFree[i] {
					log.Printf("[Planner w=%d] slot %d is going to be used", p.cfg.WorkerID, i)
					slotIsFree[i] = true
					p.freeSlotChan <- i
				} else if d != 0 && slotIsFree[i] {
					slotIsFree[i] = false
				}
			}
		}
	}
}
