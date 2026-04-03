package pipeline

import (
	"context"
	"math/rand/v2"
	"time"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/shm"
)

type Planner struct {
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

func NewPlanner(idx *index.CoreIndex, alloc *shm.Allocator, loaderChan chan *LoadJob, freeSlots chan int) *Planner {
	return &Planner{
		coreIndex:    idx,
		allocator:    alloc,
		loaderChan:   loaderChan,
		freeSlotChan: freeSlots,
	}
}

func (p *Planner) Initiate(ctx context.Context) error {

	shards := p.coreIndex.AllShards()
	var shardIDs []int
	for id := range shards {
		shardIDs = append(shardIDs, id)
	}
	rand.Shuffle(len(shardIDs), func(i, j int) { shardIDs[i], shardIDs[j] = shardIDs[j], shardIDs[i] })

	go func() {
		for _, sID := range shardIDs {
			targetSlot := <-p.freeSlotChan

			job := &LoadJob{
				ShardID: sID,
				SlotID:  targetSlot,
				Shard:   shards[sID],
			}

			p.loaderChan <- job
		}
	}()

	return nil
}

func (p *Planner) WatchRefCounts(ctx context.Context) {
	// array for deduplication - not spam free about free slot until it is not used
	slotIsFree := make([]bool, shm.NumSlots)

	// first slots are free
	for i := 0; i < shm.NumSlots; i++ {
		slotIsFree[i] = true
		p.freeSlotChan <- i
	}

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			for i := 0; i < shm.NumSlots; i++ {
				if p.allocator.ReadRefCount(i) == 0 && !slotIsFree[i] {
					slotIsFree[i] = true

					p.freeSlotChan <- i
				} else if p.allocator.ReadRefCount(i) != 0 && slotIsFree[i] {
					slotIsFree[i] = false
				}
			}
		}
	}
}
