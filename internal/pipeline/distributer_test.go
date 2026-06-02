package pipeline

import "testing"

// The one property that justifies F2: across every (rank, worker) reader the
// shard partitions are disjoint and their union is the whole dataset exactly
// once. If this fails, two DDP ranks would train on the same sample.
func TestDistributer_DisjointAndComplete(t *testing.T) {
	for _, numShards := range []int{1, 5, 9, 11, 64} {
		for _, worldSize := range []int{1, 2, 3, 4} {
			for _, numWorkers := range []int{1, 2, 4} {
				seen := make([]int, numShards)
				for rank := 0; rank < worldSize; rank++ {
					for wID := 0; wID < numWorkers; wID++ {
						d := NewDistributer(WorkerConfig{
							WorkerID:   wID,
							NumWorkers: numWorkers,
							Rank:       rank,
							WorldSize:  worldSize,
						})
						for i := 0; i < numShards; i++ {
							if d.Owns(i) {
								seen[i]++
							}
						}
					}
				}
				for i, c := range seen {
					if c != 1 {
						t.Fatalf("shards=%d world=%d workers=%d: shard %d owned %d times (want 1)",
							numShards, worldSize, numWorkers, i, c)
					}
				}
			}
		}
	}
}

// WorldSize<=1 must reduce exactly to the legacy per-worker rule
// (i % NumWorkers == WorkerID), so the single-process path is unchanged.
func TestDistributer_DegeneratesToLegacy(t *testing.T) {
	for _, ws := range []int{0, 1} { // 0 (zero value) normalizes to 1
		for numWorkers := 1; numWorkers <= 9; numWorkers++ {
			for wID := 0; wID < numWorkers; wID++ {
				d := NewDistributer(WorkerConfig{
					WorkerID: wID, NumWorkers: numWorkers, WorldSize: ws,
				})
				for i := 0; i < 50; i++ {
					want := i%numWorkers == wID
					if d.Owns(i) != want {
						t.Fatalf("ws=%d workers=%d w=%d i=%d: Owns=%v want=%v",
							ws, numWorkers, wID, i, d.Owns(i), want)
					}
				}
			}
		}
	}
}
