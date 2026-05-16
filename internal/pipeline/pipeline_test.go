package pipeline

import (
	"reflect"
	"testing"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/shm"
)

func TestSlotRange_PartitionsCoverAllSlots(t *testing.T) {
	for numWorkers := 1; numWorkers <= shm.NumSlots; numWorkers++ {
		seen := make(map[int]int)
		for w := 0; w < numWorkers; w++ {
			start, end := SlotRange(w, numWorkers)
			if start < 0 || end > shm.NumSlots || start > end {
				t.Fatalf("numWorkers=%d worker=%d: bad range [%d,%d)", numWorkers, w, start, end)
			}
			for s := start; s < end; s++ {
				seen[s]++
			}
		}
		if len(seen) != shm.NumSlots {
			t.Fatalf("numWorkers=%d: covered %d/%d slots", numWorkers, len(seen), shm.NumSlots)
		}
		for slot, count := range seen {
			if count != 1 {
				t.Fatalf("numWorkers=%d slot=%d covered %d times", numWorkers, slot, count)
			}
		}
	}
}

func TestSlotRange_BalancedSplit(t *testing.T) {
	cases := []struct {
		numWorkers int
		want       []int
	}{
		{1, []int{9}},
		{2, []int{5, 4}},
		{3, []int{3, 3, 3}},
		{4, []int{3, 2, 2, 2}},
		{9, []int{1, 1, 1, 1, 1, 1, 1, 1, 1}},
	}
	for _, tc := range cases {
		var sizes []int
		for w := 0; w < tc.numWorkers; w++ {
			start, end := SlotRange(w, tc.numWorkers)
			sizes = append(sizes, end-start)
		}
		for i, got := range sizes {
			if got != tc.want[i] {
				t.Errorf("numWorkers=%d: sizes=%v, want=%v", tc.numWorkers, sizes, tc.want)
				break
			}
		}
	}
}

func TestPipePath(t *testing.T) {
	if got := PipePath(0); got != "/tmp/datasetfs_pipe_0" {
		t.Errorf("PipePath(0)=%s", got)
	}
	if got := PipePath(7); got != "/tmp/datasetfs_pipe_7" {
		t.Errorf("PipePath(7)=%s", got)
	}
}

// mockCoreIndex builds a CoreIndex with N shards (IDs 0..N-1).
func mockCoreIndex(numShards int) *index.CoreIndex {
	ci := index.NewIndex()
	for i := 0; i < numShards; i++ {
		ci.ShardMap[i] = &index.Shard{Number: i, Type: index.Base, TotalSize: 1000}
	}
	return ci
}

func TestShardsForWorker_DeterministicWithSeed(t *testing.T) {
	// Two planners with identical (seed, cfg) must produce identical shard orderings.
	seed := uint64(42)
	ci := mockCoreIndex(11)

	cfg := WorkerConfig{
		WorkerID:   1,
		NumWorkers: 4,
		SlotStart:  3,
		SlotEnd:    5,
		Seed:       &seed,
	}

	p1 := NewPlanner(ci, nil, nil, nil, cfg)
	p2 := NewPlanner(ci, nil, nil, nil, cfg)

	out1 := p1.shardsForWorker()
	out2 := p2.shardsForWorker()

	if !reflect.DeepEqual(out1, out2) {
		t.Fatalf("seeded planner non-deterministic: %v vs %v", out1, out2)
	}
	// Expected modulo-4 assignment of sorted shard IDs to worker 1: indices 1,5,9
	if got, want := len(out1), 3; got != want {
		t.Errorf("worker 1 with 11 shards / 4 workers: got %d shards, want %d", got, want)
	}
}

func TestShardsForWorker_DifferentSeedsDifferentOrder(t *testing.T) {
	// Different seeds should (with high probability) produce different orderings.
	ci := mockCoreIndex(11)
	s1 := uint64(1)
	s2 := uint64(2)

	cfg1 := WorkerConfig{WorkerID: 0, NumWorkers: 1, SlotStart: 0, SlotEnd: 9, Seed: &s1}
	cfg2 := WorkerConfig{WorkerID: 0, NumWorkers: 1, SlotStart: 0, SlotEnd: 9, Seed: &s2}

	out1 := NewPlanner(ci, nil, nil, nil, cfg1).shardsForWorker()
	out2 := NewPlanner(ci, nil, nil, nil, cfg2).shardsForWorker()

	if reflect.DeepEqual(out1, out2) {
		t.Fatalf("different seeds produced identical order — RNG not actually used: %v", out1)
	}
	// Sets must match (same shard assignment), only order differs.
	set1 := map[int]bool{}
	set2 := map[int]bool{}
	for _, x := range out1 {
		set1[x] = true
	}
	for _, x := range out2 {
		set2[x] = true
	}
	if !reflect.DeepEqual(set1, set2) {
		t.Fatalf("different seeds picked different shards: %v vs %v", out1, out2)
	}
}

func TestWorkerConfig_RandHelpers(t *testing.T) {
	// PlannerRand/DealerRand return nil when no seed; non-nil when seeded.
	noSeed := WorkerConfig{WorkerID: 2, NumWorkers: 4}
	if noSeed.PlannerRand() != nil || noSeed.DealerRand() != nil {
		t.Fatal("expected nil rand when Seed is nil")
	}

	seed := uint64(123)
	withSeed := WorkerConfig{WorkerID: 2, NumWorkers: 4, Seed: &seed}
	if withSeed.PlannerRand() == nil || withSeed.DealerRand() == nil {
		t.Fatal("expected non-nil rand when Seed is set")
	}

	// Planner and Dealer should get DIFFERENT streams (different sub-seeds),
	// so identical operations produce different first-draw values.
	pr := withSeed.PlannerRand()
	dr := withSeed.DealerRand()
	if pr.Uint64() == dr.Uint64() {
		t.Error("planner and dealer rand produced identical first draw — same seed stream?")
	}
}
