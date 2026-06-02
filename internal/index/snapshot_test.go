package index

import (
	"sync"
	"testing"
)

// baseIdx returns a CoreIndex with one base shard (id 0) holding files a,b and an
// empty delta shard placeholder, mirroring what NewMutationManager seeds.
func baseIdx(t *testing.T) *CoreIndex {
	t.Helper()
	ci := NewIndex()
	ci.ShardMap[0] = &Shard{Number: 0, Type: Base, TotalSize: 1000}
	ci.ShardMap[DeltaShardID] = &Shard{Number: DeltaShardID, Type: "delta", TotalSize: 0}
	if err := ci.AddFile(&Metadata{ShardID: 0, Path: "a", Offset: 0, Size: 100}); err != nil {
		t.Fatal(err)
	}
	if err := ci.AddFile(&Metadata{ShardID: 0, Path: "b", Offset: 200, Size: 100}); err != nil {
		t.Fatal(err)
	}
	return ci
}

func objNames(ss *ShardSnap) map[string]bool {
	m := map[string]bool{}
	for _, o := range ss.Objects {
		m[o.Path] = true
	}
	return m
}

// A delete after a pin must NOT change what the pinned snapshot sees, but a fresh
// pin (new generation) must reflect it. This is the core consistency guarantee.
func TestSnapshot_DeleteIsolation(t *testing.T) {
	ci := baseIdx(t)

	snap1 := ci.Pin()
	defer ci.Unpin(snap1)
	if got := objNames(snap1.Shards[0]); !got["a"] || !got["b"] || len(got) != 2 {
		t.Fatalf("snap1 should hold {a,b}, got %v", got)
	}

	if err := ci.MarkDeleted("a"); err != nil {
		t.Fatal(err)
	}

	// Pinned snapshot is immutable — still sees "a".
	if got := objNames(snap1.Shards[0]); !got["a"] || len(got) != 2 {
		t.Fatalf("pinned snap1 must be immutable after delete, got %v", got)
	}

	snap2 := ci.Pin()
	defer ci.Unpin(snap2)
	if snap2.Gen <= snap1.Gen {
		t.Fatalf("delete must bump generation: snap1=%d snap2=%d", snap1.Gen, snap2.Gen)
	}
	if got := objNames(snap2.Shards[0]); got["a"] || !got["b"] || len(got) != 1 {
		t.Fatalf("snap2 must drop deleted 'a', got %v", got)
	}
}

// An added (delta) file is invisible to an already-pinned epoch and served only
// from the next generation, with the delta shard's covering size derived from
// the object offsets.
func TestSnapshot_AddVisibleNextGenWithDeltaSize(t *testing.T) {
	ci := baseIdx(t)

	snap1 := ci.Pin()
	defer ci.Unpin(snap1)
	if n := len(snap1.Shards[DeltaShardID].Objects); n != 0 {
		t.Fatalf("delta should start empty, got %d objects", n)
	}

	// Two appended files, tar layout: data starts after the 512-byte header.
	if err := ci.AddFile(&Metadata{ShardID: DeltaShardID, Path: "c", Offset: 512, Size: 100}); err != nil {
		t.Fatal(err)
	}
	if err := ci.AddFile(&Metadata{ShardID: DeltaShardID, Path: "d", Offset: 1224, Size: 50}); err != nil {
		t.Fatal(err)
	}

	if n := len(snap1.Shards[DeltaShardID].Objects); n != 0 {
		t.Fatalf("pinned snap1 must not see added files, got %d", n)
	}

	snap2 := ci.Pin()
	defer ci.Unpin(snap2)
	delta := snap2.Shards[DeltaShardID]
	if got := objNames(delta); !got["c"] || !got["d"] || len(got) != 2 {
		t.Fatalf("snap2 delta must hold {c,d}, got %v", got)
	}
	// Covering size = max(Offset+Size) = max(612, 1274) = 1274.
	if delta.TotalSize != 1274 {
		t.Fatalf("delta TotalSize should be derived as 1274, got %d", delta.TotalSize)
	}
}

// Pin caches the snapshot per generation (same pointer until a mutation), and the
// generation counter is strictly monotonic across mutations.
func TestSnapshot_CacheAndMonotonicGen(t *testing.T) {
	ci := baseIdx(t)

	a := ci.Pin()
	b := ci.Pin()
	if a != b {
		t.Fatal("two pins of the same generation must return the cached snapshot")
	}
	ci.Unpin(b)

	prev := a.Gen
	if err := ci.MarkDeleted("b"); err != nil {
		t.Fatal(err)
	}
	c := ci.Pin()
	if c == a {
		t.Fatal("a mutation must invalidate the cached snapshot")
	}
	if c.Gen <= prev {
		t.Fatalf("generation must increase: prev=%d new=%d", prev, c.Gen)
	}
	ci.Unpin(a)
	ci.Unpin(c)
}

// MinPinnedGen reports the oldest live pin — the future vacuum safepoint.
func TestSnapshot_MinPinnedGen(t *testing.T) {
	ci := baseIdx(t)

	if _, ok := ci.MinPinnedGen(); ok {
		t.Fatal("no pins yet: MinPinnedGen must report none")
	}

	old := ci.Pin() // generation g
	if g, ok := ci.MinPinnedGen(); !ok || g != old.Gen {
		t.Fatalf("MinPinnedGen=%d,%v want %d,true", g, ok, old.Gen)
	}

	if err := ci.MarkDeleted("a"); err != nil {
		t.Fatal(err)
	}
	newer := ci.Pin() // generation g+1
	if g, _ := ci.MinPinnedGen(); g != old.Gen {
		t.Fatalf("oldest pin must stay the safepoint: got %d want %d", g, old.Gen)
	}

	ci.Unpin(old)
	if g, ok := ci.MinPinnedGen(); !ok || g != newer.Gen {
		t.Fatalf("after releasing oldest, safepoint must advance to %d, got %d,%v", newer.Gen, g, ok)
	}
	ci.Unpin(newer)
	if _, ok := ci.MinPinnedGen(); ok {
		t.Fatal("all pins released: MinPinnedGen must report none")
	}
}

// Concurrent pins and mutations must be race-free (run under `go test -race`).
func TestSnapshot_ConcurrentPinAndMutate(t *testing.T) {
	ci := baseIdx(t)
	var wg sync.WaitGroup

	// Readers: pin/inspect/unpin in a tight loop.
	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 500; i++ {
				s := ci.Pin()
				_ = len(s.Shards[0].Objects)
				ci.MinPinnedGen()
				ci.Unpin(s)
			}
		}()
	}

	// Writers: add and delete files.
	for w := 0; w < 2; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < 300; i++ {
				name := string(rune('A'+w)) + string(rune('0'+i%10))
				_ = ci.AddFile(&Metadata{ShardID: DeltaShardID, Path: name, Offset: int64(512 + i), Size: 10})
				ci.MarkDeletedTolerant(name)
			}
		}(w)
	}

	wg.Wait()
}
