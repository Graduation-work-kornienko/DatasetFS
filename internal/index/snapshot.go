package index

// MVCC / snapshot isolation (feature F1).
//
// A loading session must see a consistent view of the dataset for the whole
// epoch even while another goroutine mutates it (FUSE rm/cp → MutationManager).
// The mechanism is generational: every mutation bumps CoreIndex.gen; a reader
// calls Pin() once at epoch start to obtain an immutable Snapshot of the current
// generation and Unpin()s it when the session ends. Because the Snapshot holds
// *value copies* of Metadata and a *frozen* per-shard size, a later in-place
// Deleted flip or an O_APPEND grow of the delta tar cannot change what a pinned
// epoch sees — it keeps reading its generation until the next epoch re-pins.

// Snapshot is an immutable, point-in-time view of the dataset shared by all
// workers of one loading session. Never mutate it after Pin returns it.
type Snapshot struct {
	Gen    uint64
	Shards map[int]*ShardSnap
}

// ShardSnap is the frozen view of one shard within a Snapshot. Objects are value
// copies of the shard's non-deleted entries; TotalSize is the number of bytes the
// loader must read into the slot to cover every object.
type ShardSnap struct {
	Number    int
	Type      ShardType
	TotalSize int64
	Objects   []Metadata
}

// bumpGen advances the generation and invalidates the cached snapshot. Callers
// must already hold i.Mu (every mutating method does).
func (i *CoreIndex) bumpGen() {
	i.gen++
	i.cachedSnap = nil
}

// materializeLocked builds an immutable Snapshot of the current state. Caller
// holds i.Mu. Delta shards may have no reliable manifest TotalSize — their
// on-disk tar files grow by append — so we derive the covering size from the objects
// themselves (max Offset+Size). This is also what makes delta shards correct
// after a WAL replay, which never records a shard size.
func (i *CoreIndex) materializeLocked() *Snapshot {
	snap := &Snapshot{Gen: i.gen, Shards: make(map[int]*ShardSnap, len(i.ShardMap))}
	for id, sh := range i.ShardMap {
		ss := &ShardSnap{Number: id, Type: sh.Type, TotalSize: sh.TotalSize}
		var cover int64
		for _, m := range sh.Objects {
			if m.Deleted {
				continue
			}
			ss.Objects = append(ss.Objects, *m) // value copy: immune to later in-place mutation
			if end := m.Offset + m.Size; end > cover {
				cover = end
			}
		}
		if sh.Type == Delta || id < 0 {
			ss.TotalSize = cover
		}
		snap.Shards[id] = ss
	}
	return snap
}

// Pin returns an immutable Snapshot of the current generation and records the
// pin so MinPinnedGen can report it (the vacuum safepoint). The snapshot is
// cached and shared across concurrent pins of the same generation; a mutation
// invalidates the cache so the next Pin re-materializes. Pair every Pin with an
// Unpin.
func (i *CoreIndex) Pin() *Snapshot {
	i.Mu.Lock()
	defer i.Mu.Unlock()
	if i.cachedSnap == nil || i.cachedSnap.Gen != i.gen {
		i.cachedSnap = i.materializeLocked()
	}
	if i.pinned == nil {
		i.pinned = make(map[uint64]int)
	}
	i.pinned[i.cachedSnap.Gen]++
	return i.cachedSnap
}

// Unpin releases a pin taken by Pin. Safe to call with a nil snapshot.
func (i *CoreIndex) Unpin(s *Snapshot) {
	if s == nil {
		return
	}
	i.Mu.Lock()
	defer i.Mu.Unlock()
	if i.pinned == nil {
		return
	}
	if c := i.pinned[s.Gen]; c <= 1 {
		delete(i.pinned, s.Gen)
	} else {
		i.pinned[s.Gen] = c - 1
	}
}

// MinPinnedGen returns the smallest generation currently pinned by a live
// session, or (0,false) if none. Vacuum will use this as a GC safepoint so it
// never reclaims bytes a running epoch can still see (G13/F1b); the primitive is
// exposed now even though vacuum stays maintenance-gated in F1.
func (i *CoreIndex) MinPinnedGen() (uint64, bool) {
	i.Mu.RLock()
	defer i.Mu.RUnlock()
	var min uint64
	found := false
	for g := range i.pinned {
		if !found || g < min {
			min, found = g, true
		}
	}
	return min, found
}

// Generation returns the current generation counter (number of mutations since
// load). Useful for tests and for stamping reader-visible batch metadata.
func (i *CoreIndex) Generation() uint64 {
	i.Mu.RLock()
	defer i.Mu.RUnlock()
	return i.gen
}
