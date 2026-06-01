package index

import (
	"encoding/json"
	"fmt"
	"maps"
	"sync"
)

type ShardType string

const (
	Base ShardType = "base"
)

type Shard struct {
	Number    int       `json:"-"`          // Number of shard
	Type      ShardType `json:"type"`       // Type of shard: base, delta
	TotalSize int64     `json:"total_size"` // Size of shard in bytes

	Objects []*Metadata `json:"-"`
}

type Metadata struct {
	ShardID int    `json:"c_id"`    // name of tar archive that stores object
	Offset  int64  `json:"offset"`  // Offset in shard
	Size    int64  `json:"size"`    // Size of object in bytes
	Deleted bool   `json:"deleted"` // Thumbstone, whether object deleted from dataset
	Path    string `json:"path"`    // Path of object as it was stored as single file

	// Store object metadata, e.g. label
	ObjectMetadata json.RawMessage `json:"meta"`
}

type CoreIndex struct {
	Mu sync.RWMutex

	LastShard int                  // Number of last shard
	FileMap   map[string]*Metadata // For FUSE and Mutation: object by object path
	ShardMap  map[int]*Shard       // For Planner: Shard by name
}

func NewIndex() *CoreIndex {
	return &CoreIndex{
		ShardMap: make(map[int]*Shard),
		FileMap:  make(map[string]*Metadata),
	}
}

func (i *CoreIndex) AllShards() map[int]*Shard {
	i.Mu.RLock()
	defer i.Mu.RUnlock()
	return maps.Clone(i.ShardMap)
}

func (i *CoreIndex) MarkDeleted(filename string) error {
	i.Mu.Lock()
	defer i.Mu.Unlock()
	if _, ok := i.FileMap[filename]; !ok {
		return fmt.Errorf("no such file %s", filename)
	}
	i.FileMap[filename].Deleted = true
	return nil
}

// MarkDeletedTolerant marks a file deleted, treating a missing file as a no-op.
// WAL replay must be idempotent: re-applying a tombstone for a file that a
// later checkpoint already dropped (or that never made it into the recovered
// index) must not abort recovery. Interactive deletes go through MarkDeleted,
// which still errors on a missing file.
func (i *CoreIndex) MarkDeletedTolerant(filename string) {
	i.Mu.Lock()
	defer i.Mu.Unlock()
	if m, ok := i.FileMap[filename]; ok {
		m.Deleted = true
	}
}

func (i *CoreIndex) AddFile(m *Metadata) error {
	i.Mu.Lock()
	defer i.Mu.Unlock()
	if _, ok := i.ShardMap[m.ShardID]; !ok {
		return fmt.Errorf("add file %s: no such shard %d", m.Path, m.ShardID)
	}
	i.ShardMap[m.ShardID].Objects = append(i.ShardMap[m.ShardID].Objects, m)
	i.FileMap[m.Path] = m
	return nil
}

func (i *CoreIndex) Manifest() Manifest {
	i.Mu.RLock()
	defer i.Mu.RUnlock()

	mani := Manifest{
		Version:    currentVersion,
		ShardsMeta: make(map[int]Shard, len(i.ShardMap)),
		Files:      make(map[string]Metadata, len(i.FileMap)),
	}

	for id, shard := range i.ShardMap {
		mani.ShardsMeta[id] = *shard
	}

	for path, meta := range i.FileMap {
		mani.Files[path] = *meta
	}

	return mani
}

func (i *CoreIndex) AppendShard(shard *Shard) error {
	i.Mu.Lock()
	defer i.Mu.Unlock()
	i.ShardMap[shard.Number] = shard
	for _, e := range shard.Objects {
		i.FileMap[e.Path] = e
	}
	return nil
}

// Reload atomically replaces the in-memory index from a freshly loaded
// manifest. Used after a background vacuum rewrites the dataset so the running
// daemon reflects the compacted shards without a restart. Callers must ensure
// no pipeline is concurrently reading (see ipc.BeginMaintenance); the delta
// shard placeholder is NOT re-seeded here — see MutationManager.EnsureDelta.
func (i *CoreIndex) Reload(m *Manifest) {
	i.Mu.Lock()
	defer i.Mu.Unlock()

	i.FileMap = make(map[string]*Metadata, len(m.Files))
	i.ShardMap = make(map[int]*Shard, len(m.ShardsMeta))

	for id, s := range m.ShardsMeta {
		sc := s
		sc.Number = id
		sc.Objects = nil
		i.ShardMap[id] = &sc
	}
	for path, meta := range m.Files {
		mc := meta
		i.FileMap[path] = &mc
		if sh, ok := i.ShardMap[mc.ShardID]; ok {
			sh.Objects = append(sh.Objects, &mc)
		}
	}
}
