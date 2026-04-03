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

func (i *CoreIndex) MarkDeleted(filename string) error {
	i.Mu.Lock()
	defer i.Mu.Unlock()
	if _, ok := i.FileMap[filename]; !ok {
		return fmt.Errorf("no such file %s", filename)
	}
	i.FileMap[filename].Deleted = true
	return nil
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

func (i *CoreIndex) AllShards() map[int]*Shard {
	return maps.Clone(i.ShardMap)
}
