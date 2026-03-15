package index

import (
	"encoding/json"
	"sync"
)

type Shard struct {
	Number    int    `json:"-"`          // Number of shard
	Type      string `json:"type"`       // Type of shard: base, delta
	TotalSize int64  `json:"total_size"` // Size of shard in bytes

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
	mu sync.RWMutex

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

func (i *CoreIndex) Manifest() Manifest {
	i.mu.RLock()
	defer i.mu.RUnlock()

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
	return nil
}
