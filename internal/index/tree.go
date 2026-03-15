package index

import (
	"encoding/json"
	"sync"
)

type Chunk struct {
	Path      string `json:"path"`       // Path of chunk
	Type      string `json:"type"`       // Type of chunk: base, delta
	TotalSize int    `json:"total_size"` // Size of chunk in bytes

	Objects []*Metadata `json:"-"`
}

type Metadata struct {
	ChunkID string `json:"c_id"`    // name of tar archive that stores object
	Offset  int64  `json:"offset"`  // Offset in chunk
	Size    int    `json:"size"`    // Size of object in bytes
	Deleted bool   `json:"deleted"` // Thumbstone, whether object deleted from dataset
	Path    string `json:"path"`    // Path of object as it was stored as single file

	// Store object metadata, e.g. label
	ObjectMetadata json.RawMessage `json:"meta"`
}

type CoreIndex struct {
	mu sync.RWMutex

	FileMap  map[string]*Metadata // For FUSE and Mutation: object by object path
	ChunkMap map[string]*Chunk    // For Planner: Chunk by name
}

func (i *CoreIndex) Manifest() Manifest {
	i.mu.RLock()
	defer i.mu.RUnlock()

	mani := Manifest{
		Version:    currentVersion,
		ChunksMeta: make(map[string]Chunk, len(i.ChunkMap)),
		Files:      make(map[string]Metadata, len(i.FileMap)),
	}

	for id, chunk := range i.ChunkMap {
		mani.ChunksMeta[id] = *chunk
	}

	for path, meta := range i.FileMap {
		mani.Files[path] = *meta
	}

	return mani
}
