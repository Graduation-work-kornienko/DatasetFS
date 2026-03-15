package index

import (
	"encoding/json"
	"os"
)

/*

Manifest format
{
  "version": 1.0,

  "chunks_meta": {
    "0": {"path": "/data/chunk_0.bin", "total_size": 104857600, "type": "base"},
    "1": {"path": "/data/chunk_1.bin", "total_size": 104857600, "type": "base"},
    "delta_1": {"path": "/data/deltas/delta_1.bin", "total_size": 54000, "type": "delta"}
  },

  "files": {
    "train/cat.jpg": {"c_id": "0", "offset": 0, "size": 14500, "meta": {"label": 1}},
    "train/dog.jpg": {"c_id": "1", "offset": 0, "size": 22000, "meta": {"label": 2}},
    "train/new.jpg": {"c_id": "delta_1", "offset": 0, "size": 54000, "meta": {"label": 1}}
  }
}

*/

const (
	currentVersion = "1.0"
	fileName       = "metadata.jsonl"
)

// Manifest represents how index stores in disk
// Only Manifest structure interract with disk, not CoreIndex
type Manifest struct {
	Version    string              `json:"version"`
	ChunksMeta map[string]Chunk    `json:"chunks_meta"`
	Files      map[string]Metadata `json:"files"`
}

func (m *Manifest) Store() error {
	file, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(m)
}

func LoadCoreIndex(filepath string) (*CoreIndex, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var mani Manifest
	if err := json.NewDecoder(file).Decode(&mani); err != nil {
		return nil, err
	}

	coreIdx := &CoreIndex{
		FileMap:  make(map[string]*Metadata, len(mani.Files)),
		ChunkMap: make(map[string]*Chunk, len(mani.ChunksMeta)),
	}

	for id, chunkValue := range mani.ChunksMeta {
		chunkCopy := chunkValue
		coreIdx.ChunkMap[id] = &chunkCopy
	}

	for path, metaValue := range mani.Files {
		metaCopy := metaValue
		coreIdx.FileMap[path] = &metaCopy

		if chunk, exists := coreIdx.ChunkMap[metaCopy.ChunkID]; exists {
			chunk.Objects = append(chunk.Objects, &metaCopy)
		}
	}

	return coreIdx, nil
}
