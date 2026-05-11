package index

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

/*

Manifest format
{
  "version": 1.0,

  "shards_meta": {
    "0": {"total_size": 104857600, "type": "base"},
    "1": {"total_size": 104857600, "type": "base"},
    "2": {"total_size": 54000, "type": "delta"}
  },

  "files": {
    "cat.jpg": {"s_id": "0", "offset": 0, "size": 14500, "meta": {"label": 1}},
    "dog.jpg": {"s_id": "1", "offset": 0, "size": 22000, "meta": {"label": 2}},
    "new.jpg": {"s_id": "delta_1", "offset": 0, "size": 54000, "meta": {"label": 1}}
  }
}

*/

const (
	currentVersion         = "1.0"
	manifestFileName       = "metadata.jsonl"
	ShardSize        int64 = 100 * 1024 * 1024
)

// Manifest represents how index stores in disk
// Only Manifest structure interract with disk, not CoreIndex
type Manifest struct {
	Version    string              `json:"version"`
	ShardsMeta map[int]Shard       `json:"shards_meta"`
	Files      map[string]Metadata `json:"files"`
	Root       string              `json:"-"` // Root path of manifest
}

func NewManifest(root string) *Manifest {
	return &Manifest{
		Version:    currentVersion,
		ShardsMeta: make(map[int]Shard),
		Files:      make(map[string]Metadata),
		Root:       root,
	}
}

func (m *Manifest) Load() error {
	file, err := os.Open(filepath.Join(m.Root, manifestFileName))
	if err != nil {
		return err
	}
	defer file.Close()

	if err := json.NewDecoder(file).Decode(m); err != nil {
		return err
	}
	return nil
}

func (m *Manifest) Store() error {
	filepath := filepath.Join(m.Root, manifestFileName)
	fmt.Println(filepath)
	file, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(m)
}

func (m *Manifest) AppendShard(shard *Shard) error {
	m.ShardsMeta[shard.Number] = *shard
	for _, e := range shard.Objects {
		m.Files[e.Path] = *e
	}
	return nil
}

func (m *Manifest) LoadCoreIndex() (*CoreIndex, error) {

	coreIdx := &CoreIndex{
		FileMap:  make(map[string]*Metadata, len(m.Files)),
		ShardMap: make(map[int]*Shard, len(m.ShardsMeta)),
	}

	for id, shardValue := range m.ShardsMeta {
		shardCopy := shardValue
		coreIdx.ShardMap[id] = &shardCopy
	}

	for path, metaValue := range m.Files {
		metaCopy := metaValue
		coreIdx.FileMap[path] = &metaCopy

		if shard, exists := coreIdx.ShardMap[metaCopy.ShardID]; exists {
			shard.Objects = append(shard.Objects, &metaCopy)
		}
	}

	for id, shard := range coreIdx.ShardMap {
		fmt.Println(id, len(shard.Objects))
	}

	return coreIdx, nil
}
