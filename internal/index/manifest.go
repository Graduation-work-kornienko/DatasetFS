package index

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// RemoteStorageInterface defines the interface for remote storage operations
type RemoteStorageInterface interface {
	GetLocalPath(ctx context.Context, url string) (string, error)
}

// IsURL checks if a string is a URL
func IsURL(s string) bool {
	return strings.HasPrefix(s, "http://") || strings.HasPrefix(s, "https://") || strings.HasPrefix(s, "ydisk://")
}

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
	currentVersion                = "1.0"
	manifestParquetFileName       = "metadata.parquet"
	ShardSize               int64 = 100 * 1024 * 1024
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

func (m *Manifest) Load(remoteStorage RemoteStorageInterface) error {
	// Check if we need to download the manifest from remote storage
	var localRoot string
	var err error
	if IsURL(m.Root) {
		// We need to download the manifest first
		if remoteStorage == nil {
			return fmt.Errorf("cannot load remote manifest: no remote storage configured")
		}
		localRoot, err = remoteStorage.GetLocalPath(context.Background(), m.Root)
		if err != nil {
			return fmt.Errorf("failed to download manifest: %w", err)
		}
	} else {
		localRoot = m.Root
	}

	parquetPath := filepath.Join(localRoot, manifestParquetFileName)
	if _, err := os.Stat(parquetPath); err == nil {
		manifest, err := LoadParquetManifest(localRoot)
		if err != nil {
			return fmt.Errorf("load parquet manifest: %w", err)
		}
		*m = *manifest
		return nil
	} else {
		return fmt.Errorf("metadata.parquet not found in %s: %w", localRoot, err)
	}
}

func (m *Manifest) Store() error {
	return StoreParquetManifest(m)
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

	return coreIdx, nil
}
