package index

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/parquet-go/parquet-go"
)

// ParquetManifest represents the manifest data structure for Parquet format
type ParquetManifest struct {
	Version    string             `parquet:"version"`
	ShardsMeta []ParquetShardMeta `parquet:"shards_meta"`
	Files      []ParquetFileEntry `parquet:"files"`
}

// ParquetShardMeta represents shard metadata for Parquet format
type ParquetShardMeta struct {
	Number    int32  `parquet:"number"`
	Type      string `parquet:"type"`
	TotalSize int64  `parquet:"total_size"`
}

// ParquetFileEntry represents file entry for Parquet format
type ParquetFileEntry struct {
	Path           string `parquet:"path"`
	ShardID        int32  `parquet:"shard_id"`
	Offset         int64  `parquet:"offset"`
	Size           int64  `parquet:"size"`
	Deleted        bool   `parquet:"deleted"`
	ObjectMetadata []byte `parquet:"object_metadata"`
}

// LoadParquetManifest loads a manifest from a Parquet file
func LoadParquetManifest(root string) (*Manifest, error) {
	path := filepath.Join(root, "metadata.parquet")
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open parquet manifest %s: %w", path, err)
	}
	defer file.Close()

	reader := parquet.NewGenericReader[ParquetManifest](file)
	manifests := make([]ParquetManifest, 1)
	_, err = reader.Read(manifests)
	reader.Close()
	// A single-row manifest comes back as (1, io.EOF): the reader signals
	// end-of-file together with the last row. Only a non-EOF error is fatal.
	if err != nil && err != io.EOF {
		return nil, fmt.Errorf("read parquet manifest: %w", err)
	}

	// Convert ParquetManifest to Manifest
	pManifest := manifests[0]
	manifest := &Manifest{
		Version:    pManifest.Version,
		ShardsMeta: make(map[int]Shard),
		Files:      make(map[string]Metadata),
		Root:       root,
	}

	// Convert shards
	for _, s := range pManifest.ShardsMeta {
		manifest.ShardsMeta[int(s.Number)] = Shard{
			Number:    int(s.Number),
			Type:      ShardType(s.Type),
			TotalSize: s.TotalSize,
		}
	}

	// Convert files
	for _, f := range pManifest.Files {
		meta := Metadata{
			ShardID:        int(f.ShardID),
			Offset:         f.Offset,
			Size:           f.Size,
			Deleted:        f.Deleted,
			Path:           f.Path,
			ObjectMetadata: f.ObjectMetadata,
		}
		manifest.Files[f.Path] = meta
	}

	return manifest, nil
}

// StoreParquetManifest stores a manifest to a Parquet file
func StoreParquetManifest(m *Manifest) error {
	path := filepath.Join(m.Root, "metadata.parquet")
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create parquet manifest %s: %w", path, err)
	}
	defer file.Close()

	// Convert Manifest to ParquetManifest
	pManifest := ParquetManifest{
		Version:    m.Version,
		ShardsMeta: make([]ParquetShardMeta, 0, len(m.ShardsMeta)),
		Files:      make([]ParquetFileEntry, 0, len(m.Files)),
	}

	// Convert shards
	for id, shard := range m.ShardsMeta {
		pManifest.ShardsMeta = append(pManifest.ShardsMeta, ParquetShardMeta{
			Number:    int32(id),
			Type:      string(shard.Type),
			TotalSize: shard.TotalSize,
		})
	}

	// Convert files
	for path, meta := range m.Files {
		// Handle nil ObjectMetadata
		var objMeta []byte
		if meta.ObjectMetadata != nil {
			objMeta = meta.ObjectMetadata
		}
		pManifest.Files = append(pManifest.Files, ParquetFileEntry{
			Path:           path,
			ShardID:        int32(meta.ShardID),
			Offset:         meta.Offset,
			Size:           meta.Size,
			Deleted:        meta.Deleted,
			ObjectMetadata: objMeta,
		})
	}

	writer := parquet.NewGenericWriter[ParquetManifest](file)
	_, err = writer.Write([]ParquetManifest{pManifest})
	if err != nil {
		file.Close()
		return fmt.Errorf("write parquet manifest: %w", err)
	}
	if err := writer.Close(); err != nil {
		return fmt.Errorf("close parquet writer: %w", err)
	}

	return nil
}

// ConvertJSONManifestToParquet converts a JSON manifest to Parquet format
func ConvertJSONManifestToParquet(root string) error {
	// First load the JSON manifest
	jsonPath := filepath.Join(root, "metadata.jsonl")
	jsonFile, err := os.Open(jsonPath)
	if err != nil {
		return fmt.Errorf("open json manifest %s: %w", jsonPath, err)
	}
	defer jsonFile.Close()

	var manifest Manifest
	if err := json.NewDecoder(jsonFile).Decode(&manifest); err != nil {
		return fmt.Errorf("decode json manifest: %w", err)
	}
	manifest.Root = root

	// Convert and store as Parquet
	return StoreParquetManifest(&manifest)
}
