package index

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func sampleManifest(root string) *Manifest {
	return &Manifest{
		Version: "1.0",
		Root:    root,
		ShardsMeta: map[int]Shard{
			0: {Number: 0, Type: Base, TotalSize: 1234},
			1: {Number: 1, Type: Base, TotalSize: 5678},
		},
		Files: map[string]Metadata{
			"a.png": {ShardID: 0, Offset: 512, Size: 100, Path: "a.png", ObjectMetadata: json.RawMessage(`{"label":"cat"}`)},
			"b.png": {ShardID: 1, Offset: 512, Size: 200, Path: "b.png", Deleted: true},
			"c.png": {ShardID: 1, Offset: 824, Size: 50, Path: "c.png"}, // nil ObjectMetadata
		},
	}
}

func TestParquetManifest_RoundTrip(t *testing.T) {
	dir := t.TempDir()
	m := sampleManifest(dir)

	require.NoError(t, StoreParquetManifest(m))
	require.FileExists(t, filepath.Join(dir, "metadata.parquet"))

	got, err := LoadParquetManifest(dir)
	require.NoError(t, err)

	require.Len(t, got.Files, 3)
	require.Len(t, got.ShardsMeta, 2)

	a := got.Files["a.png"]
	require.Equal(t, 0, a.ShardID)
	require.Equal(t, int64(512), a.Offset)
	require.Equal(t, int64(100), a.Size)
	require.JSONEq(t, `{"label":"cat"}`, string(a.ObjectMetadata))

	require.True(t, got.Files["b.png"].Deleted)
	require.Empty(t, got.Files["c.png"].ObjectMetadata)
	require.Equal(t, int64(5678), got.ShardsMeta[1].TotalSize)
	require.Equal(t, Base, got.ShardsMeta[1].Type)
}

// TestManifest_StoreAndLoadUseParquet is the regression test for the io.EOF bug:
// Store writes parquet and Load must read it back without falling back to JSON.
func TestManifest_StoreAndLoadUseParquet(t *testing.T) {
	dir := t.TempDir()

	m := sampleManifest(dir)
	require.NoError(t, m.Store())
	require.FileExists(t, filepath.Join(dir, "metadata.parquet"))

	reloaded := NewManifest(dir)
	require.NoError(t, reloaded.Load(nil), "Load must read the parquet manifest, not choke on io.EOF")
	require.Len(t, reloaded.Files, 3)

	idx, err := reloaded.LoadCoreIndex()
	require.NoError(t, err)
	require.Contains(t, idx.FileMap, "a.png")
	require.Equal(t, int64(100), idx.FileMap["a.png"].Size)
}
