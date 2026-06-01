package vacuum

import (
	"bytes"
	"crypto/rand"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
)

// buildDataset writes `files` (name -> bytes) into a DatasetFS dataset at root,
// one shard per call group, and returns the dataset root. It mirrors the
// converter's folder path: source bytes live as real files that AppendShard
// packs into shards.
func buildDataset(t *testing.T, files map[string][]byte) string {
	t.Helper()
	root := t.TempDir()
	src := t.TempDir()

	strg := storage.New(root, nil)
	mani := index.NewManifest(root)

	names := make([]string, 0, len(files))
	for name := range files {
		names = append(names, name)
	}
	sort.Strings(names)

	objects := make([]*index.Metadata, 0, len(names))
	for _, name := range names {
		p := filepath.Join(src, name)
		if err := os.WriteFile(p, files[name], 0644); err != nil {
			t.Fatal(err)
		}
		objects = append(objects, &index.Metadata{
			Path:           p,
			Size:           int64(len(files[name])),
			ObjectMetadata: []byte(fmt.Sprintf(`{"label":%q}`, name)),
		})
	}

	shard := &index.Shard{Number: 0, Type: index.Base, Objects: objects}
	if err := strg.AppendShard(shard); err != nil {
		t.Fatalf("AppendShard: %v", err)
	}
	if err := mani.AppendShard(shard); err != nil {
		t.Fatalf("manifest.AppendShard: %v", err)
	}
	if err := mani.Store(); err != nil {
		t.Fatalf("manifest.Store: %v", err)
	}
	return root
}

// readLive reloads the manifest at root and returns name -> bytes for every
// live (non-deleted) file, read straight out of its shard by (Offset, Size).
func readLive(t *testing.T, root string) map[string][]byte {
	t.Helper()
	mani := index.NewManifest(root)
	if err := mani.Load(nil); err != nil {
		t.Fatalf("reload manifest: %v", err)
	}
	strg := storage.New(root, nil)
	out := make(map[string][]byte)
	for path, meta := range mani.Files {
		if meta.Deleted {
			continue
		}
		f, err := os.Open(strg.ShardPath(meta.ShardID))
		if err != nil {
			t.Fatalf("open shard %d: %v", meta.ShardID, err)
		}
		buf := make([]byte, meta.Size)
		if _, err := f.ReadAt(buf, meta.Offset); err != nil {
			f.Close()
			t.Fatalf("read %s: %v", path, err)
		}
		f.Close()
		out[filepath.Base(path)] = buf
	}
	return out
}

func markDeleted(t *testing.T, root string, basename string) {
	t.Helper()
	mani := index.NewManifest(root)
	if err := mani.Load(nil); err != nil {
		t.Fatal(err)
	}
	found := false
	for path, meta := range mani.Files {
		if filepath.Base(path) == basename {
			meta.Deleted = true
			mani.Files[path] = meta
			found = true
		}
	}
	if !found {
		t.Fatalf("file %s not in manifest", basename)
	}
	if err := mani.Store(); err != nil {
		t.Fatal(err)
	}
}

func TestVacuum_ReclaimsDeletedAndPreservesLive(t *testing.T) {
	files := map[string][]byte{
		"a.bin": randBytes(t, 10_000),
		"b.bin": randBytes(t, 7_000),
		"c.bin": randBytes(t, 13_000),
	}
	root := buildDataset(t, files)

	before := manifestPhysical(t, root)
	markDeleted(t, root, "b.bin")

	stats, err := Run(Options{Root: root, Verbose: true})
	if err != nil {
		t.Fatalf("vacuum: %v", err)
	}
	if stats.LiveFiles != 2 {
		t.Fatalf("live files = %d, want 2", stats.LiveFiles)
	}

	live := readLive(t, root)
	if len(live) != 2 {
		t.Fatalf("after vacuum: %d live files, want 2", len(live))
	}
	if _, gone := live["b.bin"]; gone {
		t.Fatal("deleted file b.bin survived vacuum")
	}
	for _, name := range []string{"a.bin", "c.bin"} {
		if !bytes.Equal(live[name], files[name]) {
			t.Fatalf("%s bytes corrupted by vacuum", name)
		}
	}

	after := manifestPhysical(t, root)
	if after >= before {
		t.Fatalf("physical size did not shrink: before=%d after=%d", before, after)
	}
}

func TestVacuum_DryRunChangesNothing(t *testing.T) {
	files := map[string][]byte{
		"a.bin": randBytes(t, 5_000),
		"b.bin": randBytes(t, 5_000),
	}
	root := buildDataset(t, files)
	markDeleted(t, root, "a.bin")

	snap := snapshotDir(t, root)
	if _, err := Run(Options{Root: root, DryRun: true, Verbose: true}); err != nil {
		t.Fatalf("dry-run: %v", err)
	}
	if got := snapshotDir(t, root); got != snap {
		t.Fatalf("dry-run mutated the dataset:\nbefore=%s\nafter =%s", snap, got)
	}
}

func TestVacuum_EmptyDatasetAfterAllDeleted(t *testing.T) {
	files := map[string][]byte{"a.bin": randBytes(t, 3_000)}
	root := buildDataset(t, files)
	markDeleted(t, root, "a.bin")

	stats, err := Run(Options{Root: root})
	if err != nil {
		t.Fatalf("vacuum: %v", err)
	}
	if stats.LiveFiles != 0 || stats.NewShards != 0 {
		t.Fatalf("expected empty dataset, got live=%d shards=%d", stats.LiveFiles, stats.NewShards)
	}
	if len(readLive(t, root)) != 0 {
		t.Fatal("expected no live files")
	}
}

// TestVacuum_FragmentationAndReload exercises the body of the daemon's
// background vacuumer goroutine: measure fragmentation, compact, then reload
// the in-memory index from the compacted manifest.
func TestVacuum_FragmentationAndReload(t *testing.T) {
	files := map[string][]byte{
		"a.bin": randBytes(t, 8_000),
		"b.bin": randBytes(t, 8_000),
		"c.bin": randBytes(t, 8_000),
	}
	root := buildDataset(t, files)
	markDeleted(t, root, "b.bin")

	// Fragmentation should reflect the one tombstoned file (~1/3 of bytes).
	m := index.NewManifest(root)
	if err := m.Load(nil); err != nil {
		t.Fatal(err)
	}
	if frag := Fragmentation(m); frag < 0.2 || frag > 0.5 {
		t.Fatalf("fragmentation = %.3f, want ~0.33", frag)
	}

	if _, err := Run(Options{Root: root}); err != nil {
		t.Fatalf("vacuum: %v", err)
	}

	// Reload a CoreIndex from the compacted manifest (what the goroutine does).
	nm := index.NewManifest(root)
	if err := nm.Load(nil); err != nil {
		t.Fatal(err)
	}
	if frag := Fragmentation(nm); frag != 0 {
		t.Fatalf("post-vacuum fragmentation = %.3f, want 0", frag)
	}
	idx := index.NewIndex()
	idx.Reload(nm)

	// FileMap keys are the (full) stored paths; index them by basename.
	byBase := make(map[string]*index.Metadata)
	for path, meta := range idx.FileMap {
		byBase[filepath.Base(path)] = meta
	}
	if _, gone := byBase["b.bin"]; gone {
		t.Fatal("deleted file still present after reload")
	}
	for _, name := range []string{"a.bin", "c.bin"} {
		meta, ok := byBase[name]
		if !ok {
			t.Fatalf("%s missing from reloaded index", name)
		}
		// Reloaded shard objects must be wired up for the planner.
		sh, ok := idx.ShardMap[meta.ShardID]
		if !ok || len(sh.Objects) == 0 {
			t.Fatalf("shard %d for %s has no objects after reload", meta.ShardID, name)
		}
	}
}

// --- helpers ---

func randBytes(t *testing.T, n int) []byte {
	t.Helper()
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		t.Fatal(err)
	}
	return b
}

func manifestPhysical(t *testing.T, root string) int64 {
	t.Helper()
	mani := index.NewManifest(root)
	if err := mani.Load(nil); err != nil {
		t.Fatal(err)
	}
	var total int64
	for _, s := range mani.ShardsMeta {
		total += s.TotalSize
	}
	return total
}

// snapshotDir returns a stable string of (name,size) for shard/manifest files,
// used to assert dry-run made no changes.
func snapshotDir(t *testing.T, root string) string {
	t.Helper()
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		info, err := e.Info()
		if err != nil {
			t.Fatal(err)
		}
		names = append(names, fmt.Sprintf("%s:%d", e.Name(), info.Size()))
	}
	sort.Strings(names)
	return fmt.Sprintf("%v", names)
}
