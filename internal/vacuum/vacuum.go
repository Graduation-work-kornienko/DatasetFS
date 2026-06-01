// Package vacuum compacts a DatasetFS dataset: it drops tombstoned files,
// repacks the live ones into fresh contiguous shards, and rewrites the
// manifest. The logic lives here (not in cmd/) so both the `vacuum` CLI and
// the daemon's background vacuumer goroutine share one implementation.
//
// Safety model: vacuum reads live bytes out of the EXISTING shards and writes
// the compacted result to temporary files; the live dataset is only mutated at
// the very end via rename. --dry-run touches nothing on disk. A crash strictly
// between the shard renames and the manifest rename can leave new shard bytes
// under the old manifest — recover by restoring metadata.*.backup, deleting any
// shard_*.tmp / metadata.*.tmp, and re-running. The caller MUST ensure the
// dataset is quiescent (daemon stopped or in a maintenance window): vacuum
// rewrites the shards the loader reads from.
package vacuum

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	internalio "github.com/Graduation-work-kornienko/DatasetFS/internal/io"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
)

// Options configures a vacuum run.
type Options struct {
	Root         string // dataset root (manifest + shards)
	DryRun       bool   // compute and report the plan without touching disk
	Verbose      bool   // log progress
	MaxShardSize int64  // soft cap per output shard (a single file is never split)
	PreserveWAL  bool   // keep the WAL instead of truncating after a successful commit
	Background   bool   // yield periodically; pair with Throttle to limit read bandwidth
	Throttle     int64  // read bandwidth limit in bytes/sec (0 = unlimited)
	WALFormat    string // "json" or "binary" (defaults to "json")
}

// Stats summarizes what a run did (or would do, for --dry-run).
type Stats struct {
	LiveFiles   int
	NewShards   int
	BytesBefore int64
	BytesAfter  int64
}

const (
	// DefaultMaxShardSize matches index.ShardSize (100 MiB).
	DefaultMaxShardSize = index.ShardSize
)

// Run executes a vacuum according to opts.
func Run(opts Options) (Stats, error) {
	var stats Stats

	if opts.MaxShardSize <= 0 {
		opts.MaxShardSize = DefaultMaxShardSize
	}

	if fi, err := os.Stat(opts.Root); err != nil {
		return stats, fmt.Errorf("root path: %w", err)
	} else if !fi.IsDir() {
		return stats, fmt.Errorf("root path is not a directory: %s", opts.Root)
	}

	// Finish any half-completed previous run before touching anything.
	if !opts.DryRun {
		if err := completePendingCommit(opts.Root, opts.Verbose); err != nil {
			return stats, fmt.Errorf("recover pending commit: %w", err)
		}
	}

	// Load the durable state, then fold in any mutations the WAL recorded since
	// the last checkpoint so we don't lose FUSE-added (delta) files.
	manifest := index.NewManifest(opts.Root)
	if err := manifest.Load(nil); err != nil {
		return stats, fmt.Errorf("load manifest: %w", err)
	}
	coreIdx, err := manifest.LoadCoreIndex()
	if err != nil {
		return stats, fmt.Errorf("load core index: %w", err)
	}

	// Open + replay the WAL to fold in mutations since the last checkpoint.
	// In --dry-run we must not create a wal.log where none exists, so skip the
	// open entirely when the file is absent (an absent WAL means no pending
	// mutations anyway).
	var wal index.WAL
	walExists := fileExists(filepath.Join(opts.Root, walFileName))
	if walExists || !opts.DryRun {
		wal, err = index.OpenWALWithFormat(opts.Root, opts.WALFormat)
		if err != nil {
			return stats, fmt.Errorf("open WAL: %w", err)
		}
		defer wal.Close()
		if applied, rerr := wal.Replay(coreIdx); rerr != nil {
			return stats, fmt.Errorf("WAL replay: %w", rerr)
		} else if opts.Verbose && applied > 0 {
			fmt.Printf("[vacuum] replayed %d mutation(s) from WAL\n", applied)
		}
	}

	// Collect live files from the WAL-replayed index (FileMap), NOT manifest.Files.
	liveFiles := make([]fileEntry, 0, len(coreIdx.FileMap))
	coreIdx.Mu.RLock()
	for path, meta := range coreIdx.FileMap {
		if !meta.Deleted {
			mc := *meta
			liveFiles = append(liveFiles, fileEntry{path: path, meta: &mc})
		}
	}
	coreIdx.Mu.RUnlock()

	// Sort by (shard, offset) for sequential reads within each source shard.
	sort.Slice(liveFiles, func(i, j int) bool {
		if liveFiles[i].meta.ShardID == liveFiles[j].meta.ShardID {
			return liveFiles[i].meta.Offset < liveFiles[j].meta.Offset
		}
		return liveFiles[i].meta.ShardID < liveFiles[j].meta.ShardID
	})

	for _, s := range manifest.ShardsMeta {
		stats.BytesBefore += s.TotalSize
	}

	strg := storage.New(opts.Root, nil)

	var limiter *internalio.Limiter
	if opts.Throttle > 0 {
		limiter = internalio.NewLimiter(opts.Throttle)
	}

	res, err := compactShards(strg, liveFiles, opts, limiter)
	if err != nil {
		return stats, fmt.Errorf("compact: %w", err)
	}
	stats.LiveFiles = len(liveFiles)
	stats.NewShards = len(res.Manifest.ShardsMeta)
	for _, s := range res.Manifest.ShardsMeta {
		stats.BytesAfter += s.TotalSize
	}

	if opts.DryRun {
		if opts.Verbose {
			fmt.Printf("[vacuum] DRY-RUN: %d live files → %d shards, %d → %d bytes (no changes written)\n",
				stats.LiveFiles, stats.NewShards, stats.BytesBefore, stats.BytesAfter)
		}
		return stats, nil
	}

	// --- Commit ---------------------------------------------------------
	if err := backupManifest(opts.Root); err != nil {
		res.cleanupTemp()
		return stats, fmt.Errorf("backup manifest: %w", err)
	}

	// Persist the new manifest to a temp file alongside the temp shards.
	res.Manifest.Root = opts.Root
	manifestTmp, manifestFinal, err := storeManifestTemp(res.Manifest)
	if err != nil {
		res.cleanupTemp()
		return stats, fmt.Errorf("write temp manifest: %w", err)
	}

	// Rename temp shards into place, then the manifest last (commit point).
	if err := res.swapIntoPlace(manifestTmp, manifestFinal); err != nil {
		return stats, fmt.Errorf("swap into place: %w", err)
	}

	// Remove shards that the new manifest no longer references (incl. delta -1).
	if err := cleanupOldShards(opts.Root, res.keepIDs()); err != nil {
		return stats, fmt.Errorf("cleanup old shards: %w", err)
	}

	if !opts.PreserveWAL {
		if err := wal.Truncate(); err != nil {
			return stats, fmt.Errorf("truncate WAL: %w", err)
		}
	}

	removeManifestBackup(opts.Root)

	if opts.Verbose {
		fmt.Printf("[vacuum] done: %d live files → %d shards, %d → %d bytes\n",
			stats.LiveFiles, stats.NewShards, stats.BytesBefore, stats.BytesAfter)
	}
	return stats, nil
}

// Fragmentation reports the fraction of physical shard bytes that are wasted on
// tombstoned (deleted) files: deletedBytes / physicalBytes. Unlike a
// physical-minus-logical estimate, this does not count tar headers/padding as
// waste, so it doesn't trip on datasets of many small files.
func Fragmentation(m *index.Manifest) float64 {
	var deleted int64
	for _, meta := range m.Files {
		if meta.Deleted {
			deleted += meta.Size
		}
	}
	var physical int64
	for _, s := range m.ShardsMeta {
		physical += s.TotalSize
	}
	if physical == 0 {
		return 0
	}
	return float64(deleted) / float64(physical)
}

// fileEntry is one live file to repack.
type fileEntry struct {
	path string
	meta *index.Metadata
}

// --- manifest backup / atomic store helpers -----------------------------

var manifestNames = []string{"metadata.parquet", "metadata.jsonl"}

// walFileName mirrors the (unexported) constant in package index.
const walFileName = "wal.log"

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// backupManifest copies whichever manifest file(s) exist to <name>.backup.
func backupManifest(root string) error {
	found := false
	for _, name := range manifestNames {
		p := filepath.Join(root, name)
		data, err := os.ReadFile(p)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return err
		}
		if err := os.WriteFile(p+".backup", data, 0644); err != nil {
			return err
		}
		found = true
	}
	if !found {
		return fmt.Errorf("no manifest file (metadata.parquet/.jsonl) in %s", root)
	}
	return nil
}

func removeManifestBackup(root string) {
	for _, name := range manifestNames {
		os.Remove(filepath.Join(root, name+".backup"))
	}
}

// storeManifestTemp writes m to a temp directory inside root and returns the
// temp path plus the final path it should be renamed to. It uses the same
// format selection as Manifest.Store (parquet first, json fallback).
func storeManifestTemp(m *index.Manifest) (tmpPath, finalPath string, err error) {
	tmpDir, err := os.MkdirTemp(m.Root, ".vacuum-manifest-")
	if err != nil {
		return "", "", err
	}
	mm := *m
	mm.Root = tmpDir
	if serr := mm.Store(); serr != nil {
		os.RemoveAll(tmpDir)
		return "", "", serr
	}
	for _, name := range manifestNames {
		src := filepath.Join(tmpDir, name)
		if _, statErr := os.Stat(src); statErr == nil {
			return src, filepath.Join(m.Root, name), nil
		}
	}
	os.RemoveAll(tmpDir)
	return "", "", fmt.Errorf("temp manifest produced no known file")
}
