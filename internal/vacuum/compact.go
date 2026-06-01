package vacuum

import (
	"archive/tar"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	internalio "github.com/Graduation-work-kornienko/DatasetFS/internal/io"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
)

// compactResult holds the in-memory manifest plus the temp shard files written
// for it (empty when DryRun).
type compactResult struct {
	Manifest  *index.Manifest
	tempPaths map[int]string // new shard id -> shard_<id>.tmp path
	strg      *storage.Storage
}

func (r *compactResult) keepIDs() map[int]bool {
	keep := make(map[int]bool, len(r.Manifest.ShardsMeta))
	for id := range r.Manifest.ShardsMeta {
		keep[id] = true
	}
	return keep
}

// cleanupTemp removes any temp shard files (used when the commit is aborted).
func (r *compactResult) cleanupTemp() {
	for _, p := range r.tempPaths {
		os.Remove(p)
	}
}

// swapIntoPlace renames temp shards to their final names, then renames the temp
// manifest into place (the commit point), then fsyncs the root directory.
func (r *compactResult) swapIntoPlace(manifestTmp, manifestFinal string) error {
	for id, tmp := range r.tempPaths {
		final := r.strg.ShardPath(id)
		if err := os.Rename(tmp, final); err != nil {
			return fmt.Errorf("rename %s -> %s: %w", tmp, final, err)
		}
	}
	if err := os.Rename(manifestTmp, manifestFinal); err != nil {
		return fmt.Errorf("rename manifest: %w", err)
	}
	// Drop the now-empty temp manifest directory and any stale other-format
	// manifest so Load (parquet-first) stays consistent.
	os.RemoveAll(filepath.Dir(manifestTmp))
	for _, name := range manifestNames {
		if filepath.Join(r.strg.Root, name) != manifestFinal {
			os.Remove(filepath.Join(r.strg.Root, name))
		}
	}
	fsyncDir(r.strg.Root)
	return nil
}

func pad512(n int64) int64 { return (512 - n%512) % 512 }

// compactShards repacks liveFiles into fresh shards. In DryRun it computes the
// resulting manifest from metadata alone and touches no files; otherwise it
// reads live bytes from the existing shards and writes shard_<n>.tmp files.
func compactShards(strg *storage.Storage, liveFiles []fileEntry, opts Options, limiter *internalio.Limiter) (*compactResult, error) {
	res := &compactResult{
		Manifest: &index.Manifest{
			Version:    "1.0",
			ShardsMeta: make(map[int]index.Shard),
			Files:      make(map[string]index.Metadata),
		},
		tempPaths: make(map[int]string),
		strg:      strg,
	}

	// Lazily opened source shards, closed at the end.
	srcShards := make(map[int]*os.File)
	defer func() {
		for _, f := range srcShards {
			f.Close()
		}
	}()
	sourceFor := func(shardID int) (*os.File, error) {
		if f, ok := srcShards[shardID]; ok {
			return f, nil
		}
		f, err := os.Open(strg.ShardPath(shardID))
		if err != nil {
			return nil, fmt.Errorf("open source shard %d: %w", shardID, err)
		}
		srcShards[shardID] = f
		return f, nil
	}

	newShardID := 0
	var off int64 // running tar offset for the current shard == its TotalSize
	var curFile *os.File
	var curTW *tar.Writer
	var curObjects []*index.Metadata

	startShard := func() error {
		off = 0
		curObjects = nil
		if opts.DryRun {
			return nil
		}
		tmp := strg.ShardPath(newShardID) + ".tmp"
		f, err := os.Create(tmp)
		if err != nil {
			return fmt.Errorf("create temp shard %d: %w", newShardID, err)
		}
		curFile = f
		curTW = tar.NewWriter(f)
		res.tempPaths[newShardID] = tmp
		return nil
	}
	finishShard := func() error {
		res.Manifest.ShardsMeta[newShardID] = index.Shard{
			Number:    newShardID,
			Type:      index.Base,
			TotalSize: off,
			Objects:   curObjects,
		}
		if opts.DryRun {
			return nil
		}
		if err := curTW.Close(); err != nil {
			return fmt.Errorf("close tar writer: %w", err)
		}
		if err := curFile.Sync(); err != nil {
			return fmt.Errorf("fsync shard: %w", err)
		}
		if err := curFile.Close(); err != nil {
			return fmt.Errorf("close shard: %w", err)
		}
		return nil
	}

	if len(liveFiles) == 0 {
		// Empty dataset: no shards at all.
		return res, nil
	}

	if err := startShard(); err != nil {
		return nil, err
	}

	for i, fe := range liveFiles {
		size := fe.meta.Size
		record := 512 + size + pad512(size)

		// Roll to a new shard if this file would overflow the soft cap (but a
		// single oversized file still gets its own shard — never split a file).
		if off+record > opts.MaxShardSize && off > 0 {
			if err := finishShard(); err != nil {
				return nil, err
			}
			newShardID++
			if err := startShard(); err != nil {
				return nil, err
			}
		}

		// Updated metadata: data sits at off+512, header at off.
		mc := *fe.meta
		mc.ShardID = newShardID
		mc.Offset = off + 512
		mc.Size = size
		mc.Deleted = false

		if !opts.DryRun {
			hdr := &tar.Header{
				Name:   filepath.Base(fe.path),
				Mode:   0600,
				Size:   size,
				Format: tar.FormatGNU,
			}
			if err := curTW.WriteHeader(hdr); err != nil {
				return nil, fmt.Errorf("write header %s: %w", fe.path, err)
			}
			src, err := sourceFor(fe.meta.ShardID)
			if err != nil {
				return nil, err
			}
			section := io.NewSectionReader(src, fe.meta.Offset, size)
			var reader io.Reader = section
			if limiter != nil {
				reader = internalio.NewLimitedReader(section, limiter)
			}
			written, err := io.Copy(curTW, reader)
			if err != nil {
				return nil, fmt.Errorf("copy %s: %w", fe.path, err)
			}
			if written != size {
				return nil, fmt.Errorf("short copy %s: wrote %d of %d", fe.path, written, size)
			}
		}

		mcCopy := mc
		curObjects = append(curObjects, &mcCopy)
		res.Manifest.Files[fe.path] = mc
		off += record

		if opts.Verbose && i%100 == 0 {
			fmt.Printf("[vacuum] packed %d/%d files\n", i+1, len(liveFiles))
		}
		if opts.Background && i%100 == 0 {
			runtime.Gosched()
		}
	}

	if err := finishShard(); err != nil {
		return nil, err
	}
	return res, nil
}
