package manager

import (
	"archive/tar"
	"context"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
	"golang.org/x/sync/errgroup"
)

type MutationManager struct {
	mu        *sync.Mutex      // Гарантирует, что мы пишем дельты по очереди
	coreIndex *index.CoreIndex // Указатель на in-memory мозг
	manifest  *index.Manifest  // manifest File
	walWriter index.WAL        // Interface for WAL operations
	storage   *storage.Storage

	lastShard *int // Number of last shard
}

func NewMutationManager(idx *index.CoreIndex, m *index.Manifest, wal index.WAL, tar *storage.Storage) *MutationManager {
	var lastShard int = 0
	deltaShardID := -1
	idx.Mu.Lock()
	if _, ok := idx.ShardMap[deltaShardID]; !ok {
		idx.ShardMap[deltaShardID] = &index.Shard{
			Number:    deltaShardID,
			Type:      "delta",
			TotalSize: 0,
			Objects:   make([]*index.Metadata, 0),
		}
	}
	idx.Mu.Unlock()
	tar.AppendShard(&index.Shard{
		Number:    -1,
		Type:      "delta",
		TotalSize: 0,
		Objects:   nil,
	})
	return &MutationManager{
		mu:        &sync.Mutex{},
		coreIndex: idx,
		manifest:  m,
		walWriter: wal,
		storage:   tar,
		lastShard: &lastShard,
	}
}

// DeleteFile обрабатывает FUSE вызов `rm`.
//
// Order matters: WAL fsync goes first, then in-memory mutation. If we crash
// between them, replay sees the WAL entry and re-marks the file deleted.
// If we crashed in the opposite order, the in-memory delete would happen but
// not survive the next manifest checkpoint — silent loss.
func (m *MutationManager) DeleteFile(filename string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.walWriter != nil {
		if err := m.walWriter.LogDelete(filename); err != nil {
			return fmt.Errorf("wal LogDelete %q: %w", filename, err)
		}
	}

	if err := m.coreIndex.MarkDeleted(filename); err != nil {
		return err
	}

	return nil
}

// AddDeltaFile обрабатывает FUSE вызов `cp` / `mv` (создание файла)
func (m *MutationManager) AddDeltaFile(logicalName string, tmpFilePath string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	stat, err := os.Stat(tmpFilePath)
	if err != nil {
		return err
	}
	fileSize := stat.Size()

	deltaPath := m.storage.ShardPath(-1)

	deltaFile, err := os.OpenFile(deltaPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer deltaFile.Close()

	deltaStat, _ := deltaFile.Stat()
	currentEndOffset := deltaStat.Size()

	tw := tar.NewWriter(deltaFile)

	hdr := &tar.Header{
		Name:   logicalName,
		Mode:   0644,
		Size:   fileSize,
		Format: tar.FormatGNU,
	}

	if err := tw.WriteHeader(hdr); err != nil {
		return err
	}

	srcFile, err := os.Open(tmpFilePath)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	written, err := io.Copy(tw, srcFile)
	if err != nil {
		return err
	}

	tw.Close()

	meta := &index.Metadata{
		ShardID: -1, // delta
		Path:    logicalName,
		Size:    written,
		Offset:  currentEndOffset + 512,
		Deleted: false,
	}

	// WAL before in-memory: a crash between tw.Close() and LogAdd leaves
	// orphan bytes in the delta tar but a consistent index. A crash between
	// LogAdd and AddFile leaves a WAL entry that replay re-applies →
	// consistent. Reverse order would lose the mutation on crash.
	if m.walWriter != nil {
		if err := m.walWriter.LogAdd(meta); err != nil {
			return fmt.Errorf("wal LogAdd %q: %w", logicalName, err)
		}
	}

	if err := m.coreIndex.AddFile(meta); err != nil {
		return err
	}

	return nil
}

// WithExclusive runs fn while holding the mutation lock, so no FUSE mutation
// (add/delete/append) interleaves. The background vacuumer uses this to rewrite
// shards and reload the index atomically with respect to writes. The delta
// shard's on-disk tar is recreated lazily on the next AddDeltaFile (it opens
// with O_CREATE), so fn only needs to restore the in-memory ShardMap[-1] entry.
func (m *MutationManager) WithExclusive(fn func() error) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return fn()
}

// Simply Append shard(for dataset initialization only)
func (m *MutationManager) AppendShard(shard *index.Shard) error {

	m.mu.Lock()
	defer m.mu.Unlock()

	shard.Number = *m.lastShard

	// 1. Append to storage
	if err := m.storage.AppendShard(shard); err != nil {
		return fmt.Errorf("tarAppender.AppendShard: %w", err)
	}

	// 2. WAL before in-memory: see DeleteFile/AddDeltaFile rationale.
	if m.walWriter != nil {
		if err := m.walWriter.LogAppendShard(shard); err != nil {
			return fmt.Errorf("wal LogAppendShard %d: %w", shard.Number, err)
		}
	}

	// 3. Append to CoreIndex
	if err := m.coreIndex.AppendShard(shard); err != nil {
		return fmt.Errorf("coreIndex.AppendShard: %w", err)
	}

	*m.lastShard++
	return nil
}

func (m *MutationManager) AppendWebDatasetShards(ctx context.Context, tarPaths []string) error {
	eg, ctx := errgroup.WithContext(ctx)
	shardChan := make(chan *index.Shard, 100)
	shardIdChan := make(chan int)

	for _, tarPath := range tarPaths {
		eg.Go(func() error {
			return m.storage.HandleWebdatasetShard(tarPath, shardIdChan, shardChan)
		})
	}

	go func() {
		id := *m.lastShard
		for {
			for {
				select {
				case shardIdChan <- id:
					id++
				case <-ctx.Done():
					return
				}
			}
		}
	}()

	// get all from metachan and
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for shard := range shardChan {
			fmt.Println("got shard", shard.Number, shard.TotalSize)
			m.coreIndex.AppendShard(shard)
		}
	}()

	if err := eg.Wait(); err != nil {
		return fmt.Errorf("dataset parsing failed: %w", err)
	}
	close(shardChan)

	wg.Wait()
	return nil
}

// Shutdown checkpoints the manifest and, on success, truncates the WAL.
// Truncate order matters: WAL must outlive the manifest write — if we lose
// power between the two, replay still recovers from the (now-redundant) WAL.
// If we truncated first, a crash before Manifest.Store would lose every
// mutation since the previous checkpoint.
func (m *MutationManager) Shutdown() {
	m.mu.Lock()
	defer m.mu.Unlock()

	manifest := m.coreIndex.Manifest()
	manifest.Root = m.storage.Root
	if err := manifest.Store(); err != nil {
		fmt.Printf("[MutationManager] Shutdown: manifest.Store failed: %v — WAL preserved\n", err)
		return
	}
	if m.walWriter != nil {
		if err := m.walWriter.Truncate(); err != nil {
			fmt.Printf("[MutationManager] Shutdown: wal.Truncate failed: %v\n", err)
		}
	}
}
