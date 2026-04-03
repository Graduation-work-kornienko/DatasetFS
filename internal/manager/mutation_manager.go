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
	walWriter *index.WAL       // Указатель на текстовый лог (append-only)
	storage   *storage.Storage

	lastShard *int // Number of last shard
}

func NewMutationManager(idx *index.CoreIndex, m *index.Manifest, wal *index.WAL, tar *storage.Storage) *MutationManager {
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

// DeleteFile обрабатывает FUSE вызов `rm`
func (m *MutationManager) DeleteFile(filename string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if err := m.coreIndex.MarkDeleted(filename); err != nil {
		return err
	}

	// m.walWriter.LogDelete(filename)

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

	if err := m.coreIndex.AddFile(meta); err != nil {
		return err
	}

	// m.walWriter.LogAdd(meta)

	return nil
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

	// 2. Append to CoreIndex
	if err := m.coreIndex.AppendShard(shard); err != nil {
		return fmt.Errorf("coreIndex.AppendShard: %w", err)
	}

	*m.lastShard++
	return nil
}

func (m *MutationManager) AppendWebDatasetShards(ctx context.Context, tarPaths []string) error {
	eg, ctx := errgroup.WithContext(ctx)
	shardChan := make(chan *index.Shard, 100)

	for _, tarPath := range tarPaths {
		eg.Go(func() error {
			return m.storage.HandleWebdatasetShard(tarPath, m.lastShard, m.mu, shardChan)
		})
	}

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

func (m *MutationManager) Shutdown() {
	m.mu.Lock()
	defer m.mu.Unlock()
	// TODO - decide, where to restore from wal log
	manifest := m.coreIndex.Manifest()
	manifest.Root = m.storage.Root
	manifest.Store()
}
