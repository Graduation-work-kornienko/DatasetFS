package manager

import (
	"context"
	"fmt"
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
	return &MutationManager{
		mu:        &sync.Mutex{},
		coreIndex: idx,
		manifest:  m,
		walWriter: wal,
		storage:   tar,
		lastShard: &lastShard,
	}
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
	go func() {
		for shard := range shardChan {
			fmt.Println("got shard", shard.Number, shard.TotalSize)
			m.coreIndex.AppendShard(shard)
		}
	}()

	if err := eg.Wait(); err != nil {
		return fmt.Errorf("dataset parsing failed: %w", err)
	}
	close(shardChan)

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
