package manager

import (
	"fmt"
	"sync"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
)

type MutationManager struct {
	mu          sync.RWMutex     // Гарантирует, что мы пишем дельты по очереди
	coreIndex   *index.CoreIndex // Указатель на in-memory мозг
	manifest    *index.Manifest  // manifest File
	walWriter   *index.WAL       // Указатель на текстовый лог (append-only)
	tarAppender *storage.Storage // Утилита для дописывания в delta.tar

	lastShard int // Number of last shard
}

func NewMutationManager(idx *index.CoreIndex, m *index.Manifest, wal *index.WAL, tar *storage.Storage) *MutationManager {
	return &MutationManager{
		coreIndex:   idx,
		manifest:    m,
		walWriter:   wal,
		tarAppender: tar,
		lastShard:   0,
	}
}

// Simply Append shard(for dataset initialization only)
func (m *MutationManager) AppendShard(shard *index.Shard) error {

	m.mu.Lock()
	defer m.mu.Unlock()

	shard.Number = m.lastShard

	// 1. Append to storage
	if err := m.tarAppender.AppendShard(shard); err != nil {
		return fmt.Errorf("tarAppender.AppendShard: %w", err)
	}

	// 2. Append to manifest
	if err := m.manifest.AppendShard(shard); err != nil {
		return fmt.Errorf("manifest.AppendShard: %w", err)
	}

	// 3. Append to CoreIndex
	if err := m.coreIndex.AppendShard(shard); err != nil {
		return fmt.Errorf("coreIndex.AppendShard: %w", err)
	}

	m.lastShard++
	return nil
}

func (m *MutationManager) Shutdown() {
	m.mu.Lock()
	defer m.mu.Unlock()
	// TODO - decide, where to restore from wal log
	manifest := m.coreIndex.Manifest()
	manifest.Store()
}
