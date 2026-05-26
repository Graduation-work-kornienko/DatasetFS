package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
)

// cleanupOldShards removes shard files that are no longer referenced in the new manifest
func cleanupOldShards(rootPath string, oldShards, newShards map[int]index.Shard) error {
	// Identify shards to remove (present in old but not in new)
	for shardID := range oldShards {
		if _, exists := newShards[shardID]; !exists {
			shardPath := filepath.Join(rootPath, fmt.Sprintf("shard_%05d.tar", shardID))
			if err := os.Remove(shardPath); err != nil && !os.IsNotExist(err) {
				return fmt.Errorf("failed to remove old shard %d: %w", shardID, err)
			}
		}
	}
	return nil
}
