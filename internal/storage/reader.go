package storage

import (
	"fmt"
	"path/filepath"
)

func (s *Storage) ShardPath(shardID int) string {
	return filepath.Join(s.Root, fmt.Sprintf(ShardFormat, shardID))
}
