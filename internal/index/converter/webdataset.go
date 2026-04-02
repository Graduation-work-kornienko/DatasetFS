package converter

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/manager"
)

// ParseWebDataset parses dataset in format
// and returns core index structure
func ParseWebDataset(ctx context.Context, mm *manager.MutationManager, root string) error {
	entities, err := os.ReadDir(root)
	if err != nil {
		return fmt.Errorf("converter.ParseWebDataset: %w", err)
	}

	tarPaths := make([]string, 0, len(entities))
	for _, e := range entities {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".tar" {
			tarPaths = append(tarPaths, e.Name())
		}
	}

	return mm.AppendWebDatasetShards(ctx, tarPaths)
}
