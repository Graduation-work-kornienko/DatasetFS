package converter

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/manager"
	"golang.org/x/sync/errgroup"
)

// ParseDatasetFolder parses dataset in format https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.DatasetFolder.html#torchvision.datasets.DatasetFolder
// and returns core index structure
func ParseDatasetFolder(ctx context.Context, mm *manager.MutationManager, root string) error {
	entities, err := os.ReadDir(root)
	if err != nil {
		return fmt.Errorf("converter.ParseDatasetFolder: %w", err)
	}

	fileChan := make(chan *index.Metadata, 1000)

	var readerWg sync.WaitGroup
	eg, ctx := errgroup.WithContext(ctx)
	for _, e := range entities {
		if e.IsDir() {
			dirName := e.Name()
			readerWg.Add(1)

			eg.Go(func() error {
				defer readerWg.Done()
				return ParseLabelDir(ctx, filepath.Join(root, dirName), dirName, fileChan)
			})
		}
	}

	go func() {
		readerWg.Wait()
		close(fileChan)
	}()

	var sizeCount int64
	tarSlice := make([]*index.Metadata, 0)

	for f := range fileChan {
		sizeCount += int64(f.Size)
		tarSlice = append(tarSlice, f)

		if sizeCount > int64(index.ShardSize) {
			shard := &index.Shard{
				Type:    "base",
				Objects: tarSlice,
			}

			eg.Go(func() error {
				return mm.AppendShard(shard)
			})

			tarSlice = make([]*index.Metadata, 0)
			sizeCount = 0
		}
	}

	if len(tarSlice) > 0 {
		shard := &index.Shard{Type: "base", Objects: tarSlice}
		eg.Go(func() error {
			return mm.AppendShard(shard)
		})
	}

	if err := eg.Wait(); err != nil {
		return fmt.Errorf("dataset parsing failed: %w", err)
	}

	return nil
}

func ParseLabelDir(ctx context.Context, folder string, label string, fileChan chan<- *index.Metadata) error {

	entities, err := os.ReadDir(folder)
	if err != nil {
		return err
	}

	for _, e := range entities {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			if !e.IsDir() {

				stat, err := os.Stat(filepath.Join(folder, e.Name()))
				if err != nil {
					return err
				}
				meta := &index.Metadata{
					// ShardID sets in storage
					// Offset sets in storage
					Size:           stat.Size(),
					Path:           filepath.Join(folder, e.Name()),
					ObjectMetadata: json.RawMessage([]byte(fmt.Sprintf("{\"label\": \"%s\"}", label))), // maybe smarter
				}
				fileChan <- meta
			}
		}

	}
	return nil
}
