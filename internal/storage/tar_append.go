package storage

import (
	"archive/tar"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
)

const (
	ShardFormat = "shard_%d"
)

type Storage struct {
	Root string // Root of dataset
}

func New(root string) *Storage {
	return &Storage{
		Root: root,
	}
}

func (t *Storage) AppendShard(shard *index.Shard) error {
	tarPath := filepath.Join(t.Root, fmt.Sprintf(ShardFormat, shard.Number))
	file, err := os.Create(tarPath)
	if err != nil {
		return err
	}
	defer file.Close()

	tw := tar.NewWriter(file)
	defer tw.Close()

	var currentOffset int64 = 0

	for _, meta := range shard.Objects {
		meta.ShardID = shard.Number
		meta.Offset = currentOffset + 512

		srcFile, err := os.Open(meta.Path)
		if err != nil {
			return err
		}

		fileNameOnly := filepath.Base(meta.Path)
		hdr := &tar.Header{
			Name: fileNameOnly,
			Mode: 0600,
			Size: int64(meta.Size),
		}

		if err := tw.WriteHeader(hdr); err != nil {
			srcFile.Close()
			return err
		}

		written, err := io.Copy(tw, srcFile)
		srcFile.Close()
		if err != nil {
			return err
		}

		var padding int64 = 0
		remainder := written % 512
		if remainder != 0 {
			padding = 512 - remainder
		}
		currentOffset += 512 + written + padding
	}

	return nil
}
