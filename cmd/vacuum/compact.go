package main

import (
	"archive/tar"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	internalio "github.com/Graduation-work-kornienko/DatasetFS/internal/io"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
)

// compactShards creates new, defragmented shards from live files
func compactShards(storage *storage.Storage, liveFiles []fileEntry, maxShardSize int64, dryRun, verbose, background bool, limiter *internalio.Limiter) (*index.Manifest, error) {
	manifest := &index.Manifest{
		Version:    "1.0",
		ShardsMeta: make(map[int]index.Shard),
		Files:      make(map[string]index.Metadata),
	}

	var currentShardID int = 0
	var currentShardSize int64 = 0
	var currentShard *index.Shard
	var currentFile *os.File
	var currentWriter *tar.Writer

	// Initialize first shard
	if err := createNewShard(storage, currentShardID, &currentFile, &currentWriter); err != nil {
		return nil, fmt.Errorf("failed to create initial shard: %w", err)
	}
	currentShard = &index.Shard{Number: currentShardID, Objects: make([]*index.Metadata, 0)}
	manifest.ShardsMeta[currentShardID] = *currentShard

	for i, fileEntry := range liveFiles {
		// Open source file
		sourceFile, err := os.Open(fileEntry.path)
		if err != nil {
			return nil, fmt.Errorf("failed to open source file %s: %w", fileEntry.path, err)
		}
		defer sourceFile.Close()

		// Get file info
		fileInfo, err := sourceFile.Stat()
		if err != nil {
			return nil, fmt.Errorf("failed to stat source file %s: %w", fileEntry.path, err)
		}

		// Check if file fits in current shard
		fileSize := fileInfo.Size()
		metadataSize := int64(512) // Tar header size
		totalSize := fileSize + metadataSize

		// Add padding to make file size multiple of 512 bytes
		padding := int64(0)
		remainder := totalSize % 512
		if remainder != 0 {
			padding = 512 - remainder
		}
		totalSize += padding

		// If file doesn't fit, create new shard
		if currentShardSize+totalSize > maxShardSize && currentShardSize > 0 {
			// Close current shard
			if err := closeShard(currentFile, currentWriter); err != nil {
				return nil, fmt.Errorf("failed to close shard %d: %w", currentShardID, err)
			}

			// Create new shard
			currentShardID++
			if err := createNewShard(storage, currentShardID, &currentFile, &currentWriter); err != nil {
				return nil, fmt.Errorf("failed to create shard %d: %w", currentShardID, err)
			}
			currentShardSize = 0
			currentShard = &index.Shard{Number: currentShardID, Objects: make([]*index.Metadata, 0)}
			manifest.ShardsMeta[currentShardID] = *currentShard
		}

		// Create tar header
		fileNameOnly := filepath.Base(fileEntry.path)
		hdr := &tar.Header{
			Name: fileNameOnly,
			Mode: 0600,
			Size: fileSize,
		}

		// Write header to tar
		if err := currentWriter.WriteHeader(hdr); err != nil {
			return nil, fmt.Errorf("failed to write header for %s: %w", fileEntry.path, err)
		}

		// Copy file data with optional rate limiting
		var reader io.Reader = sourceFile
		if limiter != nil {
			reader = internalio.NewLimitedReader(reader, limiter)
		}
		if _, err := io.Copy(currentWriter, reader); err != nil {
			return nil, fmt.Errorf("failed to copy data for %s: %w", fileEntry.path, err)
		}

		// Add padding
		if padding > 0 {
			padBuf := make([]byte, padding)
			if _, err := currentWriter.Write(padBuf); err != nil {
				return nil, fmt.Errorf("failed to write padding for %s: %w", fileEntry.path, err)
			}
		}

		// Update metadata
		metaCopy := *fileEntry.meta
		metaCopy.ShardID = currentShardID
		metaCopy.Offset = currentShardSize + 512
		metaCopy.Size = fileSize
		currentShard.Objects = append(currentShard.Objects, &metaCopy)
		manifest.Files[fileEntry.path] = metaCopy

		// Update shard size
		currentShardSize += totalSize

		// Update current shard in manifest
		manifest.ShardsMeta[currentShardID] = *currentShard

		// Progress logging
		if verbose && i%100 == 0 {
			log.Printf("Processed %d/%d files", i+1, len(liveFiles))
		}

		// Yield for background mode
		if background && i%100 == 0 {
			runtime.Gosched()
		}
	}

	// Close final shard
	if err := closeShard(currentFile, currentWriter); err != nil {
		return nil, fmt.Errorf("failed to close final shard: %w", err)
	}

	// Update total size for each shard
	for shardID, shard := range manifest.ShardsMeta {
		var totalSize int64 = 0
		for _, meta := range shard.Objects {
			totalSize += meta.Size + 512 // File size + tar header
			// Add padding
			remainder := totalSize % 512
			if remainder != 0 {
				totalSize += 512 - remainder
			}
		}
		shard.TotalSize = totalSize
		manifest.ShardsMeta[shardID] = shard
	}

	return manifest, nil
}

// createNewShard creates a new shard file and tar writer
func createNewShard(storage *storage.Storage, shardID int, file **os.File, writer **tar.Writer) error {
	filename := storage.ShardPath(shardID)
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create shard %s: %w", filename, err)
	}

	*file = f
	*writer = tar.NewWriter(f)
	return nil
}

// closeShard closes the tar writer and file
func closeShard(file *os.File, writer *tar.Writer) error {
	if writer != nil {
		if err := writer.Close(); err != nil {
			return fmt.Errorf("failed to close tar writer: %w", err)
		}
	}
	if file != nil {
		if err := file.Close(); err != nil {
			return fmt.Errorf("failed to close shard file: %w", err)
		}
	}
	return nil
}
