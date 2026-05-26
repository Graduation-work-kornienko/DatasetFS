package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/io"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
)

func main() {
	rootPath := flag.String("root", "", "Path to DatasetFS root directory (containing manifest and shards)")
	dryRun := flag.Bool("dry-run", false, "Show what would be done without making changes")
	verbose := flag.Bool("verbose", false, "Enable verbose output")
	maxShardSize := flag.Int64("max-shard-size", 100*1024*1024, "Maximum size for output shards in bytes")
	preserveWAL := flag.Bool("preserve-wal", false, "Preserve WAL after vacuum (useful for debugging)")
	background := flag.Bool("background", false, "Run vacuum in background mode with throttling")
	throttle := flag.Int64("throttle", 0, "Throttle disk bandwidth in bytes/sec (0 = unlimited)")

	flag.Parse()

	if *rootPath == "" {
		log.Fatal("--root is required")
	}

	// Execute vacuum operation
	if err := vacuum(*rootPath, *dryRun, *verbose, *maxShardSize, *preserveWAL, *background, *throttle); err != nil {
		log.Fatal(err)
	}
}

func vacuum(rootPath string, dryRun, verbose bool, maxShardSize int64, preserveWAL, background bool, throttle int64) error {
	// Validate root path
	if _, err := os.Stat(rootPath); os.IsNotExist(err) {
		return fmt.Errorf("root path does not exist: %s", rootPath)
	}

	// Check for active processes
	if isDaemonRunning(rootPath) {
		return fmt.Errorf("DatasetFS daemon is running; stop it before vacuuming")
	}

	// Create backup of manifest if not in dry-run mode
	if !dryRun {
		if err := createManifestBackup(rootPath); err != nil {
			return fmt.Errorf("failed to create manifest backup: %w", err)
		}
		defer func() {
			// Cleanup backup on successful completion
			removeManifestBackup(rootPath)
		}()

		// Ensure partial outputs are cleaned up on panic
		defer func() {
			if r := recover(); r != nil {
				cleanupPartialOutputs(rootPath)
				panic(r)
			}
		}()
	}

	// Load current state
	manifest := &index.Manifest{Root: rootPath}
	if err := manifest.Load(nil); err != nil {
		return fmt.Errorf("failed to load manifest: %w", err)
	}

	// Open WAL for replay
	wal, err := index.OpenWAL(rootPath)
	if err != nil {
		return fmt.Errorf("failed to open WAL: %w", err)
	}
	defer wal.Close()

	// Replay WAL to get latest state
	coreIndex, err := manifest.LoadCoreIndex()
	if err != nil {
		return fmt.Errorf("failed to load core index: %w", err)
	}
	if applied, err := wal.Replay(coreIndex); err != nil {
		return fmt.Errorf("WAL replay failed: %w", err)
	} else if verbose && applied > 0 {
		log.Printf("Replayed %d mutations from WAL", applied)
	}

	// Collect live files sorted by current location
	var liveFiles []fileEntry
	for path, meta := range manifest.Files {
		if !meta.Deleted {
			liveFiles = append(liveFiles, fileEntry{
				path: path,
				meta: &meta,
			})
		}
	}

	// Sort by shard ID and offset for sequential access
	sort.Slice(liveFiles, func(i, j int) bool {
		if liveFiles[i].meta.ShardID == liveFiles[j].meta.ShardID {
			return liveFiles[i].meta.Offset < liveFiles[j].meta.Offset
		}
		return liveFiles[i].meta.ShardID < liveFiles[j].meta.ShardID
	})

	// Create new storage for compacted data
	storage := &storage.Storage{Root: rootPath}

	// Create I/O limiter if throttling is enabled
	var limiter *io.Limiter
	if throttle > 0 {
		limiter = io.NewLimiter(throttle)
	}

	// Compact into new shards
	newManifest, err := compactShards(storage, liveFiles, maxShardSize, dryRun, verbose, background, limiter)
	if err != nil {
		return fmt.Errorf("failed to compact shards: %w", err)
	}

	// Update manifest
	if !dryRun {
		newManifest.Root = rootPath
		if err := newManifest.Store(); err != nil {
			return fmt.Errorf("failed to store new manifest: %w", err)
		}

		// Clean up old shards
		if err := cleanupOldShards(rootPath, manifest.ShardsMeta, newManifest.ShardsMeta); err != nil {
			return fmt.Errorf("failed to cleanup old shards: %w", err)
		}

		// Truncate WAL if requested
		if !preserveWAL {
			if err := wal.Truncate(); err != nil {
				return fmt.Errorf("failed to truncate WAL: %w", err)
			}
		}
	}

	if verbose {
		log.Printf("Vacuum completed: %d files in %d shards", len(liveFiles), len(newManifest.ShardsMeta))
	}

	return nil
}

// fileEntry represents a file to be included in compaction
type fileEntry struct {
	path string
	meta *index.Metadata
}

// isDaemonRunning checks if the DatasetFS daemon is currently running
func isDaemonRunning(rootPath string) bool {
	// Implementation would check for daemon lock files or process status
	// This is a placeholder for the actual implementation
	lockPath := filepath.Join(rootPath, ".daemon_lock")
	_, err := os.Stat(lockPath)
	return !os.IsNotExist(err)
}

// createManifestBackup creates a backup of the current manifest
func createManifestBackup(rootPath string) error {
	manifestPath := filepath.Join(rootPath, "metadata.jsonl")
	backupPath := filepath.Join(rootPath, "metadata.jsonl.backup")

	input, err := os.ReadFile(manifestPath)
	if err != nil {
		return err
	}

	return os.WriteFile(backupPath, input, 0644)
}

// removeManifestBackup removes the manifest backup file
func removeManifestBackup(rootPath string) {
	backupPath := filepath.Join(rootPath, "metadata.jsonl.backup")
	os.Remove(backupPath)
}

// cleanupPartialOutputs cleans up any partial outputs if the operation fails
func cleanupPartialOutputs(rootPath string) {
	// Remove any temporary manifest files
	tempManifest := filepath.Join(rootPath, "metadata.jsonl.tmp")
	os.Remove(tempManifest)

	// Note: We don't remove new shard files here as they might be needed for recovery
	// This is a trade-off between cleanup and recovery options
}
