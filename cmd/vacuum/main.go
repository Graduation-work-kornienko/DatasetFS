// Command vacuum compacts a DatasetFS dataset: it drops tombstoned files,
// repacks the live ones into fresh shards, and rewrites the manifest.
//
// The dataset must be quiescent — stop the daemon before running this.
package main

import (
	"flag"
	"log"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/vacuum"
)

func main() {
	rootPath := flag.String("root", "", "Path to DatasetFS root directory (manifest + shards)")
	dryRun := flag.Bool("dry-run", false, "Show what would be done without making changes")
	verbose := flag.Bool("verbose", false, "Enable verbose output")
	maxShardSize := flag.Int64("max-shard-size", int64(vacuum.DefaultMaxShardSize), "Soft cap for output shard size in bytes (a single file is never split)")
	preserveWAL := flag.Bool("preserve-wal", false, "Preserve WAL after vacuum instead of truncating it")
	background := flag.Bool("background", false, "Yield periodically; pair with --throttle to limit read bandwidth")
	throttle := flag.Int64("throttle", 0, "Throttle read bandwidth in bytes/sec (0 = unlimited)")
	walFormat := flag.String("wal-format", "json", "WAL format: json or binary")

	flag.Parse()

	if *rootPath == "" {
		log.Fatal("--root is required")
	}

	stats, err := vacuum.Run(vacuum.Options{
		Root:         *rootPath,
		DryRun:       *dryRun,
		Verbose:      *verbose,
		MaxShardSize: *maxShardSize,
		PreserveWAL:  *preserveWAL,
		Background:   *background,
		Throttle:     *throttle,
		WALFormat:    *walFormat,
	})
	if err != nil {
		log.Fatalf("vacuum failed: %v", err)
	}

	log.Printf("vacuum: %d live files in %d shards (%d → %d bytes)",
		stats.LiveFiles, stats.NewShards, stats.BytesBefore, stats.BytesAfter)
}
