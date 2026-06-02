package main

import (
	"fmt"
	"log"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/vacuum"
	"github.com/spf13/cobra"
)

// newVacuumCmd builds the `vacuum` subcommand: it drops tombstoned files,
// repacks the live ones into fresh shards, and rewrites the manifest. The
// dataset must be quiescent — stop the daemon before running this.
func newVacuumCmd() *cobra.Command {
	var (
		rootPath     string
		dryRun       bool
		verbose      bool
		maxShardSize int64
		preserveWAL  bool
		background   bool
		throttle     int64
		walFormat    string
	)

	cmd := &cobra.Command{
		Use:           "vacuum",
		Short:         "Compact a dataset: drop tombstones, repack shards, rewrite the manifest",
		SilenceUsage:  true,
		SilenceErrors: false,
		RunE: func(cmd *cobra.Command, args []string) error {
			if rootPath == "" {
				return fmt.Errorf("--root is required")
			}

			stats, err := vacuum.Run(vacuum.Options{
				Root:         rootPath,
				DryRun:       dryRun,
				Verbose:      verbose,
				MaxShardSize: maxShardSize,
				PreserveWAL:  preserveWAL,
				Background:   background,
				Throttle:     throttle,
				WALFormat:    walFormat,
			})
			if err != nil {
				return fmt.Errorf("vacuum failed: %w", err)
			}

			log.Printf("vacuum: %d live files in %d shards (%d → %d bytes)",
				stats.LiveFiles, stats.NewShards, stats.BytesBefore, stats.BytesAfter)
			return nil
		},
	}

	cmd.Flags().StringVar(&rootPath, "root", "", "Path to DatasetFS root directory (manifest + shards)")
	cmd.Flags().BoolVar(&dryRun, "dry-run", false, "Show what would be done without making changes")
	cmd.Flags().BoolVar(&verbose, "verbose", false, "Enable verbose output")
	cmd.Flags().Int64Var(&maxShardSize, "max-shard-size", int64(vacuum.DefaultMaxShardSize), "Soft cap for output shard size in bytes (a single file is never split)")
	cmd.Flags().BoolVar(&preserveWAL, "preserve-wal", false, "Preserve WAL after vacuum instead of truncating it")
	cmd.Flags().BoolVar(&background, "background", false, "Yield periodically; pair with --throttle to limit read bandwidth")
	cmd.Flags().Int64Var(&throttle, "throttle", 0, "Throttle read bandwidth in bytes/sec (0 = unlimited)")
	cmd.Flags().StringVar(&walFormat, "wal-format", "json", "WAL format: json or binary")

	return cmd
}
