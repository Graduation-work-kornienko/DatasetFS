package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/control"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/manager"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/vfs"
	"github.com/hanwen/go-fuse/v2/fs"
	"github.com/hanwen/go-fuse/v2/fuse"
	"github.com/spf13/cobra"
)

// newDaemonCmd builds the `daemon` subcommand: it loads the manifest+index,
// starts the IPC control plane, replays the WAL, optionally runs the background
// vacuumer, and (unless --no-mount) mounts the FUSE filesystem.
func newDaemonCmd() *cobra.Command {
	var (
		rootPath         string
		mountPoint       string
		noMount          bool
		mutexProfileRate int
		blockProfileRate int
		cacheDir            string
		prefetchConcurrency int
		remoteThrottle      int64
		walFormat           string
		autoVacuum          bool
		vacuumInterval   time.Duration
		vacuumThreshold  float64
		vacuumThrottle   int64
	)

	cmd := &cobra.Command{
		Use:           "daemon",
		Short:         "Mount FUSE and serve the IPC/SHM loading pipeline",
		SilenceUsage:  true,
		SilenceErrors: false,
		RunE: func(cmd *cobra.Command, args []string) error {
			if mutexProfileRate > 0 {
				runtime.SetMutexProfileFraction(mutexProfileRate)
				log.Printf("[pprof] mutex profile enabled, fraction=1/%d", mutexProfileRate)
			}
			if blockProfileRate > 0 {
				runtime.SetBlockProfileRate(blockProfileRate)
				log.Printf("[pprof] block profile enabled, rate=%d ns", blockProfileRate)
			}

			// Create remote storage if needed.
			var remoteStorage *storage.RemoteStorage
			if storage.IsURL(rootPath) {
				remoteStorage = storage.NewRemoteStorage(cacheDir, remoteThrottle)
			}

			// Resolve the local root. For a remote (HTTP) dataset, fetch ONLY the
			// manifest into the cache dir and build a streaming prefetcher; the
			// shards download in the background while training runs (G9/G14).
			var localRoot string
			var prefetcher *storage.RemotePrefetcher
			var err error
			if remoteStorage != nil {
				localRoot, prefetcher, err = prefetchRemoteManifest(remoteStorage, rootPath, cacheDir, prefetchConcurrency)
				if err != nil {
					return fmt.Errorf("prefetch remote manifest: %w", err)
				}
			} else {
				localRoot = rootPath
			}

			mnfst := index.NewManifest(localRoot)
			if err := mnfst.Load(nil); err != nil {
				return fmt.Errorf("load manifest: %w", err)
			}
			coreIdx, err := mnfst.LoadCoreIndex()
			if err != nil {
				return fmt.Errorf("load core index: %w", err)
			}
			log.Println("Loaded")

			strg := storage.New(localRoot, remoteStorage)
			// Wire the streaming prefetcher and start its background workers.
			if prefetcher != nil {
				strg.Prefetcher = prefetcher
				prefetcher.Start(cmd.Context())
			}

			// Open WAL early so MutationManager can write to it, and so Replay
			// below recovers any mutations since the last manifest checkpoint.
			wal, err := index.OpenWALWithFormat(localRoot, walFormat)
			if err != nil {
				return fmt.Errorf("open WAL: %w", err)
			}

			go control.StartServer(coreIdx, strg)

			mutMgr := manager.NewMutationManager(coreIdx, mnfst, wal, strg)

			// Replay AFTER NewMutationManager so the delta shard placeholder
			// (id=-1) is in coreIdx — replay's AddFile entries reference it.
			if applied, err := wal.Replay(coreIdx); err != nil {
				return fmt.Errorf("WAL replay failed: %w (recover by inspecting/deleting %s)", err, wal.Path())
			} else if applied > 0 {
				log.Printf("WAL: replayed %d mutation(s) since last checkpoint", applied)
			}

			// Background vacuumer (opt-in). Runs as a daemon goroutine — not a
			// separate process — so it coordinates with the loading pipeline and
			// FUSE mutations.
			if autoVacuum {
				avCtx, avCancel := context.WithCancel(context.Background())
				defer avCancel()
				go runAutoVacuum(avCtx, autoVacuumConfig{
					root:      localRoot,
					walFormat: walFormat,
					interval:  vacuumInterval,
					threshold: vacuumThreshold,
					throttle:  vacuumThrottle,
				}, coreIdx, mutMgr)
			}

			if noMount {
				log.Println("Running without FUSE mount (--no-mount)")
				c := make(chan os.Signal, 1)
				signal.Notify(c, os.Interrupt, syscall.SIGTERM)
				<-c
				log.Println("Saving manifest")
				mutMgr.Shutdown()
				if cerr := wal.Close(); cerr != nil {
					log.Printf("wal.Close: %v", cerr)
				}
				log.Println("Daemon stopped")
				return nil
			}

			root := &vfs.RootNode{
				CoreIdx: coreIdx,
				MutMgr:  mutMgr,
				Storage: strg,
			}

			log.Println("Mounting")
			os.MkdirAll(mountPoint, 0755)
			server, err := fs.Mount(mountPoint, root, &fs.Options{
				MountOptions: fuse.MountOptions{
					FsName: "DatasetFS",
					Name:   "DatasetFS",
					Options: []string{
						"local",
						"volname=DatasetFS",
						"noappledouble",
						"noapplexattr",
					},
				},
			})
			if err != nil {
				return fmt.Errorf("mount: %w", err)
			}
			log.Println("Mounted")

			c := make(chan os.Signal, 1)
			signal.Notify(c, os.Interrupt, syscall.SIGTERM)

			go func() {
				<-c
				log.Println("\nUnmounting...")
				if err := server.Unmount(); err != nil {
					log.Printf("Unmount error: %v", err)
				}
			}()

			log.Printf("Successfully mounted DatasetFS %s", mountPoint)
			server.Wait()

			log.Println("Saving manifest")
			mutMgr.Shutdown()
			if cerr := wal.Close(); cerr != nil {
				log.Printf("wal.Close: %v", cerr)
			}
			log.Println("Daemon stopped")
			return nil
		},
	}

	cmd.Flags().StringVar(&rootPath, "root", "testdata/dataset", "Path to DatasetFS converted dataset (manifest + shards)")
	cmd.Flags().StringVar(&mountPoint, "mount", "./dataset_mount", "FUSE mount point (ignored with --no-mount)")
	cmd.Flags().BoolVar(&noMount, "no-mount", false, "Skip FUSE mount (run IPC + pipeline only). Useful for tests / non-FUSE hosts")
	cmd.Flags().IntVar(&mutexProfileRate, "mutex-profile-rate", 0, "If >0, enables /debug/pprof/mutex with 1-in-N sampling. Adds ~1-3% overhead.")
	cmd.Flags().IntVar(&blockProfileRate, "block-profile-rate", 0, "If >0, enables /debug/pprof/block with rate in ns. 1 means every blocking event.")
	cmd.Flags().StringVar(&cacheDir, "cache-dir", "./dataset_cache", "Directory for caching remote datasets")
	cmd.Flags().IntVar(&prefetchConcurrency, "prefetch-concurrency", 4, "Background download workers for remote streaming (overlap with training)")
	cmd.Flags().Int64Var(&remoteThrottle, "remote-throttle", 0, "Limit aggregate remote download bandwidth in bytes/sec (0 = unlimited)")
	cmd.Flags().StringVar(&walFormat, "wal-format", "json", "Format for WAL (supported: json and binary)")
	cmd.Flags().BoolVar(&autoVacuum, "auto-vacuum", false, "Run the background vacuumer goroutine (compacts deleted files when idle)")
	cmd.Flags().DurationVar(&vacuumInterval, "vacuum-interval", 5*time.Minute, "How often the background vacuumer checks fragmentation")
	cmd.Flags().Float64Var(&vacuumThreshold, "vacuum-threshold", 0.3, "Fragmentation ratio (deleted/physical) that triggers a vacuum")
	cmd.Flags().Int64Var(&vacuumThrottle, "vacuum-throttle", 0, "Read bandwidth limit for background vacuum in bytes/sec (0 = unlimited)")

	return cmd
}
