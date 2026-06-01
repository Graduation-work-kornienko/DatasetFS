package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/ipc"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/manager"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/vfs"
	"github.com/hanwen/go-fuse/v2/fs"
	"github.com/hanwen/go-fuse/v2/fuse"
)

// prefetchRemoteDataset materializes a remote (HTTP) dataset into cacheDir:
// the manifest (parquet preferred, jsonl fallback) plus every base shard. After
// this the whole pipeline runs purely local with Root=cacheDir. Shard ids come
// from the manifest, so a plain anonymous-GET bucket policy is enough (no
// bucket listing). URLs are joined path-style — never filepath.Join, which
// would mangle "http://".
func prefetchRemoteDataset(rs *storage.RemoteStorage, rootURL, cacheDir string) (string, error) {
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return "", err
	}
	base := strings.TrimRight(rootURL, "/")
	ctx := context.Background()

	// Manifest: try parquet, then jsonl.
	gotManifest := false
	for _, name := range []string{"metadata.parquet", "metadata.jsonl"} {
		if err := rs.Fetch(ctx, base+"/"+name, filepath.Join(cacheDir, name)); err == nil {
			gotManifest = true
			log.Printf("[prefetch] manifest %s", name)
			break
		}
	}
	if !gotManifest {
		return "", fmt.Errorf("no manifest (metadata.parquet/.jsonl) at %s", base)
	}

	mnfst := index.NewManifest(cacheDir)
	if err := mnfst.Load(nil); err != nil {
		return "", fmt.Errorf("load prefetched manifest: %w", err)
	}
	for id := range mnfst.ShardsMeta {
		if id < 0 {
			continue // delta placeholder has no remote shard file
		}
		name := fmt.Sprintf("shard_%d", id)
		if err := rs.Fetch(ctx, fmt.Sprintf("%s/%s", base, name), filepath.Join(cacheDir, name)); err != nil {
			return "", fmt.Errorf("prefetch %s: %w", name, err)
		}
		log.Printf("[prefetch] %s", name)
	}
	return cacheDir, nil
}

func main() {
	rootPath := flag.String("root", "cmd/dataset_converter/test", "Path to DatasetFS converted dataset (manifest + shards)")
	mountPoint := flag.String("mount", "./dataset_mount", "FUSE mount point (ignored with --no-mount)")
	noMount := flag.Bool("no-mount", false, "Skip FUSE mount (run IPC + pipeline only). Useful for tests / non-FUSE hosts")
	mutexProfileRate := flag.Int("mutex-profile-rate", 0, "If >0, enables /debug/pprof/mutex with 1-in-N sampling. Adds ~1-3% overhead.")
	blockProfileRate := flag.Int("block-profile-rate", 0, "If >0, enables /debug/pprof/block with rate in ns. 1 means every blocking event.")
	cacheDir := flag.String("cache-dir", "./dataset_cache", "Directory for caching remote datasets")
	walFormat := flag.String("wal-format", "json", "Format for WAL(supported json and binary)")
	autoVacuum := flag.Bool("auto-vacuum", false, "Run the background vacuumer goroutine (compacts deleted files when idle)")
	vacuumInterval := flag.Duration("vacuum-interval", 5*time.Minute, "How often the background vacuumer checks fragmentation")
	vacuumThreshold := flag.Float64("vacuum-threshold", 0.3, "Fragmentation ratio (deleted/physical) that triggers a vacuum")
	vacuumThrottle := flag.Int64("vacuum-throttle", 0, "Read bandwidth limit for background vacuum in bytes/sec (0 = unlimited)")
	flag.Parse()

	if *mutexProfileRate > 0 {
		runtime.SetMutexProfileFraction(*mutexProfileRate)
		log.Printf("[pprof] mutex profile enabled, fraction=1/%d", *mutexProfileRate)
	}
	if *blockProfileRate > 0 {
		runtime.SetBlockProfileRate(*blockProfileRate)
		log.Printf("[pprof] block profile enabled, rate=%d ns", *blockProfileRate)
	}

	// Create remote storage if needed
	var remoteStorage *storage.RemoteStorage
	if storage.IsURL(*rootPath) {
		remoteStorage = storage.NewRemoteStorage(*cacheDir)
	}

	// Resolve the local root. For a remote (HTTP) dataset, prefetch the manifest
	// and every shard into the cache dir up front, then run purely local.
	var localRoot string
	var err error
	if remoteStorage != nil {
		localRoot, err = prefetchRemoteDataset(remoteStorage, *rootPath, *cacheDir)
		if err != nil {
			log.Fatalf("Failed to prefetch remote dataset: %v", err)
		}
	} else {
		localRoot = *rootPath
	}

	mnfst := index.NewManifest(localRoot)
	err = mnfst.Load(nil)
	if err != nil {
		log.Fatalf("Failed to load manifest: %v", err)
	}
	coreIdx, err := mnfst.LoadCoreIndex()
	if err != nil {
		log.Fatalf("load core index fail: %v\n", err)
	}
	log.Println("Loaded")

	strg := storage.New(localRoot, remoteStorage)

	// Open WAL early so MutationManager can write to it, and so Replay below
	// recovers any mutations that happened since the last manifest checkpoint.
	wal, err := index.OpenWALWithFormat(localRoot, *walFormat)
	if err != nil {
		log.Fatalf("open WAL: %v", err)
	}

	go ipc.StartServer(coreIdx, strg)

	mutMgr := manager.NewMutationManager(coreIdx, mnfst, wal, strg)

	// Replay AFTER NewMutationManager so the delta shard placeholder (id=-1)
	// is in coreIdx — replay's AddFile entries reference it.
	if applied, err := wal.Replay(coreIdx); err != nil {
		log.Fatalf("WAL replay failed: %v (recover by inspecting/deleting %s)", err, wal.Path())
	} else if applied > 0 {
		log.Printf("WAL: replayed %d mutation(s) since last checkpoint", applied)
	}

	// Background vacuumer (opt-in). Runs as a daemon goroutine — not a separate
	// process — so it coordinates with the loading pipeline and FUSE mutations.
	if *autoVacuum {
		avCtx, avCancel := context.WithCancel(context.Background())
		defer avCancel()
		go runAutoVacuum(avCtx, autoVacuumConfig{
			root:      localRoot,
			walFormat: *walFormat,
			interval:  *vacuumInterval,
			threshold: *vacuumThreshold,
			throttle:  *vacuumThrottle,
		}, coreIdx, mutMgr)
	}

	if *noMount {
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
		return
	}

	root := &vfs.RootNode{
		CoreIdx: coreIdx,
		MutMgr:  mutMgr,
		Storage: strg,
	}

	log.Println("Mounting")
	os.MkdirAll(*mountPoint, 0755)
	server, err := fs.Mount(*mountPoint, root, &fs.Options{
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
	log.Println("Mounted")
	if err != nil {
		log.Fatalf("Mount fail: %v\n", err)
	}

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-c
		log.Println("\nUnmounting...")
		err := server.Unmount()
		if err != nil {
			log.Printf("Unmount error: %v", err)
		}
	}()

	log.Printf("Successfully mounted DatasetFS %s", *mountPoint)
	server.Wait()

	log.Println("Saving manifest")
	mutMgr.Shutdown()
	if cerr := wal.Close(); cerr != nil {
		log.Printf("wal.Close: %v", cerr)
	}
	log.Println("Daemon stopped")
}
