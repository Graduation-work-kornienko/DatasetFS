package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"runtime"
	"syscall"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/ipc"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/manager"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/vfs"
	"github.com/hanwen/go-fuse/v2/fs"
	"github.com/hanwen/go-fuse/v2/fuse"
)

func main() {
	rootPath := flag.String("root", "cmd/dataset_converter/test", "Path to DatasetFS converted dataset (manifest + shards)")
	mountPoint := flag.String("mount", "./dataset_mount", "FUSE mount point (ignored with --no-mount)")
	noMount := flag.Bool("no-mount", false, "Skip FUSE mount (run IPC + pipeline only). Useful for tests / non-FUSE hosts")
	mutexProfileRate := flag.Int("mutex-profile-rate", 0, "If >0, enables /debug/pprof/mutex with 1-in-N sampling. Adds ~1-3% overhead.")
	blockProfileRate := flag.Int("block-profile-rate", 0, "If >0, enables /debug/pprof/block with rate in ns. 1 means every blocking event.")
	cacheDir := flag.String("cache-dir", "./dataset_cache", "Directory for caching remote datasets")
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

	// Get local path for the root (download if remote)
	var localRoot string
	var err error
	if remoteStorage != nil {
		localRoot, err = remoteStorage.GetLocalPath(context.Background(), *rootPath)
		if err != nil {
			log.Fatalf("Failed to download remote dataset: %v", err)
		}
	} else {
		localRoot = *rootPath
	}

	mnfst := index.NewManifest(localRoot)
	err = mnfst.Load(remoteStorage)
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
	wal, err := index.OpenWALWithFormat(*rootPath, "json")
	if err != nil {
		log.Fatalf("open WAL: %v", err)
	}

	go ipc.StartServer(coreIdx, *rootPath)

	mutMgr := manager.NewMutationManager(coreIdx, mnfst, wal, strg)

	// Replay AFTER NewMutationManager so the delta shard placeholder (id=-1)
	// is in coreIdx — replay's AddFile entries reference it.
	if applied, err := wal.Replay(coreIdx); err != nil {
		log.Fatalf("WAL replay failed: %v (recover by inspecting/deleting %s)", err, wal.Path())
	} else if applied > 0 {
		log.Printf("WAL: replayed %d mutation(s) since last checkpoint", applied)
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
