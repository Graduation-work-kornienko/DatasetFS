package main

import (
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
	flag.Parse()

	if *mutexProfileRate > 0 {
		runtime.SetMutexProfileFraction(*mutexProfileRate)
		log.Printf("[pprof] mutex profile enabled, fraction=1/%d", *mutexProfileRate)
	}
	if *blockProfileRate > 0 {
		runtime.SetBlockProfileRate(*blockProfileRate)
		log.Printf("[pprof] block profile enabled, rate=%d ns", *blockProfileRate)
	}

	mnfst := index.NewManifest(*rootPath)
	mnfst.Load()
	coreIdx, err := mnfst.LoadCoreIndex()
	if err != nil {
		log.Fatalf("load core index fail: %v\n", err)
	}
	log.Println("Loaded")

	strg := &storage.Storage{Root: *rootPath}

	go ipc.StartServer(coreIdx, *rootPath)

	mutMgr := manager.NewMutationManager(coreIdx, mnfst, nil, strg)

	if *noMount {
		log.Println("Running without FUSE mount (--no-mount)")
		c := make(chan os.Signal, 1)
		signal.Notify(c, os.Interrupt, syscall.SIGTERM)
		<-c
		log.Println("Saving manifest")
		mutMgr.Shutdown()
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
	log.Println("Daemon stopped")
}
