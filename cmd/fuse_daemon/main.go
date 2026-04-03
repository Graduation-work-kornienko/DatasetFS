package main

import (
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/manager"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/vfs"
	"github.com/hanwen/go-fuse/v2/fs"
	"github.com/hanwen/go-fuse/v2/fuse"
)

func main() {
	rootPath := "cmd/dataset_converter/test"
	mnfst := index.NewManifest(rootPath)
	mnfst.Load()
	coreIdx, err := mnfst.LoadCoreIndex()

	// return
	if err != nil {
		log.Fatalf("load core index fail: %v\n", err)
	}
	log.Println("Loaded")

	strg := &storage.Storage{Root: rootPath}
	mutMgr := manager.NewMutationManager(coreIdx, mnfst, nil, strg)

	root := &vfs.RootNode{
		CoreIdx: coreIdx,
		MutMgr:  mutMgr,
		Storage: strg,
	}

	log.Println("Mounting")
	mountPoint := "./dataset_mount"
	os.MkdirAll(mountPoint, 0755)
	server, err := fs.Mount(mountPoint, root, &fs.Options{
		MountOptions: fuse.MountOptions{
			Debug:  true,
			FsName: "DatasetFS",
			Name:   "DatasetFS",
			Options: []string{
				"local",
				"volname=DatasetFS",
				"noappledouble",
				"noapplexattr",
			},
			// AllowOther: true,
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

	log.Printf("Successfully mounted DatasetFS %s", mountPoint)

	server.Wait()

	log.Println("Saving manifest")
	mutMgr.Shutdown()
	log.Println("Daemon stopped")
}
