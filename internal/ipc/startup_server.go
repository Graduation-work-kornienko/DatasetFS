package ipc

import (
	"log"
	"net/http"
	"sync"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/pipeline"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/shm"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
)

var (
	mu              sync.Mutex
	currentAlloc    *shm.Allocator
	currentPipeline *pipeline.Pipeline
)

func StartServer(coreIdx *index.CoreIndex, rootPath string) {
	var err error

	http.HandleFunc("/initialize_loading", func(w http.ResponseWriter, r *http.Request) {

		mu.Lock()

		if currentPipeline != nil {
			currentPipeline.Stop()
			currentAlloc.Close()
		}

		mu.Unlock()

		currentAlloc, err = shm.NewAllocator()
		if err != nil {
			log.Fatalf("Ошибка создания Shared Memory: %v", err)
		}
		strg := &storage.Storage{Root: rootPath}

		currentPipeline = pipeline.NewPipeline(coreIdx, strg, currentAlloc)
		currentPipeline.Initiate()
	})

	http.ListenAndServe(":51409", nil)
}
