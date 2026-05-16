package ipc

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	_ "net/http/pprof" // registers /debug/pprof/* on the default mux
	"sync"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/metrics"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/pipeline"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/shm"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
)

type initRequest struct {
	NumWorkers int     `json:"num_workers"`
	Seed       *uint64 `json:"seed,omitempty"`
}

type session struct {
	alloc     *shm.Allocator
	pipelines []*pipeline.Pipeline
}

func (s *session) stop() {
	for _, p := range s.pipelines {
		p.Stop()
	}
	if s.alloc != nil {
		s.alloc.Close()
	}
	metrics.ActivePipelines.Store(0)
}

var (
	mu             sync.Mutex
	currentSession *session
)

func StartServer(coreIdx *index.CoreIndex, rootPath string) {
	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	http.HandleFunc("/metrics", metrics.Handler())

	http.HandleFunc("/initialize_loading", func(w http.ResponseWriter, r *http.Request) {
		numWorkers := 1
		var seed *uint64
		if body, err := io.ReadAll(r.Body); err == nil && len(body) > 0 {
			var req initRequest
			if jerr := json.Unmarshal(body, &req); jerr != nil {
				http.Error(w, "invalid JSON: "+jerr.Error(), http.StatusBadRequest)
				return
			}
			if req.NumWorkers > 0 {
				numWorkers = req.NumWorkers
			}
			seed = req.Seed
		}
		if numWorkers > shm.NumSlots {
			http.Error(w, "num_workers exceeds NumSlots (9)", http.StatusBadRequest)
			return
		}

		mu.Lock()
		defer mu.Unlock()

		if currentSession != nil {
			currentSession.stop()
			currentSession = nil
		}

		alloc, err := shm.NewAllocator()
		if err != nil {
			log.Printf("[IPC] Ошибка создания Shared Memory: %v", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		strg := &storage.Storage{Root: rootPath}

		s := &session{alloc: alloc}
		for wID := 0; wID < numWorkers; wID++ {
			start, end := pipeline.SlotRange(wID, numWorkers)
			cfg := pipeline.WorkerConfig{
				WorkerID:   wID,
				NumWorkers: numWorkers,
				SlotStart:  start,
				SlotEnd:    end,
				PipePath:   pipeline.PipePath(wID),
				Seed:       seed,
			}
			p := pipeline.NewPipeline(coreIdx, strg, alloc, cfg)
			p.Initiate()
			s.pipelines = append(s.pipelines, p)
		}
		currentSession = s
		metrics.ActivePipelines.Store(int32(numWorkers))

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]int{"num_workers": numWorkers})
	})

	http.ListenAndServe(":51409", nil)
}
