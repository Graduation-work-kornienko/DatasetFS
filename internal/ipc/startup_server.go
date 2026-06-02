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
	NumWorkers int           `json:"num_workers"`
	Seed       *uint64       `json:"seed,omitempty"`
	Decode     *decodeOption `json:"decode,omitempty"`
}

// decodeOption mirrors pipeline.DecodeConfig over JSON. Optional; omitted
// payload defaults to {mode: "raw"} (backwards-compatible).
type decodeOption struct {
	Mode      string `json:"mode"`
	ImageSize int    `json:"image_size"`
	// Parallelism: decode worker goroutines per pipeline. 0/omitted = auto.
	Parallelism int `json:"parallelism,omitempty"`
}

type session struct {
	alloc     *shm.Allocator
	pipelines []*pipeline.Pipeline
	// coreIdx + snap implement the per-session MVCC pin (feature F1): the whole
	// session reads one immutable generation, released on stop so the vacuum
	// safepoint (CoreIndex.MinPinnedGen) advances.
	coreIdx *index.CoreIndex
	snap    *index.Snapshot
}

func (s *session) stop() {
	for _, p := range s.pipelines {
		p.Stop()
	}
	if s.coreIdx != nil {
		s.coreIdx.Unpin(s.snap)
	}
	if s.alloc != nil {
		s.alloc.Close()
	}
	metrics.ActivePipelines.Store(0)
}

var (
	mu                sync.Mutex
	currentSession    *session
	maintenanceActive bool
)

// BeginMaintenance reserves the dataset for an exclusive maintenance operation
// (e.g. vacuum). It returns false if a loading session is active or another
// maintenance op already holds the dataset. While held, /initialize_loading
// responds 503 so no pipeline starts reading shards mid-rewrite.
func BeginMaintenance() bool {
	mu.Lock()
	defer mu.Unlock()
	if currentSession != nil || maintenanceActive {
		return false
	}
	maintenanceActive = true
	return true
}

// EndMaintenance releases the reservation taken by BeginMaintenance.
func EndMaintenance() {
	mu.Lock()
	defer mu.Unlock()
	maintenanceActive = false
}

// StartServer serves the daemon's HTTP control plane. strg is the storage the
// pipeline workers read shards from (already pointed at the local root, with
// any RemoteStorage attached by the caller).
func StartServer(coreIdx *index.CoreIndex, strg *storage.Storage) {
	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	http.HandleFunc("/metrics", metrics.Handler())

	http.HandleFunc("/initialize_loading", func(w http.ResponseWriter, r *http.Request) {
		numWorkers := 1
		var seed *uint64
		decodeCfg := pipeline.DecodeConfig{Mode: pipeline.DecodeRaw}
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
			if req.Decode != nil {
				mode := pipeline.DecodeMode(req.Decode.Mode)
				switch mode {
				case "", pipeline.DecodeRaw:
					// keep default
				case pipeline.DecodeRGBUint8:
					if req.Decode.ImageSize <= 0 {
						http.Error(w, "decode.image_size must be > 0 for mode=rgb_uint8", http.StatusBadRequest)
						return
					}
					decodeCfg = pipeline.DecodeConfig{
						Mode:        mode,
						ImageSize:   req.Decode.ImageSize,
						Parallelism: req.Decode.Parallelism,
					}
				default:
					http.Error(w, "unsupported decode.mode: "+req.Decode.Mode, http.StatusBadRequest)
					return
				}
			}
		}
		if numWorkers > shm.NumSlots {
			http.Error(w, "num_workers exceeds NumSlots (9)", http.StatusBadRequest)
			return
		}

		mu.Lock()
		defer mu.Unlock()

		if maintenanceActive {
			http.Error(w, "dataset under maintenance (vacuum); retry shortly", http.StatusServiceUnavailable)
			return
		}

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

		// Pin one immutable snapshot for the whole session so every worker reads
		// the same generation even if a mutation lands mid-setup (feature F1).
		snap := coreIdx.Pin()
		s := &session{alloc: alloc, coreIdx: coreIdx, snap: snap}
		for wID := 0; wID < numWorkers; wID++ {
			start, end := pipeline.SlotRange(wID, numWorkers)
			cfg := pipeline.WorkerConfig{
				WorkerID:   wID,
				NumWorkers: numWorkers,
				SlotStart:  start,
				SlotEnd:    end,
				PipePath:   pipeline.PipePath(wID),
				Seed:       seed,
				Decode:     decodeCfg,
			}
			p := pipeline.NewPipeline(snap, strg, alloc, cfg)
			p.Initiate()
			s.pipelines = append(s.pipelines, p)
		}
		currentSession = s
		metrics.ActivePipelines.Store(int32(numWorkers))

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"num_workers": numWorkers,
			"generation":  snap.Gen,
			"decode": map[string]any{
				"mode":        string(decodeCfg.Mode),
				"image_size":  decodeCfg.ImageSize,
				"parallelism": decodeCfg.Parallelism,
			},
		})
	})

	http.ListenAndServe(":51409", nil)
}
