package control

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	_ "net/http/pprof" // registers /debug/pprof/* on the default mux
	"sync"
	"time"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/metrics"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/pipeline"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/shm"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
)

type initRequest struct {
	NumWorkers  int                `json:"num_workers"`
	Seed        *uint64            `json:"seed,omitempty"`
	Decode      *decodeOption      `json:"decode,omitempty"`
	Distributed *distributedOption `json:"distributed,omitempty"`
}

// distributedOption carries this rank's place in a DDP job (feature F2).
// Optional; when omitted the session runs single-process (rank 0, world 1).
// The daemon serves only this rank's shard partition; the deployment runs one
// daemon per rank (one per node on multi-node) with distinct ports/SHM/pipes.
type distributedOption struct {
	Rank      int `json:"rank"`
	WorldSize int `json:"world_size"`
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
	// NOTE: s.alloc is NOT closed here — the daemon owns one shared allocator
	// (sharedAlloc) reused across sessions. Re-mmapping ~1 GB every epoch was
	// pure waste; we Reset() its refcounts on reuse instead. See initialize_loading.
	metrics.ActivePipelines.Store(0)
}

var (
	mu                sync.Mutex
	currentSession    *session
	maintenanceActive bool
	// sharedAlloc is the daemon's single SHM allocator, created lazily on the
	// first /initialize_loading and reused (Reset, not re-mmapped) every session.
	// Guarded by mu (only touched inside the handler under the lock).
	sharedAlloc *shm.Allocator
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
		// Distributed defaults: single-process (rank 0 of a world of 1).
		rank, worldSize := 0, 1
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
			if req.Distributed != nil {
				worldSize = req.Distributed.WorldSize
				rank = req.Distributed.Rank
				if worldSize < 1 {
					http.Error(w, "distributed.world_size must be >= 1", http.StatusBadRequest)
					return
				}
				if rank < 0 || rank >= worldSize {
					http.Error(w, "distributed.rank must be in [0, world_size)", http.StatusBadRequest)
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

		tStop := time.Now()
		if currentSession != nil {
			currentSession.stop()
			currentSession = nil
		}
		dStop := time.Since(tStop)

		tAlloc := time.Now()
		// One shared allocator for the daemon's lifetime, reused across sessions:
		// allocate (mmap ~1 GB) once, then just Reset() refcounts each session.
		// Saves 33–69 ms/epoch of file-recreate + mmap and avoids churning a 1 GB
		// mapping every epoch.
		if sharedAlloc == nil {
			a, err := shm.NewAllocator()
			if err != nil {
				log.Printf("[IPC] Ошибка создания Shared Memory: %v", err)
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			sharedAlloc = a
		}
		sharedAlloc.Reset()
		alloc := sharedAlloc
		dAlloc := time.Since(tAlloc)

		// Pin one immutable snapshot for the whole session so every worker reads
		// the same generation even if a mutation lands mid-setup (feature F1).
		tPipe := time.Now()
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
				Rank:       rank,
				WorldSize:  worldSize,
			}
			p := pipeline.NewPipeline(snap, strg, alloc, cfg)
			p.Initiate()
			s.pipelines = append(s.pipelines, p)
		}
		currentSession = s
		log.Printf("[IPC timing] stop=%.1fms alloc=%.1fms pipelines=%.1fms (workers=%d)",
			float64(dStop.Microseconds())/1000, float64(dAlloc.Microseconds())/1000,
			float64(time.Since(tPipe).Microseconds())/1000, numWorkers)
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
			"distributed": map[string]any{
				"rank":       rank,
				"world_size": worldSize,
			},
		})
	})

	http.ListenAndServe(":51409", nil)
}
