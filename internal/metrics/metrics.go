// Package metrics is a tiny, dependency-free observability layer for the
// datasetfs daemon: atomic counters, gauge setters, and a bounded latency tracker
// that computes percentiles on demand. Exposed as JSON at /metrics so the
// Python benchmark harness can poll without adding any extra Go deps.
package metrics

import (
	"encoding/json"
	"net/http"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// --- counters ---------------------------------------------------------------

var (
	// Total number of times BackgroundLoader successfully read a shard.
	ShardLoadsTotal atomic.Int64
	// Sum of bytes read from disk into shared-memory slots.
	BytesReadTotal atomic.Int64
	// Incremented when the planner had to wait for a free slot (i.e., all
	// of its slots are still in use by the dealer / consumers).
	SlotStarvationTotal atomic.Int64
	// Incremented when SetRefCount is called on a slot whose count was not
	// already zero — signals a lifecycle bug (consumer didn't decrement).
	RefcountOverflowTotal atomic.Int64
	// Number of batches emitted on the wire (after window shuffle, sliced
	// at BatchSize).
	DealerBatchesSentTotal atomic.Int64
	// Number of samples emitted across all batches (= total work done).
	SamplesEmittedTotal atomic.Int64
	// Epochs fully completed (end-of-epoch batch sent).
	EpochsCompletedTotal atomic.Int64

	// --- remote streaming (G9/G14) -----------------------------------------
	// Bytes pulled from remote storage into the local cache (sum over Fetch).
	RemoteBytesDownloadedTotal atomic.Int64
	// Base shards fully downloaded + cached (ready to serve).
	RemoteShardsReadyTotal atomic.Int64
	// Times a consumer (EnsureShard) had to block on a shard that wasn't ready
	// yet — i.e. training outran the prefetch. The remote analogue of a stall.
	RemoteShardWaitsTotal atomic.Int64
	// Remote shard cache outcomes. Hit = shard already present on local cache;
	// miss = had to perform an HTTP fetch.
	RemoteCacheHitsTotal   atomic.Int64
	RemoteCacheMissesTotal atomic.Int64
)

// --- gauges -----------------------------------------------------------------

var (
	// Currently-active pipelines (= num_workers for the live session).
	ActivePipelines atomic.Int32
	// Base shards not yet downloaded (remote streaming). Decremented as the
	// prefetcher + on-demand fetches complete each shard.
	RemoteShardsPending atomic.Int32
	// Daemon startup time, used to compute uptime.
	startTime = time.Now()
)

// --- latency tracker --------------------------------------------------------

// LatencyTracker holds the most recent N latency samples for one metric and
// computes percentiles on demand. Cheap, lock-protected; we expect well under
// 1000 events/s so this is plenty.
type LatencyTracker struct {
	mu      sync.Mutex
	samples []float64 // in seconds
	cap     int
	idx     int   // ring-buffer write head
	filled  bool  // whether we've wrapped at least once
	count   int64 // total samples observed (not just stored)
}

func NewLatencyTracker(cap int) *LatencyTracker {
	return &LatencyTracker{samples: make([]float64, cap), cap: cap}
}

func (t *LatencyTracker) Record(d time.Duration) {
	t.mu.Lock()
	t.samples[t.idx] = d.Seconds()
	t.idx = (t.idx + 1) % t.cap
	if t.idx == 0 {
		t.filled = true
	}
	t.count++
	t.mu.Unlock()
}

// Snapshot returns a sorted copy of the currently-stored samples + total count.
func (t *LatencyTracker) Snapshot() (sorted []float64, totalCount int64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	n := t.cap
	if !t.filled {
		n = t.idx
	}
	if n == 0 {
		return nil, t.count
	}
	out := make([]float64, n)
	copy(out, t.samples[:n])
	sort.Float64s(out)
	return out, t.count
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 100 {
		return sorted[len(sorted)-1]
	}
	rank := p / 100 * float64(len(sorted)-1)
	low := int(rank)
	high := low + 1
	if high >= len(sorted) {
		return sorted[low]
	}
	frac := rank - float64(low)
	return sorted[low]*(1-frac) + sorted[high]*frac
}

// LoadLatency tracks single-shard read+parse latency. 8192 samples is enough
// to cover ~hundreds of shards × tens of epochs without losing the tail.
var LoadLatency = NewLatencyTracker(8192)

// RemoteFetchLatency tracks per-shard remote download latency (HTTP GET +
// write to cache). Reveals the cold-start cost and how it amortizes.
var RemoteFetchLatency = NewLatencyTracker(8192)

// RemoteShardWaitLatency records actual time consumers spend blocked on remote
// shard readiness. The wait counter alone cannot quantify lost training time.
var RemoteShardWaitLatency = NewLatencyTracker(8192)

// DecodeLatency measures daemon-side decode+resize per slot in rgb_uint8 mode.
var DecodeLatency = NewLatencyTracker(8192)

// SHMWriteLatency measures large slot writes/copies into shared memory.
var SHMWriteLatency = NewLatencyTracker(8192)

// DealerEmitLatency measures binary frame serialization + FIFO write latency.
var DealerEmitLatency = NewLatencyTracker(8192)

// --- JSON snapshot ----------------------------------------------------------

type histogram struct {
	Count int64   `json:"count"`
	P50   float64 `json:"p50_seconds"`
	P95   float64 `json:"p95_seconds"`
	P99   float64 `json:"p99_seconds"`
	Max   float64 `json:"max_seconds"`
}

func (t *LatencyTracker) histogram() histogram {
	sorted, n := t.Snapshot()
	h := histogram{Count: n}
	if len(sorted) > 0 {
		h.P50 = percentile(sorted, 50)
		h.P95 = percentile(sorted, 95)
		h.P99 = percentile(sorted, 99)
		h.Max = sorted[len(sorted)-1]
	}
	return h
}

// Snapshot is the JSON shape returned by /metrics. Stable so Python parsers
// don't break on schema drift.
type Snapshot struct {
	Counters   map[string]int64     `json:"counters"`
	Gauges     map[string]float64   `json:"gauges"`
	Histograms map[string]histogram `json:"histograms"`
}

func collect() Snapshot {
	return Snapshot{
		Counters: map[string]int64{
			"shard_loads_total":             ShardLoadsTotal.Load(),
			"bytes_read_total":              BytesReadTotal.Load(),
			"slot_starvation_total":         SlotStarvationTotal.Load(),
			"refcount_overflow_total":       RefcountOverflowTotal.Load(),
			"dealer_batches_sent_total":     DealerBatchesSentTotal.Load(),
			"samples_emitted_total":         SamplesEmittedTotal.Load(),
			"epochs_completed_total":        EpochsCompletedTotal.Load(),
			"remote_bytes_downloaded_total": RemoteBytesDownloadedTotal.Load(),
			"remote_shards_ready_total":     RemoteShardsReadyTotal.Load(),
			"remote_shard_waits_total":      RemoteShardWaitsTotal.Load(),
			"remote_cache_hits_total":       RemoteCacheHitsTotal.Load(),
			"remote_cache_misses_total":     RemoteCacheMissesTotal.Load(),
		},
		Gauges: map[string]float64{
			"active_pipelines":      float64(ActivePipelines.Load()),
			"daemon_uptime_seconds": time.Since(startTime).Seconds(),
			"remote_shards_pending": float64(RemoteShardsPending.Load()),
		},
		Histograms: map[string]histogram{
			"load_latency":              LoadLatency.histogram(),
			"remote_fetch_latency":      RemoteFetchLatency.histogram(),
			"remote_shard_wait_latency": RemoteShardWaitLatency.histogram(),
			"decode_latency":            DecodeLatency.histogram(),
			"shm_write_latency":         SHMWriteLatency.histogram(),
			"dealer_emit_latency":       DealerEmitLatency.histogram(),
		},
	}
}

// Handler returns an http.HandlerFunc for /metrics.
func Handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(collect())
	}
}
