package metrics

import (
	"encoding/json"
	"math"
	"net/http/httptest"
	"testing"
	"time"
)

func approxEq(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestLatencyTracker_Percentiles(t *testing.T) {
	tr := NewLatencyTracker(1000)
	// Record 100 samples: 1ms, 2ms, ..., 100ms.
	for i := 1; i <= 100; i++ {
		tr.Record(time.Duration(i) * time.Millisecond)
	}
	h := tr.histogram()
	if h.Count != 100 {
		t.Fatalf("count: got %d, want 100", h.Count)
	}
	// p50 ≈ 0.05s (50th value), p95 ≈ 0.095s, p99 ≈ 0.099s
	if !approxEq(h.P50, 0.050, 0.005) {
		t.Errorf("p50: got %.4f, want ~0.050", h.P50)
	}
	if !approxEq(h.P95, 0.095, 0.005) {
		t.Errorf("p95: got %.4f, want ~0.095", h.P95)
	}
	if !approxEq(h.P99, 0.099, 0.005) {
		t.Errorf("p99: got %.4f, want ~0.099", h.P99)
	}
	if !approxEq(h.Max, 0.100, 0.001) {
		t.Errorf("max: got %.4f, want 0.100", h.Max)
	}
}

func TestLatencyTracker_RingBufferOverwrite(t *testing.T) {
	tr := NewLatencyTracker(10)
	// Record 100 samples, only last 10 should remain.
	for i := 1; i <= 100; i++ {
		tr.Record(time.Duration(i) * time.Millisecond)
	}
	h := tr.histogram()
	if h.Count != 100 {
		t.Errorf("count should track total observed: got %d, want 100", h.Count)
	}
	// Stored samples are 91..100ms, so max is 100ms.
	if !approxEq(h.Max, 0.100, 0.001) {
		t.Errorf("max: got %.4f, want 0.100", h.Max)
	}
	// p50 of {91..100ms} ≈ 95ms
	if !approxEq(h.P50, 0.095, 0.002) {
		t.Errorf("p50 (after overwrite): got %.4f, want ~0.095", h.P50)
	}
}

func TestPercentile_EmptyAndEdges(t *testing.T) {
	if got := percentile(nil, 50); got != 0 {
		t.Errorf("empty: got %v, want 0", got)
	}
	xs := []float64{1, 2, 3, 4, 5}
	if got := percentile(xs, 0); got != 1 {
		t.Errorf("p0: got %v, want 1", got)
	}
	if got := percentile(xs, 100); got != 5 {
		t.Errorf("p100: got %v, want 5", got)
	}
}

func TestHandler_ReturnsValidJSON(t *testing.T) {
	// Bump a few counters so the snapshot has non-zero values to verify.
	ShardLoadsTotal.Store(0)
	BytesReadTotal.Store(0)
	ShardLoadsTotal.Add(7)
	BytesReadTotal.Add(12345)

	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	Handler()(w, req)

	if w.Code != 200 {
		t.Fatalf("status: got %d, want 200", w.Code)
	}

	var snap Snapshot
	if err := json.Unmarshal(w.Body.Bytes(), &snap); err != nil {
		t.Fatalf("json decode: %v\nbody=%s", err, w.Body.String())
	}
	if snap.Counters["shard_loads_total"] < 7 {
		t.Errorf("shard_loads_total: got %d, want >=7", snap.Counters["shard_loads_total"])
	}
	if snap.Counters["bytes_read_total"] < 12345 {
		t.Errorf("bytes_read_total: got %d", snap.Counters["bytes_read_total"])
	}
	if _, ok := snap.Gauges["daemon_uptime_seconds"]; !ok {
		t.Error("daemon_uptime_seconds gauge missing")
	}
	if _, ok := snap.Histograms["load_latency"]; !ok {
		t.Error("load_latency histogram missing")
	}
}
