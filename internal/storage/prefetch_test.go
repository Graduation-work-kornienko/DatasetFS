package storage

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/metrics"
)

// shardBlob is the deterministic content of a fake remote shard.
func shardBlob(id int) []byte {
	return []byte(fmt.Sprintf("SHARD-%d-CONTENTS-%s", id, string(make([]byte, 1024))))
}

// fakeShardServer serves /shard_<id> with a small latency to exercise the
// downloading/wait path. Unknown ids 404. It counts requests served.
func fakeShardServer(t *testing.T, n int, latency time.Duration) (*httptest.Server, *int32) {
	t.Helper()
	var served int32
	mux := http.NewServeMux()
	for id := 0; id < n; id++ {
		id := id
		mux.HandleFunc(fmt.Sprintf("/shard_%d", id), func(w http.ResponseWriter, r *http.Request) {
			time.Sleep(latency)
			atomic.AddInt32(&served, 1)
			_, _ = w.Write(shardBlob(id))
		})
	}
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv, &served
}

func newPrefetcher(t *testing.T, srv *httptest.Server, n, concurrency int) (*RemotePrefetcher, string) {
	t.Helper()
	cacheDir := t.TempDir()
	rs := NewRemoteStorage(cacheDir)
	ids := make([]int, n)
	for i := range ids {
		ids[i] = i
	}
	return NewRemotePrefetcher(rs, srv.URL, cacheDir, ids, concurrency), cacheDir
}

// EnsureShard must return complete bytes for every shard even when requested
// out of the background fetch order (the planner reshuffles each epoch).
func TestRemotePrefetcher_EnsureOutOfOrder(t *testing.T) {
	const n = 8
	srv, _ := fakeShardServer(t, n, 5*time.Millisecond)
	p, cacheDir := newPrefetcher(t, srv, n, 2)
	p.Start(context.Background())

	// Request in reverse order.
	for id := n - 1; id >= 0; id-- {
		path, err := p.EnsureShard(id)
		if err != nil {
			t.Fatalf("EnsureShard(%d): %v", id, err)
		}
		got, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		if want := shardBlob(id); string(got) != string(want) {
			t.Fatalf("shard %d content mismatch (%d vs %d bytes)", id, len(got), len(want))
		}
		if path != filepath.Join(cacheDir, fmt.Sprintf(ShardFormat, id)) {
			t.Fatalf("shard %d unexpected path %s", id, path)
		}
	}
}

// Concurrent EnsureShard from many goroutines (same + different ids) must be
// race-free and always return complete bytes.
func TestRemotePrefetcher_ConcurrentEnsure(t *testing.T) {
	const n = 6
	srv, _ := fakeShardServer(t, n, 2*time.Millisecond)
	p, _ := newPrefetcher(t, srv, n, 3)
	p.Start(context.Background())

	var wg sync.WaitGroup
	errCh := make(chan error, 64)
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			id := g % n
			path, err := p.EnsureShard(id)
			if err != nil {
				errCh <- err
				return
			}
			got, _ := os.ReadFile(path)
			if string(got) != string(shardBlob(id)) {
				errCh <- fmt.Errorf("shard %d mismatch", id)
			}
		}(g)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}
}

// A shard whose URL 404s must surface an error (not hang).
func TestRemotePrefetcher_FetchError(t *testing.T) {
	const n = 2
	srv, _ := fakeShardServer(t, n, 0)
	cacheDir := t.TempDir()
	rs := NewRemoteStorage(cacheDir)
	// Include a shard id (99) the server doesn't serve.
	p := NewRemotePrefetcher(rs, srv.URL, cacheDir, []int{0, 1, 99}, 2)
	p.Start(context.Background())

	if _, err := p.EnsureShard(99); err == nil {
		t.Fatal("EnsureShard(99) expected error for missing shard, got nil")
	}
	// Valid shards still work.
	if _, err := p.EnsureShard(0); err != nil {
		t.Fatalf("EnsureShard(0): %v", err)
	}
}

// The prefetcher must not require all shards to be downloaded before a specific
// one can be served: requesting the LAST shard while many earlier ones are slow
// should complete well before all are fetched.
func TestRemotePrefetcher_DoesNotWaitForAll(t *testing.T) {
	const n = 20
	// Slow shards so the background pool can't have finished them all quickly.
	srv, served := fakeShardServer(t, n, 30*time.Millisecond)
	p, _ := newPrefetcher(t, srv, n, 2)
	p.Start(context.Background())

	start := time.Now()
	if _, err := p.EnsureShard(n - 1); err != nil {
		t.Fatalf("EnsureShard(%d): %v", n-1, err)
	}
	elapsed := time.Since(start)

	// With on-demand synchronous fetch, the last shard returns after roughly one
	// download (~30ms), not after all 20 (~300ms+). Be generous to avoid flakes.
	if elapsed > 200*time.Millisecond {
		t.Fatalf("EnsureShard(last) took %v; expected to not wait for all shards", elapsed)
	}
	if got := atomic.LoadInt32(served); got >= int32(n) {
		t.Fatalf("served %d shards already; expected the last to be fetched before all", got)
	}
}

// Smoke: the remote metrics move when shards are fetched.
func TestRemotePrefetcher_MetricsMove(t *testing.T) {
	const n = 4
	srv, _ := fakeShardServer(t, n, 0)
	p, _ := newPrefetcher(t, srv, n, 2)

	before := metrics.RemoteShardsReadyTotal.Load()
	p.Start(context.Background())
	for id := 0; id < n; id++ {
		if _, err := p.EnsureShard(id); err != nil {
			t.Fatalf("EnsureShard(%d): %v", id, err)
		}
	}
	if got := metrics.RemoteShardsReadyTotal.Load() - before; got < int64(n) {
		t.Fatalf("RemoteShardsReadyTotal moved by %d, want >= %d", got, n)
	}
	if metrics.RemoteBytesDownloadedTotal.Load() == 0 {
		t.Fatal("RemoteBytesDownloadedTotal still zero")
	}
}
