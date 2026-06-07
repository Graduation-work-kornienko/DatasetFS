package storage

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/metrics"
)

// shardState tracks one base shard through the streaming lifecycle.
type shardState int

const (
	shardPending shardState = iota
	shardDownloading
	shardReady
	shardError
)

// RemotePrefetcher turns the daemon's old blocking "download every shard before
// serving" startup into streaming-overlap: the manifest is fetched up front (so
// the index/serving can start immediately), then a bounded worker pool downloads
// the base shards in the background while training is already running.
//
// Consumers call EnsureShard(id) before reading a shard. If the shard is ready
// it returns instantly; if a worker is already fetching it, it waits; otherwise
// the consumer fetches it synchronously itself — so an out-of-prefetch-order
// request (the planner reshuffles shard order every epoch) never starves behind
// the pool. The "overlap" is the pool fetching shards N+1, N+2… while the loader
// reads shard N from shared memory.
//
// Correctness: RemoteStorage.Fetch writes atomically (temp file + rename), so a
// shard marked ready is always a complete file — safe for io.ReadFull of the
// whole shard into a slot. We never expose a partially-written shard.
type RemotePrefetcher struct {
	rs       *RemoteStorage
	baseURL  string // path-style remote root, e.g. "http://host/bucket"
	cacheDir string
	shardIDs []int // base shard ids from the manifest (fetch order)

	mu    sync.Mutex
	cond  *sync.Cond
	state map[int]shardState
	errs  map[int]error
	next  int // shared cursor into shardIDs for the worker pool

	concurrency int
}

// NewRemotePrefetcher builds a prefetcher over the given base shard ids.
func NewRemotePrefetcher(rs *RemoteStorage, baseURL, cacheDir string, shardIDs []int, concurrency int) *RemotePrefetcher {
	if concurrency < 1 {
		concurrency = 1
	}
	p := &RemotePrefetcher{
		rs:          rs,
		baseURL:     baseURL,
		cacheDir:    cacheDir,
		shardIDs:    append([]int(nil), shardIDs...),
		state:       make(map[int]shardState, len(shardIDs)),
		errs:        make(map[int]error),
		concurrency: concurrency,
	}
	p.cond = sync.NewCond(&p.mu)
	for _, id := range shardIDs {
		p.state[id] = shardPending
	}
	metrics.RemoteShardsPending.Store(int32(len(shardIDs)))
	return p
}

func (p *RemotePrefetcher) shardURL(id int) string {
	return fmt.Sprintf("%s/%s", p.baseURL, fmt.Sprintf(ShardFormat, id))
}

func (p *RemotePrefetcher) shardDst(id int) string {
	return filepath.Join(p.cacheDir, fmt.Sprintf(ShardFormat, id))
}

// Start launches the background worker pool. It returns immediately; workers run
// until ctx is cancelled or every shard has been claimed.
func (p *RemotePrefetcher) Start(ctx context.Context) {
	for w := 0; w < p.concurrency; w++ {
		go p.worker(ctx)
	}
}

// worker pulls the next still-pending shard from the shared cursor and fetches
// it. Shards already claimed by a consumer (EnsureShard) are skipped.
func (p *RemotePrefetcher) worker(ctx context.Context) {
	for {
		if ctx.Err() != nil {
			return
		}
		p.mu.Lock()
		// Advance the cursor to the next pending shard.
		var id int = -1
		for p.next < len(p.shardIDs) {
			cand := p.shardIDs[p.next]
			p.next++
			if p.state[cand] == shardPending {
				p.state[cand] = shardDownloading
				id = cand
				break
			}
		}
		p.mu.Unlock()
		if id < 0 {
			return // all shards claimed
		}
		p.fetch(ctx, id)
	}
}

// fetch downloads one shard (caller must have set its state to shardDownloading)
// and transitions it to ready/error, broadcasting to any waiters.
func (p *RemotePrefetcher) fetch(ctx context.Context, id int) {
	dst := p.shardDst(id)

	// Already on disk (e.g. a previous run cached it)? Treat as ready.
	if fi, err := os.Stat(dst); err == nil && fi.Size() > 0 {
		metrics.RemoteCacheHitsTotal.Add(1)
		p.markReady(id)
		return
	}
	metrics.RemoteCacheMissesTotal.Add(1)

	start := time.Now()
	err := p.rs.Fetch(ctx, p.shardURL(id), dst)
	if err != nil {
		p.mu.Lock()
		p.state[id] = shardError
		p.errs[id] = err
		p.cond.Broadcast()
		p.mu.Unlock()
		return
	}
	metrics.RemoteFetchLatency.Record(time.Since(start))
	if fi, statErr := os.Stat(dst); statErr == nil {
		metrics.RemoteBytesDownloadedTotal.Add(fi.Size())
	}
	p.markReady(id)
}

func (p *RemotePrefetcher) markReady(id int) {
	p.mu.Lock()
	if p.state[id] != shardReady {
		p.state[id] = shardReady
		metrics.RemoteShardsReadyTotal.Add(1)
		metrics.RemoteShardsPending.Add(-1)
	}
	p.cond.Broadcast()
	p.mu.Unlock()
}

// EnsureShard guarantees shard `id` is present in the local cache and returns
// its local path. Ready → instant; in-flight → waits; pending → fetches it now
// in the calling goroutine (so it can't starve behind the pool).
func (p *RemotePrefetcher) EnsureShard(id int) (string, error) {
	dst := p.shardDst(id)
	p.mu.Lock()
	for {
		switch p.state[id] {
		case shardReady:
			p.mu.Unlock()
			return dst, nil
		case shardError:
			err := p.errs[id]
			p.mu.Unlock()
			return "", err
		case shardDownloading:
			// A worker is fetching it — wait for completion.
			metrics.RemoteShardWaitsTotal.Add(1)
			waitStart := time.Now()
			p.cond.Wait()
			metrics.RemoteShardWaitLatency.Record(time.Since(waitStart))
			continue
		default: // shardPending (or an id not in the manifest list): claim it
			// and fetch synchronously so it can't starve behind the pool.
			metrics.RemoteShardWaitsTotal.Add(1)
			p.state[id] = shardDownloading
			p.mu.Unlock()
			waitStart := time.Now()
			p.fetch(context.Background(), id)
			metrics.RemoteShardWaitLatency.Record(time.Since(waitStart))
			p.mu.Lock()
			// Loop re-reads the state set by fetch (ready or error).
		}
	}
}
