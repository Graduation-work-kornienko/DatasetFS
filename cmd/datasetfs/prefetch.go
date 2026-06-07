package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
)

// prefetchRemoteManifest fetches ONLY the manifest of a remote (HTTP) dataset
// into cacheDir (parquet preferred, jsonl fallback) and builds a
// RemotePrefetcher over the base shards. It does NOT download the shards — that
// happens lazily/in-background via the prefetcher, so the daemon can start
// serving while shards are still arriving (streaming-overlap, thesis G9/G14).
//
// Shard ids come from the manifest, so a plain anonymous-GET bucket policy is
// enough (no bucket listing). URLs are joined path-style — never filepath.Join,
// which would mangle "http://".
//
// Returns (localRoot=cacheDir, prefetcher). The caller loads the manifest from
// localRoot for the index, then wires the prefetcher into storage and Start()s it.
func prefetchRemoteManifest(rs *storage.RemoteStorage, rootURL, cacheDir string, concurrency int) (string, *storage.RemotePrefetcher, error) {
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return "", nil, err
	}
	base := strings.TrimRight(rootURL, "/")
	ctx := context.Background()

	// Manifest: try parquet, then jsonl.
	gotManifest := false
	for _, name := range []string{"metadata.parquet", "metadata.jsonl"} {
		if err := rs.Fetch(ctx, base+"/"+name, filepath.Join(cacheDir, name)); err == nil {
			gotManifest = true
			log.Printf("[prefetch] manifest %s", name)
			break
		}
	}
	if !gotManifest {
		return "", nil, fmt.Errorf("no manifest (metadata.parquet/.jsonl) at %s", base)
	}

	mnfst := index.NewManifest(cacheDir)
	if err := mnfst.Load(nil); err != nil {
		return "", nil, fmt.Errorf("load prefetched manifest: %w", err)
	}

	// Collect base shard ids (skip the delta placeholder id<0) in ascending
	// order for a deterministic background fetch order.
	shardIDs := make([]int, 0, len(mnfst.ShardsMeta))
	for id := range mnfst.ShardsMeta {
		if id >= 0 {
			shardIDs = append(shardIDs, id)
		}
	}
	sort.Ints(shardIDs)

	prefetcher := storage.NewRemotePrefetcher(rs, base, cacheDir, shardIDs, concurrency)
	log.Printf("[prefetch] streaming %d shard(s) in background (concurrency=%d)", len(shardIDs), concurrency)
	return cacheDir, prefetcher, nil
}
