package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
)

// prefetchRemoteDataset materializes a remote (HTTP) dataset into cacheDir:
// the manifest (parquet preferred, jsonl fallback) plus every base shard. After
// this the whole pipeline runs purely local with Root=cacheDir. Shard ids come
// from the manifest, so a plain anonymous-GET bucket policy is enough (no
// bucket listing). URLs are joined path-style — never filepath.Join, which
// would mangle "http://".
func prefetchRemoteDataset(rs *storage.RemoteStorage, rootURL, cacheDir string) (string, error) {
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return "", err
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
		return "", fmt.Errorf("no manifest (metadata.parquet/.jsonl) at %s", base)
	}

	mnfst := index.NewManifest(cacheDir)
	if err := mnfst.Load(nil); err != nil {
		return "", fmt.Errorf("load prefetched manifest: %w", err)
	}
	for id := range mnfst.ShardsMeta {
		if id < 0 {
			continue // delta placeholder has no remote shard file
		}
		name := fmt.Sprintf("shard_%d", id)
		if err := rs.Fetch(ctx, fmt.Sprintf("%s/%s", base, name), filepath.Join(cacheDir, name)); err != nil {
			return "", fmt.Errorf("prefetch %s: %w", name, err)
		}
		log.Printf("[prefetch] %s", name)
	}
	return cacheDir, nil
}
