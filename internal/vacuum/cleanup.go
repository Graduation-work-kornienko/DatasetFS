package vacuum

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const shardPrefix = "shard_"

// parseShardName returns the shard id encoded in a file name like "shard_3" or
// "shard_-1". The second result is false for non-shard files (and for the
// temp/backup variants such as "shard_3.tmp").
func parseShardName(name string) (int, bool) {
	if !strings.HasPrefix(name, shardPrefix) {
		return 0, false
	}
	suffix := name[len(shardPrefix):]
	id, err := strconv.Atoi(suffix)
	if err != nil {
		return 0, false
	}
	return id, true
}

// cleanupOldShards deletes every shard_* file in root whose id is not in keep,
// including the old delta shard (id -1). It also sweeps any leftover
// shard_*.tmp staging files.
func cleanupOldShards(root string, keep map[int]bool) error {
	entries, err := os.ReadDir(root)
	if err != nil {
		return err
	}
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, shardPrefix) && strings.HasSuffix(name, ".tmp") {
			os.Remove(filepath.Join(root, name))
			continue
		}
		id, ok := parseShardName(name)
		if !ok || keep[id] {
			continue
		}
		if err := os.Remove(filepath.Join(root, name)); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

// completePendingCommit cleans up debris from a previous run that crashed
// mid-commit: stray shard_*.tmp files and abandoned .vacuum-manifest-* temp
// directories. It does NOT attempt to finish a half-done rename — that is a
// manual recovery (restore metadata.*.backup and re-run), as documented.
func completePendingCommit(root string, verbose bool) error {
	entries, err := os.ReadDir(root)
	if err != nil {
		return err
	}
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, shardPrefix) && strings.HasSuffix(name, ".tmp") {
			os.Remove(filepath.Join(root, name))
		}
		if e.IsDir() && strings.HasPrefix(name, ".vacuum-manifest-") {
			os.RemoveAll(filepath.Join(root, name))
		}
	}
	return nil
}

// fsyncDir flushes a directory entry so renames within it are durable.
func fsyncDir(dir string) {
	d, err := os.Open(dir)
	if err != nil {
		return
	}
	defer d.Close()
	d.Sync()
}
