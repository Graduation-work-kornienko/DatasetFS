package index

import (
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestBinaryWAL_RoundTripAndReplay mirrors the JSON WAL tests against the
// binary format: add/delete/shard records survive a close+reopen and replay
// into a CoreIndex in order.
func TestBinaryWAL_RoundTripAndReplay(t *testing.T) {
	dir := t.TempDir()

	w1, err := OpenWALWithFormat(dir, "binary")
	require.NoError(t, err)
	require.NoError(t, w1.LogAppendShard(&Shard{
		Number: 0, Type: Base, TotalSize: 100,
		Objects: []*Metadata{{ShardID: 0, Path: "a.jpg", Size: 50, Offset: 512}},
	}))
	require.NoError(t, w1.LogAdd(&Metadata{ShardID: -1, Offset: 512, Size: 1024, Path: "new.jpg"}))
	require.NoError(t, w1.LogDelete("a.jpg"))
	require.NoError(t, w1.Close())

	// Reopen (verifies the file header round-trips) and replay.
	w2, err := OpenWALWithFormat(dir, "binary")
	require.NoError(t, err)
	defer w2.Close()

	idx := newCoreIndexWithDelta()
	applied, err := w2.Replay(idx)
	require.NoError(t, err)
	require.Equal(t, 3, applied)

	require.Contains(t, idx.FileMap, "new.jpg")
	require.Equal(t, int64(1024), idx.FileMap["new.jpg"].Size)
	require.True(t, idx.FileMap["a.jpg"].Deleted, "delete must apply after the shard add")
}

// TestBinaryWAL_ReplayThenAppend: Replay must leave the file at EOF so a
// following append doesn't clobber existing records.
func TestBinaryWAL_ReplayThenAppend(t *testing.T) {
	dir := t.TempDir()

	w, err := OpenWALWithFormat(dir, "binary")
	require.NoError(t, err)
	require.NoError(t, w.LogDelete("first.jpg"))

	_, err = w.Replay(newCoreIndexWithDelta())
	require.NoError(t, err)

	require.NoError(t, w.LogDelete("second.jpg"))

	idx := newCoreIndexWithDelta()
	idx.FileMap["first.jpg"] = &Metadata{Path: "first.jpg"}
	idx.FileMap["second.jpg"] = &Metadata{Path: "second.jpg"}
	applied, err := w.Replay(idx)
	require.NoError(t, err)
	require.Equal(t, 2, applied, "both records must be present after append-past-replay")
	require.NoError(t, w.Close())
}

func TestBinaryWAL_TruncateClearsRecords(t *testing.T) {
	dir := t.TempDir()

	w, err := OpenWALWithFormat(dir, "binary")
	require.NoError(t, err)
	require.NoError(t, w.LogDelete("a.jpg"))
	require.NoError(t, w.LogDelete("b.jpg"))
	require.NoError(t, w.Truncate())

	// After truncate the only thing left is the file header → replay applies 0.
	require.NoError(t, w.LogDelete("c.jpg"))
	idx := newCoreIndexWithDelta()
	idx.FileMap["c.jpg"] = &Metadata{Path: "c.jpg"}
	applied, err := w.Replay(idx)
	require.NoError(t, err)
	require.Equal(t, 1, applied, "post-truncate WAL must contain only the new record")
	require.NoError(t, w.Close())
}

func TestBinaryWAL_ChecksumCorruptionAborts(t *testing.T) {
	dir := t.TempDir()

	w, err := OpenWALWithFormat(dir, "binary")
	require.NoError(t, err)
	require.NoError(t, w.LogDelete("a.jpg"))
	require.NoError(t, w.Close())

	// Flip a byte in the data region to break the record checksum.
	path := filepath.Join(dir, "wal.log")
	raw, err := os.ReadFile(path)
	require.NoError(t, err)
	require.Greater(t, len(raw), walHeaderSize+recordHeaderSize+2)
	raw[len(raw)-3] ^= 0xFF // inside the data/checksum tail
	require.NoError(t, os.WriteFile(path, raw, 0644))

	w2, err := OpenWALWithFormat(dir, "binary")
	require.NoError(t, err)
	defer w2.Close()
	_, err = w2.Replay(NewIndex())
	require.Error(t, err, "a corrupted record must abort replay")
}

// TestBinaryWAL_ConcurrentWritersDoNotCorrupt proves the new mutex: many
// goroutines appending at once must produce a fully parseable log.
func TestBinaryWAL_ConcurrentWritersDoNotCorrupt(t *testing.T) {
	dir := t.TempDir()

	w, err := OpenWALWithFormat(dir, "binary")
	require.NoError(t, err)

	const n = 200
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			_ = w.LogAdd(&Metadata{ShardID: -1, Offset: 512, Size: int64(i), Path: "f"})
		}(i)
	}
	wg.Wait()
	require.NoError(t, w.Close())

	w2, err := OpenWALWithFormat(dir, "binary")
	require.NoError(t, err)
	defer w2.Close()

	idx := newCoreIndexWithDelta()
	applied, err := w2.Replay(idx)
	require.NoError(t, err, "concurrent writes must not corrupt the log")
	require.Equal(t, n, applied)
}
