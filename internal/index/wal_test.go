package index

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// newCoreIndexWithDelta returns an empty CoreIndex preseeded with the delta
// shard (id=-1) that AddDeltaFile mutations target. Mirrors the bootstrap
// that NewMutationManager does in production.
func newCoreIndexWithDelta() *CoreIndex {
	idx := NewIndex()
	idx.ShardMap[-1] = &Shard{
		Number: -1, Type: "delta", TotalSize: 0,
		Objects: make([]*Metadata, 0),
	}
	return idx
}

func TestWAL_OpenAppendsToExistingFile(t *testing.T) {
	dir := t.TempDir()

	// First open + write something.
	w1, err := OpenWAL(dir)
	require.NoError(t, err)
	require.NoError(t, w1.LogDelete("first.jpg"))
	require.NoError(t, w1.Close())

	// Reopen — must append, not truncate.
	w2, err := OpenWAL(dir)
	require.NoError(t, err)
	require.NoError(t, w2.LogDelete("second.jpg"))
	require.NoError(t, w2.Close())

	raw, err := os.ReadFile(filepath.Join(dir, "wal.log"))
	require.NoError(t, err)
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	require.Len(t, lines, 2, "second open must append, not overwrite")
}

func TestWAL_LogAddRoundTrip(t *testing.T) {
	dir := t.TempDir()

	w, err := OpenWAL(dir)
	require.NoError(t, err)
	defer w.Close()

	meta := &Metadata{
		ShardID: -1, Offset: 1024, Size: 2048, Path: "new.jpg",
		ObjectMetadata: json.RawMessage(`{"label":"cat"}`),
	}
	require.NoError(t, w.LogAdd(meta))

	// Inspect raw line to verify on-disk format is stable JSONL.
	raw, err := os.ReadFile(filepath.Join(dir, "wal.log"))
	require.NoError(t, err)
	var e WALEntry
	require.NoError(t, json.Unmarshal(raw[:len(raw)-1], &e), "trailing \\n is expected")
	require.Equal(t, OpAdd, e.Op)
	require.NotNil(t, e.Add)
	require.Equal(t, "new.jpg", e.Add.Path)
	require.Equal(t, int64(2048), e.Add.Size)
	require.NotZero(t, e.Timestamp, "writer must stamp the record")
}

func TestWAL_LogDeleteAndShard(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir)
	require.NoError(t, err)
	defer w.Close()

	require.NoError(t, w.LogDelete("old.jpg"))
	require.NoError(t, w.LogAppendShard(&Shard{
		Number: 7, Type: "base", TotalSize: 100,
		Objects: []*Metadata{
			{ShardID: 7, Path: "a.jpg", Size: 50, Offset: 0},
			{ShardID: 7, Path: "b.jpg", Size: 50, Offset: 50},
		},
	}))

	raw, err := os.ReadFile(filepath.Join(dir, "wal.log"))
	require.NoError(t, err)
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	require.Len(t, lines, 2)
}

func TestWAL_ReplayEmpty(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir)
	require.NoError(t, err)
	defer w.Close()

	idx := NewIndex()
	applied, err := w.Replay(idx)
	require.NoError(t, err)
	require.Equal(t, 0, applied)
}

func TestWAL_ReplayAppliesAdd(t *testing.T) {
	dir := t.TempDir()

	w1, err := OpenWAL(dir)
	require.NoError(t, err)
	require.NoError(t, w1.LogAdd(&Metadata{
		ShardID: -1, Offset: 512, Size: 1024, Path: "new.jpg",
	}))
	require.NoError(t, w1.Close())

	// Simulate a crash + restart: open WAL, replay into a fresh CoreIndex.
	w2, err := OpenWAL(dir)
	require.NoError(t, err)
	defer w2.Close()

	idx := newCoreIndexWithDelta()
	applied, err := w2.Replay(idx)
	require.NoError(t, err)
	require.Equal(t, 1, applied)

	m, ok := idx.FileMap["new.jpg"]
	require.True(t, ok, "Replay must populate FileMap")
	require.Equal(t, int64(1024), m.Size)
	require.Equal(t, -1, m.ShardID)
}

func TestWAL_ReplayAppliesDelete(t *testing.T) {
	dir := t.TempDir()

	// Manifest already contains a file (simulate prior checkpoint).
	idx := newCoreIndexWithDelta()
	idx.ShardMap[0] = &Shard{Number: 0, Type: "base", TotalSize: 100, Objects: []*Metadata{
		{ShardID: 0, Path: "doomed.jpg", Size: 100, Offset: 0},
	}}
	idx.FileMap["doomed.jpg"] = idx.ShardMap[0].Objects[0]

	// WAL had a delete that the previous run never checkpointed.
	w, err := OpenWAL(dir)
	require.NoError(t, err)
	require.NoError(t, w.LogDelete("doomed.jpg"))

	applied, err := w.Replay(idx)
	require.NoError(t, err)
	require.Equal(t, 1, applied)
	require.True(t, idx.FileMap["doomed.jpg"].Deleted, "Replay must mark deleted")
	require.NoError(t, w.Close())
}

func TestWAL_ReplayPreservesOrder(t *testing.T) {
	// Add then delete the same file → final state = deleted. If replay
	// applied in reverse order, file would still be live.
	dir := t.TempDir()

	w, err := OpenWAL(dir)
	require.NoError(t, err)
	require.NoError(t, w.LogAdd(&Metadata{
		ShardID: -1, Offset: 512, Size: 100, Path: "x.jpg",
	}))
	require.NoError(t, w.LogDelete("x.jpg"))

	idx := newCoreIndexWithDelta()
	applied, err := w.Replay(idx)
	require.NoError(t, err)
	require.Equal(t, 2, applied)
	require.True(t, idx.FileMap["x.jpg"].Deleted)
	require.NoError(t, w.Close())
}

func TestWAL_TruncateClearsFile(t *testing.T) {
	dir := t.TempDir()

	w, err := OpenWAL(dir)
	require.NoError(t, err)
	require.NoError(t, w.LogDelete("a.jpg"))
	require.NoError(t, w.LogDelete("b.jpg"))

	require.NoError(t, w.Truncate())

	info, err := os.Stat(filepath.Join(dir, "wal.log"))
	require.NoError(t, err)
	require.Equal(t, int64(0), info.Size(), "Truncate must zero the file")

	// Subsequent writes must still go to the same file from the start.
	require.NoError(t, w.LogDelete("c.jpg"))
	require.NoError(t, w.Close())

	raw, err := os.ReadFile(filepath.Join(dir, "wal.log"))
	require.NoError(t, err)
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	require.Len(t, lines, 1, "post-Truncate WAL must contain only the new record")
}

func TestWAL_ReplayThenAppendStillFsyncs(t *testing.T) {
	// Regression: Replay rewinds the file to start; after returning it must
	// be at EOF so subsequent appends don't overwrite existing records.
	dir := t.TempDir()

	w, err := OpenWAL(dir)
	require.NoError(t, err)
	require.NoError(t, w.LogDelete("first.jpg"))

	_, err = w.Replay(newCoreIndexWithDelta())
	require.NoError(t, err)

	require.NoError(t, w.LogDelete("second.jpg"))
	require.NoError(t, w.Close())

	raw, err := os.ReadFile(filepath.Join(dir, "wal.log"))
	require.NoError(t, err)
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	require.Len(t, lines, 2, "Replay must leave file positioned at EOF for appends")
}

func TestWAL_MalformedEntryAborts(t *testing.T) {
	dir := t.TempDir()
	walPath := filepath.Join(dir, "wal.log")

	// Hand-craft a WAL with a parseable line followed by garbage.
	require.NoError(t, os.WriteFile(walPath,
		[]byte(`{"op":"delete","ts":1,"delete":"a.jpg"}`+"\n"+`{not json}`+"\n"),
		0644))

	w, err := OpenWAL(dir)
	require.NoError(t, err)
	defer w.Close()

	applied, err := w.Replay(NewIndex())
	require.Error(t, err, "malformed record must abort replay, not silently continue")
	require.Equal(t, 1, applied, "the first valid record was applied before the error")
	require.Contains(t, err.Error(), "malformed entry")
}
