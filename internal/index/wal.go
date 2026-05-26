package index

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// WAL = write-ahead log for mutations against the in-memory CoreIndex.
//
// Problem it solves: mutations (AddDeltaFile, DeleteFile, AppendShard) update
// the in-memory CoreIndex immediately, but the on-disk manifest is rewritten
// only at Shutdown. If the daemon crashes between mutation and shutdown, all
// mutations since the last manifest save are lost.
//
// Contract:
//   - Mutation operations call LogXxx BEFORE updating CoreIndex. Each LogXxx
//     does an O_APPEND write + fsync to /<root>/wal.log.
//   - On daemon startup: load manifest → LoadCoreIndex → OpenWAL → Replay onto
//     CoreIndex. The result is the same in-memory state the daemon had when
//     it crashed.
//   - On clean Shutdown: write manifest, THEN Truncate WAL. The order matters:
//     truncating before the manifest is durable would lose the very mutations
//     we just persisted.
//
// Failure modes (documented, not yet auto-recovered):
//   - Crash AFTER tar.Write but BEFORE LogAdd → tar has orphan bytes, index
//     has nothing. Harmless for index integrity; tar accumulates dead bytes
//     until a future vacuum pass.
//   - Crash AFTER LogAdd but BEFORE CoreIndex.AddFile → WAL replay applies
//     it; bytes are already in tar; index becomes consistent.
//   - Crash AFTER CoreIndex.AddFile but before client ack → client may retry
//     and produce a duplicate WAL entry. Replay handles "shard not found"
//     gracefully but does NOT dedupe by path. Idempotency is a known gap.
//
// Format: one JSON object per line. Each line is one of:
//   {"op":"add","ts":...,"add":{<Metadata>}}
//   {"op":"delete","ts":...,"delete":"<path>"}
//   {"op":"shard","ts":...,"shard":{<Shard with embedded Objects>}}
//
// Not (yet) implemented: log rotation, segment files, CRC per record,
// concurrent-process file locking. Reasonable for a graduation-thesis
// single-instance daemon; would need hardening for multi-instance prod.

const walFileName = "wal.log"

// WALOp identifies the kind of mutation in a WAL record.
type WALOp string

const (
	OpAdd         WALOp = "add"
	OpDelete      WALOp = "delete"
	OpAppendShard WALOp = "shard"
)

// WALEntry is one line in the WAL. Exactly one of Add/Delete/Shard is
// populated, selected by Op.
type WALEntry struct {
	Op        WALOp     `json:"op"`
	Timestamp int64     `json:"ts"`
	Add       *Metadata `json:"add,omitempty"`
	Delete    string    `json:"delete,omitempty"`
	Shard     *Shard    `json:"shard,omitempty"`
}

type WAL struct {
	mu   sync.Mutex
	file *os.File
	path string
}

// OpenWAL opens (or creates) the WAL file at <root>/wal.log in append+rw mode.
// The file is positioned at end (O_APPEND); Replay seeks to start, then back.
func OpenWAL(root string) (*WAL, error) {
	path := filepath.Join(root, walFileName)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("open wal %s: %w", path, err)
	}
	return &WAL{file: f, path: path}, nil
}

// Path returns the absolute path of the WAL file. Useful for diagnostics.
func (w *WAL) Path() string { return w.path }

// writeEntry serializes and fsyncs one record. Holds the mutex for the whole
// op so concurrent mutators see WAL records in the order they were submitted.
func (w *WAL) writeEntry(e *WALEntry) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.file == nil {
		return fmt.Errorf("wal closed")
	}

	e.Timestamp = time.Now().Unix()
	data, err := json.Marshal(e)
	if err != nil {
		return fmt.Errorf("marshal wal entry: %w", err)
	}
	data = append(data, '\n')

	if _, err := w.file.Write(data); err != nil {
		return fmt.Errorf("write wal entry: %w", err)
	}
	// fsync makes the durability guarantee real. Without this, a kernel
	// crash within ~30 s of the write loses the record even though Write
	// returned success.
	if err := w.file.Sync(); err != nil {
		return fmt.Errorf("sync wal: %w", err)
	}
	return nil
}

// LogAdd records an AddFile mutation. Caller MUST call this before
// CoreIndex.AddFile so a crash between them is replay-recoverable.
func (w *WAL) LogAdd(meta *Metadata) error {
	if meta == nil {
		return fmt.Errorf("LogAdd: meta is nil")
	}
	return w.writeEntry(&WALEntry{Op: OpAdd, Add: meta})
}

// LogDelete records a tombstone mutation.
func (w *WAL) LogDelete(path string) error {
	if path == "" {
		return fmt.Errorf("LogDelete: empty path")
	}
	return w.writeEntry(&WALEntry{Op: OpDelete, Delete: path})
}

// LogAppendShard records an entire shard's metadata (including its Objects).
// Used by dataset-init paths that bulk-append shards.
func (w *WAL) LogAppendShard(shard *Shard) error {
	if shard == nil {
		return fmt.Errorf("LogAppendShard: shard is nil")
	}
	return w.writeEntry(&WALEntry{Op: OpAppendShard, Shard: shard})
}

// Replay reads the WAL from start to EOF and applies every record to idx.
// On parse error or apply failure it returns immediately — the daemon must
// surface the problem rather than silently continue with a stale CoreIndex.
//
// Restores the file position to EOF on exit so subsequent LogXxx calls keep
// appending.
func (w *WAL) Replay(idx *CoreIndex) (applied int, err error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.file == nil {
		return 0, fmt.Errorf("wal closed")
	}

	if _, err = w.file.Seek(0, io.SeekStart); err != nil {
		return 0, fmt.Errorf("seek wal start: %w", err)
	}
	// Always end positioned at EOF for subsequent appends.
	defer w.file.Seek(0, io.SeekEnd)

	scanner := bufio.NewScanner(w.file)
	// Shard records (with Objects array) can be large; raise scanner buffer.
	scanner.Buffer(make([]byte, 0, 64*1024), 32*1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var e WALEntry
		if jerr := json.Unmarshal(line, &e); jerr != nil {
			return applied, fmt.Errorf("wal replay: malformed entry at record %d: %w", applied+1, jerr)
		}
		if aerr := applyEntry(&e, idx); aerr != nil {
			return applied, fmt.Errorf("wal replay: apply entry %d (op=%s): %w", applied+1, e.Op, aerr)
		}
		applied++
	}
	if serr := scanner.Err(); serr != nil {
		return applied, fmt.Errorf("wal scan: %w", serr)
	}
	return applied, nil
}

// applyEntry dispatches a single WAL record to the right CoreIndex method.
// Kept package-private; clients use Replay.
func applyEntry(e *WALEntry, idx *CoreIndex) error {
	switch e.Op {
	case OpAdd:
		if e.Add == nil {
			return fmt.Errorf("op=add but Add field is nil")
		}
		return idx.AddFile(e.Add)
	case OpDelete:
		if e.Delete == "" {
			return fmt.Errorf("op=delete but Delete field is empty")
		}
		return idx.MarkDeleted(e.Delete)
	case OpAppendShard:
		if e.Shard == nil {
			return fmt.Errorf("op=shard but Shard field is nil")
		}
		return idx.AppendShard(e.Shard)
	default:
		return fmt.Errorf("unknown op: %q", e.Op)
	}
}

// Truncate empties the WAL. Call ONLY after the manifest has been durably
// stored — truncating before that would lose the very mutations the manifest
// was supposed to capture.
func (w *WAL) Truncate() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.file == nil {
		return fmt.Errorf("wal closed")
	}

	if err := w.file.Truncate(0); err != nil {
		return fmt.Errorf("truncate wal: %w", err)
	}
	if _, err := w.file.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("seek wal start: %w", err)
	}
	return w.file.Sync()
}

// Close flushes and releases the WAL file descriptor.
func (w *WAL) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.file == nil {
		return nil
	}
	err := w.file.Close()
	w.file = nil
	return err
}
