package index

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// WAL format constants
const (
	walFileName     = "wal.log"
	walFormatJSON   = "json"
	walFormatBinary = "binary"
)

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

// WAL is the interface for write-ahead log operations.
// Both JSONWAL and BinaryWAL implement this interface.
type WAL interface {
	// Path returns the absolute path of the WAL file. Useful for diagnostics.
	Path() string

	// LogAdd records an AddFile mutation. Caller MUST call this before
	// CoreIndex.AddFile so a crash between them is replay-recoverable.
	LogAdd(meta *Metadata) error

	// LogDelete records a tombstone mutation.
	LogDelete(path string) error

	// LogAppendShard records an entire shard's metadata (including its Objects).
	// Used by dataset-init paths that bulk-append shards.
	LogAppendShard(shard *Shard) error

	// Replay reads the WAL from start to EOF and applies every record to idx.
	// On parse error or apply failure it returns immediately — the daemon must
	// surface the problem rather than silently continue with a stale CoreIndex.
	// Restores the file position to EOF on exit so subsequent LogXxx calls keep
	// appending.
	Replay(idx *CoreIndex) (applied int, err error)

	// Truncate empties the WAL. Call ONLY after the manifest has been durably
	// stored — truncating before that would lose the very mutations the manifest
	// was supposed to capture.
	Truncate() error

	// Close flushes and releases the WAL file descriptor.
	Close() error
}

// JSONWAL is the JSON-based WAL implementation.
type JSONWAL struct {
	mu   sync.Mutex
	file *os.File
	path string
}

// openWAL opens (or creates) the WAL file at <root>/wal.log in append+rw mode.
// The file is positioned at end (O_APPEND); Replay seeks to start, then back.
// The format parameter specifies the WAL format to use ("json" or "binary").
// If format is empty, defaults to "json".
func OpenWALWithFormat(root, format string) (WAL, error) {
	// Default to JSON format if not specified
	if format == "" {
		format = walFormatJSON
	}

	path := filepath.Join(root, walFileName)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("open wal %s: %w", path, err)
	}

	// Create appropriate WAL implementation based on format
	switch strings.ToLower(format) {
	case walFormatJSON:
		wal := &JSONWAL{file: f, path: path}
		return wal, nil
	case walFormatBinary:
		// For binary format, we need to check if the file is empty or has the correct header
		info, err := f.Stat()
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("stat wal file: %w", err)
		}

		// If file is empty, create new binary WAL
		if info.Size() == 0 {
			return NewBinaryWAL(f)
		}

		// If file exists, open for reading and verify header
		return OpenBinaryWAL(f)
	default:
		f.Close()
		return nil, fmt.Errorf("unsupported WAL format: %s", format)
	}
}

// Path returns the absolute path of the WAL file. Useful for diagnostics.
func (w *JSONWAL) Path() string { return w.path }

// writeEntry serializes and fsyncs one record. Holds the mutex for the whole
// op so concurrent mutators see WAL records in the order they were submitted.
func (w *JSONWAL) writeEntry(e *WALEntry) error {
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
func (w *JSONWAL) LogAdd(meta *Metadata) error {
	if meta == nil {
		return fmt.Errorf("LogAdd: meta is nil")
	}
	return w.writeEntry(&WALEntry{Op: OpAdd, Add: meta})
}

// LogDelete records a tombstone mutation.
func (w *JSONWAL) LogDelete(path string) error {
	if path == "" {
		return fmt.Errorf("LogDelete: empty path")
	}
	return w.writeEntry(&WALEntry{Op: OpDelete, Delete: path})
}

// LogAppendShard records an entire shard's metadata (including its Objects).
// Used by dataset-init paths that bulk-append shards.
func (w *JSONWAL) LogAppendShard(shard *Shard) error {
	if shard == nil {
		return fmt.Errorf("LogAppendShard: shard is nil")
	}
	return w.writeEntry(&WALEntry{Op: OpAppendShard, Shard: shard})
}

// Replay reads the WAL from start to EOF and applies every record to idx.
// On parse error or apply failure it returns immediately — the daemon must
// surface the problem rather than silently continue with a stale CoreIndex.
// Restores the file position to EOF on exit so subsequent LogXxx calls keep
// appending.
func (w *JSONWAL) Replay(idx *CoreIndex) (applied int, err error) {
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
		// Replay is idempotent: a tombstone for an already-absent file is a
		// no-op rather than a fatal error (see MarkDeletedTolerant).
		idx.MarkDeletedTolerant(e.Delete)
		return nil
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
func (w *JSONWAL) Truncate() error {
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
func (w *JSONWAL) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.file == nil {
		return nil
	}
	err := w.file.Close()
	w.file = nil
	return err
}
