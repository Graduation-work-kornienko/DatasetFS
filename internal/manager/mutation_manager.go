package manager

import (
	"archive/tar"
	"context"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
	"golang.org/x/sync/errgroup"
)

type MutationManager struct {
	mu        *sync.Mutex      // Гарантирует, что мы пишем дельты по очереди
	coreIndex *index.CoreIndex // Указатель на in-memory мозг
	manifest  *index.Manifest  // manifest File
	walWriter index.WAL        // Interface for WAL operations
	storage   *storage.Storage

	lastShard      *int // Number of last shard
	currentDeltaID *int // Current append-only delta shard (negative IDs)
}

type ReplacementFile struct {
	LogicalName string
	TmpFilePath string
}

type TransactionOpKind uint8

const (
	TxPut TransactionOpKind = iota + 1
	TxDelete
	TxRename
)

type TransactionOp struct {
	Kind        TransactionOpKind
	Path        string
	Target      string
	TmpFilePath string
}

func NewMutationManager(idx *index.CoreIndex, m *index.Manifest, wal index.WAL, tar *storage.Storage) *MutationManager {
	var lastShard int = 0
	currentDeltaID := index.DeltaShardID
	idx.Mu.Lock()
	for shardID, shard := range idx.ShardMap {
		if shard.Type == index.Delta || shardID < 0 {
			if shardID < currentDeltaID {
				currentDeltaID = shardID
			}
		}
		if shardID >= lastShard {
			lastShard = shardID + 1
		}
	}
	if _, ok := idx.ShardMap[currentDeltaID]; !ok {
		idx.ShardMap[currentDeltaID] = &index.Shard{
			Number:    currentDeltaID,
			Type:      index.Delta,
			TotalSize: 0,
			Objects:   make([]*index.Metadata, 0),
		}
	}
	idx.Mu.Unlock()
	_ = ensureDeltaFile(tar, currentDeltaID)
	return &MutationManager{
		mu:             &sync.Mutex{},
		coreIndex:      idx,
		manifest:       m,
		walWriter:      wal,
		storage:        tar,
		lastShard:      &lastShard,
		currentDeltaID: &currentDeltaID,
	}
}

func ensureDeltaFile(strg *storage.Storage, shardID int) error {
	f, err := os.OpenFile(strg.ShardPath(shardID), os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return err
	}
	return f.Close()
}

func tarRecordSize(payloadSize int64) int64 {
	padding := int64(0)
	if remainder := payloadSize % 512; remainder != 0 {
		padding = 512 - remainder
	}
	return 512 + payloadSize + padding
}

func tarAppendSize(payloadSize int64) int64 {
	// tar.Writer.Close appends two zero blocks. Existing single-file append path
	// creates one tiny tar stream per object, so batch planning must account for
	// the same physical bytes when computing the next object's offset.
	return tarRecordSize(payloadSize) + 1024
}

func (m *MutationManager) ensureWritableDelta(recordSize int64) (int, int64, error) {
	m.coreIndex.Mu.RLock()
	for shardID, shard := range m.coreIndex.ShardMap {
		if (shard.Type == index.Delta || shardID < 0) && shardID < *m.currentDeltaID {
			*m.currentDeltaID = shardID
		}
	}
	m.coreIndex.Mu.RUnlock()

	deltaID := *m.currentDeltaID
	deltaPath := m.storage.ShardPath(deltaID)
	if err := ensureDeltaFile(m.storage, deltaID); err != nil {
		return 0, 0, err
	}
	stat, err := os.Stat(deltaPath)
	if err != nil {
		return 0, 0, err
	}
	currentEndOffset := stat.Size()
	if currentEndOffset == 0 || currentEndOffset+recordSize <= index.ShardSize {
		return deltaID, currentEndOffset, nil
	}

	newDeltaID := deltaID - 1
	newShard := &index.Shard{
		Number:    newDeltaID,
		Type:      index.Delta,
		TotalSize: 0,
		Objects:   make([]*index.Metadata, 0),
	}
	if m.walWriter != nil {
		if err := m.walWriter.LogAppendShard(newShard); err != nil {
			return 0, 0, fmt.Errorf("wal LogAppendShard %d: %w", newDeltaID, err)
		}
	}
	if err := m.coreIndex.AppendShard(newShard); err != nil {
		return 0, 0, err
	}
	if err := ensureDeltaFile(m.storage, newDeltaID); err != nil {
		return 0, 0, err
	}
	*m.currentDeltaID = newDeltaID
	return newDeltaID, 0, nil
}

// DeleteFile обрабатывает FUSE вызов `rm`.
//
// Order matters: WAL fsync goes first, then in-memory mutation. If we crash
// between them, replay sees the WAL entry and re-marks the file deleted.
// If we crashed in the opposite order, the in-memory delete would happen but
// not survive the next manifest checkpoint — silent loss.
func (m *MutationManager) DeleteFile(filename string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	return m.deleteFileLocked(filename)
}

func (m *MutationManager) deleteFileLocked(filename string) error {
	if m.walWriter != nil {
		if err := m.walWriter.LogDelete(filename); err != nil {
			return fmt.Errorf("wal LogDelete %q: %w", filename, err)
		}
	}

	if err := m.coreIndex.MarkDeleted(filename); err != nil {
		return err
	}

	return nil
}

// AddDeltaFile обрабатывает FUSE вызов `cp` / `mv` (создание файла)
func (m *MutationManager) AddDeltaFile(logicalName string, tmpFilePath string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.addDeltaFileLocked(logicalName, tmpFilePath)
}

func (m *MutationManager) addDeltaFileLocked(logicalName string, tmpFilePath string) error {
	stat, err := os.Stat(tmpFilePath)
	if err != nil {
		return err
	}
	fileSize := stat.Size()

	recordSize := tarRecordSize(fileSize)
	deltaID, currentEndOffset, err := m.ensureWritableDelta(recordSize)
	if err != nil {
		return err
	}
	deltaPath := m.storage.ShardPath(deltaID)

	meta := &index.Metadata{
		ShardID: deltaID,
		Path:    logicalName,
		Size:    fileSize,
		Offset:  currentEndOffset + 512,
		Deleted: false,
	}

	// WAL is written before the object is made visible in the dataset state and
	// before appending bytes to the delta shard. If the process crashes after the
	// WAL write but before the physical append, recovery must ignore/repair the
	// incomplete add rather than publishing an object with missing bytes.
	if m.walWriter != nil {
		if err := m.walWriter.LogAdd(meta); err != nil {
			return fmt.Errorf("wal LogAdd %q: %w", logicalName, err)
		}
	}

	deltaFile, err := os.OpenFile(deltaPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer deltaFile.Close()

	tw := tar.NewWriter(deltaFile)

	hdr := &tar.Header{
		Name:   logicalName,
		Mode:   0644,
		Size:   fileSize,
		Format: tar.FormatGNU,
	}

	if err := tw.WriteHeader(hdr); err != nil {
		return err
	}

	srcFile, err := os.Open(tmpFilePath)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	written, err := io.Copy(tw, srcFile)
	if err != nil {
		return err
	}
	if written != fileSize {
		return fmt.Errorf("delta write size mismatch for %s: expected %d, wrote %d", logicalName, fileSize, written)
	}
	if err := tw.Close(); err != nil {
		return err
	}

	if err := m.coreIndex.AddFile(meta); err != nil {
		return err
	}
	m.coreIndex.Mu.Lock()
	if shard := m.coreIndex.ShardMap[deltaID]; shard != nil {
		shard.TotalSize = currentEndOffset + recordSize
	}
	m.coreIndex.Mu.Unlock()

	return nil
}

func (m *MutationManager) ReplaceFilesBatch(files []ReplacementFile) error {
	ops := make([]TransactionOp, 0, len(files))
	for _, file := range files {
		ops = append(ops, TransactionOp{Kind: TxPut, Path: file.LogicalName, TmpFilePath: file.TmpFilePath})
	}
	return m.ApplyTransaction(ops)
}

func (m *MutationManager) ApplyTransaction(ops []TransactionOp) error {
	if len(ops) == 0 {
		return nil
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	type plannedAdd struct {
		op         TransactionOp
		meta       *index.Metadata
		recordSize int64
	}
	type plannedRename struct {
		from string
		meta *index.Metadata
	}

	entries := make([]*index.WALEntry, 0, len(ops)*2)
	planned := make([]plannedAdd, 0)
	renames := make([]plannedRename, 0)
	deletes := make([]string, 0)
	newShards := make([]*index.Shard, 0)

	m.coreIndex.Mu.RLock()
	for shardID, shard := range m.coreIndex.ShardMap {
		if (shard.Type == index.Delta || shardID < 0) && shardID < *m.currentDeltaID {
			*m.currentDeltaID = shardID
		}
	}
	m.coreIndex.Mu.RUnlock()

	deltaID := *m.currentDeltaID
	if err := ensureDeltaFile(m.storage, deltaID); err != nil {
		return err
	}
	stat, err := os.Stat(m.storage.ShardPath(deltaID))
	if err != nil {
		return err
	}
	currentEndOffset := stat.Size()

	for _, op := range ops {
		switch op.Kind {
		case TxPut:
			stat, err := os.Stat(op.TmpFilePath)
			if err != nil {
				return err
			}
			fileSize := stat.Size()
			recordSize := tarRecordSize(fileSize)
			appendSize := tarAppendSize(fileSize)
			if currentEndOffset != 0 && currentEndOffset+recordSize > index.ShardSize {
				deltaID--
				newShard := &index.Shard{
					Number:    deltaID,
					Type:      index.Delta,
					TotalSize: 0,
					Objects:   make([]*index.Metadata, 0),
				}
				newShards = append(newShards, newShard)
				entries = append(entries, &index.WALEntry{Op: index.OpAppendShard, Shard: newShard})
				currentEndOffset = 0
			}

			meta := &index.Metadata{
				ShardID: deltaID,
				Path:    op.Path,
				Size:    fileSize,
				Offset:  currentEndOffset + 512,
				Deleted: false,
			}
			entries = append(entries,
				&index.WALEntry{Op: index.OpDelete, Delete: op.Path},
				&index.WALEntry{Op: index.OpAdd, Add: meta},
			)
			planned = append(planned, plannedAdd{op: op, meta: meta, recordSize: recordSize})
			currentEndOffset += appendSize
		case TxDelete:
			entries = append(entries, &index.WALEntry{Op: index.OpDelete, Delete: op.Path})
			deletes = append(deletes, op.Path)
		case TxRename:
			m.coreIndex.Mu.RLock()
			old := m.coreIndex.FileMap[op.Path]
			m.coreIndex.Mu.RUnlock()
			if old == nil || old.Deleted {
				return fmt.Errorf("rename %q: source not found", op.Path)
			}
			meta := *old
			meta.Path = op.Target
			meta.Deleted = false
			entries = append(entries,
				&index.WALEntry{Op: index.OpDelete, Delete: op.Path},
				&index.WALEntry{Op: index.OpDelete, Delete: op.Target},
				&index.WALEntry{Op: index.OpAdd, Add: &meta},
			)
			renames = append(renames, plannedRename{from: op.Path, meta: &meta})
		default:
			return fmt.Errorf("unknown transaction op kind: %d", op.Kind)
		}
	}

	if len(entries) == 0 {
		return nil
	}

	if m.walWriter != nil {
		if batch, ok := m.walWriter.(index.BatchWAL); ok {
			if err := batch.LogBatch(entries); err != nil {
				return fmt.Errorf("wal LogBatch: %w", err)
			}
		} else {
			for _, entry := range entries {
				switch entry.Op {
				case index.OpAppendShard:
					if err := m.walWriter.LogAppendShard(entry.Shard); err != nil {
						return err
					}
				case index.OpDelete:
					if err := m.walWriter.LogDelete(entry.Delete); err != nil {
						return err
					}
				case index.OpAdd:
					if err := m.walWriter.LogAdd(entry.Add); err != nil {
						return err
					}
				}
			}
		}
	}

	for _, shard := range newShards {
		if err := m.coreIndex.AppendShard(shard); err != nil {
			return err
		}
		if err := ensureDeltaFile(m.storage, shard.Number); err != nil {
			return err
		}
	}
	*m.currentDeltaID = deltaID

	for _, path := range deletes {
		m.coreIndex.MarkDeletedTolerant(path)
	}
	for _, rename := range renames {
		m.coreIndex.MarkDeletedTolerant(rename.from)
		m.coreIndex.MarkDeletedTolerant(rename.meta.Path)
		if err := m.coreIndex.AddFile(rename.meta); err != nil {
			return err
		}
	}

	var deltaFile *os.File
	openedDeltaID := 0
	closeDelta := func() error {
		if deltaFile == nil {
			return nil
		}
		err := deltaFile.Close()
		deltaFile = nil
		return err
	}
	defer closeDelta()

	for _, add := range planned {
		m.coreIndex.MarkDeletedTolerant(add.op.Path)
		if deltaFile == nil || openedDeltaID != add.meta.ShardID {
			if err := closeDelta(); err != nil {
				return err
			}
			deltaFile, err = os.OpenFile(m.storage.ShardPath(add.meta.ShardID), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
			if err != nil {
				return err
			}
			openedDeltaID = add.meta.ShardID
		}
		if err := writeTarRecord(deltaFile, add.meta.Path, add.op.TmpFilePath, add.meta.Size); err != nil {
			return err
		}
		if err := m.coreIndex.AddFile(add.meta); err != nil {
			return err
		}
		m.coreIndex.Mu.Lock()
		if shard := m.coreIndex.ShardMap[add.meta.ShardID]; shard != nil {
			shard.TotalSize = add.meta.Offset - 512 + add.recordSize
		}
		m.coreIndex.Mu.Unlock()
	}

	return nil
}

func writeTarRecord(dst *os.File, logicalName string, srcPath string, size int64) error {
	tw := tar.NewWriter(dst)
	hdr := &tar.Header{Name: logicalName, Mode: 0644, Size: size, Format: tar.FormatGNU}
	if err := tw.WriteHeader(hdr); err != nil {
		return err
	}
	srcFile, err := os.Open(srcPath)
	if err != nil {
		return err
	}
	defer srcFile.Close()
	written, err := io.Copy(tw, srcFile)
	if err != nil {
		return err
	}
	if written != size {
		return fmt.Errorf("delta write size mismatch for %s: expected %d, wrote %d", logicalName, size, written)
	}
	return tw.Close()
}

// WithExclusive runs fn while holding the mutation lock, so no FUSE mutation
// (add/delete/append) interleaves. The background vacuumer uses this to rewrite
// shards and reload the index atomically with respect to writes. The delta
// shard's on-disk tar is recreated lazily on the next AddDeltaFile (it opens
// with O_CREATE), so fn only needs to restore the in-memory ShardMap[-1] entry.
func (m *MutationManager) WithExclusive(fn func() error) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return fn()
}

// Simply Append shard(for dataset initialization only)
func (m *MutationManager) AppendShard(shard *index.Shard) error {

	m.mu.Lock()
	defer m.mu.Unlock()

	shard.Number = *m.lastShard

	// 1. Append to storage
	if err := m.storage.AppendShard(shard); err != nil {
		return fmt.Errorf("tarAppender.AppendShard: %w", err)
	}

	// 2. WAL before in-memory: see DeleteFile/AddDeltaFile rationale.
	if m.walWriter != nil {
		if err := m.walWriter.LogAppendShard(shard); err != nil {
			return fmt.Errorf("wal LogAppendShard %d: %w", shard.Number, err)
		}
	}

	// 3. Append to CoreIndex
	if err := m.coreIndex.AppendShard(shard); err != nil {
		return fmt.Errorf("coreIndex.AppendShard: %w", err)
	}

	*m.lastShard++
	return nil
}

func (m *MutationManager) AppendWebDatasetShards(ctx context.Context, tarPaths []string) error {
	eg, ctx := errgroup.WithContext(ctx)
	shardChan := make(chan *index.Shard, 100)
	shardIdChan := make(chan int)

	for _, tarPath := range tarPaths {
		eg.Go(func() error {
			return m.storage.HandleWebdatasetShard(tarPath, shardIdChan, shardChan)
		})
	}

	go func() {
		id := *m.lastShard
		for {
			for {
				select {
				case shardIdChan <- id:
					id++
				case <-ctx.Done():
					return
				}
			}
		}
	}()

	// get all from metachan and
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for shard := range shardChan {
			fmt.Println("got shard", shard.Number, shard.TotalSize)
			m.coreIndex.AppendShard(shard)
		}
	}()

	if err := eg.Wait(); err != nil {
		return fmt.Errorf("dataset parsing failed: %w", err)
	}
	close(shardChan)

	wg.Wait()
	return nil
}

// Shutdown checkpoints the manifest and, on success, truncates the WAL.
// Truncate order matters: WAL must outlive the manifest write — if we lose
// power between the two, replay still recovers from the (now-redundant) WAL.
// If we truncated first, a crash before Manifest.Store would lose every
// mutation since the previous checkpoint.
func (m *MutationManager) Shutdown() {
	m.mu.Lock()
	defer m.mu.Unlock()

	manifest := m.coreIndex.Manifest()
	manifest.Root = m.storage.Root
	if err := manifest.Store(); err != nil {
		fmt.Printf("[MutationManager] Shutdown: manifest.Store failed: %v — WAL preserved\n", err)
		return
	}
	if m.walWriter != nil {
		if err := m.walWriter.Truncate(); err != nil {
			fmt.Printf("[MutationManager] Shutdown: wal.Truncate failed: %v\n", err)
		}
	}
}
