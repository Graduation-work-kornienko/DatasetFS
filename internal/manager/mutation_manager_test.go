package manager

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
)

type walOp struct {
	kind    string
	shardID int
	path    string
}

type recordingWAL struct {
	t       *testing.T
	root    string
	ops     []walOp
	onAdd   func(*index.Metadata)
	onShard func(*index.Shard)
}

func (w *recordingWAL) Path() string { return filepath.Join(w.root, "wal.log") }

func (w *recordingWAL) LogAdd(meta *index.Metadata) error {
	w.ops = append(w.ops, walOp{kind: "add", shardID: meta.ShardID, path: meta.Path})
	if w.onAdd != nil {
		w.onAdd(meta)
	}
	return nil
}

func (w *recordingWAL) LogDelete(path string) error {
	w.ops = append(w.ops, walOp{kind: "delete", path: path})
	return nil
}

func (w *recordingWAL) LogAppendShard(shard *index.Shard) error {
	w.ops = append(w.ops, walOp{kind: "shard", shardID: shard.Number})
	if w.onShard != nil {
		w.onShard(shard)
	}
	return nil
}

func (w *recordingWAL) Replay(idx *index.CoreIndex) (int, error) { return 0, nil }
func (w *recordingWAL) Truncate() error                          { return nil }
func (w *recordingWAL) Close() error                             { return nil }

func writeTempObject(t *testing.T, dir, name, data string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(data), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestAddDeltaFileLogsBeforePhysicalAppend(t *testing.T) {
	root := t.TempDir()
	strg := storage.New(root, nil)
	idx := index.NewIndex()
	manifest := index.NewManifest(root)

	wal := &recordingWAL{t: t, root: root}
	wal.onAdd = func(meta *index.Metadata) {
		st, err := os.Stat(strg.ShardPath(meta.ShardID))
		if err != nil {
			t.Fatalf("delta stat during LogAdd: %v", err)
		}
		if st.Size() != 0 {
			t.Fatalf("LogAdd must happen before appending delta bytes: size=%d", st.Size())
		}
	}

	mgr := NewMutationManager(idx, manifest, wal, strg)
	src := writeTempObject(t, root, "object.bin", "abc")
	if err := mgr.AddDeltaFile("object.bin", src); err != nil {
		t.Fatal(err)
	}

	if len(wal.ops) != 1 || wal.ops[0].kind != "add" || wal.ops[0].shardID != index.DeltaShardID {
		t.Fatalf("unexpected WAL ops: %#v", wal.ops)
	}
	if _, ok := idx.FileMap["object.bin"]; !ok {
		t.Fatalf("object was not added to index")
	}
}

func TestAddDeltaFileRotatesDeltaShardWhenFull(t *testing.T) {
	root := t.TempDir()
	strg := storage.New(root, nil)
	idx := index.NewIndex()
	manifest := index.NewManifest(root)
	wal := &recordingWAL{t: t, root: root}

	mgr := NewMutationManager(idx, manifest, wal, strg)
	f, err := os.OpenFile(strg.ShardPath(index.DeltaShardID), os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(index.ShardSize); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	src := writeTempObject(t, root, "rotated.bin", "abc")
	if err := mgr.AddDeltaFile("rotated.bin", src); err != nil {
		t.Fatal(err)
	}

	if len(wal.ops) != 2 {
		t.Fatalf("expected shard+add WAL ops, got %#v", wal.ops)
	}
	if wal.ops[0].kind != "shard" || wal.ops[0].shardID != -2 {
		t.Fatalf("first WAL op must create shard_-2, got %#v", wal.ops[0])
	}
	if wal.ops[1].kind != "add" || wal.ops[1].shardID != -2 {
		t.Fatalf("second WAL op must add into shard_-2, got %#v", wal.ops[1])
	}
	meta := idx.FileMap["rotated.bin"]
	if meta == nil || meta.ShardID != -2 {
		t.Fatalf("object must be indexed in shard_-2, got %#v", meta)
	}
	if _, err := os.Stat(strg.ShardPath(-2)); err != nil {
		t.Fatalf("rotated delta shard was not created: %v", err)
	}
}

func TestApplyTransactionPutDeleteRename(t *testing.T) {
	root := t.TempDir()
	strg := storage.New(root, nil)
	idx := index.NewIndex()
	manifest := index.NewManifest(root)
	wal := &recordingWAL{t: t, root: root}
	mgr := NewMutationManager(idx, manifest, wal, strg)

	old := writeTempObject(t, root, "old.bin", "old")
	if err := mgr.AddDeltaFile("old.bin", old); err != nil {
		t.Fatal(err)
	}
	deleteMe := writeTempObject(t, root, "delete.bin", "delete")
	if err := mgr.AddDeltaFile("delete.bin", deleteMe); err != nil {
		t.Fatal(err)
	}
	put := writeTempObject(t, root, "new.bin", "new")

	if err := mgr.ApplyTransaction([]TransactionOp{
		{Kind: TxRename, Path: "old.bin", Target: "renamed.bin"},
		{Kind: TxDelete, Path: "delete.bin"},
		{Kind: TxPut, Path: "put.bin", TmpFilePath: put},
	}); err != nil {
		t.Fatal(err)
	}

	if meta := idx.FileMap["old.bin"]; meta == nil || !meta.Deleted {
		t.Fatalf("old.bin must be tombstoned, got %#v", meta)
	}
	if meta := idx.FileMap["delete.bin"]; meta == nil || !meta.Deleted {
		t.Fatalf("delete.bin must be tombstoned, got %#v", meta)
	}
	if meta := idx.FileMap["renamed.bin"]; meta == nil || meta.Deleted || meta.Path != "renamed.bin" {
		t.Fatalf("renamed.bin missing: %#v", meta)
	}
	if meta := idx.FileMap["put.bin"]; meta == nil || meta.Deleted || meta.Size != 3 {
		t.Fatalf("put.bin missing: %#v", meta)
	}
}

func TestParseTransactionFile(t *testing.T) {
	root := t.TempDir()
	putDir := filepath.Join(root, "put")
	if err := os.Mkdir(putDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(putDir, "payload_000000"), []byte("abc"), 0644); err != nil {
		t.Fatal(err)
	}

	var buf []byte
	buf = append(buf, []byte("DFTX")...)
	header := make([]byte, 8)
	binary.LittleEndian.PutUint16(header[0:2], 1)
	binary.LittleEndian.PutUint16(header[2:4], 0)
	binary.LittleEndian.PutUint32(header[4:8], 3)
	buf = append(buf, header...)
	appendRecord := func(op byte, path, aux string) {
		buf = append(buf, op)
		buf = binary.AppendUvarint(buf, uint64(len(path)))
		buf = append(buf, []byte(path)...)
		buf = binary.AppendUvarint(buf, uint64(len(aux)))
		buf = append(buf, []byte(aux)...)
	}
	appendRecord(byte(TxPut), "a.bin", "put/payload_000000")
	appendRecord(byte(TxDelete), "b.bin", "")
	appendRecord(byte(TxRename), "c.bin", "d.bin")

	path := filepath.Join(root, "ops.dfstx")
	if err := os.WriteFile(path, buf, 0644); err != nil {
		t.Fatal(err)
	}
	ops, err := ParseTransactionFile(path, root)
	if err != nil {
		t.Fatal(err)
	}
	if len(ops) != 3 {
		t.Fatalf("expected 3 ops, got %d", len(ops))
	}
	if ops[0].Kind != TxPut || ops[0].Path != "a.bin" || ops[0].TmpFilePath != filepath.Join(root, "put/payload_000000") {
		t.Fatalf("bad put op: %#v", ops[0])
	}
	if ops[1].Kind != TxDelete || ops[1].Path != "b.bin" {
		t.Fatalf("bad delete op: %#v", ops[1])
	}
	if ops[2].Kind != TxRename || ops[2].Path != "c.bin" || ops[2].Target != "d.bin" {
		t.Fatalf("bad rename op: %#v", ops[2])
	}
}
