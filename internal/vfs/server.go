package vfs

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"syscall"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/manager"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"

	"github.com/hanwen/go-fuse/v2/fs"
	"github.com/hanwen/go-fuse/v2/fuse"
)

type RootNode struct {
	fs.Inode
	CoreIdx *index.CoreIndex
	MutMgr  *manager.MutationManager
	Storage *storage.Storage
	txMu    sync.Mutex
	txs     map[string]*TxState
}

var _ = (fs.NodeReaddirer)((*RootNode)(nil))
var _ = (fs.NodeLookuper)((*RootNode)(nil))
var _ = (fs.NodeUnlinker)((*RootNode)(nil))
var _ = (fs.NodeGetattrer)((*RootNode)(nil))

func (r *RootNode) Getattr(ctx context.Context, fh fs.FileHandle, out *fuse.AttrOut) syscall.Errno {
	out.Mode = fuse.S_IFDIR | 0777

	out.Uid = uint32(os.Getuid())
	out.Gid = uint32(os.Getgid())

	return 0
}

var _ = (fs.NodeGetattrer)((*FileNode)(nil))

func (f *FileNode) Getattr(ctx context.Context, fh fs.FileHandle, out *fuse.AttrOut) syscall.Errno {
	out.Size = uint64(f.meta.Size)

	out.Mode = fuse.S_IFREG | 0644

	out.Uid = uint32(os.Getuid())
	out.Gid = uint32(os.Getgid())

	return 0
}

func (r *RootNode) Readdir(ctx context.Context) (fs.DirStream, syscall.Errno) {
	entries := []fuse.DirEntry{{Mode: fuse.S_IFDIR | 0755, Name: ".datasetfs"}}

	// Читаем все файлы из нашего In-Memory Индекса
	r.CoreIdx.Mu.RLock()
	for name, meta := range r.CoreIdx.FileMap {
		// Пропускаем удаленные файлы (Они прячутся от ОС!)
		if meta.Deleted {
			continue
		}
		entries = append(entries, fuse.DirEntry{
			Mode: fuse.S_IFREG | 0644, // Это обычный файл
			Name: name,
		})
	}
	r.CoreIdx.Mu.RUnlock()

	return fs.NewListDirStream(entries), 0
}

func (r *RootNode) Lookup(ctx context.Context, name string, out *fuse.EntryOut) (*fs.Inode, syscall.Errno) {
	if name == ".datasetfs" {
		out.Mode = fuse.S_IFDIR | 0755
		return r.NewInode(ctx, &AdminNode{root: r}, fs.StableAttr{Mode: out.Mode}), 0
	}

	r.CoreIdx.Mu.RLock()
	meta, exists := r.CoreIdx.FileMap[name]
	r.CoreIdx.Mu.RUnlock()

	if !exists || meta.Deleted {
		return nil, syscall.ENOENT
	}

	out.Size = uint64(meta.Size)
	out.Mode = fuse.S_IFREG | 0644

	fileNode := &FileNode{
		meta:    meta,
		storage: r.Storage,
	}

	return r.NewInode(ctx, fileNode, fs.StableAttr{Mode: out.Mode}), 0
}

func (r *RootNode) txMap() map[string]*TxState {
	if r.txs == nil {
		r.txs = make(map[string]*TxState)
	}
	return r.txs
}

func (r *RootNode) createTx(id string) (*TxState, error) {
	if id == "" || filepath.Base(id) != id || id == "." || id == ".." {
		return nil, fmt.Errorf("invalid tx id %q", id)
	}
	r.txMu.Lock()
	defer r.txMu.Unlock()
	if _, exists := r.txMap()[id]; exists {
		return nil, os.ErrExist
	}
	dir, err := os.MkdirTemp("", "datasetfs_tx_"+id+"_*")
	if err != nil {
		return nil, err
	}
	if err := os.Mkdir(filepath.Join(dir, "put"), 0755); err != nil {
		os.RemoveAll(dir)
		return nil, err
	}
	tx := &TxState{ID: id, Root: dir}
	r.txMap()[id] = tx
	return tx, nil
}

func (r *RootNode) lookupTx(id string) *TxState {
	r.txMu.Lock()
	defer r.txMu.Unlock()
	return r.txMap()[id]
}

func (r *RootNode) commitTx(id string) error {
	r.txMu.Lock()
	tx := r.txMap()[id]
	if tx == nil {
		r.txMu.Unlock()
		return os.ErrNotExist
	}
	delete(r.txMap(), id)
	r.txMu.Unlock()
	defer os.RemoveAll(tx.Root)

	ops, err := manager.ParseTransactionFile(filepath.Join(tx.Root, "ops.dfstx"), tx.Root)
	if err != nil {
		return err
	}
	return r.MutMgr.ApplyTransaction(ops)
}

type TxState struct {
	ID   string
	Root string
}

type AdminNode struct {
	fs.Inode
	root *RootNode
}

var _ = (fs.NodeReaddirer)((*AdminNode)(nil))
var _ = (fs.NodeLookuper)((*AdminNode)(nil))
var _ = (fs.NodeGetattrer)((*AdminNode)(nil))

func dirAttr(out *fuse.AttrOut) {
	out.Mode = fuse.S_IFDIR | 0777
	out.Uid = uint32(os.Getuid())
	out.Gid = uint32(os.Getgid())
}

func (n *AdminNode) Getattr(ctx context.Context, fh fs.FileHandle, out *fuse.AttrOut) syscall.Errno {
	dirAttr(out)
	return 0
}

func (n *AdminNode) Readdir(ctx context.Context) (fs.DirStream, syscall.Errno) {
	return fs.NewListDirStream([]fuse.DirEntry{
		{Mode: fuse.S_IFDIR | 0755, Name: "tx"},
		{Mode: fuse.S_IFDIR | 0755, Name: "commit"},
	}), 0
}

func (n *AdminNode) Lookup(ctx context.Context, name string, out *fuse.EntryOut) (*fs.Inode, syscall.Errno) {
	switch name {
	case "tx":
		out.Mode = fuse.S_IFDIR | 0755
		return n.NewInode(ctx, &TxRootNode{root: n.root}, fs.StableAttr{Mode: out.Mode}), 0
	case "commit":
		out.Mode = fuse.S_IFDIR | 0755
		return n.NewInode(ctx, &CommitRootNode{root: n.root}, fs.StableAttr{Mode: out.Mode}), 0
	default:
		return nil, syscall.ENOENT
	}
}

type TxRootNode struct {
	fs.Inode
	root *RootNode
}

var _ = (fs.NodeReaddirer)((*TxRootNode)(nil))
var _ = (fs.NodeLookuper)((*TxRootNode)(nil))
var _ = (fs.NodeMkdirer)((*TxRootNode)(nil))
var _ = (fs.NodeRenamer)((*TxRootNode)(nil))
var _ = (fs.NodeGetattrer)((*TxRootNode)(nil))

func (n *TxRootNode) Getattr(ctx context.Context, fh fs.FileHandle, out *fuse.AttrOut) syscall.Errno {
	dirAttr(out)
	return 0
}

func (n *TxRootNode) Readdir(ctx context.Context) (fs.DirStream, syscall.Errno) {
	n.root.txMu.Lock()
	defer n.root.txMu.Unlock()
	entries := make([]fuse.DirEntry, 0, len(n.root.txMap()))
	for id := range n.root.txMap() {
		entries = append(entries, fuse.DirEntry{Mode: fuse.S_IFDIR | 0755, Name: id})
	}
	return fs.NewListDirStream(entries), 0
}

func (n *TxRootNode) Lookup(ctx context.Context, name string, out *fuse.EntryOut) (*fs.Inode, syscall.Errno) {
	tx := n.root.lookupTx(name)
	if tx == nil {
		return nil, syscall.ENOENT
	}
	out.Mode = fuse.S_IFDIR | 0755
	return n.NewInode(ctx, &TxNode{root: n.root, tx: tx}, fs.StableAttr{Mode: out.Mode}), 0
}

func (n *TxRootNode) Mkdir(ctx context.Context, name string, mode uint32, out *fuse.EntryOut) (*fs.Inode, syscall.Errno) {
	tx, err := n.root.createTx(name)
	if err != nil {
		if os.IsExist(err) {
			return nil, syscall.EEXIST
		}
		return nil, syscall.EIO
	}
	out.Mode = fuse.S_IFDIR | 0755
	return n.NewInode(ctx, &TxNode{root: n.root, tx: tx}, fs.StableAttr{Mode: out.Mode}), 0
}

func (n *TxRootNode) Rename(ctx context.Context, name string, newParent fs.InodeEmbedder, newName string, flags uint32) syscall.Errno {
	if _, ok := newParent.EmbeddedInode().Operations().(*CommitRootNode); !ok {
		return syscall.EXDEV
	}
	if newName != name {
		return syscall.EINVAL
	}
	if err := n.root.commitTx(name); err != nil {
		if os.IsNotExist(err) {
			return syscall.ENOENT
		}
		return syscall.EIO
	}
	return 0
}

type CommitRootNode struct {
	fs.Inode
	root *RootNode
}

var _ = (fs.NodeReaddirer)((*CommitRootNode)(nil))
var _ = (fs.NodeGetattrer)((*CommitRootNode)(nil))

func (n *CommitRootNode) Getattr(ctx context.Context, fh fs.FileHandle, out *fuse.AttrOut) syscall.Errno {
	dirAttr(out)
	return 0
}

func (n *CommitRootNode) Readdir(ctx context.Context) (fs.DirStream, syscall.Errno) {
	return fs.NewListDirStream(nil), 0
}

type TxNode struct {
	fs.Inode
	root *RootNode
	tx   *TxState
}

var _ = (fs.NodeReaddirer)((*TxNode)(nil))
var _ = (fs.NodeLookuper)((*TxNode)(nil))
var _ = (fs.NodeCreater)((*TxNode)(nil))
var _ = (fs.NodeGetattrer)((*TxNode)(nil))

func (n *TxNode) Getattr(ctx context.Context, fh fs.FileHandle, out *fuse.AttrOut) syscall.Errno {
	dirAttr(out)
	return 0
}

func (n *TxNode) Readdir(ctx context.Context) (fs.DirStream, syscall.Errno) {
	return fs.NewListDirStream([]fuse.DirEntry{
		{Mode: fuse.S_IFDIR | 0755, Name: "put"},
		{Mode: fuse.S_IFREG | 0644, Name: "ops.dfstx"},
	}), 0
}

func (n *TxNode) Lookup(ctx context.Context, name string, out *fuse.EntryOut) (*fs.Inode, syscall.Errno) {
	switch name {
	case "put":
		out.Mode = fuse.S_IFDIR | 0755
		return n.NewInode(ctx, &TxPutNode{tx: n.tx}, fs.StableAttr{Mode: out.Mode}), 0
	case "ops.dfstx":
		if _, err := os.Stat(filepath.Join(n.tx.Root, "ops.dfstx")); err != nil {
			return nil, syscall.ENOENT
		}
		out.Mode = fuse.S_IFREG | 0644
		return n.NewInode(ctx, &TxStagedFileNode{path: filepath.Join(n.tx.Root, "ops.dfstx")}, fs.StableAttr{Mode: out.Mode}), 0
	default:
		return nil, syscall.ENOENT
	}
}

func (n *TxNode) Create(ctx context.Context, name string, flags uint32, mode uint32, out *fuse.EntryOut) (*fs.Inode, fs.FileHandle, uint32, syscall.Errno) {
	if name != "ops.dfstx" {
		return nil, nil, 0, syscall.EPERM
	}
	return createStagedFile(ctx, n, filepath.Join(n.tx.Root, "ops.dfstx"), mode, out)
}

type TxPutNode struct {
	fs.Inode
	tx *TxState
}

var _ = (fs.NodeReaddirer)((*TxPutNode)(nil))
var _ = (fs.NodeCreater)((*TxPutNode)(nil))
var _ = (fs.NodeGetattrer)((*TxPutNode)(nil))

func (n *TxPutNode) Getattr(ctx context.Context, fh fs.FileHandle, out *fuse.AttrOut) syscall.Errno {
	dirAttr(out)
	return 0
}

func (n *TxPutNode) Readdir(ctx context.Context) (fs.DirStream, syscall.Errno) {
	entries, err := os.ReadDir(filepath.Join(n.tx.Root, "put"))
	if err != nil {
		return nil, syscall.EIO
	}
	out := make([]fuse.DirEntry, 0, len(entries))
	for _, entry := range entries {
		out = append(out, fuse.DirEntry{Mode: fuse.S_IFREG | 0644, Name: entry.Name()})
	}
	return fs.NewListDirStream(out), 0
}

func (n *TxPutNode) Create(ctx context.Context, name string, flags uint32, mode uint32, out *fuse.EntryOut) (*fs.Inode, fs.FileHandle, uint32, syscall.Errno) {
	if filepath.Base(name) != name || name == "." || name == ".." {
		return nil, nil, 0, syscall.EINVAL
	}
	return createStagedFile(ctx, n, filepath.Join(n.tx.Root, "put", name), mode, out)
}

type TxStagedFileNode struct {
	fs.Inode
	path string
}

var _ = (fs.NodeGetattrer)((*TxStagedFileNode)(nil))

func (n *TxStagedFileNode) Getattr(ctx context.Context, fh fs.FileHandle, out *fuse.AttrOut) syscall.Errno {
	stat, err := os.Stat(n.path)
	if err != nil {
		return syscall.ENOENT
	}
	out.Size = uint64(stat.Size())
	out.Mode = fuse.S_IFREG | 0644
	out.Uid = uint32(os.Getuid())
	out.Gid = uint32(os.Getgid())
	return 0
}

type StagedWriteHandle struct {
	file *os.File
}

var _ = (fs.FileWriter)((*StagedWriteHandle)(nil))
var _ = (fs.FileReleaser)((*StagedWriteHandle)(nil))

func createStagedFile(ctx context.Context, parent fs.InodeEmbedder, path string, mode uint32, out *fuse.EntryOut) (*fs.Inode, fs.FileHandle, uint32, syscall.Errno) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_RDWR, 0644)
	if err != nil {
		return nil, nil, 0, syscall.EIO
	}
	out.Mode = fuse.S_IFREG | mode
	node := &TxStagedFileNode{path: path}
	return parent.EmbeddedInode().NewInode(ctx, node, fs.StableAttr{Mode: out.Mode}), &StagedWriteHandle{file: file}, 0, 0
}

func (h *StagedWriteHandle) Write(ctx context.Context, data []byte, off int64) (uint32, syscall.Errno) {
	n, err := h.file.WriteAt(data, off)
	if err != nil {
		return 0, syscall.EIO
	}
	return uint32(n), 0
}

func (h *StagedWriteHandle) Release(ctx context.Context) syscall.Errno {
	if err := h.file.Sync(); err != nil {
		_ = h.file.Close()
		return syscall.EIO
	}
	if err := h.file.Close(); err != nil {
		return syscall.EIO
	}
	return 0
}

func (r *RootNode) Unlink(ctx context.Context, name string) syscall.Errno {
	err := r.MutMgr.DeleteFile(name)
	if err != nil {
		return syscall.EIO
	}
	return 0 // Успех! Файл удален.
}

type FileNode struct {
	fs.Inode
	meta    *index.Metadata
	storage *storage.Storage
}

var _ = (fs.NodeReader)((*FileNode)(nil))
var _ = (fs.NodeOpener)((*FileNode)(nil))

type ShardHandle struct {
	file *os.File
}

var _ = (fs.FileReleaser)((*ShardHandle)(nil))

func (h *ShardHandle) Release(ctx context.Context) syscall.Errno {
	h.file.Close()
	return 0
}

func (f *FileNode) Open(ctx context.Context, flags uint32) (fh fs.FileHandle, fuseFlags uint32, errno syscall.Errno) {
	if flags&(syscall.O_WRONLY|syscall.O_RDWR) != 0 {
		return nil, 0, syscall.EPERM
	}

	shardPath := f.storage.ShardPath(f.meta.ShardID)

	fmt.Println(shardPath)

	file, err := os.Open(shardPath)
	if err != nil {
		return nil, 0, syscall.EIO
	}

	return &ShardHandle{file: file}, 0, 0
}

func (f *FileNode) Read(ctx context.Context, f1 fs.FileHandle, dest []byte, off int64) (fuse.ReadResult, syscall.Errno) {
	handle := f1.(*ShardHandle)

	absoluteOffset := f.meta.Offset + off

	n, err := handle.file.ReadAt(dest, absoluteOffset)
	if err != nil && err != io.EOF {
		return nil, syscall.EIO
	}

	return fuse.ReadResultData(dest[:n]), 0
}

var _ = (fs.NodeCreater)((*RootNode)(nil))

func (r *RootNode) Create(ctx context.Context, name string, flags uint32, mode uint32, out *fuse.EntryOut) (node *fs.Inode, fh fs.FileHandle, fuseFlags uint32, errno syscall.Errno) {

	tmpFile, err := os.CreateTemp("", "mlfs_upload_*")
	if err != nil {
		return nil, nil, 0, syscall.EIO
	}

	// create empty filenode - os needs to see it first
	fileNode := &FileNode{
		meta:    &index.Metadata{Path: name, Size: 0},
		storage: r.Storage,
	}

	out.Mode = fuse.S_IFREG | mode

	writeHandle := &WriteHandle{
		name:     name,
		tmpFile:  tmpFile,
		mutMgr:   r.MutMgr,
		coreIdx:  r.CoreIdx,
		fileNode: fileNode,
	}

	return r.NewInode(ctx, fileNode, fs.StableAttr{Mode: out.Mode}), writeHandle, 0, 0
}

type WriteHandle struct {
	name     string
	tmpFile  *os.File
	mutMgr   *manager.MutationManager
	coreIdx  *index.CoreIndex
	fileNode *FileNode
}

var _ = (fs.FileWriter)((*WriteHandle)(nil))
var _ = (fs.FileReleaser)((*WriteHandle)(nil))

// Write сохраняет прилетающие куски 64KB во временный файл
func (w *WriteHandle) Write(ctx context.Context, data []byte, off int64) (uint32, syscall.Errno) {
	n, err := w.tmpFile.WriteAt(data, off)
	if err != nil {
		return 0, syscall.EIO
	}
	return uint32(n), 0
}

func (w *WriteHandle) Release(ctx context.Context) syscall.Errno {
	w.tmpFile.Sync()
	tmpPath := w.tmpFile.Name()
	w.tmpFile.Close()

	defer os.Remove(tmpPath)

	err := w.mutMgr.AddDeltaFile(w.name, tmpPath)
	if err != nil {
		return syscall.EIO
	}

	w.coreIdx.Mu.RLock()
	meta := w.coreIdx.FileMap[w.name]
	w.coreIdx.Mu.RUnlock()
	if meta == nil || meta.Deleted {
		return syscall.EIO
	}
	w.fileNode.meta = meta

	return 0
}
