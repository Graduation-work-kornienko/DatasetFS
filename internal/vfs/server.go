package vfs

import (
	"context"
	"fmt"
	"io"
	"os"
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
	var entries []fuse.DirEntry

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
		name:    name,
		tmpFile: tmpFile,
		mutMgr:  r.MutMgr,
		coreIdx: r.CoreIdx,
	}

	return r.NewInode(ctx, fileNode, fs.StableAttr{Mode: out.Mode}), writeHandle, 0, 0
}

type WriteHandle struct {
	name    string
	tmpFile *os.File
	mutMgr  *manager.MutationManager
	coreIdx *index.CoreIndex
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

	return 0
}
