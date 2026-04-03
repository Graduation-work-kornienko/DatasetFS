package shm

import (
	"fmt"
	"os"
	"sync/atomic"
	"syscall"
	"unsafe"
)

const (
	NumSlots  = 9
	SlotSize  = 110 * 1024 * 1024 // 110 MB
	TotalSize = NumSlots * SlotSize
)

type Allocator struct {
	dataMap []byte
	refsMap []int32

	dataFile *os.File
	refsFile *os.File
}

func NewAllocator() (*Allocator, error) {
	df, err := createShmFile("/tmp/mlfs_data.bin", int64(TotalSize))
	if err != nil {
		return nil, err
	}

	// numslots * sizeof(int32)
	rf, err := createShmFile("/tmp/mlfs_refs.bin", int64(NumSlots*4))
	if err != nil {
		return nil, err
	}

	// Флаги: Чтение + Запись, Shared
	dataMem, err := syscall.Mmap(int(df.Fd()), 0, TotalSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap data failed: %w", err)
	}

	refsMem, err := syscall.Mmap(int(rf.Fd()), 0, NumSlots*4, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap refs failed: %w", err)
	}

	refsPtr := (*int32)(unsafe.Pointer(&refsMem[0]))
	refsAtomicArray := unsafe.Slice(refsPtr, NumSlots)

	return &Allocator{
		dataMap:  dataMem,
		refsMap:  refsAtomicArray,
		dataFile: df,
		refsFile: rf,
	}, nil
}

func (a *Allocator) GetSlotBuffer(slotID int) []byte {
	start := slotID * SlotSize
	end := start + SlotSize
	return a.dataMap[start:end]
}

func (a *Allocator) SetRefCount(slotID int, count int32) {
	atomic.StoreInt32(&a.refsMap[slotID], count)
}

func (a *Allocator) ReadRefCount(slotID int) int32 {
	return atomic.LoadInt32(&a.refsMap[slotID])
}

func (a *Allocator) Close() {
	if a.dataMap != nil {
		syscall.Munmap(a.dataMap)
	}

	if a.refsMap != nil {
		byteLen := len(a.refsMap) * 4

		byteSlice := unsafe.Slice((*byte)(unsafe.Pointer(&a.refsMap[0])), byteLen)

		syscall.Munmap(byteSlice)
	}

	if a.dataFile != nil {
		a.dataFile.Close()
	}
	if a.refsFile != nil {
		a.refsFile.Close()
	}
}

func createShmFile(path string, size int64) (*os.File, error) {
	os.Remove(path)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0666)
	if err != nil {
		return nil, err
	}

	_, err = f.Seek(size-1, os.SEEK_SET)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("ошибка Seek при аллокации %s: %w", path, err)
	}
	_, err = f.Write([]byte{0})
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("ошибка аллокации пространства для %s: %w", path, err)
	}
	f.Seek(0, os.SEEK_SET)
	return f, nil
}
