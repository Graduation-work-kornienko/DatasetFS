package shm

import (
	"fmt"
	"os"
	"sync/atomic"
	"syscall"
	"unsafe"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/metrics"
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

// Reset zeroes every slot's refcount so a REUSED allocator presents all slots
// as free to a new loading session. The ~1 GB data buffer is left untouched
// (each slot is overwritten lazily as its first shard loads), which is the
// whole point: re-mmapping TotalSize on every /initialize_loading cost 33–69 ms
// per epoch for no benefit, since the size never changes. Must be called only
// after the previous session's goroutines have been joined (no concurrent SHM
// access), which server.go guarantees by stopping the old session first.
func (a *Allocator) Reset() {
	for i := range a.refsMap {
		atomic.StoreInt32(&a.refsMap[i], 0)
	}
}

func (a *Allocator) GetSlotBuffer(slotID int) []byte {
	start := slotID * SlotSize
	end := start + SlotSize
	return a.dataMap[start:end]
}

func (a *Allocator) SetRefCount(slotID int, count int32) {
	// If the slot's prior refcount is nonzero when we set a new one, the
	// consumer never finished draining the previous batch — a lifecycle bug
	// that would silently corrupt downstream pipelines.
	prev := atomic.SwapInt32(&a.refsMap[slotID], count)
	if prev != 0 {
		metrics.RefcountOverflowTotal.Add(1)
	}
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
