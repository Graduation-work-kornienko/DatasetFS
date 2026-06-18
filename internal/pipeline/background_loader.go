package pipeline

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/metrics"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/shm"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
)

type BackgroundLoader struct {
	storage      *storage.Storage
	allocator    *shm.Allocator
	loaderChan   <-chan *LoadJob
	metadataChan chan<- *SlotMeta
	freeSlotChan chan int
}

func NewBackgroundLoader(strg *storage.Storage, alloc *shm.Allocator, req <-chan *LoadJob, res chan<- *SlotMeta, freeSlotChan chan int) *BackgroundLoader {
	return &BackgroundLoader{
		storage:      strg,
		allocator:    alloc,
		loaderChan:   req,
		metadataChan: res,
		freeSlotChan: freeSlotChan,
	}
}

func (b *BackgroundLoader) Launch(ctx context.Context) {
	// Closing metadataChan downstream lets DealerWorker know the epoch is done.
	defer close(b.metadataChan)

	for {
		select {
		case <-ctx.Done():
			return
		case job, ok := <-b.loaderChan:
			if !ok {
				// Planner has scheduled all shards; drain done.
				return
			}

			loadStart := time.Now()
			// EnsureShard returns a guaranteed-complete local path: a no-op for
			// local datasets, or (for remote/streaming) blocks until the shard
			// has finished downloading into the cache. The atomic temp+rename in
			// RemoteStorage.Fetch means the file is whole before io.ReadFull.
			shardPath, err := b.storage.EnsureShard(job.ShardID)
			if err != nil {
				log.Printf("[Loader] ❌ Ошибка получения шарда %d: %v", job.ShardID, err)
				continue
			}
			file, err := os.Open(shardPath)
			if err != nil {
				log.Printf("[Loader] ❌ Ошибка открытия шарда %d: %v", job.ShardID, err)
				continue
			}

			targetSlice := b.allocator.GetSlotBuffer(job.SlotID)

			readStart := time.Now()
			n, err := io.ReadFull(file, targetSlice[:job.Shard.TotalSize])
			closeErr := file.Close()
			readElapsed := time.Since(readStart)
			metrics.StorageReadLatency.Record(readElapsed)
			metrics.SHMWriteLatency.Record(readElapsed)

			if err != nil {
				log.Printf("[Loader] ❌ Ошибка io.ReadFull для шарда %d: %v", job.ShardID, err)
				continue
			}
			if closeErr != nil {
				log.Printf("[Loader] ❌ Ошибка закрытия шарда %d: %v", job.ShardID, closeErr)
				continue
			}

			// Successful load: record latency and bytes.
			metrics.LoadLatency.Record(time.Since(loadStart))
			metrics.ShardLoadsTotal.Add(1)
			metrics.BytesReadTotal.Add(int64(n))

			metaStart := time.Now()
			var validMeta []*Metadata
			globalSlotStartOffset := int64(job.SlotID * shm.SlotSize)

			for _, meta := range job.Shard.Objects {
				// Snapshot Objects are already live-filtered (see
				// CoreIndex.materializeLocked); the Deleted guard is kept as a
				// cheap invariant check.
				if !meta.Deleted {
					localMeta := Metadata{
						Metadata: meta,
						SlotID:   job.SlotID,
					}

					localMeta.Offset = globalSlotStartOffset + meta.Offset
					localMeta.SlotID = job.SlotID

					validMeta = append(validMeta, &localMeta)
				}

			}
			metrics.MetadataBuildLatency.Record(time.Since(metaStart))

			if len(validMeta) == 0 {
				select {
				case b.freeSlotChan <- job.SlotID:
				case <-ctx.Done():
					return
				}
				continue
			}

			// ctx-guarded: on session teardown the downstream stage may have
			// stopped draining; we must not block here or Stop() would hang and
			// the allocator could be unmapped while we still reference the slot.
			select {
			case b.metadataChan <- &SlotMeta{Objects: validMeta, SlotID: job.SlotID}:
			case <-ctx.Done():
				return
			}
		}
	}
}

// streamingReader wraps a reader to simultaneously read from source, write to cache, and provide data
type streamingReader struct {
	reader       io.Reader
	cacheWriter  *os.File
	cachePath    string
	cacheCreated bool
}

func (sr *streamingReader) Read(p []byte) (int, error) {
	n, err := sr.reader.Read(p)

	// If we have data and haven't created the cache yet, create it
	if n > 0 && !sr.cacheCreated {
		err := os.MkdirAll(filepath.Dir(sr.cachePath), 0755)
		if err != nil {
			return n, fmt.Errorf("failed to create cache directory: %w", err)
		}
		sr.cacheWriter, err = os.Create(sr.cachePath)
		if err != nil {
			return n, fmt.Errorf("failed to create cache file: %w", err)
		}
		sr.cacheCreated = true
	}

	// Write to cache if we have a writer and data
	if sr.cacheWriter != nil && n > 0 {
		_, werr := sr.cacheWriter.Write(p[:n])
		// Don't return error immediately, but remember it
		if werr != nil {
			// If we have a write error, close the cache writer and remove the file
			sr.cacheWriter.Close()
			sr.cacheWriter = nil
			os.Remove(sr.cachePath)
			return n, fmt.Errorf("failed to write to cache: %w", werr)
		}
	}

	return n, err
}

func (sr *streamingReader) Close() error {
	var closeErr error

	// Close the cache file if it was created
	if sr.cacheWriter != nil {
		closeErr = sr.cacheWriter.Close()
		// If there was an error closing, remove the cache file
		if closeErr != nil {
			os.Remove(sr.cachePath)
		}
	}

	// Close the response body if it's different from the cache writer
	if sr.reader != sr.cacheWriter {
		if closer, ok := sr.reader.(io.Closer); ok {
			cerr := closer.Close()
			// If we don't have a previous close error, use this one
			if closeErr == nil {
				closeErr = cerr
			}
		}
	}

	// Return the close error if there was one
	return closeErr
}

// openFile opens a file with streaming download and caching
func (b *BackgroundLoader) openFile(path string) (io.ReadCloser, error) {
	// Check if we have remote storage available
	if b.storage.RemoteStorage == nil {
		// No remote storage, just open the file locally
		return os.Open(path)
	}

	// Check if file exists locally
	if _, err := os.Stat(path); err == nil {
		// File exists locally, open it
		return os.Open(path)
	}

	// File doesn't exist locally, check if path is a URL
	if !index.IsURL(path) {
		// Not a URL, return error
		return nil, fmt.Errorf("file not found: %s", path)
	}

	// Create a streaming reader that downloads and caches on-the-fly
	resp, err := b.storage.RemoteStorage.DownloadStream(context.Background(), path)
	if err != nil {
		return nil, fmt.Errorf("failed to download %s: %w", path, err)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("download failed with status %d", resp.StatusCode)
	}

	// Determine cache path
	filename := filepath.Base(path)
	if filename == "." || filename == "/" {
		filename = "index"
	}
	cachePath := filepath.Join(b.storage.RemoteStorage.CacheDir, filename)

	// Create streaming reader
	return &streamingReader{
		reader:       resp.Body,
		cachePath:    cachePath,
		cacheWriter:  nil,
		cacheCreated: false,
	}, nil
}
