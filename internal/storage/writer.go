package storage

import (
	"archive/tar"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
)

type WDSMetadataFile struct {
	Annotations json.RawMessage `json:"annotations"`
}

// tarPath - path of Webdataset Shard
// shardID - pointer to int, refer to global shared number of DatasetFS shard
// mu - mutex to synchronize shardID between goroutines
// metaChan - channel, receive index.Metadata to CoreIndex
func (s *Storage) HandleWebdatasetShard(
	tarPath string,
	shardID *int,
	mu *sync.Mutex,
	shardChan chan<- *index.Shard,
) error {

	var currentFile *os.File
	var currentTw *tar.Writer
	var currentSize int64
	var currentShardId int
	var currentTargetOffset int64 = 0

	currentObjects := make([]*index.Metadata, 0)
	metaKeeper := make(map[string]*index.Metadata, 5000)
	mu.Lock()
	currentShardId = *shardID
	*shardID++
	mu.Unlock()

	sourceFile, err := os.Open(tarPath)
	if err != nil {
		return fmt.Errorf("failed to open source tar: %w", err)
	}
	defer sourceFile.Close()

	tarReader := tar.NewReader(sourceFile)

	if err := s.createWriter(&currentFile, &currentTw, currentShardId); err != nil {
		return err
	}

	cnt := 0

	for {
		cnt++
		header, err := tarReader.Next()
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			return fmt.Errorf("error reading tar header for file %s: %w", tarPath, err)
		}

		if header.Typeflag == tar.TypeDir {
			continue
		}

		filename := strings.TrimSuffix(filepath.Base(header.Name), filepath.Ext(header.Name))
		var meta *index.Metadata
		if metaKeeper[filename] == nil {
			meta = &index.Metadata{}
			metaKeeper[filename] = meta
		} else {
			meta = metaKeeper[filename]
		}

		if filepath.Ext(header.Name) == ".json" {
			var wdsData WDSMetadataFile
			jsonBytes, err := io.ReadAll(tarReader)
			if err != nil {
				return err
			}

			if err := json.Unmarshal(jsonBytes, &wdsData); err != nil {
				return err
			}
			meta.ObjectMetadata = wdsData.Annotations
			continue
		}

		// fmt.Println(meta)
		if cnt%100 == 0 {
			fmt.Println(cnt, currentSize)
		}
		// fmt.Println(currentSize)

		if currentSize > index.ShardSize {
			fmt.Println("change")
			if currentTw != nil {
				currentTw.Close()
				currentFile.Close()
			}

			// Save previous shard
			shard := index.Shard{
				Number:    currentShardId,
				Type:      index.Base,
				TotalSize: currentSize,
				Objects:   currentObjects,
			}
			shardChan <- &shard
			currentObjects = make([]*index.Metadata, 0)

			mu.Lock()
			currentShardId = *shardID
			*shardID++
			mu.Unlock()

			if err := s.createWriter(&currentFile, &currentTw, currentShardId); err != nil {
				return err
			}

			currentSize = 0
			currentTargetOffset = 0
		}

		cleanFileName := filepath.Base(header.Name)

		cleanHdr := &tar.Header{
			Name:   cleanFileName,
			Mode:   0600,
			Size:   header.Size,
			Format: tar.FormatGNU,
		}

		meta.ShardID = currentShardId
		meta.Path = cleanFileName
		meta.Offset = currentTargetOffset + 512
		meta.Size = header.Size
		currentObjects = append(currentObjects, meta)

		if err := currentTw.WriteHeader(cleanHdr); err != nil {
			return fmt.Errorf("failed to write tar header for %s: %w", header.Name, err)
		}

		written, err := io.Copy(currentTw, tarReader)
		if err != nil {
			return fmt.Errorf("failed to pipe data for %s: %w", header.Name, err)
		}

		if written != header.Size {
			return fmt.Errorf("size mismatch: expected %d, wrote %d", header.Size, written)
		}

		delete(metaKeeper, filename)

		var padding int64 = 0
		remainder := written % 512
		if remainder != 0 {
			padding = 512 - remainder
		}

		currentSize += header.Size
		currentTargetOffset += written + padding + 512
	}

	fmt.Println(cnt)

	if currentTw != nil {
		currentTw.Close()
		currentFile.Close()
	}

	if currentSize > 0 {
		// Save previous shard
		shard := index.Shard{
			Number:    currentShardId,
			Type:      index.Base,
			TotalSize: currentSize,
			Objects:   currentObjects,
		}
		shardChan <- &shard
	}

	return nil
}

func (s *Storage) createWriter(file **os.File, tw **tar.Writer, shardID int) error {
	fmt.Println(shardID)
	filename := s.ShardPath(shardID)
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create shard %s: %w", filename, err)
	}
	*file = f
	*tw = tar.NewWriter(f)
	return nil
}
