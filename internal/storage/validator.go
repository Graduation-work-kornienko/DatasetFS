package storage

import (
	"archive/tar"
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
)

func (s *Storage) ValidateDataset(coreIdx *index.CoreIndex) error {

	shardMap := coreIdx.AllShards()

	var totalFilesChecked int
	var errorsFound []string

	for shardID, shardInfo := range shardMap {
		fmt.Printf("Проверка чанка: %d (%s)\n", shardID, shardInfo.Type)

		err := s.validateSingleChunk(shardID, shardInfo)
		if err != nil {
			errStr := fmt.Sprintf("Ошибка в чанке %d: %v", shardID, err)
			fmt.Println(errStr)
			errorsFound = append(errorsFound, errStr)
			continue
		}

		totalFilesChecked += len(shardInfo.Objects)
	}

	if len(errorsFound) > 0 {
		return fmt.Errorf("валидация провалена. Найдено %d ошибок в чанках", len(errorsFound))
	}

	fmt.Printf("Успех! Датасет консистентен. Проверено файлов: %d\n", totalFilesChecked)
	return nil
}

// validateSingleChunk открывает один физический .tar файл и проверяет все оффсеты
func (s *Storage) validateSingleChunk(shardID int, shardInfo *index.Shard) error {
	shardPath := s.ShardPath(shardID)

	file, err := os.Open(shardPath)
	if err != nil {
		return fmt.Errorf("не удалось открыть файл %s: %w", shardPath, err)
	}
	defer file.Close()

	// Копируем слайс метаданных, чтобы отсортировать его по Offset
	// Это критично для скорости NVMe диска (Sequential Seek)
	objects := make([]*index.Metadata, len(shardInfo.Objects))
	copy(objects, shardInfo.Objects)
	sort.Slice(objects, func(i, j int) bool {
		return objects[i].Offset < objects[j].Offset
	})

	// Размер заголовка TAR
	headerBuf := make([]byte, 512)

	// 4. Прыгаем по оффсетам и читаем заголовки
	for _, meta := range objects {
		// В Индексе лежит смещение на сырые данные.
		// Значит, заголовок TAR начинается ровно на 512 байт раньше.
		expectedHeaderOffset := meta.Offset - 512

		if expectedHeaderOffset < 0 {
			return fmt.Errorf("критическая ошибка: отрицательное смещение заголовка для файла %s", meta.Path)
		}

		_, err := file.Seek(expectedHeaderOffset, os.SEEK_SET)
		if err != nil {
			return fmt.Errorf("ошибка Seek (%s): %w", meta.Path, err)
		}

		_, err = io.ReadFull(file, headerBuf)
		if err != nil {
			return fmt.Errorf("не удалось прочитать заголовок (%s): %w", meta.Path, err)
		}

		tarReader := tar.NewReader(bytes.NewReader(headerBuf))

		hdr, err := tarReader.Next()
		if err != nil {
			return fmt.Errorf("битый TAR заголовок по смещению %d (%s): %w", expectedHeaderOffset, meta.Path, err)
		}

		expectedName := filepath.Base(meta.Path)

		if hdr.Name != expectedName {
			return fmt.Errorf("несовпадение имен! Индекс: '%s', Физически в TAR лежит: '%s'", expectedName, hdr.Name)
		}

		if hdr.Size != int64(meta.Size) {
			return fmt.Errorf("коррупция размера! Индекс: %d, TAR Заголовок: %d", meta.Size, hdr.Size)
		}
	}

	return nil
}
