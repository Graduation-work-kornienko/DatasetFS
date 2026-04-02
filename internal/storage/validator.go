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
	fmt.Println("🚀 Внимание: Начинаем глубокую валидацию датасета...")

	shardMap := coreIdx.AllShards()

	var totalFilesChecked int
	var errorsFound []string

	// 2. Идем по каждому физическому архиву (Чанку/Дельте)
	for shardID, shardInfo := range shardMap {
		fmt.Printf("🔍 Проверка чанка: %d (%s)\n", shardID, shardInfo.Type)

		err := s.validateSingleChunk(shardID, shardInfo)
		if err != nil {
			errStr := fmt.Sprintf("❌ Ошибка в чанке %d: %v", shardID, err)
			fmt.Println(errStr)
			errorsFound = append(errorsFound, errStr)
			continue
		}

		totalFilesChecked += len(shardInfo.Objects)
	}

	if len(errorsFound) > 0 {
		return fmt.Errorf("валидация провалена. Найдено %d ошибок в чанках", len(errorsFound))
	}

	fmt.Printf("✅ Успех! Датасет консистентен. Проверено файлов: %d\n", totalFilesChecked)
	return nil
}

// validateSingleChunk открывает один физический .tar файл и проверяет все оффсеты
func (s *Storage) validateSingleChunk(shardID int, shardInfo *index.Shard) error {
	shardPath := s.ShardPath(shardID)

	// 1. Открываем физический файл на диске
	file, err := os.Open(shardPath)
	if err != nil {
		return fmt.Errorf("не удалось открыть файл %s: %w", shardPath, err)
	}
	defer file.Close()

	// 2. Копируем слайс метаданных, чтобы отсортировать его по Offset
	// Это критично для скорости NVMe диска (Sequential Seek)
	objects := make([]*index.Metadata, len(shardInfo.Objects))
	copy(objects, shardInfo.Objects)
	sort.Slice(objects, func(i, j int) bool {
		return objects[i].Offset < objects[j].Offset
	})

	// 3. Выделяем буфер ровно на 512 байт (Размер заголовка TAR)
	// Мы переиспользуем эту память миллион раз, не нагружая Garbage Collector!
	headerBuf := make([]byte, 512)

	// 4. Прыгаем по оффсетам и читаем заголовки
	for _, meta := range objects {
		// В Индексе лежит смещение на СЫРЫЕ ПИКСЕЛИ.
		// Значит, заголовок TAR начинается ровно на 512 байт раньше.
		expectedHeaderOffset := meta.Offset - 512

		if expectedHeaderOffset < 0 {
			return fmt.Errorf("критическая ошибка: отрицательное смещение заголовка для файла %s", meta.Path)
		}

		// Прыжок головки диска (или указателя NVMe)
		_, err := file.Seek(expectedHeaderOffset, os.SEEK_SET)
		if err != nil {
			return fmt.Errorf("ошибка Seek (%s): %w", meta.Path, err)
		}

		// Читаем физические 512 байт с диска
		_, err = io.ReadFull(file, headerBuf)
		if err != nil {
			return fmt.Errorf("не удалось прочитать заголовок (%s): %w", meta.Path, err)
		}

		// 5. Передаем сырые байты стандартному парсеру Go
		tarReader := tar.NewReader(bytes.NewReader(headerBuf))

		// Читаем распарсенный заголовок
		hdr, err := tarReader.Next()
		if err != nil {
			return fmt.Errorf("битый TAR заголовок по смещению %d (%s): %w", expectedHeaderOffset, meta.Path, err)
		}

		// 6. МАГИЯ ВАЛИДАЦИИ: Сверяем Физику с Логикой (Индексом)
		expectedName := filepath.Base(meta.Path) // Мы сохраняли только имена в TAR'ах (без полного пути)

		if hdr.Name != expectedName {
			return fmt.Errorf("несовпадение имен! Индекс: '%s', Физически в TAR лежит: '%s'", expectedName, hdr.Name)
		}

		if hdr.Size != int64(meta.Size) {
			return fmt.Errorf("коррупция размера! Индекс: %d, TAR Заголовок: %d", meta.Size, hdr.Size)
		}
	}

	return nil
}
