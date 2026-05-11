package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"testing"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/manager"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
	"github.com/stretchr/testify/require"
)

func TestWebdataset(t *testing.T) {

	targetDir = "test"
	sourceDir = "."
	err := os.RemoveAll("test")
	require.NoError(t, err)

	err = os.MkdirAll(targetDir, 0755)
	require.NoError(t, err)

	coreIndex := index.NewIndex()
	manifest := index.NewManifest(targetDir)
	wal := &index.WAL{}
	storage := storage.New(targetDir)

	ctx := context.Background()
	mutationManager := manager.NewMutationManager(coreIndex, manifest, wal, storage)

	err = ParseWebDataset(ctx, mutationManager, sourceDir)
	require.NoError(t, err)

	mutationManager.Shutdown()

	fmt.Printf("Successfully converted dataset from %s to %s\n", sourceDir, targetDir)

	mnfst := index.NewManifest(targetDir)

	err = mnfst.Load()
	require.NoError(t, err)
	coreIdx, err := mnfst.LoadCoreIndex()
	// Запускаем твой новый модуль
	err = storage.ValidateDataset(coreIdx)
	if err != nil {
		log.Fatalf("Датасет испорчен: %v", err)
	}
}
