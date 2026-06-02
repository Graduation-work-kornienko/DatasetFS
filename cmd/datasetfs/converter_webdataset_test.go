package main

import (
	"context"
	"testing"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/manager"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"
	"github.com/stretchr/testify/require"
)

func TestWebdataset(t *testing.T) {
	// Exercise the parse → manifest → validate path against an empty source dir
	// (no .tar shards), so the test is self-contained and leaves no artifacts.
	// ParseWebDataset opens tar names relative to cwd, so a dedicated empty dir
	// keeps it hermetic regardless of where `go test` runs.
	sourceDir = t.TempDir()
	targetDir = t.TempDir()

	coreIndex := index.NewIndex()
	manifest := index.NewManifest(targetDir)
	strg := storage.New(targetDir, nil)

	ctx := context.Background()
	mutationManager := manager.NewMutationManager(coreIndex, manifest, nil, strg)

	require.NoError(t, ParseWebDataset(ctx, mutationManager, sourceDir))
	mutationManager.Shutdown()

	mnfst := index.NewManifest(targetDir)
	require.NoError(t, mnfst.Load(nil))
	coreIdx, err := mnfst.LoadCoreIndex()
	require.NoError(t, err)
	require.NoError(t, strg.ValidateDataset(coreIdx))
}
