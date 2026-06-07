package main

import (
	"context"
	"fmt"
	"os"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/manager"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"

	"github.com/spf13/cobra"

	// Import the parquet package to ensure it's included in the build.
	_ "github.com/parquet-go/parquet-go"
)

// sourceDir/targetDir back the converter flags. They are package-level so the
// webdataset test can drive ParseWebDataset directly without a cobra invocation.
var (
	sourceDir string
	targetDir string
)

// newConverterCmd builds the `converter` parent command grouping the format
// converters (dataset-folder, webdataset).
func newConverterCmd() *cobra.Command {
	converterCmd := &cobra.Command{
		Use:   "converter",
		Short: "Convert other dataset formats to DatasetFS format",
		Long:  "Convert many files (or a WebDataset) to DatasetFS format",
	}

	datasetFolder := &cobra.Command{
		Use:   "dataset-folder",
		Short: "Convert a DatasetFolder format to DatasetFS format",
		Long:  "Convert a DatasetFolder format to DatasetFS format",
		RunE:  generateConvertCommand(ParseDatasetFolder),
	}
	datasetFolder.Flags().StringVarP(&sourceDir, "source", "s", "", "Source directory containing the dataset to convert")
	datasetFolder.Flags().StringVarP(&targetDir, "target", "t", "", "Target directory for the DatasetFS output")
	datasetFolder.MarkFlagRequired("source")
	datasetFolder.MarkFlagRequired("target")

	webDataset := &cobra.Command{
		Use:   "webdataset",
		Short: "Convert a Webdataset format to DatasetFS format",
		Args:  cobra.ExactArgs(2),
		RunE:  generateConvertCommand(ParseWebDataset),
	}

	converterCmd.AddCommand(datasetFolder)
	converterCmd.AddCommand(webDataset)
	return converterCmd
}

type parseFunc func(context.Context, *manager.MutationManager, string) error

func generateConvertCommand(f parseFunc) func(*cobra.Command, []string) error {
	return func(cmd *cobra.Command, args []string) error {
		if err := os.MkdirAll(targetDir, 0755); err != nil {
			return fmt.Errorf("failed to create target directory: %w", err)
		}

		coreIndex := index.NewIndex()
		manifest := index.NewManifest(targetDir)
		storage := storage.New(targetDir, nil)

		ctx := context.Background()
		// nil WAL log, as it is not used in dataset converting
		mutationManager := manager.NewMutationManager(coreIndex, manifest, nil, storage)

		if err := f(ctx, mutationManager, sourceDir); err != nil {
			return fmt.Errorf("failed to parse dataset folder: %w", err)
		}

		mutationManager.Shutdown()

		fmt.Printf("Successfully converted dataset from %s to %s\n", sourceDir, targetDir)
		return nil
	}
}
