package main

import (
	"context"
	"fmt"
	"os"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/index/converter"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/manager"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/storage"

	"github.com/spf13/cobra"
)

var (
	rootCmd = &cobra.Command{
		Use:   "datasetconverter",
		Short: "Tool to convert other datasets to DatasetFS format",
		Long:  "Tool to convert many files(or WebDataset) to DatasetFS format",
	}

	datasetFolder = &cobra.Command{
		Use:   "dataset-folder",
		Short: "Convert a DatasetFolder format to DatasetFS format",
		Long:  "Convert a DatasetFolder format to DatasetFS format",
		Args:  cobra.ExactArgs(2),
		RunE:  generateConvertCommand(converter.ParseDatasetFolder),
	}

	webDataset = &cobra.Command{
		Use:   "webdataset",
		Short: "Convert a Webdataset format to DatasetFS format",
		Args:  cobra.ExactArgs(2),
		RunE:  generateConvertCommand(converter.ParseWebDataset),
	}

	sourceDir string
	targetDir string
)

func init() {
	datasetFolder.Flags().StringVarP(&sourceDir, "source", "s", "", "Source directory containing the dataset to convert")
	datasetFolder.Flags().StringVarP(&targetDir, "target", "t", "", "Target directory for the DatasetFS output")
	datasetFolder.MarkFlagRequired("source")
	datasetFolder.MarkFlagRequired("target")
	rootCmd.AddCommand(datasetFolder)
	rootCmd.AddCommand(webDataset)
}

type parseFunc func(context.Context, *manager.MutationManager, string) error

func generateConvertCommand(f parseFunc) func(*cobra.Command, []string) error {
	return func(cmd *cobra.Command, args []string) error {
		if err := os.MkdirAll(targetDir, 0755); err != nil {
			return fmt.Errorf("failed to create target directory: %w", err)
		}

		coreIndex := index.NewIndex()
		manifest := index.NewManifest(targetDir)
		wal := &index.WAL{}
		storage := storage.New(targetDir)

		ctx := context.Background()
		mutationManager := manager.NewMutationManager(coreIndex, manifest, wal, storage)

		if err := f(ctx, mutationManager, sourceDir); err != nil {
			return fmt.Errorf("failed to parse dataset folder: %w", err)
		}

		mutationManager.Shutdown()

		fmt.Printf("Successfully converted dataset from %s to %s\n", sourceDir, targetDir)
		return nil
	}
}

func main() {
	err := rootCmd.Execute()
	if err != nil {
		os.Exit(1)
	}
}
