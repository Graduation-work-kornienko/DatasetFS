package main

import (
	"fmt"
	"os"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/spf13/cobra"
)

// convertManifestCmd handles the manifest conversion command
func convertManifestCmd(cmd *cobra.Command, args []string) error {
	sourceDir, _ := cmd.Flags().GetString("source")

	if sourceDir == "" {
		return fmt.Errorf("source directory is required")
	}

	// Check if source directory exists
	if _, err := os.Stat(sourceDir); os.IsNotExist(err) {
		return fmt.Errorf("source directory does not exist: %s", sourceDir)
	}

	// Convert JSON manifest to Parquet
	if err := index.ConvertJSONManifestToParquet(sourceDir); err != nil {
		return fmt.Errorf("failed to convert manifest: %w", err)
	}

	fmt.Printf("Successfully converted manifest from JSON to Parquet format in %s\n", sourceDir)
	return nil
}
