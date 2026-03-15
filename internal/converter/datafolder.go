package converter

import (
	"fmt"
	"os"
)

// ParseDatasetFolder parses dataset in format https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.DatasetFolder.html#torchvision.datasets.DatasetFolder
// and returns index manifest structure
func ParseDatasetFolder(root string) (any, error) {
	entities, err := os.ReadDir(root)
	if err != nil {
		return nil, fmt.Errorf("converter.ParseDatasetFolder: %w", err)
	}

	for _, e := range entities {
		if e.IsDir() {

		}
	}
	return nil, nil
}
