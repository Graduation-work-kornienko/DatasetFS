package datasetconverter

import (
	"os"

	"github.com/spf13/cobra"
)

var (
	rootCmd = &cobra.Command{
		Use:   "datasetconverter",
		Short: "Tool to convert other datasets to DatasetFS format",
		Long:  "Tool to convert many files(or WebDataset) to DatasetFS format",
	}
)

func main() {
	err := rootCmd.Execute()
	if err != nil {
		os.Exit(1)
	}
}

func init() {

}
