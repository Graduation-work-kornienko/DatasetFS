// Command datasetfs is the single entrypoint for DatasetFS. It bundles the
// three tools that used to be separate binaries as cobra subcommands:
//
//	datasetfs daemon      — mount FUSE + serve the IPC/SHM loading pipeline
//	datasetfs vacuum       — compact a dataset (drop tombstones, repack shards)
//	datasetfs converter ... — convert ImageFolder/WebDataset → DFS shards
//
// Because the daemon pulls in the cgo libjpeg-turbo decoder (internal/pipeline),
// the whole binary requires the cgo build env — or the datasetfs_purego tag.
package main

import (
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "datasetfs",
	Short: "DatasetFS — a FUSE+SHM dataset filesystem for ML training",
	Long:  "DatasetFS bundles the daemon, the vacuum compactor and the dataset converter into one binary.",
}

func init() {
	rootCmd.AddCommand(newDaemonCmd())
	rootCmd.AddCommand(newVacuumCmd())
	rootCmd.AddCommand(newConverterCmd())
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}
