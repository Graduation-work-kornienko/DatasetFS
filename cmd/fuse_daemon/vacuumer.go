package main

import (
	"context"
	"log"
	"time"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/ipc"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/manager"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/vacuum"
)

// autoVacuumConfig configures the in-daemon background vacuumer. It replaces the
// former standalone cmd/background_vacuum program: running it inside the daemon
// lets it coordinate with the loading pipeline (via ipc maintenance) and with
// FUSE mutations (via MutationManager.WithExclusive) instead of shelling out.
type autoVacuumConfig struct {
	root      string
	walFormat string
	interval  time.Duration
	threshold float64
	throttle  int64
}

// runAutoVacuum periodically checks fragmentation and, when it crosses the
// threshold AND no loading session is active, compacts the dataset in place and
// reloads the in-memory index. It is gated off by default (--auto-vacuum) so it
// never perturbs benchmark runs.
func runAutoVacuum(ctx context.Context, cfg autoVacuumConfig, coreIdx *index.CoreIndex, mutMgr *manager.MutationManager) {
	log.Printf("[auto-vacuum] enabled: interval=%s threshold=%.0f%% throttle=%d B/s",
		cfg.interval, cfg.threshold*100, cfg.throttle)
	ticker := time.NewTicker(cfg.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m := index.NewManifest(cfg.root)
			if err := m.Load(nil); err != nil {
				log.Printf("[auto-vacuum] load manifest: %v", err)
				continue
			}
			frag := vacuum.Fragmentation(m)
			if frag < cfg.threshold {
				continue
			}

			// Reserve the dataset; skip this tick if a pipeline is running.
			if !ipc.BeginMaintenance() {
				log.Printf("[auto-vacuum] fragmentation %.1f%% but a session is active — skipping", frag*100)
				continue
			}
			log.Printf("[auto-vacuum] fragmentation %.1f%% ≥ %.0f%% — compacting", frag*100, cfg.threshold*100)

			err := mutMgr.WithExclusive(func() error {
				if _, e := vacuum.Run(vacuum.Options{
					Root:       cfg.root,
					WALFormat:  cfg.walFormat,
					Background: true,
					Throttle:   cfg.throttle,
				}); e != nil {
					return e
				}
				// Reflect the compacted dataset in the live index.
				nm := index.NewManifest(cfg.root)
				if e := nm.Load(nil); e != nil {
					return e
				}
				coreIdx.Reload(nm)
				// Restore the delta placeholder removed by the compaction; its
				// tar is recreated lazily on the next FUSE write.
				coreIdx.Mu.Lock()
				if _, ok := coreIdx.ShardMap[-1]; !ok {
					coreIdx.ShardMap[-1] = &index.Shard{
						Number: -1, Type: "delta", Objects: make([]*index.Metadata, 0),
					}
				}
				coreIdx.Mu.Unlock()
				return nil
			})

			ipc.EndMaintenance()
			if err != nil {
				log.Printf("[auto-vacuum] failed: %v", err)
			} else {
				log.Printf("[auto-vacuum] done")
			}
		}
	}
}
