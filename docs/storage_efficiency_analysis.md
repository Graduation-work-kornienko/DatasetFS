# DatasetFS Storage Efficiency Analysis

DatasetFS stores payloads in tar shards and metadata in compact binary control files.

## Manifest

The current manifest is `metadata.parquet`.

- It stores shard metadata and object metadata in a typed Parquet schema.
- It avoids the large text overhead of the former JSON manifest.
- It is the only runtime manifest format used by the daemon, vacuum, remote prefetch, and benchmark tooling.

## WAL

The optimized WAL format is binary and is selected by default for mutable daemon runs.

- Read-only benchmarks use `--no-wal` to avoid measuring WAL overhead.
- Mutation benchmarks explicitly use `--wal-format binary`.
- Vacuum reads/replays the binary WAL before checkpointing a new Parquet manifest and truncating the WAL.

## Benchmark Implication

Benchmark results should not include JSON manifest or JSONL WAL overhead. DatasetFS measurements use Parquet manifests, and the only WAL-enabled benchmark uses the binary WAL path.
