# DatasetFS Manifest Format

DatasetFS uses a single Parquet manifest file named `metadata.parquet`.

## Current State

- `metadata.parquet` is the only supported runtime manifest.
- New Python preparation code writes Parquet directly.
- The Go daemon, vacuum command, remote prefetch path, and benchmark preflight all require Parquet.
- Legacy JSON manifest fallback and the JSON-to-Parquet conversion CLI were removed to avoid dual source-of-truth drift.

## Layout

The manifest is a one-row Parquet file with three top-level fields:

- `version`: manifest format version.
- `shards_meta`: list of shard descriptors: `number`, `type`, `total_size`.
- `files`: list of object descriptors: `path`, `shard_id`, `offset`, `size`, `deleted`, `object_metadata`.

`object_metadata` stores per-object metadata as JSON bytes inside the Parquet row. The payload bytes remain in DatasetFS tar shards.

## Remote Storage

Remote DatasetFS roots must expose:

- `metadata.parquet`
- `shard_<id>` files referenced by the manifest

Remote preflight downloads and validates `metadata.parquet`, probes referenced shards, and fails if a legacy `metadata.jsonl` endpoint is present.

## WAL

The optimized WAL format is binary. Read-only benchmarks start the daemon with `--no-wal`; mutation benchmarks use `--wal-format binary`.
