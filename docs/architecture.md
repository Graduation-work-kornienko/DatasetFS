# Архитектура DatasetFS

DatasetFS — runtime и формат хранения датасетов для ML-обучения. Система состоит из Go-демона, Python-клиента для PyTorch и benchmark harness. Основной training path не требует FUSE: данные идут через shared memory и named pipes; FUSE mount используется для POSIX-доступа и сценариев мутаций.

## Схема

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ Go daemon: cmd/datasetfs daemon                                             │
│                                                                            │
│ HTTP control plane :51409                                                   │
│ /healthz /initialize_loading /metrics /metrics/pipeline /debug/pprof/*      │
│                                                                            │
│ Index layer                                                                 │
│ metadata.parquet -> CoreIndex -> immutable Snapshot per loading session     │
│ WAL, generation counter, pinned generations for mutation/vacuum safety       │
│                                                                            │
│ Storage layer                                                               │
│ local tar shards, append-only delta shards, HTTP remote prefetch/cache       │
│                                                                            │
│ Per-worker pipeline                                                         │
│ Planner -> BackgroundLoader -> optional Decoder -> Dealer                   │
│                                                                            │
│ Shared memory allocator                                                     │
│ /tmp/mlfs_data.bin + /tmp/mlfs_refs.bin                                     │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │
                                  │ mmap + session FIFO binary frames
                                  ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ Python client: clients/python/dataset_fs.py                                 │
│ DatasetFS(IterableDataset): opens the session FIFO, mmaps SHM, parses        │
│ binary frames, decodes/transforms samples, decrements slot refcounts         │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │ batches
                                  ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ torch.utils.data.DataLoader + benchmark/training code                       │
└────────────────────────────────────────────────────────────────────────────┘
```

## Формат на диске

DatasetFS root содержит:

```text
metadata.parquet
shard_0
shard_1
...
shard_-1          # delta shard для мутаций, если они были
datasetfs.wal     # WAL, если включен
```

`metadata.parquet` хранит список объектов и шардов: `path`, `shard_id`, `offset`, `size`, `deleted`, `meta`. Runtime загружает этот manifest в `CoreIndex`, а pipeline читает только immutable `Snapshot`, закрепленный за loading-сессией.

Формат шардов — tar-like shard files. Для базовых датасетов converter пишет объекты в shard files и сохраняет offsets. Для мутаций `MutationManager` пишет новые payload'ы в delta shards, tombstone'ит удаленные записи и фиксирует изменения в WAL.

## Жизненный цикл training-сессии

1. Пользователь запускает daemon: `bin/datasetfs daemon --no-mount --root <datasetfs-root>`.
2. Daemon загружает `metadata.parquet`, строит `CoreIndex`, открывает WAL, поднимает HTTP control plane и, если не указан `--no-mount`, монтирует FUSE.
3. Python создает `DatasetFS(num_workers=N, seed=..., decode_mode=...)`.
4. Клиент отправляет `POST /initialize_loading` на `http://localhost:51409`.
5. Daemon останавливает предыдущую loading-сессию, переиспользует общий SHM allocator, пинит immutable snapshot текущего поколения и создает `N` pipeline'ов.
6. Ответ `/initialize_loading` содержит `session_id`, `pipe_template`, snapshot `generation`, подтвержденные decode/distributed параметры.
7. Каждый PyTorch worker открывает свой FIFO вида `/tmp/datasetfs_pipe_<session_id>_<worker_id>`.
8. `Planner` выбирает шарды и порядок объектов для worker'а, учитывая seed, worker id и DDP partition (`rank/world_size`, если задан).
9. `BackgroundLoader` читает shard bytes в выделенный SHM slot.
10. Если включен `decode_mode="rgb_uint8"`, `Decoder` декодирует JPEG и resize'ит изображения в packed RGB uint8 HWC внутри slot'а.
11. `Dealer` формирует binary frame с метаданными объектов и пишет его в FIFO.
12. Python читает frame, берет view на bytes из mmap, применяет `decode_fn`/`transform` или fast path `rgb_uint8`, отдает sample в DataLoader и батчем уменьшает refcount slot'а.
13. Когда refcount становится 0, `Planner.WatchRefCounts` возвращает slot в пул.

## Компоненты Go

| Пакет | Роль | Важные файлы |
|---|---|---|
| `cmd/datasetfs` | Единый CLI: `daemon`, `converter`, `vacuum` | `main.go`, `daemon.go`, `converter*.go`, `vacuum.go` |
| `internal/control` | HTTP control plane, lifecycle сессий, shared allocator reuse, maintenance gate | `server.go` |
| `internal/index` | Manifest, Parquet storage, CoreIndex, Snapshot/MVCC, WAL | `manifest.go`, `parquet_manifest.go`, `tree.go`, `snapshot.go`, `binary_wal.go` |
| `internal/storage` | Чтение/запись шардов, tar append, remote HTTP prefetch/cache | `reader.go`, `tar_append.go`, `prefetch.go`, `remote.go` |
| `internal/pipeline` | Data plane: planner, loader, decoder, dealer, binary wire protocol | `pipeline.go`, `planner.go`, `background_loader.go`, `decoder.go`, `dealer.go` |
| `internal/shm` | Shared-memory slots и atomic refcounts | `allocator.go` |
| `internal/manager` | Мутации, delta shards, WAL запись и shutdown checkpoint | `mutation_manager.go`, `txlog.go` |
| `internal/vacuum` | Compaction: перепаковка live objects и atomic swap manifest/shards | `vacuum.go`, `compact.go`, `cleanup.go` |
| `internal/vfs` | FUSE nodes для POSIX create/read/list/unlink | `server.go` |
| `internal/metrics` | Atomic counters, latency histograms, `/metrics` handlers | `metrics.go` |

## Python-клиент

Основной класс: `clients/python/dataset_fs.py::DatasetFS`.

Ключевые параметры:

- `num_workers` — effective worker count для daemon; максимум 9 из-за `shm.NumSlots`.
- `seed` — детерминированный shuffle across workers.
- `decode_fn` — bytes-like object -> decoded object; по умолчанию PIL image decode.
- `transform` — decoded object -> tensor/value.
- `decode_mode="raw"` — daemon отдает bytes исходного payload'а.
- `decode_mode="rgb_uint8"` + `decode_image_size` — daemon отдает уже декодированный RGB HWC uint8.
- `decode_parallelism` — число decode goroutines per pipeline; 0 означает auto.
- `rank/world_size` — DDP partition; предполагается один daemon на rank с отдельными SHM/FIFO/port settings на уровне запуска.

`DatasetFS` наследуется от `IterableDataset`, поэтому порядок и sharding контролируются daemon'ом. Для custom audio/text/video payload'ов нужно передать module-level `decode_fn` и `transform`, чтобы они были picklable при Python multiprocessing `spawn`.

## IPC и shared memory

### HTTP control plane

`POST /initialize_loading` принимает JSON:

```json
{
  "num_workers": 4,
  "seed": 42,
  "decode": {"mode": "rgb_uint8", "image_size": 224, "parallelism": 2},
  "distributed": {"rank": 0, "world_size": 1}
}
```

Ответ содержит подтвержденную конфигурацию:

```json
{
  "num_workers": 4,
  "session_id": 12,
  "pipe_template": "/tmp/datasetfs_pipe_12_{worker_id}",
  "generation": 7,
  "decode": {"mode": "rgb_uint8", "image_size": 224, "parallelism": 2},
  "distributed": {"rank": 0, "world_size": 1}
}
```

Повторный `/initialize_loading` — это re-init: daemon останавливает старые pipeline goroutines, не размэпливает общий allocator, сбрасывает refcounts и создает новую session-specific FIFO группу. Это закрывает класс багов, где старый iterator и новая эпоха делили один pipe.

### SHM files

- `/tmp/mlfs_data.bin` — data mmap. Сейчас 9 slots × 110 MB.
- `/tmp/mlfs_refs.bin` — int32 refcount на slot.
- Slot partitioning: `SlotRange(workerID, numWorkers)` раздает contiguous диапазоны; первые `9 % N` workers получают на один slot больше.

Python уменьшает refcount батчем после обработки frame. Go переиспользует slot только когда refcount стал 0.

### Binary FIFO frame

Начиная с opt 03 wire protocol не JSON. `Dealer` пишет length-prefixed binary frame, а Python парсит фиксированный columnar блок через `numpy.frombuffer`.

```text
magic u32 = 0x44465331  # "DFS1"
total_len u32           # bytes after magic+total_len
generation u64          # snapshot generation
item_count u32
blob_len u32

item_count × 28 bytes:
  slot_id i32
  offset  i64
  size    i64
  path_len u32
  meta_len u32

blobs:
  path bytes + meta JSON bytes for every item
```

Пустой frame (`item_count == 0`) означает конец эпохи. `generation` помогает тестам обнаружить torn reads при concurrent mutation.

## Decode modes

| Mode | Что лежит в SHM slot | Что делает Python | Когда использовать |
|---|---|---|---|
| `raw` | исходные shard bytes | `decode_fn` + `transform` | универсальный путь: изображения, аудио, текст, произвольные bytes |
| `rgb_uint8` | packed RGB uint8 HWC фиксированного размера | `np.frombuffer` -> reshape -> `transform` | изображения, когда нужно убрать PIL decode/resize из Python path |

`rgb_uint8` реализован как optional stage между `BackgroundLoader` и `Dealer`. JPEG backend по умолчанию использует libjpeg-turbo через cgo; pure-Go fallback собирается через `make build-purego`.

## Мутации, snapshots и vacuum

DatasetFS поддерживает POSIX-мутации через FUSE path и внутренний `MutationManager`.

- Каждая mutation bump'ает generation в `CoreIndex`.
- Loading session пинит immutable `Snapshot`; все workers одной эпохи видят одну generation.
- Delete помечает object tombstone'ом.
- Add/replace пишет payload в delta shard и добавляет запись в индекс.
- WAL фиксирует mutation operations и replay'ится при старте.
- Vacuum перепаковывает live objects, удаляет tombstones и rewrites manifest; background vacuum включается флагом `--auto-vacuum` и проходит через maintenance gate.

Это важно для online-learning сценария: текущая эпоха не должна видеть «рваное» состояние датасета, даже если параллельно происходят writes/deletes.

## Remote storage

Daemon может стартовать с HTTP root:

```bash
bin/datasetfs daemon --no-mount \
  --root http://localhost:8000/datasetfs \
  --cache-dir runs/remote_cache \
  --prefetch-concurrency 4
```

Текущая реализация: manifest скачивается в cache, shards догружаются background prefetcher'ом. Для бенчмарков remote path сравнивается с WebDataset HTTP при одинаковом throttle/staging subset.

## Benchmark harness

`benchmarks/datasetfs_bench` строит сопоставимые DataLoader'ы для разных форматов и пишет единые CSV/plots/reports. Он не является частью hot path DatasetFS, но задает воспроизводимую методологию измерений.

Основные части:

- `loaders/` — ImageFolder, WebDataset, DatasetFS, LMDB, HDF5, TFRecord, HuggingFace, FFCV, synthetic.
- `train/loop.py` — общий training loop и steady-state метрики.
- `metrics/` — training, system sampler, daemon sampler.
- `runner/` — single-run, sweep, real-universal, remote preflight, mutation, pipeline memory.
- `reporting/` — plots и Markdown reports.

## Ограничения и подводные камни

- `num_workers` ограничен 9 effective workers, потому что allocator имеет 9 slots.
- FUSE path нужен для POSIX и mutation сценариев; большинство training/benchmark запусков используют `--no-mount`.
- Дефолтная сборка требует cgo + libjpeg-turbo. На macOS Makefile ожидает Homebrew путь `/opt/homebrew/opt/jpeg-turbo/lib/pkgconfig`.
- `go test ./...` может быть менее надежным, чем `make go-test`, потому что Makefile выставляет нужный cgo env.
- На macOS DataLoader workers используют `spawn`; нельзя передавать lambdas/closures в `decode_fn`, `transform`, `collate_fn`.
- `main.py` в корне — устаревший ручной скрипт. Актуальные сценарии находятся в Makefile и `benchmarks/datasetfs_bench`.
- Design docs про vacuum/remote/manifest описывают замысел; фактическое состояние сверять с этим файлом, `docs/status.md`, Makefile и тестами.
