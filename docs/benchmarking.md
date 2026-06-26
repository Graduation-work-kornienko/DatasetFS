# Бенчмарки DatasetFS

Каталог бенчмарков: [benchmarks/datasetfs_bench/](../benchmarks/datasetfs_bench/). Makefile содержит основные reproducible targets.

Цель бенчмарков — не «любой ценой обогнать WebDataset», а доказать свойства DatasetFS как системы:

- конкурентная скорость чтения и обучения;
- воспроизводимые измерения на одинаковых transforms/model/training loop;
- объяснимость bottleneck'ов через training, system и daemon метрики;
- уникальные сценарии: server-side decode, remote storage, online mutations, vacuum, разные модальности.

## Методология

Каждый benchmark cell обычно соответствует `loader × seed × axis value`. Runner пишет:

- `config.yaml` — копия конфигурации;
- `host_info.json` — fingerprint машины;
- `summary.csv` — per-epoch/cell итоговые строки;
- timeseries CSV для daemon/system samplers, если включены;
- plots и `REPORT.md`, если есть reporting step.

Основное правило после фикса measurement artifact 2026-06-03: timing включает `iter(dl)`, а headline-графики используют steady-state throughput (`steady_samples_per_second`) с отброшенными warmup batches, если колонка доступна. Это важно на macOS: `spawn` DataLoader workers раньше несправедливо прятался для map-style loaders и попадал в таймер для `IterableDataset` loaders.

## Форматы и loaders

| Loader | Реализация | Для чего нужен |
|---|---|---|
| `datasetfs` | Go daemon + Python client | Тестируемая система |
| `datasetfs-rgb` | тот же DatasetFS, но `dfs_decode_mode=rgb_uint8` | Уникальный режим server-side decode |
| `imagefolder` | `torchvision.datasets.ImageFolder` | Базовый plain-files baseline |
| `webdataset` | `webdataset.WebDataset` | Shard-based baseline и прямой архитектурный конкурент |
| `lmdb` | LMDB key-value store | Формат-матрица G1 |
| `hdf5` | HDF5 | Формат-матрица G1 |
| `tfrecord` | TFRecord reader без полного TensorFlow | Формат-матрица G1 |
| `huggingface` | HuggingFace/Arrow/Parquet dataset | Современный dataset ecosystem baseline |
| `ffcv` | FFCV loader | Linux-only performance baseline |
| `synthetic` | in-memory/generated samples | Compute ceiling, без storage bottleneck |

Все image loaders используют общий transform из `benchmarks/datasetfs_bench/loaders/_common.py`, чтобы сравнение было про storage/loading path, а не про разные transforms.

## Метрики

### Training-side

Источник: `benchmarks/datasetfs_bench/metrics/training.py` и `train/loop.py`.

| Колонка | Смысл |
|---|---|
| `samples_per_second` | Полный throughput эпохи |
| `steady_samples_per_second` | Headline throughput без warmup batches; предпочтительная метрика для графиков |
| `time_to_first_batch` | TTFB, включая DataLoader iterator startup |
| `stall_fraction` | Доля времени ожидания batch относительно compute/fetch |
| `fetch_p50/p95/p99` | Латентность `next(loader)` |
| `compute_p50` | Латентность model forward/backward |
| `n_samples`, `n_batches`, `wall_seconds` | Контекст размера прогона |

### System-side

Источник: `benchmarks/datasetfs_bench/metrics/system.py`, sampler interval обычно 0.2 s.

| Колонка | Смысл |
|---|---|
| `sys_cpu_pct_mean/p95` | System CPU utilization |
| `sys_mem_*` | RAM usage на машине |
| `sys_disk_read_bytes`, `sys_disk_write_bytes` | Disk I/O за cell |
| `sys_tracked_rss_*` | RSS process tree: Python + daemon PID для DatasetFS |
| process-labeled RSS columns | Раздельная динамика Python/daemon, где включено |

### Daemon-side

Источник: `internal/metrics` + `benchmarks/datasetfs_bench/metrics/daemon.py`.

| Метрика | Смысл |
|---|---|
| `shard_loads_total`, `bytes_read_total` | Сколько daemon реально прочитал |
| `dealer_batches_sent_total`, `samples_emitted_total` | Data plane output |
| `slot_starvation_total`, `refcount_overflow_total` | Health-сигналы allocator/refcount lifecycle |
| `epochs_completed_total`, `active_pipelines` | Sanity состояния сессии |
| `load_latency_*` | Латентность загрузки shard/slot |
| `frame_encode_latency_*`, `pipe_write_latency_*`, `dealer_emit_latency_*` | Transport path после opt 03 |
| `metrics/pipeline` | Дополнительный pipeline endpoint, если нужен finer breakdown |

### Feature-specific

| Сценарий | Дополнительные метрики |
|---|---|
| Mutation G3/G13 | mutation attempted/succeeded/failed, mutation latency, consistency violations, generation stability |
| Vacuum G4 | fragmentation ratio, bytes reclaimed, latency/stall during maintenance |
| Remote G9/G14 | remote bytes, cache behavior, cold-start TTFB, throttle/cap context |
| Pipeline memory | RSS plateau, cycles, replacement count |

## Основные Makefile targets

### Подготовка данных

| Target | Что делает |
|---|---|
| `make data` | Подготовить все стандартные датасеты/форматы |
| `make data-imagenette` | Imagenette в imagefolder/webdataset/datasetfs |
| `make data-imagewoof` | Imagewoof в стандартных форматах |
| `make data-audio` | Speech Commands V2 для audio сценариев |
| `make data-formats-extra` | LMDB/HDF5/TFRecord/HuggingFace/FFCV для image matrix |
| `make data-speech-commands-replicated` | 10× replicated Speech Commands для small-files scaling |

### Быстрые и базовые бенчмарки

| Target | Назначение |
|---|---|
| `make bench-smoke` | Sanity-check runner'а и reporting |
| `make bench-mvp` | Исторический MVP headline: ImageFolder/WebDataset/DatasetFS |
| `make bench-sweep-workers` | Throughput vs `num_workers` |
| `make bench-sweep-batch` | Throughput vs `batch_size` |
| `make bench-decode-compare` | raw vs `rgb_uint8` в compute-bound режиме |
| `make bench-decode-compare-simplecnn` | raw vs `rgb_uint8` в loader-bound режиме |
| `make bench-decode-parallelism` | Sweep по `dfs_decode_parallelism` |

### Формат-матрица и модальности

| Target | Назначение |
|---|---|
| `make bench-format-images` | G1: image formats на Imagenette + Imagewoof |
| `make bench-format-images-smoke` | Быстрая single-dataset версия |
| `make bench-format-audio` | G1/G8 на Speech Commands |
| `make bench-format-publaynet` | >RAM/local matrix на PubLayNet subset |
| `make bench-publaynet-sequence` | Disk-safe PubLayNet sequence с archive/delete шагами |
| `make bench-small-files-scaling` | Scaling по числу объектов |
| `make bench-real-universal` | Real-dataset universality matrix; пишет `missing.csv` |
| `make bench-real-universal-strict` | То же, но падает, если не все датасеты подготовлены |

### Remote, mutations, vacuum, stability

| Target | Назначение |
|---|---|
| `make bench-remote-preflight` | Проверка staged remote subset и config перед запуском |
| `make bench-remote-streaming` | G9/G14: DatasetFS HTTP remote vs WebDataset HTTP |
| `make bench-mutation` | G3/G13 synthetic concurrent FUSE mutations + epoch drain |
| `make bench-mutation-smoke` | Быстрая проверка mutation benchmark |
| `make bench-mutation-imagewoof` | Imagewoof endurance: training runs + deletes |
| `make bench-mutation-imagewoof-smoke` | Быстрая Imagewoof mutation проверка |
| `make bench-mutation-format-compare` | DatasetFS FUSE/WAL vs ImageFolder vs WebDataset rewrite cost |
| `make bench-mutation-format-compare-smoke` | Быстрый format mutation compare |
| `make bench-pipeline-memory` | RSS plateau under repeated sessions and replacements |
| `make bench-pipeline-memory-smoke` | Быстрая RSS plateau проверка |
| `make bench-vacuum-compaction` | Mutation + background vacuum matrix |
| `make bench-vacuum-compaction-smoke` | Быстрый vacuum smoke |

## Конфиги

Конфиги лежат в [benchmarks/datasetfs_bench/configs/](../benchmarks/datasetfs_bench/configs/). Важные группы:

- `smoke.yaml`, `mvp.yaml` — базовые single-run сценарии.
- `workers_sweep.yaml`, `batch_sweep.yaml`, `sweep_smoke.yaml` — классические sweeps.
- `decode_compare*.yaml`, `decode_parallelism_sweep.yaml` — opt 01/02.
- `format_matrix_images*.yaml`, `format_matrix_audio.yaml`, `format_matrix_publaynet.yaml` — G1/G8/G10.
- `remote_streaming.yaml`, `remote_cache_fair*.yaml` — remote/cache сценарии.
- `real_universal_datasets.yaml` — universality matrix.

Runner поддерживает несколько entries одного format с разными именами, например `datasetfs` и `datasetfs-rgb`, через dict-записи в `loaders`.

## Текущие выводы

### Fair format matrix

После переноса `iter(dl)` внутрь timed region и перехода графиков на steady-state метрику выяснилось, что ранний вывод «DatasetFS самый медленный» был артефактом измерения. Для map-style loaders worker spawn скрывался вне таймера, а для IterableDataset попадал в первый `next()`.

Текущий qualitative итог из `HANDOFF.md`:

- `datasetfs-rgb` лидирует на image SimpleCNN steady-state сценарии;
- `datasetfs` raw находится в середине matrix, а не внизу;
- audio путь показывает, что когда decode дешевый, transport становится заметнее, но DatasetFS остается в конкурентном диапазоне;
- FFCV полноценно сравнивается только на Linux.

### Server-side decode

Opt 01 показала, что перенос JPEG decode в daemon архитектурно корректен, но pure Go backend был медленным. libjpeg-turbo закрыл ResNet-18 gap до почти паритета. Opt 02 добавила parallel decode и показала, что в loader-bound микросценарии `rgb_uint8` резко обгоняет raw PIL; в end-to-end SimpleCNN плато наступает уже при малом K, потому что bottleneck уходит из decode.

### Transport

Opt 03 заменила JSON-over-pipe на binary frame, добавила zero-copy SHM view и batched refcount. Чистый transport ceiling вырос примерно 1.4×, realistic audio path около +10%. На тяжелых image/model сценариях transport скрыт decode/compute.

### Remote storage

Remote path сейчас измеряется как HTTP staged subset с cache/prefetch для DatasetFS и WebDataset HTTP baseline. Это не полноценный S3 SDK backend; цель бенча — cold-start/prefetch/cache behavior и сравнение remote streaming при контролируемом размере subset/throttle.

### Mutations and vacuum

Mutation benchmarks используют реальный FUSE path: writes/deletes параллельно draining/training. Главные sanity outputs — consistency violations должны быть 0, mutation latency bounded, throughput degradation объясним. Vacuum benchmarks проверяют, что background compaction удерживает fragmentation без разрушения loading path.

## Reporting

| Скрипт | Что строит |
|---|---|
| `reporting/plots.py` | Базовые throughput/latency графики |
| `reporting/sweep_plots.py` | Line plots по sweep axis |
| `reporting/format_matrix_plots.py` | Format matrix bar charts |
| `reporting/wait_compute_plots.py` | Fetch/compute/stall breakdown |
| `reporting/daemon_timeseries_plots.py` | Daemon metrics over time |
| `reporting/system_timeseries_plots.py` | CPU/RSS/disk timeseries |
| `reporting/remote_plots.py` | Remote-specific plots |
| `reporting/mutation_plots.py` | Mutation and vacuum scenario plots |
| `reporting/pipeline_memory_plots.py` | RSS plateau plots |
| `reporting/real_universal_plots.py` | Universality matrix plots |
| `reporting/report.py` | Markdown report aggregation |

## Практические правила запуска

- Перед thesis-grade runs сначала запускать соответствующий `*-smoke` target.
- Для cold-cache сравнения нужен passwordless `sudo purge` на macOS или `drop_caches` на Linux; без него runner помечает cache state как uncontrolled.
- Для mutation/FUSE targets нужен macFUSE и рабочий mount/unmount.
- Для remote targets сначала запускать `bench-remote-preflight`.
- Для FFCV нужен Linux и `requirements-linux.txt`.
- Для внешних больших датасетов не полагаться на auto-download внутри benchmark target: сначала подготовить данные отдельным data/prep target или script.

## Что еще стоит измерить

- Linux/GPU numbers: текущие macOS/CPU результаты не закрывают production inference/training сценарии.
- Long-format Parquet для всех timeseries, чтобы reporting не зависел от shape конкретного CSV.
- Более явный cache-hit/cache-miss учет для remote prefetch.
- Отдельные microbenchmarks для WAL binary/json и manifest parquet/json historical comparison.
- End-to-end loss/accuracy curves для крупных benchmarks, если нужно доказательство ML-equivalence в дополнение к correctness tests.
