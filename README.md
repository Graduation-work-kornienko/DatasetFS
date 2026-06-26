# DatasetFS

DatasetFS — экспериментальная файловая система и runtime для загрузки датасетов в ML-обучении. Проект сделан как дипломная работа: цель не только «быстро прочитать картинки», а показать формат и инфраструктуру, которые остаются удобными для больших датасетов, разных типов данных, удаленного хранения, мутаций и воспроизводимых бенчмарков.

Коротко: Go-демон читает DatasetFS-датасет из tar-шардов, управляет индексом и пайплайном загрузки, кладет данные в shared memory, а Python-клиент отдает их в `torch.utils.data.DataLoader` через `IterableDataset`.

## Что умеет проект

- Единый бинарник `datasetfs` с подкомандами `daemon`, `converter`, `vacuum`.
- Собственный формат хранения: tar-шарды + `metadata.parquet` с индексом объектов.
- Python-клиент `clients/python/dataset_fs.py` для PyTorch `DataLoader`.
- Multi-worker loading: демон создает отдельный пайплайн и FIFO для каждого worker'а.
- IPC без копирования больших структур через `/tmp/mlfs_data.bin`, `/tmp/mlfs_refs.bin` и named pipes.
- Опциональный server-side JPEG decode: режим `rgb_uint8` декодирует и resize'ит изображения на стороне Go-демона.
- Поддержка не только изображений: аудио и другие byte payload'ы проходят через пользовательский `decode_fn` и `transform` в Python.
- WAL, snapshot pinning, vacuum/compaction и HTTP remote storage как инфраструктура для online/remote сценариев.
- Бенчмарк-харнесс для сравнения с ImageFolder, WebDataset, LMDB, HDF5, TFRecord, HuggingFace и FFCV на Linux.

## Быстрый старт

### 1. Установить зависимости

Go:

```bash
go version
```

Проектный модуль объявляет `go 1.25.5`. Для стандартной сборки нужен cgo и `libjpeg-turbo`, потому что daemon использует быстрый JPEG backend.

macOS:

```bash
brew install jpeg-turbo
```

Python:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r tests/requirements.txt
```

Linux-only extras для FFCV находятся в `requirements-linux.txt`.

### 2. Собрать бинарник

```bash
make build
```

Если нужен pure-Go вариант без `libjpeg-turbo`:

```bash
make build-purego
```

### 3. Подготовить данные

Для стандартных локальных форматов:

```bash
make data-imagenette
```

Это подготовит структуры вида:

```text
data/formats/<dataset>/imagefolder/
data/formats/<dataset>/webdataset/
data/formats/<dataset>/datasetfs/
```

Ручная конвертация DatasetFolder/ImageFolder-подобной структуры:

```bash
bin/datasetfs converter dataset-folder \
  --source data/formats/imagenette/imagefolder \
  --target data/formats/imagenette/datasetfs
```

### 4. Запустить daemon

Для обучения и бенчмарков обычно используется режим без FUSE mount:

```bash
bin/datasetfs daemon --no-mount --root data/formats/imagenette/datasetfs
```

Демон поднимает HTTP control plane на `http://localhost:51409`:

- `GET /healthz` — проверка жизни.
- `POST /initialize_loading` — создание loading-сессии под нужное число workers.
- `GET /metrics` — JSON-метрики демона.
- `/debug/pprof/*` — Go pprof endpoints.

### 5. Читать из Python

Минимальный пример:

```python
from torch.utils.data import DataLoader
from clients.python.dataset_fs import DatasetFS

dataset = DatasetFS(num_workers=2, seed=42)
loader = DataLoader(dataset, batch_size=64, num_workers=2)

for batch in loader:
    ...
```

Для server-side decode изображений:

```python
dataset = DatasetFS(
    num_workers=2,
    seed=42,
    decode_mode="rgb_uint8",
    decode_image_size=224,
)
```

Для аудио или другого payload'а передаются свои `decode_fn` и `transform`.

## Архитектура

DatasetFS состоит из трех основных частей.

```text
┌──────────────────────────────────────────────────────────────┐
│ Go daemon                                                     │
│ cmd/datasetfs daemon                                          │
│                                                              │
│ HTTP control plane: /healthz, /initialize_loading, /metrics   │
│ Index: manifest/parquet, WAL, snapshots                       │
│ Storage: tar shards, local/remote cache                       │
│ Pipeline: Planner -> BackgroundLoader -> Decoder -> Dealer    │
│ SHM allocator: data slots + refcounts                         │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                │ shared memory + FIFO frames
                                ▼
┌──────────────────────────────────────────────────────────────┐
│ Python client                                                  │
│ clients/python/dataset_fs.py                                   │
│ DatasetFS(IterableDataset) -> PyTorch DataLoader               │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│ Benchmark harness                                              │
│ benchmarks/datasetfs_bench                                     │
│ unified loaders, training loop, metrics, reporting             │
└──────────────────────────────────────────────────────────────┘
```

Главный поток выполнения:

1. `datasetfs daemon --root <datasetfs-root>` загружает `metadata.parquet`, строит in-memory `CoreIndex` и открывает control plane.
2. Python-клиент создает `DatasetFS(...)` и отправляет `POST /initialize_loading` с числом workers, seed и настройками decode.
3. Демон пинит snapshot индекса, создает пайплайн на каждого worker'а и выделяет каждому worker'у диапазон SHM-слотов.
4. `Planner` выбирает шарды и порядок объектов, `BackgroundLoader` читает shard bytes в shared memory.
5. Если включен `decode_mode="rgb_uint8"`, `Decoder` на стороне Go декодирует JPEG в packed RGB uint8.
6. `Dealer` отправляет Python worker'у бинарный frame с `slot_id`, `offset`, `size`, `path` и metadata.
7. Python worker читает frame из FIFO, берет bytes из mmap, применяет decode/transform и уменьшает refcount слота.
8. Когда refcount слота становится 0, daemon может переиспользовать слот для следующего шарда.

Подробная версия с IPC-протоколом, режимами decode и подводными камнями: [docs/architecture.md](docs/architecture.md).

## Структура репозитория

| Путь | Назначение |
|---|---|
| `cmd/datasetfs/` | CLI и подкоманды `daemon`, `converter`, `vacuum` |
| `internal/index/` | Manifest, Parquet metadata, in-memory index, snapshots, WAL |
| `internal/storage/` | Чтение/запись tar-шардов, remote prefetch/cache |
| `internal/pipeline/` | Планирование, фоновые загрузчики, decode stage, отправка batches |
| `internal/shm/` | Shared-memory allocator и refcount slots |
| `internal/control/` | HTTP control plane и lifecycle loading-сессий |
| `internal/manager/` | Mutation manager и запись изменений |
| `internal/vacuum/` | Compaction/vacuum живых объектов |
| `internal/vfs/` | FUSE-ноды для POSIX mount path |
| `clients/python/` | Python `DatasetFS` для PyTorch |
| `benchmarks/datasetfs_bench/` | Бенчмарки, loaders, training loop, metrics, plots |
| `scripts/datasets/` | Подготовка датасетов и форматов |
| `tests/` | Python/e2e тесты |
| `docs/` | Архитектура, статус, бенчмарки, дизайн-доки и оптимизации |

## Формат данных

DatasetFS-root содержит:

```text
metadata.parquet
shard_0
shard_1
...
```

`metadata.parquet` хранит:

- список шардов и их физические размеры;
- список объектов;
- путь объекта;
- `shard_id`, `offset`, `size` внутри tar-шарда;
- object metadata, например label.

Такой формат сохраняет преимущества shard-based хранения, но оставляет отдельный индекс, который можно обновлять, снапшотить, компактировать и анализировать отдельно от payload bytes.

## Тесты

Быстрые Go unit tests:

```bash
make go-test
```

Базовый Python correctness набор:

```bash
make test
```

Широкий gate перед экспериментами:

```bash
make thesis-code-gate
```

Часть тестов требует внешних условий: macFUSE, Docker/MinIO, подготовленных датасетов или Linux-only зависимостей. Каталог покрытия и оговорки: [docs/testing.md](docs/testing.md).

## Бенчмарки

Бенчмарки живут в `benchmarks/datasetfs_bench/` и запускаются через Makefile.

Смок всего harness'а:

```bash
make bench-smoke
```

MVP headline-прогон:

```bash
make bench-mvp
```

Sweep по числу workers:

```bash
make bench-sweep-workers
```

Сравнение raw vs server-side decode:

```bash
make bench-decode-compare
make bench-decode-compare-simplecnn
```

Format matrix и сценарии audio/remote/mutation тоже описаны в Makefile. Методология, текущие метрики и интерпретация результатов: [docs/benchmarking.md](docs/benchmarking.md).

## Документация

Если нужно быстро войти в проект, читать в таком порядке:

1. [docs/architecture.md](docs/architecture.md) — подробная архитектура и IPC.
2. [docs/benchmarking.md](docs/benchmarking.md) — методология бенчмарков и текущие результаты.
3. [docs/testing.md](docs/testing.md) — тесты и известные gaps.
4. [docs/optimizations/](docs/optimizations/) — журнал оптимизаций: server-side decode, parallel decode, transport.

## Важные ограничения

- FUSE mount path существует, но большинство тестов и бенчмарков используют `--no-mount`; основной training path идет через SHM/FIFO.
- Дефолтная сборка требует cgo и `libjpeg-turbo`; для окружений без системных зависимостей есть `make build-purego`.
- Shared memory сейчас ограничена 9 слотами по 110 MB, поэтому `DatasetFS(num_workers=...)` не должен превышать 9 effective workers.
- На macOS Python multiprocessing часто использует `spawn`, поэтому функции, передаваемые в DataLoader workers, должны быть picklable.
- `main.py` в корне — старый ручной скрипт сравнения; для актуального запуска и измерений используйте `benchmarks/datasetfs_bench` и Makefile.

## Лицензия

См. [LICENSE](LICENSE).
