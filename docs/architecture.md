# Архитектура DatasetFS

DatasetFS — файловая система для обучения ML-моделей на больших датасетах.
Состоит из **трёх слоёв**, общающихся через named pipes + shared memory:

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Go daemon (cmd/datasetfs daemon)                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  HTTP control plane (internal/control): /healthz /initialize_loading│    │
│  │                                     /metrics /debug/pprof/*        │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Per-worker Pipeline (internal/pipeline)                           │    │
│  │  Planner → BackgroundLoader → [Decoder?] → DealerWorker            │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────┐ ┌──────────────────────────────────┐     │
│  │ SHM allocator (internal/shm) │ │ Storage (internal/storage)       │     │
│  │ 9 slots × 110 MB + refcounts │ │ tar-shards on disk               │     │
│  └──────────────────────────────┘ └──────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────────────┘
                          │ /tmp/mlfs_data.bin (mmap, SHARED)
                          │ /tmp/mlfs_refs.bin (mmap, atomic int32 per slot)
                          │ /tmp/datasetfs_pipe_<worker_id> (FIFO, JSON-per-line)
                          ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  Python client (clients/python/dataset_fs.py)                              │
│  - DatasetFS(IterableDataset): __iter__ читает pipe, mmap'ает SHM,         │
│    декодит (PIL или skip-if-rgb_uint8), вызывает transform                 │
│  - Один DatasetFS на main process, PyTorch DataLoader spawn'ит N workers,  │
│    каждый получает свой pipe_<worker_id>                                   │
└────────────────────────────────────────────────────────────────────────────┘
                          │ batches → torch.utils.data.DataLoader
                          ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  Benchmark harness (benchmarks/datasetfs_bench)                            │
│  Уравнивает измерения между DFS / WebDataset / ImageFolder                 │
└────────────────────────────────────────────────────────────────────────────┘
```

## Жизненный цикл

### Подготовка данных (одноразово на датасет)

`scripts/datasets/prepare_formats.py` качает датасет (fastai imagenette / imagewoof
/ Speech Commands V2) и собирает три формата:

- `data/formats/<ds>/imagefolder/` — symlink-фермa: `class/file.jpg`
- `data/formats/<ds>/webdataset/` — tar-шарды с `<key>.jpg + <key>.cls`
- `data/formats/<ds>/datasetfs/` — собственный формат: tar-шарды + `metadata.parquet`

Имеется конвертер `datasetfs converter` (Go, подкоманда единого бинарника),
которого `prepare_formats` дёргает для построения DFS-формата из imagefolder-структуры.

### Обучение

1. Запускается daemon: `bin/datasetfs daemon --no-mount --root <path>`. Daemon
   читает `metadata.parquet`, строит `CoreIndex` (in-memory: `FileMap`, `ShardMap`),
   ждёт IPC.
2. Python создаёт `DatasetFS(num_workers=N, ...)`. В `__init__` шлёт `POST
   /initialize_loading` с `num_workers + seed + decode-config`. Daemon
   разбивает 9 SHM-слотов по N воркеров, инициализирует N пайплайнов.
3. `DataLoader(ds, num_workers=N, ...)` — PyTorch спавнит N подпроцессов.
   Каждый вызывает `DatasetFS.__iter__`, который открывает свой
   `/tmp/datasetfs_pipe_<worker_id>`.
4. На стороне daemon'а Planner шуффлит шарды для своего воркера, отправляет
   `LoadJob` в BackgroundLoader. Тот читает шард в SHM-слот. Если
   `decode_mode = rgb_uint8` — между Loader и Dealer вставлен Decoder,
   который JPEG-декодит + resize'ит, перезаписывая слот пакованным RGB uint8.
5. DealerWorker набирает `WindowSize = 3` слота, шуффлит объекты внутри окна,
   шлёт `Batch` как JSON-line в pipe.
6. Python читает pipe, `mmap`'ает слот по `offset+size`, копирует bytes,
   декодит (PIL → tensor, либо `np.frombuffer` для rgb_uint8), декрементит
   refcount слота через `/tmp/mlfs_refs.bin`.
7. Когда refcount слота = 0, Planner.WatchRefCounts возвращает слот в пул.

## Компоненты

### Go daemon

| Пакет | Что делает | Ключевые файлы |
|---|---|---|
| `cmd/datasetfs` | единый бинарник (cobra): подкоманды `daemon` (Manifest+CoreIndex, IPC, опц. mount FUSE), `vacuum`, `converter` | `main.go`, `daemon.go`, `vacuum.go`, `converter*.go` |
| `internal/index` | Manifest (на диске) + CoreIndex (в памяти): метаданные объектов и шардов | `tree.go`, `manifest.go`, `wal.go` |
| `internal/storage` | tar-шарды: чтение, запись, валидация | `reader.go`, `writer.go`, `tar_append.go`, `validator.go` |
| `internal/shm` | Allocator 9×110 MB + refcounts через `/tmp/mlfs_refs.bin`; atomic ops для синхронизации с Python | `allocator.go` |
| `internal/pipeline` | Per-worker `Pipeline` собирает Planner + BackgroundLoader + (Decoder?) + DealerWorker. Конкурентная зона: горутины общаются через каналы. | `pipeline.go`, `planner.go`, `background_loader.go`, `decoder.go`, `dealer.go` |
| `internal/pipeline (cgo)` | Swappable JPEG backend: `decoder_libjpeg.go` (libjpeg-turbo через cgo, default) и `decoder_purego.go` (stdlib, build tag `datasetfs_purego`) | см. [docs/optimizations/01](optimizations/01-server-side-decode.md) |
| `internal/control` | HTTP server `:51409`: `/healthz`, `/initialize_loading` (создание сессии, поддерживает re-init), `/metrics` (JSON-метрики), `/debug/pprof/*` | `server.go` |
| `internal/metrics` | Атомарные counter'ы + latency-гистограмма; экспортит JSON | `metrics.go` |
| `internal/manager` | MutationManager (AddDeltaFile, DeleteFile) — фичу есть, тестов **нет** | `mutation_manager.go` |
| `internal/vfs` | FUSE-ноды (POSIX-чтение через mount-point) — **untested**, бенчмарки не используют (все идут с `--no-mount`) | — |

### Python-клиент

`clients/python/dataset_fs.py`:

- `DatasetFS(IterableDataset)` — основной класс
- `decode_fn` — пользовательская функция bytes → intermediate (по умолчанию PIL.Image.open). Игнорируется в `decode_mode="rgb_uint8"`.
- `transform` — пользовательский torchvision-композ. Должен соответствовать вы­ходу `decode_fn` (PIL.Image → tensor) или режиму rgb_uint8 (numpy HWC uint8 → tensor).
- Аудио-датасеты: пользователь передаёт свой `decode_fn` (например, `soundfile.read(BytesIO)`) и `transform`. Поддержка протестирована через [tests/test_speech_commands*.py](../tests/test_speech_commands.py).

### Benchmark harness

`benchmarks/datasetfs_bench/`:

| Подпакет | Что |
|---|---|
| `loaders/` | Унифицированные обёртки: `imagefolder.py`, `webdataset_loader.py`, `datasetfs.py`. Все используют общую `make_image_transform(image_size)` чтобы убрать transform как confounding variable |
| `models/registry.py` | `simplecnn`, `resnet18`, `resnet50` |
| `train/loop.py` | format-agnostic `train_one_epoch(...) → EpochStats` |
| `metrics/` | `training.py` (per-epoch), `daemon.py` (scrape /metrics), `system.py` (psutil sampler) |
| `runner/` | `daemon_ctl.py`, `cache_control.py`, `single_run.py`, `sweep.py`, `profile_run.py` |
| `reporting/` | `plots.py` (bar charts MVP), `sweep_plots.py` (line plots vs axis) |
| `configs/` | YAML-описания сценариев (см. [benchmarking.md](benchmarking.md)) |

## IPC-протокол

### HTTP control plane

`POST /initialize_loading` принимает:

```json
{
  "num_workers": 4,
  "seed": 42,
  "decode": {"mode": "raw", "image_size": 0}
}
```

`mode ∈ {"raw", "rgb_uint8"}`. Ответ — JSON с подтверждённой конфигурацией (Python сверяет, что daemon согласился). Повторный вызов — **re-init**: daemon корректно тушит предыдущую сессию, открывает новую.

`GET /metrics` отдаёт JSON со счётчиками (см. [benchmarking.md](benchmarking.md)).

`/debug/pprof/*` включён по умолчанию; mutex/block профили **выключены** до выставления флагов `--mutex-profile-rate=5 --block-profile-rate=10000`.

### Data plane

- **SHM data**: `/tmp/mlfs_data.bin`, mmap MAP_SHARED, размер `9 × 110 MB = 990 MB`. Слот идентифицируется `slot_id ∈ [0..8]`. Внутри слота — packed bytes (raw shard или packed decoded RGB).
- **SHM refs**: `/tmp/mlfs_refs.bin`, 9 × int32. Atomic-операции с обеих сторон. Python: `struct.pack("<i", new_val)`. Go: `atomic.LoadInt32` / `atomic.SwapInt32`.
- **Pipe**: `/tmp/datasetfs_pipe_<worker_id>`, FIFO, JSON-line на батч. Формат:
  ```json
  {"items": [
    {"slot_id": 3, "offset": 314572800, "size": 27648,
     "path": "data/raw/imagenette2/train/n03394916/n03394916_40715.JPEG",
     "meta": {"label": "n03394916"}},
    ...
  ]}
  ```
  Пустой `items: []` — сигнал конца эпохи.

## Decode modes

| Mode | Что в slot'е | Что в pipe | Что в Python |
|---|---|---|---|
| **`raw`** (по умолчанию) | сырой шард, items указывают на JPEG-байты внутри | offsets указывают на JPEG-данные | PIL.Image.open → resize → ToTensor |
| **`rgb_uint8`** | packed RGB uint8 HWC, items указывают на `image_size² × 3` байт | offsets указывают на packed RGB | np.frombuffer + reshape(H,W,3) → ToTensor (без PIL) |

Server-side decode реализован как опциональный стейдж между BackgroundLoader и DealerWorker. Условное включение — в `pipeline.NewPipeline`, по `cfg.Decode.IsServerSide()`. Полная история — в [docs/optimizations/01-server-side-decode.md](optimizations/01-server-side-decode.md).

## Конвенции и подводные камни

### Сборка

- **cgo обязателен для daemon'а** в дефолтном билде (libjpeg-turbo). Makefile выставляет `CGO_ENABLED=1` + `PKG_CONFIG_PATH=/opt/homebrew/opt/jpeg-turbo/lib/pkgconfig`. Для pure-Go fallback — `make build-purego`.
- **macOS Homebrew**: `brew install jpeg-turbo`. На Linux: `apt install libturbojpeg0-dev` + переменная `PKG_CONFIG_PATH` корректируется.

### Python multi-worker

- **macOS / Python 3.13 default start method = `spawn`**. Всё, что передаётся в worker'ов, должно быть picklable. Транзитивно: lambdas / closures **нельзя** в `collate_fn`, `transform`, `decode_fn`. Использовать module-level функции + `functools.partial`.
- **DFS multi-worker**: один pipe на worker (`/tmp/datasetfs_pipe_<worker_id>`). Daemon создаёт N пайплайнов через `POST /initialize_loading {"num_workers": N}`. Max N = **9** (= NumSlots).

### Шардинг и слоты

- `shm.NumSlots = 9`, `shm.SlotSize = 110 MB`.
- **Слот-партиция**: каждому воркеру даётся contiguous диапазон слотов. Первые `(9 % N)` воркеров получают по +1 слоту.
- **Dealer window**: `DealerWorker` блокируется только на первой `SlotMeta`, потом non-blockingly дренирует до `WindowSize=3`. Раньше блокировалось до полного окна — был deadlock при «шарды > слотов на воркера», см. историю в [internal/pipeline/dealer.go:55-56](../internal/pipeline/dealer.go).
- **Detector starvation**: `metrics.SlotStarvationTotal` — счётчик случаев, когда Planner не нашёл свободный слот.

### Известные баги в истории

- **TotalSize bug**: converter в одной из ранних версий писал 0 в `total_size` манифеста. Loader читал 0 байт → пустой SHM → silent PIL decode failure → refcount не декрементился → deadlock для воркеров с shards>slots. Фикс в [internal/storage/tar_append.go](../internal/storage/tar_append.go)
- ~~`cmd/fuse_daemon/main_test.go` вызывал `main()` и виснул~~ — этого файла больше нет (бинарники объединены в `cmd/datasetfs`). `go test ./internal/... ./cmd/datasetfs/...` (или `make go-test`) безопасен.

### Imagefolder и Speech Commands

- `imagefolder_index` keys = **`"class/filename"`**, не basename. Speech Commands V2 имеет одинаковые basename'ы между классами (одна и та же запись разных слов).
- **Imagefolder prep** строит per-class symlinks в `data/formats/<ds>/imagefolder/`, фильтрует по `ds.classes` (исключает `_background_noise_` и т.п.). DFS converter (Go) разворачивает symlink через `os.Stat`.

### Audio

- `torchaudio.load(BytesIO)` требует `torchcodec`, которого нет на macOS PyPI. **Использовать `soundfile.read` напрямую** в `decode_fn`. `torchaudio` для transforms (MelSpectrogram etc.) — нормально.

### Cache control в бенчмарках

- `cache_control.drop_page_cache()` запускает `sudo -n purge` на macOS / `drop_caches` на Linux. **Требует passwordless sudo**. У текущего пользователя настроено через `/etc/sudoers.d/datasetfs-bench`.
- Если sudo не настроен — runner предупреждает и продолжает с `cache_state=uncontrolled`.
- Drop вызывается **между ячейками** (per `loader×seed`), **не между эпохами**. Внутри ячейки warmup epoch разогревает кэш для measured epoch.

### Daemon-аргументы

```
datasetfs daemon --no-mount --root <path>
                 [--mutex-profile-rate N]   # 1-in-N sampling, ~1-3% overhead
                 [--block-profile-rate ns]  # block events >ns
```
