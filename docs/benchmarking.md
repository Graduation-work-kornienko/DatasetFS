# Бенчмарки DatasetFS

Каталог: [benchmarks/datasetfs_bench/](../benchmarks/datasetfs_bench/). Цель — измерить **DFS на тех же осях**, что и конкурирующие dataloader-форматы (ImageFolder, WebDataset), и при необходимости — DFS в разных конфигурациях между собой (например, два режима decode).

## Цели бенчмарков (для диплома)

1. **Конкурентоспособная производительность.** DFS не должен проигрывать ImageFolder/WebDataset в типовых сценариях обучения. Где проигрывает — должен быть понятный аналитический ответ почему.
2. **Гибкость / уникальные свойства.** Дать данные под фичи, которых нет у конкурентов: server-side decode, concurrent mutations (план), S3 streaming (план), различные типы данных (images + audio уже, video в плане).
3. **Аналитика узких мест.** Бенчмарки + профили должны однозначно указывать, где сидит bottleneck в каждом сценарии. Без этого нельзя честно оптимизировать.

## Сравнительная группа

| Loader | Тип | Что сравнивает |
|---|---|---|
| **`imagefolder`** | `torchvision.datasets.ImageFolder` поверх symlink-фермы | Канонический бейзлайн «один файл — один сэмпл». Лучшее, что есть «из коробки» |
| **`webdataset`** | `webdataset.WebDataset` поверх tar-шардов | Современный shard-based format для больших датасетов. Прямой конкурент DFS архитектурно |
| **`datasetfs`** | DFS daemon + Python client | Тестируемая система |

Все три используют общий `make_image_transform(image_size)` (см. [benchmarks/datasetfs_bench/loaders/_common.py](../benchmarks/datasetfs_bench/loaders/_common.py)) → одинаковые pixel-статистики на выходе.

## Каталог конфигов

Все конфиги в [benchmarks/datasetfs_bench/configs/](../benchmarks/datasetfs_bench/configs/).

### Single-run (один прогон, без оси sweep'а)

| Файл | Цель | Время | Команда |
|---|---|---|---|
| `smoke.yaml` | Sanity-check всего runner'а. SimpleCNN, 1 seed, 1 epoch, 20 batches, **uncontrolled cache** | ~2 мин | `make bench-smoke` |
| `mvp.yaml` | **Headline graph** Phase 2 — bar chart throughput 3 loader'а. ResNet-18, 3 seeds, 2 epoch × 80 batches, **cold cache** | ~15-20 мин | `make bench-mvp` |

### Sweep (Cartesian product оси × loaders × seeds)

| Файл | Ось | Цель | Время | Команда |
|---|---|---|---|---|
| `sweep_smoke.yaml` | num_workers ∈ [2, 4] | Verification смок'а sweep-runner'а | ~2 мин | `make bench-sweep-smoke` |
| `workers_sweep.yaml` | num_workers ∈ [0, 2, 4, 8] | **Headline 1** — как масштабируется throughput по воркерам. ResNet-18, 3 seeds | ~25 мин | `make bench-sweep-workers` |
| `batch_sweep.yaml` | batch_size ∈ [16, 32, 64, 128] | **Headline 2** — амортизация per-batch overhead. ResNet-18, 3 seeds | ~25 мин | `make bench-sweep-batch` |
| `decode_compare.yaml` | dfs_decode_mode ∈ [raw, rgb_uint8] | **Optimization 01 A/B** в **compute-bound** региме (ResNet-18) | ~6 мин | `make bench-decode-compare` |
| `decode_compare_simplecnn.yaml` | dfs_decode_mode ∈ [raw, rgb_uint8] | **Optimization 01 A/B** в **loader-bound** региме (SimpleCNN). Ожидаемое место для большого выигрыша | ~3-4 мин | `make bench-decode-compare-simplecnn` |

## Метрики

Каждая ячейка (cell = loader × seed × axis-value) записывается в `summary.csv` или, для sweep, дополнительно агрегируется в `sweep_summary.csv` с `axis_<name>` колонками.

### Per-epoch (training-side, `benchmarks/datasetfs_bench/metrics/training.py`)

| Колонка | Что | Зачем смотрим |
|---|---|---|
| `samples_per_second` | **Главный** показатель throughput | Headline-метрика. Чем больше — тем лучше |
| `n_samples`, `n_batches` | Размер прогона | Контекст для sps |
| `wall_seconds` | Wall-clock эпохи | sps = n_samples / wall_seconds |
| `time_to_first_batch` (TTFB) | Время до первого батча | Качество startup'а. Чем меньше — тем интерактивнее. Растёт при server-side decode (decoder синхронно обрабатывает первый slot) |
| `stall_fraction` | Доля времени, когда модель ждёт данные | Loader-Health. <5% — отлично, >10% — loader не успевает |
| `fetch_p50`, `fetch_p95`, `fetch_p99` | Латенси `iter(loader).next()` per-batch | Где «толстый хвост» в loader-side |
| `compute_p50` | Латенси `forward+backward` per-batch | Loader-vs-compute baseline |

### Системные (psutil, `benchmarks/datasetfs_bench/metrics/system.py`)

Сэмплер с интервалом 0.2 сек, агрегирует за время ячейки.

| Колонка | Что |
|---|---|
| `sys_cpu_pct_mean`, `sys_cpu_pct_p95` | System-wide CPU% (8 ядер ⇒ ceiling 800%) |
| `sys_mem_max_bytes`, `sys_mem_mean_bytes` | System-wide used RAM (всё, не только наше дерево) |
| `sys_disk_read_bytes` | Cumulative disk-read дельта от начала ячейки. **Главное для оценки cold-cache I/O** |
| `sys_tracked_rss_max_bytes`, `sys_tracked_rss_mean_bytes` | Сумма RSS дерева процессов (Python tree + daemon) |
| `sys_n_samples` | Количество семплов в окне (sanity check) |

### Daemon `/metrics` (Go counter'ы, `benchmarks/datasetfs_bench/metrics/daemon.py`)

JSON scrape перед/после ячейки, дельта counter'ов.

| Колонка | Что | Когда смотреть |
|---|---|---|
| `daemon_shard_loads_total_delta` | Сколько шардов daemon загрузил | Sanity (= шардов в датасете, ±шум) |
| `daemon_bytes_read_total_delta` | Сколько байт daemon прочитал с диска (mmap-семантика — реальные access'ы, не cached lookup'ы) | Контраст с `sys_disk_read_bytes`: если `daemon_bytes >> sys_disk_read` → daemon идёт из page cache |
| `daemon_dealer_batches_sent_total_delta` | Сколько `Batch` JSON-сообщений ушло в pipe | Анализ pipe overhead'а: меньше = эффективнее |
| `daemon_samples_emitted_total_delta` | Сколько сэмплов daemon отправил | = sum по dealer batches |
| `daemon_epochs_completed_total_delta` | Сколько эпох daemon завершил (ожидается = `cfg.epochs`) | Sanity |
| `daemon_slot_starvation_total_delta` | Сколько раз Planner не нашёл свободный слот | Loader-health: 0 — норма, >0 — slots тесные |
| `daemon_refcount_overflow_total_delta` | Сколько раз `SetRefCount` нашёл ненулевой prev value | Должно быть 0; иначе lifecycle-bug |
| `daemon_active_pipelines` | Текущее число пайплайнов | Sanity (= num_workers) |
| `daemon_daemon_uptime_seconds` | Сколько живёт daemon | Контекст для stability-тестов |
| `daemon_load_latency_p50`, `_p95`, `_p99`, `_max` | Гистограмма времени загрузки одного шарда | **Главный** показатель «как daemon-side чувствует себя». Рост p99 = contention или back-pressure |
| `daemon_load_latency_count` | Количество семплов в окне | Sanity |

## Текущие результаты — что измерили

### Headline (`bench-mvp`, MVP-смок — SimpleCNN, single seed, warm cache)

| Loader | sps | TTFB | stall | sys_disk | daemon_bytes |
|---|---|---|---|---|---|
| imagefolder | 744 | 0.17 с | 33% | 89 MB | — |
| webdataset | 345 | 1.14 с | 68% | 254 MB | — |
| **datasetfs** | **308** | **1.48 с** | **71%** | **0.2 MB** | **900 MB** |

**Ключевой инсайт**: DFS `sys_disk_read = 0.2 MB` при `daemon_bytes_read = 900 MB`. Daemon идёт **из page cache**, не с диска. DFS — **не I/O-bound**, узкое горлышко — где-то в daemon→Python пайплайне. Это направление аналитической части диплома.

### Workers sweep (ResNet-18, cold cache, 3 seeds)

| workers | DFS sps | DFS stall | DFS daemon p99 |
|---|---|---|---|
| 0 | 97.8 | 16.4% | 71 ms |
| 2 | 99.5 | 4.5% | **34 ms** ← sweet spot |
| 4 | 97.4 | 4.8% | 66 ms |
| 8 | 91.7 | 5.7% | 131 ms |

Тогда казалось, что p50/p99 растёт от воркеров → contention на стороне daemon'а. После профилирования стало понятно — это **простой** daemon'а в ожидании Python-стороны. См. [optimizations/01](optimizations/01-server-side-decode.md).

### Batch sweep (ResNet-18, cold cache, 3 seeds)

| batch | DFS sps | stall | daemon p99 |
|---|---|---|---|
| 16 | 75.8 | 12.8% | 65 ms |
| 32 | 89.1 | 7.3% | 65 ms |
| 64 | 95.8 | 4.2% | 65 ms |
| 128 | 101.7 | 2.8% | 69 ms |

Daemon p99 **не зависит** от batch_size → daemon оперирует шардами, не батчами. Рост sps — амортизация per-batch overhead в Python + BLAS efficiency на ResNet-18.

### Optimization 01 — server-side decode A/B (ResNet-18, compute-bound)

| iter | mode | sps | stall | TTFB |
|---|---|---|---|---|
| 1 (pure Go) | raw | 97.0 | 5.0% | 1.96 с |
| 1 (pure Go) | rgb_uint8 | 83.8 (-14%) | 12.7% | 5.79 с |
| **2 (libjpeg-turbo)** | raw | 96.1 | 4.5% | 1.77 с |
| **2 (libjpeg-turbo)** | rgb_uint8 | 94.8 (**-1.4%**) | 5.9% | 2.38 с |

В compute-bound региме обе моды упираются в ResNet-18 на CPU (~100 sps ceiling). Backend-swap closed gap, но не дал чистый «win». **Loader-bound A/B (SimpleCNN) ещё впереди** — там должен проявиться чистый выигрыш.

## Sweep-методология

Полный sweep — `(loader, seed) × axis-value`, каждая ячейка:

1. **Cache drop** между ячейками (`sudo -n purge`/`drop_caches`). Требует passwordless sudo, настроен через `/etc/sudoers.d/datasetfs-bench`. Без него `cache_state="uncontrolled"`.
2. **Daemon restart** между seed'ами для DFS (свежая сессия — никаких state-leak'ов).
3. **Warmup epoch** (`warmup_epochs=1`) — её sps **не** идёт в headline аггрегаты (фильтр `warmup=False`).
4. **3 seeds** для error bars.

Drop кэша — **between cells (loader×seed)**, не between epochs. Внутри ячейки warmup epoch и measured epoch идут «как в реальном training scenario». См. дискуссию в [optimizations/01](optimizations/01-server-side-decode.md) (раздел про cache-drop методологию).

## Что хорошо измеряем

- Throughput, TTFB, stall — full coverage
- I/O от диска и из daemon отдельно — позволяет различать disk-bound vs page-cache-served
- Daemon-side latency percentiles — даёт картинку daemon-health'а
- System metrics включают daemon-PID — единое RSS-измерение Python+Go

## Что **не** измеряем, но стоит добавить

| Метрика | Зачем | Сложность |
|---|---|---|
| **Pipe back-pressure** (counter `pipe_blocked_writes` в daemon) | Сейчас не видно, блокируется ли Dealer на pipe-write. С добавлением — увидим, кто узкое горлышко: daemon-side compute или pipe-saturation | low (30 мин в `internal/pipeline/dealer.go`) |
| **Dealer window utilization** (gauge `dealer_window_utilization`) | Сейчас не знаем фактического среднего заполнения окна (WindowSize=3 — насколько часто доходит до 3?). С добавлением — увидим, нужно ли увеличить window | low |
| **Decoder latency** (histogram `decode_latency` per-slot) | Optimization 01 не имеет своей гистограммы — латенси sliding window'ом из `load_latency` (а это `BackgroundLoader`, не `Decoder`) | low |
| **Per-worker stall breakdown** | Сейчас stall — единичная метрика по training-side. Не видно, какой воркер тормозит | medium |
| **GPU metrics** (если/когда тренировка пойдёт на GPU) | Сейчас всё CPU; для thesis'а GPU неактуально, но для практичности добавить sm_util, mem_used | medium (внешний инструмент — `nvidia-smi` polling) |
| **End-to-end loss/accuracy** в бенчмарке | Сейчас бенчмарки замеряют только throughput, не learning. Доказать «training не сломан» — это работа тестов, но bench с loss-curves даёт более полную картину | medium |
| **Cache-hit rate в SHM-слотах** | Сейчас нет понимания, насколько хорошо переиспользуются слоты между шардами | medium-high |
| **Decoded sample cache (CoorDL-style)** | Если будем экспериментировать с кэшированием decoded RGB на стороне daemon'а — нужны метрики hit/miss | high |
| **Per-shard fairness** | Если планировщик выдаёт shard'ы неравномерно — увидим только в hot path. Метрика «количество сэмплов per shard» в эпохе помогла бы | low-medium |

## Что хотим достичь по итогам

| Сценарий | Желаемый результат | Текущий результат | Зазор |
|---|---|---|---|
| Headline bar chart (SimpleCNN, MVP) | DFS ≥ WebDataset throughput | DFS 308 < WebDataset 345 | -10%, gap покрыт ResNet-A/B (см. ниже) |
| Headline bar chart (ResNet-18) | DFS на уровне imagefolder/webdataset | DFS ~97 vs IF ~107 vs WDS ~99 | ±5-10%, **достигнуто** |
| Workers scaling | Sweet spot ясно идентифицирован, deg pattern объяснён | sweet spot = 2 workers; деградация — простой daemon'а | ✓ объяснено в [opt 01](optimizations/01-server-side-decode.md) |
| Batch scaling | Линейный рост throughput с batch (до ceiling) | подтверждено | ✓ |
| Cold-cache I/O | DFS лучше webdataset по `sys_disk` | DFS 0.2 MB vs WDS 254 MB | **✓ DFS лучше** (page-cache from mmap) |
| Server-side decode (ResNet-18) | rgb_uint8 ≥ raw | -1.4% (paritet) | SimpleCNN A/B ещё впереди |
| Server-side decode (SimpleCNN) | rgb_uint8 даёт значимый win | Не измерено | **TODO** |
| Stability across 20 sessions | retention ≥85%, RSS growth ≤2× | 99.76% retention, 1.00× growth | ✓ |
| Concurrent training + mutations | DFS работает, WebDataset нет | Не измерено | **TODO в plan'е** |
| S3 streaming + cold-start | DFS работает | Не измерено | **TODO в plan'е** |

## Бенчмарки следующих фаз (план)

См. [docs/status.md](status.md). Кратко:

- **Concurrent mutations bench** — обучение + асинхронные `AddDeltaFile`/`DeleteFile`. WebDataset не умеет (tar immutable). Нужен новый конфиг + расширение runner'а.
- **Cold-start / S3 streaming** — daemon стримит шард по сети. Нужна реализация S3-backend + новый конфиг.
- **Imagewoof / Speech Commands** в основном headline-сете — для обобщения вне Imagenette. Конфиги — копия `mvp.yaml` с переключённым `dataset.*`. Дёшево добавить.

## Reporting

| Скрипт | Что строит | Когда вызывается |
|---|---|---|
| `reporting/plots.py` | Bar chart throughput-с-error-bars, latency-table.md | `bench-smoke`, `bench-mvp` |
| `reporting/sweep_plots.py` | Line plot vs axis (throughput, stall) | `bench-sweep-*`, `bench-decode-compare*` |

Известное ограничение `sweep_plots.py`: axis значения парсятся как `float`, поэтому строки (`raw`, `rgb_uint8`) дают warning «No artists with labels». CSV всё равно корректный, plot strictly required только для headline'а; для A/B по строкам — таблица из CSV информативнее.
