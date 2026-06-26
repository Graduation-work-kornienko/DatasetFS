# Тесты DatasetFS

Каталог тестов: [tests/](../tests/). Go unit tests находятся рядом с пакетами в `internal/**` и `cmd/datasetfs/**`.

Рекомендуемые entrypoints:

```bash
make go-test
make test
make thesis-code-gate
```

Не все тесты одинаково портативны. Часть сценариев требует подготовленных датасетов, macFUSE, Docker/MinIO, Linux-only зависимостей или длительного CPU time.

## Категории покрытия

| Категория | Цель | Файлы |
|---|---|---|
| Manifest/index/storage | Проверить `metadata.parquet`, shard offsets/sizes, WAL, snapshots, vacuum | `test_manifest.py`, `internal/index/*_test.go`, `internal/storage/*_test.go`, `internal/vacuum/*_test.go` |
| Data correctness | DFS выдает тот же payload/labels без дублей и пропусков | `test_correctness.py`, `test_imagewoof.py`, `test_speech_commands.py` |
| Training correctness | Данные пригодны для обучения, loss падает и сравним с baseline | `test_training.py`, `test_speech_commands_training.py`, `test_universal_datasets_training.py` |
| Decode path | `rgb_uint8` эквивалентен PIL resize/to-tensor | `test_decode.py`, `internal/pipeline/decoder_test.go` |
| Pipeline lifecycle | Re-init, session FIFO, crash/EOF bounded behavior, cleanup, RSS plateau | `test_deferred_gaps.py`, `test_pipeline_leak.py`, `test_stability.py` |
| FUSE/mutations | POSIX path, snapshot consistency under concurrent writes/deletes | `test_deferred_gaps.py`, `test_mutation_consistency.py`, `internal/manager/*_test.go` |
| Remote storage | HTTP/MinIO remote path and preflight/reporting | `test_remote_minio.py`, `test_remote_preflight.py`, `test_remote_plots.py` |
| Benchmark/reporting code | CSV/plots/reporting не ломаются | `test_*plots.py`, `test_benchmark_report.py`, `test_training_metrics.py`, `test_wait_compute_plots.py` |
| Format matrix loaders | Extra formats отдают совместимые batches | `test_format_loaders.py`, `test_format_mutation_benchmark.py`, `test_real_universal_*` |

## Makefile targets

| Target | Что запускает | Зависимости/оговорки |
|---|---|---|
| `make go-test` | `go test ./internal/... ./cmd/datasetfs/...` с cgo env | Нужен libjpeg-turbo для default build |
| `make test` | `test-manifest`, `test-correctness`, `test-imagewoof`, `test-training` | Нужны подготовленные image datasets |
| `make test-manifest` | `tests/test_manifest.py` | Читает Parquet manifest |
| `make test-correctness` | Imagenette data correctness | Дольше smoke-тестов |
| `make test-imagewoof` | Cross-dataset correctness на Imagewoof | Нужен Imagewoof |
| `make test-training` | Imagenette training correctness | CPU-heavy |
| `make test-audio` | Speech Commands correctness + training | Нужен `make data-audio` |
| `make test-mutation` | Real FUSE mutation consistency | Нужен macFUSE |
| `make test-deferred-gaps` | Crash/cleanup/FUSE deferred coverage | FUSE parts skip without macFUSE |
| `make test-datasetfs-writer` | Python writer/helper path | Быстрый |
| `make test-real-universal-prep` | Prep scaffolding for real-universal datasets | Быстрый |
| `make test-universal-datasets` | Synthetic text/audio/image+tabular training through daemon | Без внешних downloads |
| `make test-reporting` | py_compile + reporting unit tests | Без daemon |
| `make test-remote` | MinIO + remote HTTP training | Docker + `minio` SDK; skips if unavailable |
| `make test-stability` | 20 loading sessions against one daemon | Несколько минут |
| `make test-decode` | Pixel correctness for `rgb_uint8` | Быстрый |
| `make test-format-loaders` | Extra storage formats batch invariants | Нужен `make data-formats-extra` |
| `make test-pipeline-leak` | Skipped samples must not leak slots | Быстрый regression |
| `make test-pipeline-rss` | Repeated session RSS plateau | Synthetic dataset |
| `make test-pipeline-rss-mutation` | RSS plateau under bounded FUSE replacements | Нужен macFUSE |
| `make thesis-code-gate` | Fast pre-experiment gate | Сборка daemon + несколько Python/Go suites |

## Что проверяют основные тесты

### `test_manifest.py`

Проверяет соответствие `metadata.parquet` фактическим shard files:

- каждый manifest object указывает на существующий shard;
- `offset/size` попадают в реальные границы shard;
- `total_size` shard'а не расходится с физикой;
- нет дубликатов object paths;
- metadata читается через актуальный Parquet path.

### `test_correctness.py` и `test_imagewoof.py`

Проверяют image loading path:

- completeness: за эпоху приходят все expected objects;
- no duplicates;
- multi-worker disjointness;
- seed/shuffle behavior;
- byte hash equivalence с raw imagefolder;
- labels совпадают с ImageFolder class directory;
- edge values `num_workers ∈ {0,1,2,4,8}`;
- re-init resilience.

`test_imagewoof.py` повторяет ключевые checks на другом датасете, чтобы не overfit'иться на Imagenette.

### `test_speech_commands.py` и `test_speech_commands_training.py`

Проверяют generic raw payload path на audio:

- custom `decode_fn` через `soundfile.read(BytesIO)`;
- shape/sample_rate/non-silent sanity;
- MelSpectrogram/transform path;
- loss decreases на audio model.

### `test_training.py`

Проверяет, что DFS path не только байтово корректен, но и пригоден для обучения:

- loss decreases на DFS;
- final loss parity с ImageFolder baseline при одинаковом seed/model setup.

### `test_decode.py`

Проверяет server-side decode:

- DFS `rgb_uint8` output сравнивается с `PIL.Image.open + resize(BILINEAR)`;
- shape `(H, W, 3)`, dtype `uint8`;
- mean/p95/max pixel diff в допустимых пределах;
- `transforms.ToTensor()` получает float CHW `[0, 1]`.

Тест backend-агностичен: должен проходить и с libjpeg-turbo, и с pure-Go decoder.

### `test_deferred_gaps.py`

Закрывает старые deferred gaps:

- concurrent loaders/session FIFO failure mode: новая session не делит pipe со старым iterator;
- daemon crash mid-epoch: iterator завершается bounded failure/EOF, не висит бесконечно;
- cleanup verification: после stop не остаются `/tmp/mlfs_*` и `datasetfs_pipe_*`;
- FUSE POSIX smoke: create/read/list/unlink через mount path.

FUSE test skips, если macFUSE недоступен.

### `test_mutation_consistency.py`

Проверяет F1/snapshot behavior:

- запускает daemon с FUSE mount;
- читает DatasetFS epoch;
- параллельно делает writes/deletes через POSIX mount;
- проверяет, что epoch видит один pinned generation и не получает torn reads.

### `test_pipeline_leak.py`

Регрессии pipeline lifecycle:

- skipped samples (`decode_fn -> None`) все равно декрементят refcount;
- иначе slots залипают, когда shards > slots;
- repeated session restarts не должны приводить к RSS creep;
- отдельная проверка RSS plateau under bounded FUSE replacements.

### `test_stability.py`

Длинный health gate:

- много loading sessions против одного daemon process;
- проверяет throughput retention;
- проверяет daemon/Python RSS growth;
- ловит refcount drift, goroutine/FD leaks, allocator lifecycle issues.

### Reporting tests

`test_daemon_timeseries_plots.py`, `test_real_universal_plots.py`, `test_remote_plots.py`, `test_system_timeseries_plots.py`, `test_pipeline_memory_plots.py`, `test_training_stage_plots.py`, `test_wait_compute_plots.py`, `test_benchmark_report.py`, `test_training_metrics.py` защищают benchmark infrastructure от silent breakage. Они особенно важны, потому что дипломные графики строятся scripts, а не вручную.

## Go unit tests

Go tests покрывают внутренние инварианты:

- `internal/index`: Parquet manifest, WAL, binary WAL, snapshots.
- `internal/storage`: prefetch/cache behavior и shard helpers.
- `internal/pipeline`: dealer/distributer/decoder/pipeline invariants, binary wire, parallel decode.
- `internal/manager`: mutation manager and delta shard behavior.
- `internal/vacuum`: compaction, dry-run, fragmentation/reload cases.
- `internal/metrics`: counters/histograms handler behavior.
- `cmd/datasetfs`: converter/vacuum/daemon CLI-adjacent checks.

Использовать `make go-test`, а не голый `go test ./...`, чтобы получить правильные cgo flags для libjpeg-turbo.

## Fixtures и conventions

`tests/conftest.py`:

- собирает `bin/datasetfs` один раз на pytest session;
- готовит datasets idempotently;
- предоставляет daemon managers для Imagenette/Imagewoof/Speech Commands;
- поддерживает `.restart()` для fresh loading session;
- чистит `/tmp/mlfs_*` и `/tmp/datasetfs_pipe_*`;
- дает `.pid` для psutil/RSS checks.

`tests/helpers.py`:

- `imagefolder_index(root)` строит ключи вида `class/filename`, что важно для Speech Commands/ImageFolder с одинаковыми basenames в разных классах;
- содержит byte/hash helpers и shared assertions.

Правила при добавлении тестов:

- Все `decode_fn`, `transform`, `collate_fn` должны быть module-level функциями; никаких lambdas/closures для spawn-mode.
- Для длинных тестов ставить `pytest.mark.timeout(N)` и запускать с `-s`, если нужен progress.
- Если тест делает несколько эпох/итераций, явно re-init daemon session через fixture API.
- Не полагаться на `main.py`; для integration scenarios использовать benchmark loaders или dedicated helpers.
- FUSE/Docker/Linux-only тесты должны gracefully skip, если окружение не готово.

## Оставшиеся gaps

| Gap | Почему важно | Статус |
|---|---|---|
| Linux/GPU integration tests | Проверить production-like path, FFCV и GPU starvation | Не закрыто |
| Multi-rank DDP e2e | `rank/world_size` path есть в client/control/pipeline, но нужен полноценный distributed training test | Частично инфраструктура есть, e2e gap |
| Long-running remote + mutation + vacuum combined | Интегративный сценарий G13/G14 | Бенчи есть по частям, нужен полный endurance |
| Crash recovery with WAL replay under real mutation load | Проверить не только bounded failure, но и восстановление состояния | Частично Go/WAL tests, e2e gap |
| Video / very large object path | `SlotSize=110 MB` ограничивает payload class | Future work |
