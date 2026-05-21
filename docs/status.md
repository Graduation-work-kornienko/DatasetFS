# Состояние проекта DatasetFS

Снимок состояния на **2026-05-22**. Обновлять при значимых переходах (смена фазы, завершение оптимизации, новое направление).

## Где мы по фазам плана

| Фаза | Что | Статус |
|---|---|---|
| 0 | Multi-worker DFS client (Python `num_workers`, Go per-worker pipelines, slot partitioning, seed) | ✅ Done |
| 1 | Correctness suite (images + audio + manifest + training) | ✅ Done |
| 2 | MVP benchmark (3 loaders, ResNet-18, headline bar chart) | ✅ Done |
| 3 | Metrics + sweep infrastructure | ✅ Done (workers/batch sweeps, daemon `/metrics`, psutil, cache-control, stability) |
| **Optimization track** | Server-side decode (opt 01) → ... | 🔄 **Opt 01 в Iter 2** |
| 4 | Full matrix (FFCV, HF Parquet, ResNet-50, дополнительные датасеты в headline) | Не начат |
| 5 | Polish (notebook, LaTeX, README) | Не начат |

## Что сейчас лежит на столе

### Optimization 01 — server-side decode (в работе)

Полный контекст: [optimizations/01-server-side-decode.md](optimizations/01-server-side-decode.md).

- Итерация 1 (pure Go) — **завершена**. Архитектура подтверждена, gap -14% vs PIL.
- Итерация 2 (libjpeg-turbo через cgo) — **запущена**:
  - ✅ Декодер вынесен в swappable интерфейс (`jpegDecoder`)
  - ✅ `decoder_purego.go` (build tag `datasetfs_purego`) + `decoder_libjpeg.go` (default, cgo + TurboJPEG)
  - ✅ Makefile собирает с `CGO_ENABLED=1 PKG_CONFIG_PATH=...`
  - ✅ Pytest-фикстура `daemon_binary` синхронизирована с Makefile'ом
  - ✅ `test-decode` проходит (max diff 1 vs PIL — даже лучше pure Go)
  - ✅ ResNet-18 A/B прогнан: rgb_uint8 = -1.4% от raw (closed gap с -14%)
  - ⏸ **Следующий шаг — `make bench-decode-compare-simplecnn`** (SimpleCNN A/B, loader-bound, ~3-4 мин)
  - ⏸ После этого — финализация opt 01 в журнале (статус → завершено)

### Phase 3 doings — итоги

Закрытые задачи:
- `make bench-sweep-workers` + `make bench-sweep-batch` — оба прогнаны, данные собраны (есть в [benchmarking.md](benchmarking.md))
- `make test-stability` — 99.76% retention, 1.00× RSS growth
- Cold-cache control (passwordless `sudo purge` через `/etc/sudoers.d/datasetfs-bench`)
- Profiling-harness `benchmarks/datasetfs_bench/runner/profile_run.py` — снимает CPU/mutex/block/goroutine/heap + Python cProfile одновременно
- Mutex/block profile flags на daemon'е (`--mutex-profile-rate`, `--block-profile-rate`)

## План оптимизаций

Очередь после opt 01, перетриажированная под **«гибкая ФС с современными практиками + конкурентоспособная производительность»** (см. [memory/project_datasetfs_goal](../.claude/projects/-Users-true-danil-12-Graduation-work-DatasetFS/memory/project_datasetfs_goal.md)).

### Tier A — следующие кандидаты (после opt 01)

**Opt 02 — Concurrent training + mutations bench.** Уникальная фича DFS, которой нет у WebDataset (tar immutable). У нас есть `MutationManager` ([internal/manager/mutation_manager.go](../internal/manager/mutation_manager.go)), но **нет тестов** и нет бенчмарка. План:
1. Написать тест: обучение идёт, в это время отдельный поток вызывает `AddDeltaFile` / `DeleteFile`
2. Бенч-конфиг с метриками: «насколько просел throughput из-за параллельной мутации»
3. Сравнение: WebDataset вообще не запускается в таком режиме → качественный аргумент

**Opt 03 — S3 streaming + Cold Start bench.** Минимальная версия: манифест с URL'ами вместо локальных файлов, daemon скачивает шард при load'е. Бонус: Cold Start bench, где DFS должен показать «учимся пока тянем данные». Реальная стоимость — 1-2 недели работы.

**Opt 04 — Pipeline optimization next layer.** Когда decode уже не bottleneck — кто следующий? Из профиля видно:
- JSON encoding в DealerWorker — небольшой, но не нулевой
- SHM-чтение из Python (`bytes(data_mmap[off:off+size])` — memcpy)
- pipe-write blocking

Открытое направление; до этого нужно завершить opt 01.

### Tier B — стоит вписать в diploma

**CoorDL / Plumber related work** — раздел в дипломе про современные подходы. Имплементация cross-worker decoded cache (CoorDL-style) — возможна в DFS благодаря центральному daemon'у, но дорогая. Анализ-only обязателен.

**Video / макрообъекты** — гибкость по типам данных. SlotSize=110 MB не вмещает видеоклип; нужен либо больший слот, либо chunked-доступ. Архитектурно интересно, но дорого.

### Tier C — future work (упомянуть в дипломе как направления)

- Distributed training (одно-узловой DDP)
- Background vacuumer / compaction (если mutations попадают в бенчмарк)
- FUSE-mount mode (`internal/vfs/` сейчас untested)
- HuggingFace Parquet как ещё один бейзлайн

## Открытые вопросы / TODO

- **`sweep_plots.py` плохо рисует string-ось** (raw / rgb_uint8) — warning «No artists with labels». CSV корректный, но плот не информативен. Фикс: ветка для категориальных осей. Низкий приоритет.
- **Numpy non-writable warning** в `transforms.ToTensor()` для rgb_uint8: торч ругается «not writable». На корректность не влияет (`.contiguous()` создаёт writable копию), но pollute'ит логи. Косметика — `np.frombuffer(...).copy()` уберёт варн ценой одного memcpy.
- **HANDOFF.md устарел** в отношении новых docs/ и optimizations/. Можно дописать pointer на `docs/README.md` (как entry point — толще, чем HANDOFF, но если новая сессия — пройдёт через HANDOFF).

## Что лежит на ветке (git)

На момент снимка `main`. Git-статус — серия изменений по opt 01:
- `internal/pipeline/decoder.go` (новый orchestration)
- `internal/pipeline/decoder_purego.go`, `decoder_libjpeg.go` (новые backends)
- `internal/pipeline/pipeline.go` (DecodeMode/DecodeConfig + optional decoder stage)
- `internal/ipc/startup_server.go` (decode-config parse в `/initialize_loading`)
- `cmd/fuse_daemon/main.go` (pprof flags)
- `clients/python/dataset_fs.py` (decode_mode + rgb_uint8 path)
- `benchmarks/datasetfs_bench/loaders/datasetfs.py` + `_common.py` (decode_mode passthrough)
- `benchmarks/datasetfs_bench/runner/single_run.py` (dfs_decode_mode → spec)
- `benchmarks/datasetfs_bench/runner/profile_run.py` (новый)
- `benchmarks/datasetfs_bench/configs/decode_compare.yaml`, `decode_compare_simplecnn.yaml` (новые)
- `tests/test_decode.py` (новый)
- `tests/test_stability.py` (новый)
- `tests/conftest.py` (cgo env в фикстуре + `.pid` property)
- `Makefile` (новые таргеты, CGO_ENV, build-purego)
- `go.mod`, `go.sum` (golang.org/x/image)
- `HANDOFF.md` (обновлено по фазам)
- `docs/` (новая папка с этой документацией)

Коммиты пока **не созданы**. На следующий вход стоит проверить — может быть, имеет смысл сгруппировать в логические коммиты (opt 01 plumbing / opt 01 decoder / opt 01 cgo / phase 3 stability / docs).

## Memory pointers (если новая сессия)

`MEMORY.md` подгружается автоматически. Файлы памяти:
- [memory/user_role.md](../.claude/projects/-Users-true-danil-12-Graduation-work-DatasetFS/memory/user_role.md) — CS-студент, диплом, Russian
- [memory/project_datasetfs_goal.md](../.claude/projects/-Users-true-danil-12-Graduation-work-DatasetFS/memory/project_datasetfs_goal.md) — основная установка: гибкая ФС, не «обогнать WebDataset на throughput». **Перетриаж 2026-05-17** после ошибочной первичной формулировки.
- [memory/project_deferred_tests.md](../.claude/projects/-Users-true-danil-12-Graduation-work-DatasetFS/memory/project_deferred_tests.md) — 6 deferred тестов, 1 закрыт (stability).
- [memory/project_handoff_pointer.md](../.claude/projects/-Users-true-danil-12-Graduation-work-DatasetFS/memory/project_handoff_pointer.md) — указывает читать HANDOFF.md первым.

## Bootstrap для следующей сессии

1. Прочитать [HANDOFF.md](../HANDOFF.md) (быстрый обзор, 2 мин)
2. Прочитать [docs/status.md](status.md) (этот файл) — где мы сейчас
3. Прочитать [docs/optimizations/01-server-side-decode.md](optimizations/01-server-side-decode.md) (если продолжаем opt 01)
4. Запустить `make bench-decode-compare-simplecnn` (3-4 мин) — закрыть SimpleCNN A/B
5. Дописать «A/B — SimpleCNN» в [optimizations/01](optimizations/01-server-side-decode.md), перевести статус → завершено
6. Решить, что брать в opt 02 (см. Tier A выше)
