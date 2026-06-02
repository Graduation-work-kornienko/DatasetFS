# Состояние проекта DatasetFS

Снимок состояния на **2026-06-01**. Обновлять при значимых переходах (смена фазы, завершение оптимизации, новое направление).

## Где мы по фазам плана

| Фаза | Что | Статус |
|---|---|---|
| 0 | Multi-worker DFS client (Python `num_workers`, Go per-worker pipelines, slot partitioning, seed) | ✅ Done |
| 1 | Correctness suite (images + audio + manifest + training) | ✅ Done |
| 2 | MVP benchmark (3 loaders, ResNet-18, headline bar chart) | ✅ Done |
| 3 | Metrics + sweep infrastructure | ✅ Done (workers/batch sweeps, daemon `/metrics`, psutil, cache-control, stability) |
| **Optimization track** | Server-side decode (opt 01) → параллельный декод (opt 02) → ... | ✅ **Opt 01 + 02 завершены** |
| 4 | Full matrix (FFCV, HF Parquet, ResNet-50, дополнительные датасеты в headline) | Не начат |
| 5 | Polish (notebook, LaTeX, README) | Не начат |

## Новые фичи: vacuum / parquet-манифест / бинарный WAL / remote storage (2026-06-01)

Эти четыре фичи были добавлены отдельной итерацией, затем проверены и
доведены до рабочего состояния (исходно код не собирался / терял данные).
Дизайн-доки в `docs/{vacuum_design,remote_storage_analysis,manifest_format_migration}.md`
**описывают замысел, а не текущую реализацию** — ниже факт.

| Фича | Состояние | Тесты |
|---|---|---|
| Parquet-манифест | ✅ работает (Store пишет parquet и удаляет stale jsonl; Load читает parquet, jsonl — фолбэк). Был баг: `io.EOF` трактовался как фатальный → манифест не читался | `internal/index/parquet_manifest_test.go` |
| Бинарный WAL | ✅ работает, потокобезопасен (добавлен мьютекс); `Truncate` переписывает заголовок | `internal/index/binary_wal_test.go` (+race) |
| Vacuum | ✅ переписан: читает живые байты **из существующих шардов** по `(Offset,Size)`, пишет во временные шарды, атомарный swap; `--dry-run` ничего не трогает. Общий пакет `internal/vacuum`, CLI `cmd/vacuum` — тонкая обёртка | `internal/vacuum/vacuum_test.go` |
| Background vacuumer | ✅ теперь **горутина в демоне** (`--auto-vacuum`, off by default), а не отдельная программа (удалена `cmd/background_vacuum`). Координация через `ipc.BeginMaintenance` + `MutationManager.WithExclusive`; после vacuum — `CoreIndex.Reload` | покрыт `TestVacuum_FragmentationAndReload` + boot-smoke |
| Remote storage | ✅ работает по HTTP через **prefetch-в-кэш** при старте (`--root http://… --cache-dir …`). `ipc.StartServer` теперь принимает `*storage.Storage`. S3-SDK/`s3://` — вне рамок (анонимный HTTP-бакет MinIO) | `tests/test_remote_minio.py` (MinIO в Docker + обучение) |

Команды: `make go-test` (Go-юниты), `make test-remote` (MinIO+обучение, скипается без Docker/`minio` SDK), `go run ./cmd/datasetfs vacuum --root <ds> --dry-run`.

Известные пред-существующие баги (НЕ из этих фич, не трогал): converter
webdataset-пути пишет `TotalSize = сумма raw-размеров` без tar-заголовков/паддинга
(`internal/storage/writer.go`), и `go test ./...` без cgo-тега падает на decoder
(`make build-purego` использует `-tags datasetfs_purego`).

## Что сейчас лежит на столе

### Optimization 01 + 02 — server-side decode → параллельный декод ✅ ЗАВЕРШЕНЫ

Полный контекст: [optimizations/01-server-side-decode.md](optimizations/01-server-side-decode.md),
[optimizations/02-parallel-decode.md](optimizations/02-parallel-decode.md).

- **Opt 01** (decode в демоне, libjpeg-turbo через cgo): архитектура подтверждена,
  ResNet-18 A/B gap -14% → -1.4%. Открытый вопрос про loader-bound выигрыш закрыт opt 02.
- **Opt 02** (параллельный декод, пул K воркер-горутин на пайплайн):
  - Корень проблемы: декод демона был однопоточным → при малом числе воркеров демон не
    успевал кормить даже одного консумера (`rgb_uint8` был медленнее raw PIL).
  - ✅ Микро (num_workers=0): **487 → 3136 sps (6.4×)**, обогнал raw PIL (689) в **4.6×**.
  - ✅ End-to-end K-sweep (num_workers=1, SimpleCNN): K=1 378 → K=2 511 sps (+35%), плато при K≥2.
  - ✅ Кноб `decode.parallelism` (auto = NumCPU/NumWorkers); bench-ось `dfs_decode_parallelism`.
  - ✅ Дешёвый выигрыш: refcount poll 100 мс → 2 мс ([planner.go](../internal/pipeline/planner.go)).
  - ✅ Побочно устранён **use-after-Munmap SIGSEGV** на teardown сессии (Pipeline.Stop теперь
    джойнит горутины до Munmap; loader/dealer-отправки сделаны ctx-aware).
  - ✅ `test-decode` (max diff 1), `go test -race` зелёный на обоих build-тегах.

### Phase 3 doings — итоги

Закрытые задачи:
- `make bench-sweep-workers` + `make bench-sweep-batch` — оба прогнаны, данные собраны (есть в [benchmarking.md](benchmarking.md))
- `make test-stability` — 99.76% retention, 1.00× RSS growth
- Cold-cache control (passwordless `sudo purge` через `/etc/sudoers.d/datasetfs-bench`)
- Profiling-harness `benchmarks/datasetfs_bench/runner/profile_run.py` — снимает CPU/mutex/block/goroutine/heap + Python cProfile одновременно
- Mutex/block profile flags на daemon'е (`--mutex-profile-rate`, `--block-profile-rate`)

## Конечная цель: каталог графиков + очередь фич

Конечная цель диплома (формулировка пользователя 2026-06-01) — **обширный каталог
объёмных графиков** по многим осям, каждый демонстрирует свойство системы
(современность / конкурентность / гибкость). Полный каталог **G1–G14** и две новые
фичи (F1, F2) — в [HANDOFF.md](../HANDOFF.md) → «Benchmark & graph catalog» и
«Roadmap to thesis completion». Здесь — очередь работ под этот каталог.

**Дисциплина метрик (применяется к КАЖДОМУ бенчмарку, требование 2026-06-01).**
У каждого бенча — исчерпывающий набор метрик; по каждой метрике зафиксировано
*зачем* её отслеживаем и *какую гипотезу* она подтверждает/опровергает; и проверка
достаточности: *позволяют ли метрики заключить «X быстрее/лучше» И объяснить почему?*
Если у результата ≥2 объяснения, неразличимых метриками, — набор недостаточен.
(Именно так шли opt 01→02: throughput говорил «rgb медленнее», и только daemon-CPU
+ Python-idle% + per-stage профиль вскрыли *почему* — последовательный декод.)
Полная таксономия метрик — в [HANDOFF.md](../HANDOFF.md) → «Metrics discipline»,
инвентарь и пробелы — в [benchmarking.md](benchmarking.md).

### Tier A — фичи под флагманские графики (наибольший вес для диплома)

**F1 — Snapshot-консистентная мутация при обучении (→ график G3). НОВОЕ.**
Ключевая мысль пользователя: мутировать датасет во время обучения нужно (online
learning), но running-эпоха не должна видеть «рваное» полу-применённое состояние.
→ нужен механизм консистентности: тренировка **пинит снапшот / поколение манифеста**
на старте эпохи; мутации создают *новое* поколение (MVCC / copy-on-write); ридер
держит свой вид до перепина. Кирпичи есть (`MutationManager.WithExclusive`, vacuum
temp→swap, `CoreIndex.Reload`), но **снапшот-изоляция не спроектирована**. Это
флагманский аргумент гибкости (WebDataset обходит проблему запретом мутаций).
Сюда же — отложенный тест #6 (тесты на мутации) и бенч «throughput vs темп мутаций».

**F2 — Распределённое обучение (→ график G7). НОВОЕ.** Демон должен стать
rank/world-size-aware (шардинг поверх существующего per-worker слот-партишена).
Сначала одно-узловой DDP, потом мульти-узловой. График: масштабирование throughput
vs число процессов/GPU/узлов.

**G13 — End-to-end «реальный режим» (вдолгую). НОВОЕ.** Интегративный бенч поверх
F1: обучение + конкурентные мутации много эпох, активный vacuumer, запись WAL.
Один прогон доказывает, что вся история online-learning держится — корректность
(consistency violations = 0), эффективность (vacuum держит фрагментацию, WAL дёшев),
стабильность (нет дрейфа/утечек — расширяет `tests/test_stability.py`).

**Формат-матрица (→ график G1).** Лоадеры и format-prep для LMDB, TFRecord, HDF5,
HuggingFace (Arrow/Parquet), FFCV (Linux). Сейчас готовы только ImageFolder/
WebDataset/HF/DFS. Самый объёмный пункт по числу графиков.

**G12 — Мультимодальная / сложно-структурированная модель. НОВОЕ.** Модель с
многополевым сэмплом (image+text или image+audio+tabular) — доказывает, что DFS
корректно отдаёт гетерогенные per-sample структуры, и нагружает metadata/collate-путь,
а не только сырые байты картинки. Нужны мультимодальный датасет + collate.

### Tier B — дешёвые графики (фичи built, нужен только бенч)

- **Background vacuumer on/off (→ G4)** — throughput с `--auto-vacuum`, фрагментация
  по эпохам, latency в окна обслуживания.
- **WAL формат (→ G5)** — JSONL vs binary WAL: write tput, размер лога, recovery.
- **Манифест формат (→ G6)** — JSONL vs Parquet: load time, RAM, размер.
- **Remote / S3 (→ G9 кривая + G14 выделенный сценарий)** — cold-start, prefetch
  overlap, cache hit ratio, end-to-end «учимся пока тянем» из remote.
- **Pipeline next layer (opt 03, → конкурентность)** — после decode переузким местом
  могут стать JSON-over-pipe в `DealerWorker`, SHM-memcpy в Python, pipe-write.
  Профилировать после ре-замера headline.

### Tier C — широта данных / типов

- ResNet-50 + 5 seeds для thesis-grade headline; PubLayNet (size-scaling), доп. аудио.
- **Реальные числа на Linux + GPU** (сейчас всё macOS/CPU — крупнейший пробел).
- Video / макрообъекты — `SlotSize=110 MB` не вмещает видеоклип; нужен больший слот
  или chunked-доступ. Архитектурно интересно, дорого → future work.
- CoorDL/Plumber related-work раздел (cross-worker decoded cache — анализ-only).
- FUSE-mount mode (`internal/vfs/` untested, отложенный тест #5).

## Открытые вопросы / TODO

- **`sweep_plots.py` плохо рисует string-ось** (raw / rgb_uint8) — warning «No artists with labels». CSV корректный, но плот не информативен. Фикс: ветка для категориальных осей. Низкий приоритет.
- **Numpy non-writable warning** в `transforms.ToTensor()` для rgb_uint8: торч ругается «not writable». На корректность не влияет (`.contiguous()` создаёт writable копию), но pollute'ит логи. Косметика — `np.frombuffer(...).copy()` уберёт варн ценой одного memcpy.
- **HANDOFF.md актуализирован 2026-06-01** — содержит каталог графиков G1–G11, фичи F1/F2, roadmap до защиты. Источник правды для новой сессии.

## Что лежит на ветке (git)

На момент снимка `main`. Git-статус — серия изменений по opt 01:
- `internal/pipeline/decoder.go` (новый orchestration)
- `internal/pipeline/decoder_purego.go`, `decoder_libjpeg.go` (новые backends)
- `internal/pipeline/pipeline.go` (DecodeMode/DecodeConfig + optional decoder stage)
- `internal/control/server.go` (decode-config parse в `/initialize_loading`)
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
