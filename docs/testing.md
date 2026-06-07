# Тесты DatasetFS

Каталог: [tests/](../tests/). Запуск: `pytest`, либо точечные Makefile-таргеты.

## Категории

| Категория | Цель | Файлы |
|---|---|---|
| **Корректность данных** | Доказать, что DFS отдаёт ровно те же байты/лейблы, что и raw imagefolder, без дублей, без пропусков, без коллизий | `test_manifest.py`, `test_correctness.py`, `test_imagewoof.py`, `test_speech_commands.py` |
| **Корректность обучения** | Доказать, что на DFS-данных модель учится не хуже, чем на ImageFolder (loss-decreasing + loss-parity) | `test_training.py`, `test_speech_commands_training.py` |
| **Стабильность daemon'а** | Доказать, что 20+ сессий не дают деградации throughput / роста RSS | `test_stability.py` |
| **Server-side decode** | Доказать, что rgb_uint8 мод выдаёт пиксели, эквивалентные PIL Resize | `test_decode.py` |

## Запуск

```bash
# Phase 1 verification gate (~30-40 мин CPU суммарно)
make test               # = test-manifest + test-correctness + test-imagewoof + test-training

# Точечно
make test-manifest      # ~30 сек
make test-correctness   # ~5-10 мин
make test-imagewoof     # ~3 мин
make test-training      # ~10-20 мин (тренировка ResNet/SimpleCNN)

# Аудио (требует `make data-audio` — Speech Commands V2 ~2.4 GB)
make test-audio         # = test-audio-correctness + test-audio-training

# Phase 3
make test-stability     # ~4 мин: 20 сессий
make test-decode        # ~15 сек: 20 файлов pixel-сравнение vs PIL

# Go-тесты
make go-test            # = go test ./internal/... ./cmd/datasetfs/...
```

## По файлам

### `test_manifest.py` — целостность манифеста

Проверяет соответствие `metadata.parquet` на диске и того, что реально лежит в
tar-шардах. Цели:
- Размер каждого объекта в манифесте == offset/size в tar-шарде
- ShardID в манифесте == имя файла шарда
- `total_size` шарда в манифесте корректный (см. историю про TotalSize bug)
- Нет дубликатов path/key

**Покрытие**: Imagenette (полный датасет).

### `test_correctness.py` — image correctness (Imagenette)

P0+P1 тесты (по терминологии плана):

- **Completeness** — DFS отдаёт ровно `len(imagefolder)` объектов за эпоху
- **No duplicates** — каждый объект ровно один раз
- **Multi-worker disjoint** — при `num_workers=4` объекты разных воркеров не пересекаются
- **Shuffle разнообразие** — два прогона с разными seed дают разный порядок
- **Byte-hash equivalence** — для каждого path: `sha256(DFS bytes) == sha256(raw file)`
- **Labels per-file** — лейбл, выданный DFS, совпадает с папкой ImageFolder
- **Edge num_workers** — `num_workers ∈ {0, 1, 2, 4, 8}` все корректно
- **Re-init resilience** — несколько `/initialize_loading` подряд, каждая сессия чистая
- **Seed validation** — некорректный seed (None, negative, non-int) → понятная ошибка

**Покрытие**: Imagenette. Не покрывает: rgb_uint8 mode (отдельно в test_decode), mutations, FUSE-mount.

### `test_imagewoof.py` — cross-dataset smoke

Прогоняет ключевые тесты из `test_correctness.py` на Imagewoof. Цель — убедиться,
что прохождение test_correctness — не overfit на одном датасете.

### `test_speech_commands.py` — audio correctness

Те же P0+P1 чек-листы, плюс аудио-специфические:
- **sample_rate** — выданный сэмпл декодится с правильной частотой
- **shape** — после decode/transform тензор имеет ожидаемую форму
- **non-silent** — не все сэмплы — тишина (catch decode-failure-as-zeros)

Использует `soundfile.read(BytesIO)` как `decode_fn` (см. конвенции в [architecture.md](architecture.md#audio)).

### `test_training.py` — training correctness (Imagenette)

Два теста, оба используют `SimpleCNN` + SGD, 3 эпохи:

- **`test_loss_decreases_imagenette`** — на DFS loss падает монотонно (с ±1% запасом на шум), финальный loss ≥ 5% ниже random baseline `ln(num_classes)`. Catches: corrupt data, mislabeled samples, broken collate, shuffle without replacement breaking SGD.
- **`test_loss_parity_with_imagefolder`** — train с тем же seed на DFS и raw ImageFolder, final losses различаются <25%. Catches: tampering с byte content, mis-mapping label→idx.

### `test_speech_commands_training.py`

Аналог для аудио: SimpleCNN на MelSpectrogram-фичах, проверка что loss падает.
Цель — доказать **data-agnostic** работу клиента (decode_fn-точка расширения работает).

### `test_stability.py` — Phase 3 stability gate

Один тест, ~4 мин:

- 20 сессий × 1 эпоха на одном daemon-процессе
- Каждая сессия = свежий `/initialize_loading`
- Метрики: per-epoch throughput (sps), per-epoch daemon RSS (psutil)
- Ассерты:
  - **Throughput retention**: `mean(last 3 epochs) / epoch_2 ≥ 0.85` (catches refcount drift, goroutine leaks, FD leaks)
  - **RSS growth**: `final_rss / epoch_2_rss ≤ 2.0` (catches memory leaks, allocator не освобождает SHM)

**Закрытый gap**: deferred test #1 из [memory:project_deferred_tests](../.claude/projects/-Users-true-danil-12-Graduation-work-DatasetFS/memory/project_deferred_tests.md). Last run: retention 99.76%, growth 1.00×.

### `test_decode.py` — Phase 3 server-side decode

Два теста:

- **`test_rgb_uint8_decode_matches_pil`** — 20 случайных файлов из Imagenette. Для каждого: DFS rgb_uint8 → uint8 HWC ndarray vs `PIL.Image.open + resize(BILINEAR)` → ndarray. Asserts:
  - Same shape (H, W, 3), dtype uint8
  - `mean abs diff < 5/255` per-pixel (среднее по 20 файлам)
  - `p95 diff < 25/255`
  - `max diff < 90/255` (защита от RGB↔BGR swap или stride bugs)
- **`test_rgb_uint8_with_to_tensor_yields_float_chw`** — end-to-end: rgb_uint8 + `transforms.ToTensor()` → правильный float32 CHW [0,1].

**Backend-агностичный**: оба теста проходят и на pure-Go decoder'е, и на libjpeg-turbo. С libjpeg-turbo пиксели ещё ближе к PIL (PIL внутри тоже libjpeg).

## Что покрыто хорошо

- Корректность данных на images + audio
- Multi-worker сценарии (0, 1, 2, 4, 8)
- Re-init поведение daemon'а
- Long-running stability (20 сессий)
- Server-side decode pixel-correctness

## Coverage gaps (deferred tests из memory)

| # | Gap | Зачем нужен | Сложность |
|---|---|---|---|
| ~~1~~ | ~~**Long-running stability**~~ | ~~Refcount drift, FD/goroutine/memory leaks~~ | ~~Done~~ |
| ~~2~~ | ~~**Concurrent loaders failure mode**~~ | ~~Покрыто `tests/test_deferred_gaps.py`: session-specific FIFO paths не дают stale iterator и новой эпохе делить один pipe; старый iterator завершается bounded EOF/error или дочитывает корректные данные~~ | ~~done~~ |
| 3 | **Daemon crash mid-epoch** | Покрыто `tests/test_deferred_gaps.py`: kill daemon во время итерации должен завершиться bounded failure, без бесконечного hang | done |
| 4 | **Cleanup verification** | Покрыто `tests/test_deferred_gaps.py`: после `daemon.stop()` assert'им отсутствие `/tmp/mlfs_*` и `datasetfs_pipe_*` | done |
| ~~5~~ | ~~**FUSE-mount mode**~~ | ~~Покрыто `tests/test_deferred_gaps.py`: POSIX create/read/list/unlink через FUSE mount работает; `Create` обновляет inode metadata после commit в delta shard~~ | ~~done~~ |
| 6 | **Mutations** (`DeleteFile` / `AddDeltaFile`) | Покрыто `tests/test_mutation_consistency.py`: real FUSE writes/deletes + DatasetFS epoch snapshot consistency | done |

Полный контекст: [memory/project_deferred_tests.md](../.claude/projects/-Users-true-danil-12-Graduation-work-DatasetFS/memory/project_deferred_tests.md).

## Соглашения о тестовых фикстурах

`tests/conftest.py`:

- **Session-scoped**: `repo_root`, `data_root`, `daemon_binary` (билдит daemon с cgo+libjpeg-turbo один раз на сессию), `converter_binary`, `imagenette_prepared`, `imagewoof_prepared`, `speech_commands_prepared` (готовят данные идемпотентно — пропускают, если уже на диске).
- **Function-scoped**: `daemon` (поднимает daemon на Imagenette), `daemon_imagewoof`, `daemon_speech_commands`. Поддерживает `.restart()` для тестов, итерирующих несколько эпох. Cleanup `/tmp` в teardown.
- `.pid` property у `DaemonManager` — для psutil-сэмплинга RSS (нужен `test_stability`).

`tests/helpers.py`:

- `imagefolder_index(root)` — `dict[path → class]`. Ключ — `"class/filename"` (см. конвенцию в [architecture.md](architecture.md#imagefolder-и-speech-commands)).
- Прочие хелперы для хеш-сравнения, etc.

## Pattern guidelines (что соблюдать при добавлении тестов)

1. **Spawn-mode picklability**: все функции, передаваемые в DataLoader (`collate_fn`, `transform`, `decode_fn`) — module-level. Никаких lambdas/closures.
2. **Daemon re-init между эпохами**: если тест итерирует несколько раз — вызывать `daemon.restart()` между прогонами. Pipe-state иначе может interfere.
3. **Go-тесты**: `make go-test` (= `go test ./internal/... ./cmd/datasetfs/...`). Прежний висящий `cmd/fuse_daemon/main_test.go` удалён вместе с объединением бинарников.
4. **Timeout-маркер**: длинные тесты помечать `pytest.mark.timeout(N)`. Дефолт pytest-timeout — 0 (нет лимита).
5. **`-s` для прогрессивного вывода**: добавлять `-s` в `pytest` для длинных тестов (stability, training) чтобы видеть epoch-by-epoch прогресс. Уже в Makefile-таргетах.
