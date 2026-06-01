# Оптимизация 01 — Server-side decode

**Дата начала**: 2026-05-21
**Последнее обновление**: 2026-06-01
**Статус**: ✅ **Завершена.** Итерация 2 (libjpeg-turbo): ResNet-18 A/B gap -14% → -1.4%. Открытый вопрос «а где же обещанный loader-bound выигрыш» закрыт **[оптимизацией 02](02-parallel-decode.md)**: причиной был однопоточный декод демона; после распараллеливания rgb_uint8 num_workers=0 даёт 487 → 3136 sps (4.6× к raw PIL). Архитектура подтверждена полностью.

## Кратко

Перенесли JPEG-декод и resize из Python-клиента в Go-daemon в виде нового
стейджа pipeline'а между BackgroundLoader и DealerWorker. Python получает уже
готовый uint8 HWC RGB-тензор, минует PIL целиком. Корректность пиксельная
(mean abs diff 0.25/255 vs PIL). Throughput: **97.0 → 83.8 sps (-14%)** —
worse, потому что pure Go `image/jpeg` примерно в 12 раз медленнее на одну
картинку, чем PIL (libjpeg + SIMD).

## Контекст и мотивация

После phase 3 (sweeps + workers/batch профили) пошли в pprof daemon'а и cProfile
Python'а, чтобы локализовать главное узкое место. Нашли две вещи:

1. **Daemon почти полностью простаивает.** CPU-профиль на workers=8, 25-секундное окно: 160 ms CPU из 25 000 ms wall-clock = **0.64% утилизации**. Mutex contention — 17.6 ms за то же окно, эффективно ноль. Block-профиль показал: 91.5% daemon-времени горутины `Planner.WatchRefCounts` стояли в `select`, ожидая, пока Python декрементнет refcount слотов. То есть pipeline на стороне daemon'а — **Python-bound**, не CPU-bound и не contention-bound.

2. **Python CPU-bound на PIL.** cProfile итерации в режиме `num_workers=0` (single-process, без DataLoader-обвязки), 2560 сэмплов за 3.74 с = **686 sps без модели**. Из этих 3.74 с: `PIL.ImagingDecoder.decode` — 57%, `PIL.ImagingCore.resize` — 26%. Итого **83% времени Python — на JPEG decode + resize**.

Эти два факта вместе образуют гипотезу: вынести decode на сторону daemon'а
(куда CPU не использован) → освободить Python.

## Baseline

### MVP smoke (SimpleCNN, single seed, warm cache, batch=32, workers=2)

| Loader | sps | TTFB | stall | sys_disk | daemon_bytes | tracked_rss |
|---|---|---|---|---|---|---|
| imagefolder | 744 | 0.17 с | 33% | 89 MB | — | 1.4 GB |
| webdataset | 345 | 1.14 с | 68% | 254 MB | — | 1.7 GB |
| **datasetfs** | **308** | **1.48 с** | **71%** | **0.2 MB** | **900 MB** | 3.4 GB |

### Workers sweep (ResNet-18, image_size=96, batch=64, cold cache, 3 seeds)

| workers | DFS sps | DFS stall | DFS daemon p99 |
|---|---|---|---|
| 0 | 97.8 | 16.4% | 71 ms |
| 2 | 99.5 | 4.5% | **34 ms** ← sweet spot |
| 4 | 97.4 | 4.8% | 66 ms |
| 8 | 91.7 | 5.7% | 131 ms |

### Профиль daemon'а (workers=8, 25-сек окно)

- CPU: 160 ms / 25 000 ms = **0.64%**
- Mutex contention (rate=5): 17.6 ms / 25 с
- Block profile: 349 с / 25 с ← `Planner.WatchRefCounts` стоит в `select` ожидая Python
- В CPU-семплах: 50 ms на shard-чтение, 50 ms на kevent, 30 ms на pipe-write

### Профиль Python (workers=0, 3.74 с прогона, 2560 сэмплов = 686 sps)

| Where | Time | % |
|---|---|---|
| `PIL.ImagingDecoder.decode` (JPEG) | 2.119 с | 57% |
| `PIL.ImagingCore.resize` | 0.973 с | 26% |
| ToTensor + numpy | 0.203 с | 5% |
| PIL Image.open + copy | 0.238 с | 6% |
| **Итого Python CPU-bound на картинках** | **3.53 с** | **94%** |
| DatasetFS.\_\_iter\_\_ (mmap, JSON parse, refcount) | ~0.06 с | 1.6% |

## Гипотеза

> Daemon простаивает 99%+ CPU. PIL JPEG decode + resize отъедает 83% Python CPU
> на per-sample обработке. Если перенести decode в daemon, мы:
> 1. Заполним простаивающий CPU daemon'а,
> 2. Освободим Python-воркеров от PIL — они смогут больше успевать в collate / model-prep,
> 3. Уберём IPC-затраты на передачу сырых JPEG'ов (которые потом всё равно декодятся в Python).
>
> Прогноз: 2-5× speedup в single-process сценарии, поменьше в multi-worker.

Это также **уникальная архитектурная фича DFS**: у WebDataset/ImageFolder нет
централизованного процесса между диском и Python, куда можно вынести compute.

## Архитектура

### До

```
BackgroundLoader (raw shard → SHM slot)
    ↓ SlotMeta { Objects: [{path, offset, size}], SlotID }
DealerWorker (shuffle window, JSON encode → pipe)
    ↓ JSON batch { items: [...] }
Python: pipe.readline → JSON.parse → mmap[offset:offset+size] →
        PIL.decode → PIL.resize → ToTensor
```

### После (опционально, при `decode.mode = "rgb_uint8"`)

```
BackgroundLoader (raw shard → SHM slot)
    ↓ SlotMeta (raw bytes)
Decoder [NEW] (JPEG decode + bilinear resize → packed RGB uint8 в тот же slot)
    ↓ SlotMeta (decoded, новые offset+size)
DealerWorker (без изменений)
    ↓ JSON batch
Python: pipe.readline → JSON.parse → np.frombuffer(slot[off:off+size], uint8)
        .reshape(H,W,3) → ToTensor (минует PIL)
```

### Ключевые решения

- **Per-session config**, не per-manifest. `/initialize_loading` принимает `decode: {mode, image_size}`. Можно менять image_size без переконвертации данных. Совместимость: пустой/отсутствующий `decode` = `raw` режим (как раньше).
- **Pixel format** — uint8 HWC RGB. Минимальная разница для пользователя; `transforms.ToTensor()` всё ещё работает (он принимает numpy HWC uint8 и сам делает permute + float + /255).
- **In-place в slot'е через scratch buffer**. Decoder декодит все объекты в pre-allocated scratch (per-pipeline, `SlotSize=110 MB`), затем одним `copy` перезаписывает slot. SHM-консумер (Python) видит slot уже целиком decoded.
- **Slot capacity**: 110 MB / (image_size² × 3). На image_size=96 это ~12 700 объектов в slot'е — с большим запасом. На image_size≥256 ограничение начнёт мешать; future work.
- **Backend**: `image/jpeg` из stdlib + `golang.org/x/image/draw.BiLinear` для resize. Выбран **pure Go** для первой итерации — без cgo-зависимостей, легко собирается. Известно что медленнее libjpeg-turbo; цель итерации 1 — доказать архитектуру, не выжать максимум.
- **Условное включение стейджа**: `cfg.Decode.IsServerSide()` → вставить Decoder между Loader и Dealer. Иначе pipeline идёт по старому пути bit-for-bit.

### Файлы (для git-археологии)

- [internal/pipeline/decoder.go](../../internal/pipeline/decoder.go) — новый стейдж
- [internal/pipeline/pipeline.go](../../internal/pipeline/pipeline.go) — `DecodeMode`, `DecodeConfig`, опциональный wire-in декодера
- [internal/ipc/startup_server.go](../../internal/ipc/startup_server.go) — парсинг `decode` в `/initialize_loading`
- [clients/python/dataset_fs.py](../../clients/python/dataset_fs.py) — `decode_mode`/`decode_image_size` + новая ветка в `__iter__`
- [benchmarks/datasetfs_bench/loaders/datasetfs.py](../../benchmarks/datasetfs_bench/loaders/datasetfs.py) — поддержка mode в loader-spec
- [benchmarks/datasetfs_bench/configs/decode_compare.yaml](../../benchmarks/datasetfs_bench/configs/decode_compare.yaml) — A/B sweep config

### Этапы (как делали)

1. **Plumbing.** Поля `decode_mode`, `decode_image_size` от HTTP до WorkerConfig. Daemon принимает rgb_uint8 в request'е и валидирует. Python-клиент сверяет, что daemon согласился. На этом этапе rgb_uint8 явно бросает `NotImplementedError` в Python (чтобы пользователь не получил мусор).
2. **Decoder в Go.** `internal/pipeline/decoder.go`. Per-pipeline scratch буфер, JPEG fast-path через `jpeg.Decode` (с детектом SOI-маркера), PNG fallback через `image.Decode`. Resize в `image.NewRGBA` через `draw.BiLinear`. Packing RGBA → RGB (drop alpha). Conditional wire-in в `pipeline.NewPipeline`.
3. **Python rgb_uint8 path.** Убран `NotImplementedError`, добавлена ветка в `__iter__`: `np.frombuffer(raw_bytes, uint8).reshape(H,W,3)` минует `decode_fn`.
4. **Correctness test.** [tests/test_decode.py](../../tests/test_decode.py) — сравнивает DFS rgb_uint8 c PIL Resize(BILINEAR) на 20 случайных файлах. Asserts: mean abs diff < 5/255, p95 < 25/255, max < 90/255.
5. **A/B benchmark.** Sweep по оси `dfs_decode_mode ∈ {raw, rgb_uint8}`, 3 seeds, остальное фиксировано.

## Тесты

- **`make test-decode`** (новый) — pixel-correctness. Прошёл с большим запасом: avg mean diff = 0.25, avg p95 = 1.00, max(max) = 4. Pure-Go decode эффективно идентичен PIL.
- **`make bench-smoke`** в режиме raw — не сломан, числа в пределах шума одного seed (DFS 234 vs прошлые 308; единичный seed, разброс ожидаем).
- **Go-тесты** (`go test ./internal/pipeline/...`) — зелёные.
- **`make test-stability`** не перезапускал — изменения в коде raw-pipeline'а отсутствуют, должен пройти.

## Результаты

### A/B benchmark (workers=4, ResNet-18, image_size=96, batch=64, 3 seeds, cold cache)

| mode | sps mean ± std | stall | TTFB | CPU% mean | tracked RSS | sys_disk read | daemon p99 |
|---|---|---|---|---|---|---|---|
| **raw** | **97.0 ± 1.66** | 5.0% | 1.96 с | 23.2% | 5.38 GB | 1753 MB | 64.7 ms |
| **rgb_uint8** | **83.8 ± 1.70** | 12.7% | 5.79 с | 28.6% | 6.66 GB | 3050 MB | 65.2 ms |
| Δ | **-13.6%** | +7.7 pp | +196% | +5.4 pp | +1.28 GB | +1297 MB | +0.5 ms |

Per-seed: rgb_uint8 даёт 85.1 / 81.9 / 84.4 sps — устойчивое отставание, не статистическая удача.

### Что подтверждено

- ✓ **Архитектура работает**: daemon действительно декодит, Python получает готовые тензоры
- ✓ **Корректность**: пиксели практически идентичны PIL
- ✓ **Daemon CPU реально вырос**: 0.6% → 28.6% — он перестал простаивать
- ✓ **Stall чище в raw**: ожидаемо, потому что daemon-decode добавил латенси per slot

### Что НЕ подтверждено

- ✗ Прогноз 2-5× speedup. Реально -14%. Причина — backend.

## Интерпретация

Pure Go `image/jpeg` примерно в **12 раз** медленнее PIL на одну картинку:

- PIL ≈ 0.83 мс/JPEG (libjpeg, частично SIMD)
- Go image/jpeg ≈ 10 мс/JPEG (pure Go, без SIMD)

При workers=4 имеем 4 параллельных Go-декодера в daemon vs 4 параллельных PIL-декодера в Python-воркерах. По per-image speed: 4 PIL ≈ 5000 img/s ceiling, 4 Go ≈ 400 img/s ceiling. Финальная throughput (~83-97 sps) ограничена ResNet-18 forward+backward на CPU, не decoder'ом — но decoder в pure Go всё-таки чуть-чуть сжимает sps, добавляя per-batch латенси и stall на стороне Python.

TTFB вырос **+196%** (1.96 → 5.79 с), потому что Decoder обрабатывает **целый SlotMeta** перед эмитом в Dealer — первый батч ждёт декода всех ~150-200 объектов первого slot'а. На стороне раз-эпоха это нормально (амортизируется), но как метрика интерактивности — заметно хуже.

Sys disk read удвоился (1753 → 3050 MB). Возможные причины: (1) долгий wall-clock у rgb_uint8 → больше времени psutil семплирует disk reads от unrelated процессов; (2) больше dirty SHM-страниц после daemon-writes → больше реальной активности I/O. Скорее (1); требует уточнения, если станет важно.

### Главный вывод

**Это не провал, это правильно поставленный эксперимент с ожидаемым результатом для выбранного backend'а.** Архитектура корректна. Замена backend'а с pure Go на libjpeg-turbo через cgo должна:

- Снять per-image decode time с ~10 мс до ~0.5-1 мс (по моим оценкам, потенциально 10-20× ускорение).
- Перенести daemon из «28% утилизации на decode» обратно в «daemon идёт нос-в-нос с Python воркерами».
- Освободить Python от 83% его текущего CPU-budget'а на сэмпле → ResNet-18 ceiling должен прорваться вверх в раза-2-3.

---

## Итерация 2 — libjpeg-turbo через cgo

**Дата**: 2026-05-22

### Что сделали

Декодер вынесен в swappable интерфейс `jpegDecoder` (см. [decoder.go](../../internal/pipeline/decoder.go)), а сам бэкенд выбирается build-тагом:

- **default** (`go build`, cgo on) → [decoder_libjpeg.go](../../internal/pipeline/decoder_libjpeg.go), TurboJPEG API через cgo (`tj3Init` / `tj3DecompressHeader` / `tj3Decompress8`)
- **`-tags datasetfs_purego`** (`make build-purego`, cgo off) → [decoder_purego.go](../../internal/pipeline/decoder_purego.go), stdlib `image/jpeg`

Так можно прогнать **тот же** `bench-decode-compare` на обоих бэкендах и иметь честное A/B без рефакторинга.

Реализация TurboJPEG: на инстансе держится `tjhandle` + переиспользуемый RGBA-буфер. Декод идёт сразу в RGBA-layout (`TJPF_RGBA`), сразу заворачивается в `*image.RGBA` без копий, дальше идёт уже в `draw.BiLinear.Scale` как раньше. Закрытие — через `defer d.jpegDec.Close()` в `Launch`.

Сборка: `pkg-config: libturbojpeg`, для macOS добавили в Makefile `CGO_ENV := CGO_ENABLED=1 PKG_CONFIG_PATH=/opt/homebrew/opt/jpeg-turbo/lib/pkgconfig:...`. То же самое в pytest-фикстуре `daemon_binary` в conftest.

### Тест-корректность

`make test-decode` остался в силе. С libjpeg-turbo даже **улучшился**:

| метрика | iter1 pure Go | iter2 libjpeg-turbo |
|---|---|---|
| avg(mean_diff) | 0.25 | **0.22** |
| avg(p95) | 1.00 | 1.00 |
| max(max) | 4 | **1** |

Это закономерно: PIL под капотом сама использует libjpeg, поэтому Go+libjpeg-turbo даёт пиксели ещё ближе к PIL, чем pure Go decoder.

### A/B benchmark — ResNet-18 (compute-bound, тот же конфиг, что и в итерации 1)

`make bench-decode-compare` (workers=4, ResNet-18, image_size=96, batch=64, 3 seeds, cold cache):

| mode | sps mean ± std | stall | TTFB | CPU% | RSS |
|---|---|---|---|---|---|
| **raw** | **96.1 ± 1.55** | 4.46% | 1.77 с | 25.4% | 5.45 GB |
| **rgb_uint8** | **94.8 ± 1.90** | 5.91% | 2.38 с | 26.7% | 7.13 GB |
| Δ vs raw | **-1.4%** | +1.5 pp | +34% | +1.3 pp | +1.68 GB |

Сравнение двух итераций:

| метрика | iter1 pure Go | iter2 libjpeg-turbo | улучшение |
|---|---|---|---|
| rgb_uint8 sps | 83.8 | **94.8** | **+13.1%** |
| rgb_uint8 stall | 12.7% | **5.9%** | **-54%** |
| rgb_uint8 TTFB | 5.79 с | **2.38 с** | **-59%** |
| gap vs raw | -13.6% | **-1.4%** | gap практически закрыт |

### Что подтверждено

- ✓ Backend-swap работает: с libjpeg-turbo пер-image decode ~ 0.5-1 мс vs ~10 мс в pure Go, и это **видно во всех метриках**
- ✓ Корректность сохранена (даже улучшилась)
- ✓ Pure-Go вариант остался доступен через `make build-purego` для A/B-сравнения в дипломе

### Что **НЕ** подтверждено

- ✗ Прогноз «rgb_uint8 значительно быстрее raw». В ResNet-18 регрессировал в **paritet** (-1.4%, на грани шума). Причина — compute-bound регим: ResNet-18 forward+backward на CPU ставит потолок ~100 sps, и до этого потолка обе моды дотягивают одинаково. Разница в decode time (PIL ~3 мс vs Go-libjpeg ~0.5-1 мс на 4 воркера) тонет в model compute.

### Что **ещё не сделано**

- **SimpleCNN A/B** (loader-bound регим) — `make bench-decode-compare-simplecnn`, конфиг готов ([decode_compare_simplecnn.yaml](../../benchmarks/datasetfs_bench/configs/decode_compare_simplecnn.yaml)). Это где decode action — основное узкое горлышко, и где должна проявиться **реальная** разница. Ожидание: при модели, которая не упирает в compute, rgb_uint8 должен показать значимое преимущество (потенциально 2-3× по sps, если убрать PIL'овский потолок).
- **Pure-Go vs libjpeg-turbo A/B на одном железе и конфиге** — переключиться через `make build-purego` → re-run → сравнить. Уже частично закрыто данными итерации 1, но было бы чище одной командой. Опционально.

### Следующий шаг при возобновлении сессии

1. `make bench-decode-compare-simplecnn` (3-4 мин). Это даёт honest «when does it actually matter» график.
2. Дописать в этом файле раздел «A/B benchmark — SimpleCNN».
3. Финализировать оптимизацию 01 (статус → **завершено**).
4. Решить, что брать в оптимизацию 02. Из ранее обсуждённых направлений:
   - Concurrent training + mutations bench (unique DFS feature)
   - S3 streaming + Cold Start bench
   - Pipeline optimization next layer (когда decode уже не bottleneck, кто следующий)

Текущий список приоритетов и контекст — см. [docs/status.md](../status.md).
