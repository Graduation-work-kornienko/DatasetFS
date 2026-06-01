# Оптимизация 02 — Параллельный server-side decode

**Дата**: 2026-06-01
**Статус**: ✅ Завершена. Подтверждена количественно (микро-профиль + end-to-end sweep).

## Кратко

Декод-стейдж демона (opt 01) был **однопоточным на пайплайн** — один
`Decoder`-горутин последовательно декодил все объекты слота. При малом числе
воркеров демон декодил на одном ядре и не успевал кормить даже одного
Python-консумера, поэтому `rgb_uint8` оказывался **медленнее** клиентского PIL
(см. opt 01: гипотеза «2-3× в loader-bound» не подтвердилась). Мы распараллелили
декод на пул из K воркер-горутин внутри каждого пайплайна.

**Headline (num_workers=0, изолирует декод-стейдж):**

| режим | sps | комментарий |
|---|---|---|
| raw (PIL в Python) | 689 | baseline |
| rgb_uint8 ДО (последовательный) | **487** | медленнее raw |
| rgb_uint8 ПОСЛЕ (параллельный, K=12) | **3136** | **6.4× к ДО, 4.6× к raw** |

Заодно: устранён латентный **use-after-Munmap SIGSEGV** на teardown сессии и
снижен интервал опроса refcount'ов слотов 100 мс → 2 мс.

## Контекст и мотивация

Opt 01 вынесла JPEG decode + resize в демон (`rgb_uint8`). Архитектура работала,
но прогноз «2-3× speedup в loader-bound» не подтвердился: на ResNet-18 (compute-
bound) получили паритет, а loader-bound сценарий не был замерен. Я профилировал
именно loader-bound регим (`num_workers=0`, single-process, чистая итерация без
модели):

| режим | num_workers=0 |
|---|---|
| raw (PIL decode в Python) | 689 sps |
| rgb_uint8 (decode в демоне) | **487 sps** ← медленнее! |

Декод **убрал 85% Python-работы** (PIL), но end-to-end стал **медленнее**.
Профиль объяснил почему:

- **Python простаивает 97%** времени в `select.select`, ожидая данных от демона.
- **Демон загружен на 84% ОДНОГО ядра** (при 12 ядрах в системе). Разбивка CPU
  демона: libjpeg-turbo decode 52%, resize (`draw.scaleX/Y_RGBA`) 18%, packRGB
  11.6%, isJPEG 12.8% (артефакт first-touch SHM-страниц, сама функция читает
  3 байта).

Узкое место сместилось из Python в **однопоточный декод демона**. Один
`Decoder`-горутин на пайплайн ([decoder.go](../../internal/pipeline/decoder.go),
старый `decodeSlot`) обрабатывал объекты строго последовательно. При
`num_workers=0/1` (один пайплайн) это один занятый core против 11 простаивающих.

## Гипотеза

> Декод каждого объекта независим, а упакованный выход фиксированного размера
> (`perItem = ImageSize²·3`), поэтому объект `i` всегда ложится в `scratch[i·perItem]`
> без сериализации. Значит декод слота можно раздать пулу горутин. Демон начнёт
> использовать простаивающие ядра, и `rgb_uint8` наконец обгонит raw в loader-bound
> региме — подтвердив исходную гипотезу opt 01, которая упиралась лишь в
> однопоточность бэкенда.

## Архитектура / реализация

### Параллельный декодер ([decoder.go](../../internal/pipeline/decoder.go))

- **Постоянный пул K воркер-горутин** на каждый `Decoder`, создаётся в
  `NewDecoder`, живёт всю сессию (амортизирует cgo `tj3Init` и рост RGBA-буфера —
  слот ~1000 объектов переиспользует пул ~1000 раз).
- `decodeWorker{ jpegDec, resized }` — у каждого **свой** TurboJPEG-handle (handle
  не потокобезопасен) и свой resize-буфер. Внутри воркера `Decode→Scale→pack`
  остаётся последовательным (сохраняет инвариант алиасинга `buf` в
  [decoder_libjpeg.go](../../internal/pipeline/decoder_libjpeg.go)).
- **Партиционирование** — буферизованный канал `jobs` индексов объектов; K воркеров
  тянут из него (work-stealing балансирует крупные JPEG лучше, чем `i%K`-страйпинг).
- **Фиксированные оффсеты с дырами**: объект `i` → `scratch[i·perItem]`. Никакого
  курсора → ноль сериализации. Неудачные декоды оставляют `results[i]=nil` (дыру);
  Python читает каждый объект по своему `(offset,size)`, дыры безвредны.
- **Детерминизм сохранён**: после `wg.Wait()` собираем выживших в исходном порядке
  слота (фильтр `nil`). Refcount = число выживших, совпадает с декрементами Python.
- **Бэкенды не тронуты**: `decoder_libjpeg.go`/`decoder_purego.go` без изменений;
  `newJPEGDecoder()` вызывается K раз, по инстансу на воркера. Оба build-тега собираются.

### Параллелизм-кноб ([pipeline.go](../../internal/pipeline/pipeline.go), [startup_server.go](../../internal/ipc/startup_server.go))

`DecodeConfig.Parallelism` (0 = auto). `resolveParallelism` = `max(1, NumCPU/NumWorkers)`,
чтобы суммарно `NumWorkers·K` не пересабскрайбило ядра. Прокидывается через
`/initialize_loading` → `decode.parallelism` (и в bench-ось `dfs_decode_parallelism`).

### Дешёвый выигрыш B — refcount poll 100 мс → 2 мс ([planner.go](../../internal/pipeline/planner.go))

`WatchRefCounts` опрашивал refcount'ы слотов раз в 100 мс — латентность
переиспользования слота до 100 мс на цикл. `ReadRefCount` — atomic load по ≤9
int'ам, поэтому 2 мс почти бесплатны по CPU и убирают latency-floor (помогает и
raw, и rgb).

### Побочно: фикс use-after-Munmap SIGSEGV (teardown-гонка)

При прогоне K-sweep демон падал с `SIGSEGV` в `tj3Decompress8`. Корень: re-init
сессии (`session.stop()`) делал `alloc.Close()` (**Munmap**), а `Pipeline.Stop()`
лишь отменял ctx и **не ждал** горутин. Если предыдущая эпоха не дочитана (обучение
рвёт её на `max_batches`), decode/loader-воркеры ещё читали/писали mmap → доступ к
освобождённой памяти. Баг латентно существовал и до opt 02 (один decoder-горутин),
но K воркеров + обучение вскрыли его надёжно.

Фикс: `Pipeline` получил `sync.WaitGroup`, все горутины (planner/loader/decoder/
dealer + scheduling-горутина) трекаются, `Stop()` теперь **блокируется до их
завершения** перед Munmap. Чтобы join не зависал, сделаны ctx-aware блокирующие
отправки в [background_loader.go](../../internal/pipeline/background_loader.go) и
добавлен ctx-watcher, закрывающий pipe в
[dealer.go](../../internal/pipeline/dealer.go) (иначе `encoder.Encode` висит на
полном pipe, если консумер ушёл рано).

## Тесты

- **[decoder_test.go](../../internal/pipeline/decoder_test.go)** (новый):
  `decodeSlot` сохраняет порядок при ошибке в середине, оффсеты = `index·perItem`;
  K=1 и K=4 дают **побайтово идентичный** упакованный выход; auto-K резолв.
- **Race-детектор** зелёный на обоих тегах: `go test -race ./internal/pipeline/...`
  (cgo) и `-race -tags datasetfs_purego`.
- **`make test-decode`**: пиксели vs PIL — avg mean diff **0.23**, max **1**
  (идентично opt 01; параллелизм не меняет пиксели).
- Teardown-репро (ранний break + 10× re-init): 0 SIGSEGV после фикса.

## Результаты

### Микро-профиль (num_workers=0, изолирует декод-стейдж)

| режим | sps | daemon CPU | Python `select.select` |
|---|---|---|---|
| raw (PIL) | 689 | 0.3% (idle) | — |
| rgb_uint8 ДО | 487 | 84% одного ядра | **97% (idle, ждёт демон)** |
| **rgb_uint8 ПОСЛЕ (K=12)** | **3136** | **~2.5+ ядер** | **69%** |

- **6.4× к последовательному rgb, 4.6× к raw PIL.** Гипотеза opt 01 не просто
  подтверждена — превышена.
- packRGB в профиле ПОСЛЕ упал до 3% (был 11.6% одного ядра) — распараллеливание
  размазало его по ядрам, поэтому отказ от TJPF_RGB-оптимизации (cheap-win A) был
  оправдан.

### End-to-end K-scaling (num_workers=1, SimpleCNN, warm cache, 3 seeds)

Конфиг [decode_parallelism_sweep.yaml](../../benchmarks/datasetfs_bench/configs/decode_parallelism_sweep.yaml).

| K | sps mean ± std | stall% | sys CPU% |
|---|---|---|---|
| 1 | 378.1 ± 4.5 | 45.7 | 16.3 |
| **2** | **511.1 ± 1.4** | **20.2** | 23.8 |
| 4 | 486.5 ± 7.6 | 19.6 | 27.2 |
| 8 | 492.2 ± 3.3 | 23.7 | 27.2 |

- **K=1 → K=2: +35% sps, stall 46% → 20%.** K=1 воспроизводит до-оптимизационный
  последовательный декод.
- Плато при K≥2: с одним Python-консумером (num_workers=1) узкое место уходит из
  декода в DataLoader-IPC + SimpleCNN compute. Декод перестал быть горлышком —
  цель достигнута.

## Интерпретация

В loader-bound региме потолок определяется тем, успевает ли демон отдавать
декодированные тензоры быстрее, чем консумер их потребляет. Однопоточный декод
(~500 img/s на ядро) проигрывал даже 4 PIL-воркерам Python. Параллельный пул
поднимает потолок демона до `K × (ядро)` — в num_workers=0 это выливается в 6.4×,
в num_workers=1 эффект упирается в IPC одного консумера уже при K=2.

Главный архитектурный вывод для диплома: **централизованный демон DFS — это место,
куда можно вынести и отмасштабировать compute**, чего нет у WebDataset/ImageFolder
(нет процесса между диском и Python). Opt 01 показала возможность, opt 02 —
масштабируемость.

## Следующий шаг

- Опционально: cheap-win A (TJPF_RGB, убрать packRGB) — отложено, после
  распараллеливания pack не в топе профиля (3%). Брать только если повторный
  профиль покажет иначе.
- Opt 03 — кандидаты в [docs/status.md](../status.md): concurrent training +
  mutations bench (уникальная фича DFS), либо следующий слой пайплайна (JSON-over-
  pipe в dealer, SHM-memcpy в Python).
