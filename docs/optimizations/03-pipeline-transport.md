# Optimization 03 — Транспорт пайплайна (общий путь, не только картинки)

## Кратко

Заменили JSON-over-pipe между демоном и Python-клиентом на компактный бинарный
length-prefixed фрейм; добавили zero-copy чтение SHM (memoryview) и батч-декремент
refcount (одна запись на слот вместо одной на сэмпл). Попутно закрыт латентный баг:
пропущенные сэмплы (decode→None / mismatch / transform-исключение) не декрементили
refcount → слот не переиспользовался. Чистый транспорт (num_workers=0, тёплый кэш)
вырос **144.5k → ~205k sps (+~42%, 1.4×)**; реалистичный аудио-путь (soundfile)
**17.6k → 19.3k sps (+~10%)**.

## Контекст и мотивация

Opt 01/02 перенесли JPEG-декод в демон. Но это помогает **только изображениям**.
Для аудио (`soundfile.read`) и любого типа, который нельзя декодировать на сервере,
декод дёшев — и тогда доминирующей становится та часть пайплайна, что **одинакова для
всех типов данных**: транспорт метаданных daemon→Python и работа Python на сэмпл.

Профиль raw-режима на **изображениях** (`profiling/raw_w0_20260601T194740`) это
маскирует: PIL `decode`+`resize` = 3.0s из 3.7s, а `dataset_fs.py:__iter__` self-time
= 0.06s. Стоит убрать тяжёлый декод (аудио) — и видно настоящую цену транспорта.

Это «opt 03 / следующий слой пайплайна» из роадмапа. Три decode-независимые статьи
расхода, которые мы атаковали:

1. **JSON-over-pipe** — на каждый батч `encoding/json` в Go + `json.loads` в Python,
   плюс построение per-item dict. Поля `c_id`(ShardID) и `deleted` (всегда false —
   снапшот их отфильтровал) — мёртвый груз.
2. **Копия из SHM** — `bytes(data_mmap[off:off+size])` на каждый сэмпл (для
   rgb_uint8 — лишняя полная копия буфера перед `np.frombuffer`).
3. **Per-sample refcount** — `struct.pack` в `refs_mmap` на каждый сэмпл.

Чтобы это **измерить** (бенч был только image), добавлен аудио-лоадер (Speech
Commands, soundfile → log-mel) — он же график G8 (общность типов данных).

## Baseline (ДО)

`profiling/raw_transport_bench.py`, `data/formats/speech_commands/datasetfs`,
num_workers=0, тёплый кэш, 12800 сэмплов, macOS dev (arm64):

| Вектор | sps (медиана) | Что меряет |
|---|---|---|
| `noop` (декод-заглушка) | **144.5k** | чистый транспорт (потолок) |
| `soundfile` (реальный аудио-декод) | **17.6k** | реалистичный «дешёвый-декод» путь |

## Гипотеза

При дешёвом декоде per-sample стоимость определяется транспортом. Бинарный фрейм
(без рефлексии JSON, без мёртвых полей, с одним векторизованным `np.frombuffer` по
колоночному блоку) + zero-copy SHM + батч-refcount поднимут потолок транспорта;
выигрыш виден именно в transport-bound режиме (аудио), а не на image (там стена —
PIL/мел/модель). Falsifiable: если транспорт не был узким местом, `noop` sps не
вырастет.

## Архитектура / реализация

**Бинарный wire-формат** (little-endian), один `Write` в трубу на батч.
Источник правды: `internal/pipeline/dealer.go` `encodeFrame` ↔
`clients/python/dataset_fs.py` `ITEM_DTYPE`/парсер.

```
HEADER    magic u32=0x44465331 | total_len u32 | generation u64 | item_count u32 | blob_len u32
COLUMNAR  item_count × 28 байт (SoA): slot_id i32, offset i64, size i64, path_len u32, meta_len u32
BLOBS     по item, в порядке: path (utf8), затем meta (raw JSON; может быть 0)
```

- `total_len` считается ПОСЛЕ 8-байтового префикса (magic+total_len): Python читает
  8 байт, затем ровно `total_len`. Пустой фрейм (`item_count==0`) = конец эпохи.
- **Go**: переиспользуемый `bytes.Buffer` (dealer однопоточный на воркера), ручные
  `binary.LittleEndian.PutUintXX` (без рефлексивного `binary.Write`). Трюк с
  закрытием трубы по ctx-cancel сохранён.
- **Python**: труба открыта `os.open(..., O_RDONLY)`, `_read_exact` поверх `os.read`;
  `select` гейтит ПЕРВЫЙ байт фрейма (сохраняет 30s idle-timeout / конец эпохи).
  Колоночный блок парсится одним `np.frombuffer(body, ITEM_DTYPE, count, offset=16)`
  — без per-item `struct.unpack`. `json.loads(meta)` только когда `meta_len>0`.
  `assert ITEM_DTYPE.itemsize == 28` ловит дрейф протокола Go↔Python.

**Zero-copy SHM**: `view = memoryview(data_mmap)[off:off+size]` отдаётся в decode.
Честно: реальный выигрыш — на `rgb_uint8` (`np.frombuffer(view)` теперь без лишней
копии); для PIL/soundfile `io.BytesIO` всё равно копирует (но мы убрали явный
пред-`bytes()`, так что raw-путь — на одну копию меньше). Контракт безопасности:
sub-view освобождается (`view.release()`) **до** yield — иначе брошенный на
`GeneratorExit` генератор оставил бы живой экспорт, и `mmap.close()` падает
(`BufferError`). decode/transform материализуют owned-объект до release.

**Батч-refcount + фикс утечки**: `consumed[slot_id]` инкрементируется в НАЧАЛЕ цикла
по каждому item — даже пропущенному — затем одна запись на слот после фрейма
(`_decrement_refcount_by`). Сумма по фреймам = числу объектов слота, так что
планнер видит 0 корректно. Раньше пропуск (None/mismatch/исключение) не декрементил
→ слот с любым пропуском не переиспользовался (стена при shards>slots). Теперь
учитываются все.

**Аудио-бенч**: `SimpleCNN(in_channels=1)` (модель `simplecnn_audio`),
`_common.audio_decode_fn`+`audio_melspec_transform` (soundfile→log-mel, lazy
MelSpectrogram per-process для spawn), `modality` в spec лоадера и в `single_run`,
конфиг `configs/audio_speech_commands.yaml`, таргет `make bench-audio`.

## Тесты

- **Go**: `make go-test` (cgo) зелёный; `go test -race ./internal/pipeline/...`
  зелёный; `build-purego` собирается; `gofmt`/`go vet` чисто.
- **Корректность (wire-гейт, сквозь клиент)**: `test-correctness` (15), `test-imagewoof`
  (2), `test-audio` correctness (8), `test-decode` (2, rgb-пиксели = PIL),
  `test-training` (2, loss-parity DFS vs ImageFolder) — все зелёные. Эти тесты
  проходят только если бинарный фрейм + батч-refcount байт-корректны.
- **Регрессия утечки**: новый `tests/test_pipeline_leak.py` — decode_fn пропускает
  половину; на датасете с shards>slots эпоха обязана пройти целиком (выживает ≈
  половина). На старом коде слоты бы залипли. Зелёный (2.8s).
- Пред-существующий, НЕ связанный с opt 03: `test_manifest_imagenette` падает, т.к.
  DatasetFS использует `metadata.parquet`; тесты должны читать Parquet manifest
  (наследие parquet-миграции; данные/тест мои правки не трогали).

## Результаты (ПОСЛЕ)

| Вектор | ДО | ПОСЛЕ | Δ |
|---|---|---|---|
| `noop` (чистый транспорт) | 144.5k | **~205k** | **+~42% (1.4×)** |
| `soundfile` (аудио-декод) | 17.6k | **~19.3k** | +~10% |
| e2e аудио-обучение (mel+SimpleCNN, w=4, 35 классов) | — | ~1130–1385 sps | (transport уже не стена) |

## Интерпретация

Гипотеза подтверждена: потолок транспорта вырос в transport-bound режиме (`noop`
+42%) — JSON-кодек/парс и per-item работа были реальной ценой, скрытой за PIL на
картинках. На `soundfile` выигрыш меньше (+10%), т.к. сам аудио-декод уже забирает
основную долю; на полном e2e-обучении (mel-спектр + fwd/bwd) узким местом становится
compute, а не транспорт — это честная граница применимости: opt 03 ускоряет именно
тот режим, где живут «дешёвые-декод» типы данных. Побочно закрыт slot-leak (важно
для корректности под пропусками, а не только для скорости).

## Следующий шаг

- Профиль раздельных вкладов (JSON vs SHM-копия vs refcount) — сейчас они сняты
  суммарно; для диплома можно изолировать через флаги микробенча.
- Multi-worker замер транспорта (sweep по num_workers) — текущие цифры w=0.
- Кандидаты на opt 04 (после ре-замера headline): pipe-write батчинг (несколько
  фреймов за один write), либо перенос мел-спектра/ресемпла в демон по аналогии с
  rgb_uint8 (server-side audio decode) — закрыло бы и e2e-режим.
