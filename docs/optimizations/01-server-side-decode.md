# Оптимизация 01 — Server-side decode

**Дата:** 2026-05-21 → 2026-05-22  
**Статус:** завершена. Исторически это первая оптимизация data path: JPEG decode + resize вынесены из Python в Go-daemon. Финальный практический выигрыш раскрыт только вместе с [opt 02](02-parallel-decode.md), где decode стал параллельным.

## Кратко

До opt 01 DatasetFS отдавал Python raw JPEG bytes. Python worker делал `PIL.Image.open`, resize и `ToTensor`. Профиль показал, что Python тратит большую часть времени именно на PIL decode/resize, а Go-daemon почти простаивает. Мы добавили optional stage в pipeline:

```text
BackgroundLoader -> Decoder -> Dealer
```

В режиме `decode_mode="rgb_uint8"` daemon декодирует JPEG, resize'ит картинку и кладет packed RGB uint8 HWC bytes обратно в SHM slot. Python пропускает PIL и делает `np.frombuffer(...).reshape(H,W,3)` перед transform.

## Исходный bottleneck

Профиль raw path на изображениях:

- daemon CPU utilization был около 0.6% в 25-секундном окне;
- daemon в основном ждал refcount decrement от Python;
- Python `num_workers=0` тратил большую часть времени на `PIL.ImagingDecoder.decode` и `PIL.ImagingCore.resize`;
- `DatasetFS.__iter__` без decode занимал малую долю self-time.

Гипотеза: если перенести JPEG decode/resize на сторону daemon'а, мы загрузим простаивающий Go process и освободим Python worker'ов.

## Реализация

Затронутые части:

- [internal/pipeline/pipeline.go](../../internal/pipeline/pipeline.go) — `DecodeMode`, `DecodeConfig`, conditional decoder stage.
- [internal/pipeline/decoder.go](../../internal/pipeline/decoder.go) — server-side decode orchestration.
- [internal/pipeline/decoder_libjpeg.go](../../internal/pipeline/decoder_libjpeg.go) — default libjpeg-turbo backend через cgo.
- [internal/pipeline/decoder_purego.go](../../internal/pipeline/decoder_purego.go) — pure-Go fallback под build tag `datasetfs_purego`.
- [internal/control/server.go](../../internal/control/server.go) — `decode` block в `/initialize_loading`.
- [clients/python/dataset_fs.py](../../clients/python/dataset_fs.py) — `decode_mode`, `decode_image_size`, rgb_uint8 fast path.
- [benchmarks/datasetfs_bench/loaders/datasetfs.py](../../benchmarks/datasetfs_bench/loaders/datasetfs.py) — прокидывание decode mode из benchmark configs.

Запрос клиента:

```json
{
  "num_workers": 4,
  "decode": {"mode": "rgb_uint8", "image_size": 224}
}
```

Ключевые решения:

- decode policy per session, а не per dataset;
- pixel format: uint8 HWC RGB, чтобы `transforms.ToTensor()` продолжал работать;
- packed output fixed-size для каждого объекта: `image_size * image_size * 3`;
- raw path остается default и нужен для аудио/текста/произвольных bytes;
- cgo/libjpeg-turbo default, pure-Go fallback для окружений без системной зависимости.

## Итерация 1: pure Go backend

Первая реализация использовала `image/jpeg` и `golang.org/x/image/draw.BiLinear`. Она доказала correctness и архитектуру, но была медленнее PIL/libjpeg.

Исторический результат ResNet-18 A/B:

| mode | sps | вывод |
|---|---|---|
| raw | ~97 | baseline |
| rgb_uint8 pure-Go | ~84 | примерно -14%, backend слишком медленный |

Главный вывод: перенос compute в daemon работает, но pure Go JPEG не подходит для performance path.

## Итерация 2: libjpeg-turbo

Бэкенд заменен на libjpeg-turbo через cgo. Makefile выставляет `CGO_ENABLED=1` и `PKG_CONFIG_PATH` для Homebrew `jpeg-turbo`; pure-Go fallback остался через `make build-purego`.

Исторический результат ResNet-18 A/B:

| mode | sps mean | stall | TTFB | вывод |
|---|---:|---:|---:|---|
| raw | ~96 | ~4.5% | ~1.8s | baseline |
| rgb_uint8 libjpeg-turbo | ~95 | ~5.9% | ~2.4s | gap почти закрыт |

ResNet-18 на CPU оказался compute-bound: даже после ускорения decode обе моды упирались в model compute около 100 sps. Поэтому opt 01 сама по себе не дала headline speedup, но закрыла архитектурную часть.

## Корректность

Основная проверка: `make test-decode`.

Что проверяется:

- output shape `(H, W, 3)`;
- dtype `uint8`;
- pixel diff относительно `PIL.Image.open + resize(BILINEAR)`;
- `transforms.ToTensor()` получает корректный float CHW tensor.

Тест должен проходить и с libjpeg-turbo, и с pure-Go decoder. Небольшой pixel diff ожидаем из-за различий JPEG/resize implementations.

## Почему понадобилась opt 02

После libjpeg-turbo оставался неочевидный вопрос: почему loader-bound сценарий не показывает обещанный большой выигрыш? Профиль показал, что decode stage был последовательным на pipeline: один decoder goroutine обрабатывал весь slot. Это переносило bottleneck из Python PIL в один Go core.

[Opt 02](02-parallel-decode.md) решила именно эту проблему: decode стал parallel внутри pipeline, появился `decode.parallelism`, а loader-bound микросценарий показал многократный рост `rgb_uint8` относительно последовательного decode и raw PIL.

## Текущее состояние

В текущей архитектуре opt 01 дала:

- public API `decode_mode="rgb_uint8"` в Python client;
- `decode` config в `/initialize_loading`;
- optional decoder stage в Go pipeline;
- default fast JPEG backend через libjpeg-turbo;
- pure-Go fallback;
- основу для `datasetfs-rgb` в format matrix.

Opt 01 не следует читать как финальный speedup claim. Финальный decode story = opt 01 + opt 02: сначала вынесли decode в daemon, затем распараллелили его.
