# Оптимизация 02 — Parallel server-side decode

**Дата:** 2026-06-01  
**Статус:** завершена. Закрывает главный bottleneck, оставшийся после [opt 01](01-server-side-decode.md): последовательный decode внутри daemon pipeline.

## Кратко

Opt 01 перенесла JPEG decode + resize из Python в daemon, но decoder был один goroutine на pipeline. В loader-bound режиме это означало: Python больше не тратит время на PIL, но теперь ждет один Go core, который последовательно декодит весь slot.

Opt 02 добавила пул decode workers внутри каждого pipeline. Количество worker'ов задается `decode.parallelism`; 0 означает auto: `max(1, runtime.NumCPU() / num_workers)`.

Исторический headline для isolated loader-bound сценария `num_workers=0`:

| mode | sps | вывод |
|---|---:|---|
| raw, PIL in Python | ~689 | baseline |
| rgb_uint8, sequential daemon decode | ~487 | перенос bottleneck в один Go core |
| rgb_uint8, parallel daemon decode | ~3136 | decode bottleneck снят |

## Bottleneck после opt 01

Профиль `rgb_uint8` до opt 02:

- Python почти все время ждал данные в `select.select`;
- daemon был загружен примерно на один core;
- CPU flamegraph daemon'а был dominated by libjpeg-turbo decode, resize и pack RGB;
- остальные cores простаивали.

Это означало, что архитектурная идея opt 01 верная, но реализация не использовала параллелизм железа.

## Гипотеза

Декод объектов внутри slot'а независим. Так как output каждого объекта имеет фиксированный размер `image_size * image_size * 3`, можно заранее вычислить offset объекта `i` в scratch buffer. Значит, объекты можно декодировать параллельно без shared cursor и без сериализации записи.

Ожидаемый результат: daemon перестанет быть one-core bottleneck'ом, `rgb_uint8` обгонит raw PIL в loader-bound сценариях, а в compute-bound сценариях останется близко к ceiling модели.

## Реализация

Ключевой файл: [internal/pipeline/decoder.go](../../internal/pipeline/decoder.go).

Что сделано:

- постоянный пул `K` decode goroutines на `Decoder`;
- у каждого worker'а свой JPEG decoder handle, потому что TurboJPEG handle не потокобезопасен;
- jobs channel раздает индексы объектов;
- объект `i` пишет output в `scratch[i * perItem : (i+1) * perItem]`;
- после `WaitGroup` результаты собираются в исходном порядке;
- неудачные decode'ы фильтруются, а offsets оставшихся объектов остаются корректными;
- raw path не меняется.

Control plane:

```json
{
  "decode": {"mode": "rgb_uint8", "image_size": 224, "parallelism": 4}
}
```

Benchmark configs используют это как ось `dfs_decode_parallelism`.

## Safe teardown fix

Opt 02 вскрыла старый lifecycle bug: re-init мог отменить session и размэпить allocator, пока loader/decoder/dealer goroutines еще трогали SHM. С несколькими decoder goroutines это стало воспроизводиться как SIGSEGV/use-after-Munmap.

Фикс:

- `Pipeline` получил `sync.WaitGroup`;
- все goroutines, которые касаются SHM или pipe, запускаются через tracked wrapper;
- `Pipeline.Stop()` делает `cancel()` и затем `wg.Wait()`;
- blocking sends/write paths стали ctx-aware;
- dealer закрывает pipe при ctx cancellation, чтобы не висеть на full FIFO.

Позже control plane также стал переиспользовать один shared allocator across sessions и делать `Reset()` refcounts вместо постоянного remap около 1 GB; это уменьшило startup/re-init overhead и снизило риск lifecycle churn.

## Дополнительный latency fix

`Planner.WatchRefCounts` опрашивал slot refcounts раз в 100 ms. Для 9 int32 atomic loads это слишком грубо. Интервал был уменьшен до 2 ms, что снижает latency переиспользования slots без заметной CPU цены.

## Тесты

Покрытие opt 02:

- `internal/pipeline/decoder_test.go` — порядок, offsets, K=1 vs K>1 equivalence, auto-K resolve.
- `go test -race ./internal/pipeline/...` — race safety для pipeline pieces.
- `make test-decode` — pixel correctness после parallel decode.
- `test_pipeline_leak.py` и deferred lifecycle tests — регрессии around early stop/re-init/refcounts.
- `make go-test` — общий Go gate.

## Результаты

### Isolated loader-bound path

| mode | sps | интерпретация |
|---|---:|---|
| raw PIL | ~689 | Python decode bottleneck |
| rgb_uint8 sequential | ~487 | daemon one-core bottleneck |
| rgb_uint8 parallel | ~3136 | decode больше не bottleneck |

### End-to-end K sweep

Исторический `bench-decode-parallelism`, `num_workers=1`, SimpleCNN:

| K | sps mean | stall | вывод |
|---:|---:|---:|---|
| 1 | ~378 | ~46% | sequential bottleneck |
| 2 | ~511 | ~20% | основной выигрыш |
| 4 | ~486 | ~20% | плато |
| 8 | ~492 | ~24% | oversubscription/другой bottleneck |

Главный вывод: в e2e режиме decode bottleneck уходит уже при малом K; дальше ограничивают DataLoader IPC, transform/model compute и общий transport path.

## Текущее состояние

Opt 02 сейчас отражена в архитектуре так:

- `DecodeConfig.Parallelism` в Go;
- `decode_parallelism` в Python client;
- `dfs_decode_parallelism` в benchmark configs;
- auto-K при omitted/0 parallelism;
- safe `Pipeline.Stop()` с join'ом goroutines;
- refcount polling с низкой latency.

## Граница применимости

Parallel decode помогает, когда JPEG decode/resize были bottleneck'ом. Если сценарий compute-bound, например тяжелая модель на CPU/GPU, `rgb_uint8` может быть близок к raw, а не кратно быстрее. В format matrix это поэтому показывается как отдельный loader label `datasetfs-rgb`, а не как универсальная замена `datasetfs`.

Следующий общий bottleneck после opt 02 стал transport daemon→Python; он закрывался в [opt 03](03-pipeline-transport.md).
