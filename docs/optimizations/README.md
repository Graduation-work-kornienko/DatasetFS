# Журнал оптимизаций DatasetFS

Папка фиксирует историю производительных изменений DatasetFS: что было bottleneck'ом, какая была гипотеза, что изменили в архитектуре, чем проверили корректность и какие получили числа.

Этот журнал не заменяет [architecture.md](../architecture.md). Архитектура описывает текущее состояние, а файлы здесь объясняют, почему система пришла именно к нему.

## Как читать

Рекомендуемый порядок:

1. [01-server-side-decode.md](01-server-side-decode.md) — почему появился optional decode stage в daemon.
2. [02-parallel-decode.md](02-parallel-decode.md) — почему decode stage стал parallel и почему одного libjpeg-turbo backend'а было недостаточно.
3. [03-pipeline-transport.md](03-pipeline-transport.md) — почему JSON-over-pipe заменен на binary frame и почему это важно для аудио/cheap-decode данных.

## Текущее состояние optimizations

| # | Оптимизация | Статус | Текущее влияние на архитектуру |
|---|---|---|---|
| [01](01-server-side-decode.md) | Server-side JPEG decode + resize | Завершена | В pipeline есть optional `Decoder` stage; client поддерживает `decode_mode="rgb_uint8"`; default build использует libjpeg-turbo через cgo |
| [02](02-parallel-decode.md) | Parallel server-side decode | Завершена | `decode.parallelism` прокидывается через `/initialize_loading`; auto-K = `NumCPU/NumWorkers`; `Pipeline.Stop()` join'ит goroutines до allocator teardown |
| [03](03-pipeline-transport.md) | Binary wire + zero-copy SHM view + batched refcount | Завершена | FIFO protocol теперь binary length-prefixed frame, не JSON; Python парсит columnar block через `np.frombuffer`; refcount уменьшается батчем per slot |

## Главные выводы

- Opt 01 подтвердила архитектурную идею: daemon может выполнять compute между storage и Python, чего нет у ImageFolder/WebDataset.
- Pure Go JPEG был слишком медленным; libjpeg-turbo закрыл gap в compute-bound ResNet-18 сценарии до почти паритета.
- Opt 02 показала, что server-side decode раскрывается только после parallelism: последовательный daemon decode переносил bottleneck из Python в один Go core.
- Opt 03 показала, что после устранения image decode bottleneck следующий общий слой — transport. Binary frame особенно важен для аудио и других cheap-decode payload'ов.
- Побочные correctness fixes оказались не менее важны, чем speedups: join goroutines before unmap, skipped-sample refcount accounting, session-specific FIFO paths.

## Связанные команды

```bash
make test-decode
make test-pipeline-leak
make bench-decode-compare
make bench-decode-compare-simplecnn
make bench-decode-parallelism
make bench-audio
```

## Шаблон для будущих оптимизаций

Новый файл `NN-short-name.md` должен отвечать на вопросы:

1. Что было измерено как bottleneck?
2. Какая falsifiable-гипотеза была проверена?
3. Что изменилось в data path?
4. Какие тесты доказывают корректность?
5. Какие числа до/после?
6. Где граница применимости результата?
7. Что стало следующим bottleneck'ом?
