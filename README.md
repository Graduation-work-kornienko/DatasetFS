# DatasetFS
File System, uses FUSE for easy file reading andmodification, and shared memory and advanced core for Dataloading(while training)

## Инструкция по запуску

1) Запустить файловую систему - `go run ./cmd/datasetfs daemon`
2) Запустить скрипт тестирования - `python main.py`

## Подробнее про скрипт тестирования

Так как в качестве референса мы взяли формат WebDataset, и улучшаем его, то первым этапом было бы логично сравнить их эффективности работы.

Мы создаем два DataLoader. Один принимает в качестве Dataset объект DatasetFS, другой - WebDataset. Изначально берется датасет WebDataset и конвертируется в формат DatasetFS. На двух этих DataLoader запускается обучение одной и той же обычной CNN. Чтобы отслеживать, что результат обучения заметно не ухудшается из-за того, что для DatasetFS мы по сути делали свой stochastic shuffler, мы смотрим, чтобы результаты обучения были в среднем похожи.

TODO: нужны бенчмарки. Нужны оценки по времени, технические характеристики обучения.
