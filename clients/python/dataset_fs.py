import json
import mmap
import struct
from torch.utils.data import IterableDataset
from PIL import Image
import select
import PIL
import io
import torchvision.transforms as transforms
import requests

class DatasetFS(IterableDataset):
    def __init__(self,
                 shm_data_path="/tmp/mlfs_data.bin",
                 shm_refs_path="/tmp/mlfs_refs.bin",
                 pipe_path="/tmp/datasetfs_pipe",
                 transform=None):

        self.shm_data_path = shm_data_path
        self.shm_refs_path = shm_refs_path
        self.pipe_path = pipe_path
        self.timeout_seconds = 0.1

        self.transform = transform or transforms.Compose([
            transforms.ToTensor()
        ])

        requests.get('http://localhost:51409/initialize_loading')

    def _decrement_refcount(self, refs_mmap, slot_id):
        offset = slot_id * 4

        current_val_bytes = refs_mmap[offset : offset+4]
        current_val = struct.unpack("<i", current_val_bytes)[0]

        new_val = current_val - 1
        refs_mmap[offset : offset+4] = struct.pack("<i", new_val)

    def __iter__(self):
        print(f"[Python] Подключение к DatasetFS")

        with open(self.shm_data_path, "r+b") as data_f, \
             open(self.shm_refs_path, "r+b") as refs_f:

            data_mmap = mmap.mmap(data_f.fileno(), 0, access=mmap.ACCESS_READ)
            refs_mmap = mmap.mmap(refs_f.fileno(), 0, access=mmap.ACCESS_WRITE)

            print(f"[Python] Ожидание Pipe {self.pipe_path}")

            with open(self.pipe_path, "r") as pipe:
                print("[Python] Pipe подключен")

                while True:
                    ready_to_read, _, _ = select.select([pipe], [], [], self.timeout_seconds)

                    if not ready_to_read:
                        # Если список пустой, значит прошло timeout_seconds, а данных нет
                        print(f"[Python] Таймаут ({self.timeout_seconds}s): данных нет. Завершаем эпоху.")
                        break

                    # 2. Если данные есть, читаем одну строку
                    line = pipe.readline()

                    if not line:
                        # EOF (конец файла/пайпа)
                        print("[Python] Получен EOF. Завершаем эпоху.")
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        batch_meta = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"[Python] Ошибка парсинга JSON: {e}")
                        continue

                    if not batch_meta.get("items"):
                        print("[Python] Датасет полностью прочитан (Пустой батч). Конец эпохи.")
                        break


                    for item in batch_meta["items"]:
                        slot_id = item["slot_id"]
                        offset = item["offset"]
                        size = item["size"]

                        raw_jpeg = data_mmap[offset : offset + size]

                        # TODO:  nvJPEG
                        try:
                            image = Image.open(io.BytesIO(raw_jpeg))
                            tensor = self.transform(image)

                            result = {"image": tensor}

                            if "meta" in item and item["meta"]:
                                meta_data = item["meta"]

                                if isinstance(meta_data, dict):
                                    result.update(meta_data)
                                elif isinstance(meta_data, list):
                                    result["annotations"] = meta_data
                                else:
                                    result["meta_raw"] = meta_data

                            # decrement counters
                            self._decrement_refcount(refs_mmap, slot_id)

                            yield result
                        except (PIL.UnidentifiedImageError, OSError):
                            continue


            data_mmap.close()
            refs_mmap.close()
