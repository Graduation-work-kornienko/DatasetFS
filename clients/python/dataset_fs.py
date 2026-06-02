import io
import json
import mmap
import select
import struct

import numpy as np
import requests
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset, get_worker_info


MAX_WORKERS = 9

# Server-side decode modes — must match pipeline.DecodeMode in the Go daemon.
DECODE_RAW = "raw"            # legacy: daemon serves raw JPEG/etc bytes
DECODE_RGB_UINT8 = "rgb_uint8"  # daemon serves pre-decoded uint8 HWC RGB


def _default_decode(raw_bytes: bytes):
    """Default decoder: bytes → PIL.Image (for image datasets).

    Returning None signals that this sample should be skipped (decode failed).
    """
    from PIL import Image, UnidentifiedImageError
    try:
        return Image.open(io.BytesIO(raw_bytes))
    except (UnidentifiedImageError, OSError):
        return None


class DatasetFS(IterableDataset):
    """Streams samples from a running DatasetFS daemon via shared memory + named pipes.

    Audio/non-image use:
        pass a custom `decode_fn(raw_bytes) -> intermediate` and an optional
        `transform(intermediate) -> tensor`. Default decode_fn opens PIL images.

    Server-side decode (decode_mode="rgb_uint8"): the daemon JPEG-decodes and
    resizes to (decode_image_size, decode_image_size, 3) uint8 HWC. Profiling
    shows PIL decode+resize was ~83% of per-sample Python time while daemon CPU
    was 0.6% utilized, so offloading is a large net win for image datasets.
    Custom decode_fn is ignored in this mode (server already decoded).

    The yielded dict has at minimum {"image": tensor or transformed value, "path": str},
    plus any object-level metadata fields (e.g., "label") emitted by the converter.
    """

    def __init__(self,
                 num_workers=0,
                 seed=None,
                 shm_data_path="/tmp/mlfs_data.bin",
                 shm_refs_path="/tmp/mlfs_refs.bin",
                 pipe_path_template="/tmp/datasetfs_pipe_{worker_id}",
                 decode_fn=None,
                 transform=None,
                 timeout_seconds=30.0,
                 daemon_url="http://localhost:51409",
                 decode_mode=DECODE_RAW,
                 decode_image_size=None,
                 decode_parallelism=0):
        effective_workers = max(num_workers, 1)
        if effective_workers > MAX_WORKERS:
            raise ValueError(
                f"num_workers={num_workers} exceeds MAX_WORKERS={MAX_WORKERS} "
                f"(daemon has 9 shared-memory slots to partition)"
            )
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError(f"seed must be a non-negative int or None, got {seed!r}")
        if decode_mode not in (DECODE_RAW, DECODE_RGB_UINT8):
            raise ValueError(
                f"decode_mode must be one of {DECODE_RAW!r}/{DECODE_RGB_UINT8!r}, "
                f"got {decode_mode!r}"
            )
        if decode_mode == DECODE_RGB_UINT8:
            if not isinstance(decode_image_size, int) or decode_image_size <= 0:
                raise ValueError(
                    "decode_mode='rgb_uint8' requires decode_image_size as a "
                    f"positive int, got {decode_image_size!r}"
                )

        self.num_workers = num_workers
        self.seed = seed
        self._effective_workers = effective_workers
        self.shm_data_path = shm_data_path
        self.shm_refs_path = shm_refs_path
        self.pipe_path_template = pipe_path_template
        self.timeout_seconds = timeout_seconds
        self.daemon_url = daemon_url
        self.decode_mode = decode_mode
        self.decode_image_size = decode_image_size
        self.decode_parallelism = decode_parallelism

        self.decode_fn = decode_fn or _default_decode
        self.transform = transform or transforms.Compose([
            transforms.ToTensor()
        ])

        payload = {"num_workers": effective_workers}
        if seed is not None:
            payload["seed"] = seed
        if decode_mode != DECODE_RAW:
            payload["decode"] = {"mode": decode_mode, "image_size": decode_image_size}
            if decode_parallelism and decode_parallelism > 0:
                payload["decode"]["parallelism"] = decode_parallelism

        resp = requests.post(
            f"{daemon_url}/initialize_loading",
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        # Verify the daemon agreed to the mode we asked for; otherwise a stale
        # daemon could silently downgrade and we'd misinterpret SHM contents.
        try:
            ack = resp.json()
        except ValueError:
            ack = {}
        ack_mode = (ack.get("decode") or {}).get("mode", DECODE_RAW)
        if ack_mode != decode_mode:
            raise RuntimeError(
                f"daemon acknowledged decode_mode={ack_mode!r} but client asked for "
                f"{decode_mode!r}. Likely a daemon version mismatch — rebuild it."
            )

    def _decrement_refcount(self, refs_mmap, slot_id):
        offset = slot_id * 4

        current_val_bytes = refs_mmap[offset : offset+4]
        current_val = struct.unpack("<i", current_val_bytes)[0]

        new_val = current_val - 1
        refs_mmap[offset : offset+4] = struct.pack("<i", new_val)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        pipe_path = self.pipe_path_template.format(worker_id=worker_id)

        print(f"[Python worker={worker_id}] Подключение к DatasetFS pipe={pipe_path}")

        with open(self.shm_data_path, "r+b") as data_f, \
             open(self.shm_refs_path, "r+b") as refs_f:

            data_mmap = mmap.mmap(data_f.fileno(), 0, access=mmap.ACCESS_READ)
            refs_mmap = mmap.mmap(refs_f.fileno(), 0, access=mmap.ACCESS_WRITE)

            print(f"[Python worker={worker_id}] Ожидание Pipe {pipe_path}")

            with open(pipe_path, "r") as pipe:
                print(f"[Python worker={worker_id}] Pipe подключен")

                while True:
                    ready_to_read, _, _ = select.select([pipe], [], [], self.timeout_seconds)

                    if not ready_to_read:
                        print(f"[Python worker={worker_id}] Таймаут ({self.timeout_seconds}s): данных нет. Завершаем эпоху.")
                        break

                    line = pipe.readline()

                    if not line:
                        print(f"[Python worker={worker_id}] Получен EOF. Завершаем эпоху.")
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        batch_meta = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"[Python worker={worker_id}] Ошибка парсинга JSON: {e}")
                        continue

                    if not batch_meta.get("items"):
                        print(f"[Python worker={worker_id}] Датасет полностью прочитан (Пустой батч). Конец эпохи.")
                        break

                    # Snapshot generation this batch was served from (feature F1).
                    # Constant for the whole epoch across all workers; a test
                    # asserts this to detect torn reads under concurrent mutation.
                    batch_generation = batch_meta.get("generation")

                    for item in batch_meta["items"]:
                        slot_id = item["slot_id"]
                        offset = item["offset"]
                        size = item["size"]

                        raw_bytes = bytes(data_mmap[offset : offset + size])

                        if self.decode_mode == DECODE_RGB_UINT8:
                            # Daemon already JPEG-decoded + resized; slot bytes
                            # are a packed (H, W, 3) uint8 HWC tensor. Skip the
                            # user-supplied decode_fn entirely.
                            h = w = self.decode_image_size
                            expected = h * w * 3
                            if size != expected:
                                # Defensive: daemon misconfig or mode mismatch.
                                # Skip rather than feeding wrong-shape garbage
                                # into the transform.
                                continue
                            decoded = np.frombuffer(
                                raw_bytes, dtype=np.uint8,
                            ).reshape(h, w, 3)
                        else:
                            decoded = self.decode_fn(raw_bytes)
                            if decoded is None:
                                # decoder signaled skip (e.g., corrupted image)
                                continue

                        try:
                            tensor = self.transform(decoded)
                        except Exception:
                            # transform failure: skip this sample but keep going
                            continue

                        result = {"image": tensor}

                        if batch_generation is not None:
                            result["dfs_generation"] = batch_generation

                        if "path" in item:
                            result["path"] = item["path"]

                        if "meta" in item and item["meta"]:
                            meta_data = item["meta"]
                            if isinstance(meta_data, dict):
                                result.update(meta_data)
                            elif isinstance(meta_data, list):
                                result["annotations"] = meta_data
                            else:
                                result["meta_raw"] = meta_data

                        self._decrement_refcount(refs_mmap, slot_id)

                        yield result

            data_mmap.close()
            refs_mmap.close()
