import io
import json
import mmap
import os
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

# ---- Binary wire protocol (opt 03) ------------------------------------------
# Must stay byte-compatible with internal/pipeline/dealer.go encodeFrame.
# Frame = HEADER (magic u32, total_len u32, generation u64, item_count u32,
# blob_len u32) + COLUMNAR (item_count × ITEM_DTYPE) + BLOBS (path+meta bytes).
# Replaces the old newline-delimited JSON: for cheap-decode data (audio) the
# JSON encode/parse + per-item dict was the dominant per-sample cost.
_FRAME_MAGIC = 0x44465331          # "DFS1"
_FRAME_HDR_LEN = 8                 # magic + total_len
_FRAME_REST_LEN = 16               # generation + item_count + blob_len

# Structured dtype for the columnar block — one vectorized np.frombuffer parses
# all items at once (no per-item struct.unpack). align=False => packed layout
# matching the Go encoder exactly. itemsize MUST equal dealer.go ItemWireSize.
ITEM_DTYPE = np.dtype([
    ("slot_id",  "<i4"),
    ("offset",   "<i8"),
    ("size",     "<i8"),
    ("path_len", "<u4"),
    ("meta_len", "<u4"),
], align=False)
assert ITEM_DTYPE.itemsize == 28, (
    f"ITEM_DTYPE.itemsize={ITEM_DTYPE.itemsize}, expected 28 to match "
    f"dealer.go ItemWireSize — Go/Python wire protocol drift"
)


def _read_exact(fd, n):
    """Read exactly n bytes from raw fd, looping over short reads. Returns the
    bytes, or None on clean EOF (writer closed) at/within a frame boundary."""
    if n == 0:
        return b""
    chunks = []
    remaining = n
    while remaining > 0:
        b = os.read(fd, remaining)
        if not b:
            return None  # EOF — writer (daemon dealer) closed the pipe
        chunks.append(b)
        remaining -= len(b)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


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
                 decode_parallelism=0,
                 rank=0,
                 world_size=1):
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
        # Distributed (DDP) placement (feature F2). Each DDP rank runs its own
        # daemon (distinct daemon_url + shm_*_path + pipe_path_template), and the
        # daemon serves only this rank's disjoint shard partition. Defaults
        # rank=0/world_size=1 = single-process, identical to the legacy path.
        if not isinstance(world_size, int) or world_size < 1:
            raise ValueError(f"world_size must be a positive int, got {world_size!r}")
        if not isinstance(rank, int) or not (0 <= rank < world_size):
            raise ValueError(
                f"rank must be an int in [0, world_size={world_size}), got {rank!r}"
            )

        self.rank = rank
        self.world_size = world_size
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
        if world_size > 1:
            payload["distributed"] = {"rank": rank, "world_size": world_size}

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
        # Confirm the daemon partitioned for the rank/world we asked for; a stale
        # daemon ignoring the `distributed` block would serve the full dataset to
        # every rank (duplicate samples across ranks → biased gradient).
        ack_dist = ack.get("distributed") or {}
        ack_world = ack_dist.get("world_size", 1)
        if ack_world != world_size:
            raise RuntimeError(
                f"daemon acknowledged world_size={ack_world!r} but client asked for "
                f"{world_size!r}. Likely a daemon version mismatch — rebuild it."
            )
        self.session_id = ack.get("session_id")
        self.pipe_path_template = ack.get("pipe_template") or self.pipe_path_template

    def _decrement_refcount_by(self, refs_mmap, slot_id, n):
        """Subtract n from a slot's refcount in one read-modify-write (opt 03:
        batched, was one write per sample). The Go planner recycles a slot when
        its refcount hits 0; a slot's objects may span several frames, so each
        frame subtracts only the count it consumed — the sum reaches 0 exactly
        when the last frame holding the slot is drained. Only this worker writes
        this slot (slots are partitioned per worker), so no locking is needed."""
        if n == 0:
            return
        offset = slot_id * 4
        current_val = struct.unpack_from("<i", refs_mmap, offset)[0]
        struct.pack_into("<i", refs_mmap, offset, current_val - n)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        pipe_path = self.pipe_path_template.format(worker_id=worker_id)

        print(f"[Python worker={worker_id}] Подключение к DatasetFS pipe={pipe_path}")

        with open(self.shm_data_path, "r+b") as data_f, \
             open(self.shm_refs_path, "r+b") as refs_f:

            data_mmap = mmap.mmap(data_f.fileno(), 0, access=mmap.ACCESS_READ)
            refs_mmap = mmap.mmap(refs_f.fileno(), 0, access=mmap.ACCESS_WRITE)
            # Zero-copy window into the slot buffers (opt 03): per-item slices of
            # this are handed to decode without the old bytes() copy. Released
            # before close (close() raises if any export is still live).
            data_view = memoryview(data_mmap)

            print(f"[Python worker={worker_id}] Ожидание Pipe {pipe_path}")

            # Binary pipe (opt 03): readline() can't frame binary, so we read the
            # raw fd with os.read + exact-length reads. O_RDONLY on a FIFO blocks
            # until the daemon's dealer opens the write end — same rendezvous as
            # the old open(pipe_path, "r").
            fd = os.open(pipe_path, os.O_RDONLY)
            print(f"[Python worker={worker_id}] Pipe подключен")
            decoded = view = None
            try:
                while True:
                    # select gates the FIRST byte of each frame so the 30s idle
                    # timeout / end-of-epoch semantics are preserved.
                    ready_to_read, _, _ = select.select([fd], [], [], self.timeout_seconds)
                    if not ready_to_read:
                        print(f"[Python worker={worker_id}] Таймаут ({self.timeout_seconds}s): данных нет. Завершаем эпоху.")
                        break

                    hdr = _read_exact(fd, _FRAME_HDR_LEN)
                    if hdr is None:
                        print(f"[Python worker={worker_id}] Получен EOF. Завершаем эпоху.")
                        break
                    magic, total_len = struct.unpack("<II", hdr)
                    if magic != _FRAME_MAGIC:
                        raise RuntimeError(
                            f"bad frame magic {magic:#x} (expected {_FRAME_MAGIC:#x}); "
                            f"Go/Python wire protocol mismatch — rebuild the daemon"
                        )

                    body = _read_exact(fd, total_len)
                    if body is None:
                        print(f"[Python worker={worker_id}] EOF в середине фрейма. Завершаем эпоху.")
                        break

                    # Snapshot generation this frame was served from (feature F1).
                    # Constant for the whole epoch across all workers; a test
                    # asserts this to detect torn reads under concurrent mutation.
                    generation, item_count, _blob_len = struct.unpack_from("<QII", body, 0)
                    if item_count == 0:
                        print(f"[Python worker={worker_id}] Датасет полностью прочитан (Пустой фрейм). Конец эпохи.")
                        break

                    # One vectorized parse of the fixed columnar block; no
                    # per-item struct.unpack.
                    cols = np.frombuffer(
                        body, dtype=ITEM_DTYPE, count=item_count, offset=_FRAME_REST_LEN,
                    )
                    slots = cols["slot_id"]
                    offsets = cols["offset"]
                    sizes = cols["size"]
                    path_lens = cols["path_len"]
                    meta_lens = cols["meta_len"]

                    # Variable section follows the columnar block.
                    cursor = _FRAME_REST_LEN + item_count * ITEM_DTYPE.itemsize

                    # Batched refcount (opt 03): count EVERY item toward its slot
                    # — even those we skip below — so a slot with a decode failure
                    # still reaches 0 and the planner recycles it (fixes a latent
                    # slot leak). One decrement per slot after the frame loop.
                    consumed = {}

                    for i in range(item_count):
                        slot_id = int(slots[i])
                        consumed[slot_id] = consumed.get(slot_id, 0) + 1

                        offset = int(offsets[i])
                        size = int(sizes[i])
                        pl = int(path_lens[i])
                        ml = int(meta_lens[i])

                        path = None
                        if pl:
                            path = body[cursor:cursor + pl].decode("utf-8")
                        cursor += pl
                        meta = None
                        if ml:
                            meta = json.loads(body[cursor:cursor + ml])
                        cursor += ml

                        # Zero-copy slice into the slot buffer, handed to decode
                        # without the old bytes() copy. The slice MUST be released
                        # before we yield: a generator paused at yield that is then
                        # abandoned (GeneratorExit) would otherwise leave a live
                        # export, and mmap.close() raises while any export is alive.
                        # Decode/transform materialize owned objects, so releasing
                        # the view here is safe (the slot itself can't recycle until
                        # the per-slot decrement after this frame's loop).
                        view = data_view[offset:offset + size]
                        decoded = None
                        if self.decode_mode == DECODE_RGB_UINT8:
                            # Daemon already JPEG-decoded + resized; slot bytes
                            # are a packed (H, W, 3) uint8 HWC tensor. frombuffer
                            # aliases the mmap (truly zero-copy now); ToTensor
                            # materializes an owned float tensor.
                            h = w = self.decode_image_size
                            if size == h * w * 3:
                                decoded = np.frombuffer(view, dtype=np.uint8).reshape(h, w, 3)
                            # else: daemon misconfig / mode mismatch → skip below
                        else:
                            decoded = self.decode_fn(view)

                        tensor = None
                        if decoded is not None:
                            try:
                                tensor = self.transform(decoded)
                            except Exception:
                                tensor = None  # transform failure: skip, keep going

                        # Drop any slot-aliasing array (rgb frombuffer view), then
                        # release the slice — before yield, before any continue.
                        decoded = None
                        view.release()
                        view = None

                        if tensor is None:
                            # decoder signaled skip (None), size mismatch, or
                            # transform failure — already counted in `consumed`.
                            continue

                        result = {"image": tensor, "dfs_generation": generation}
                        if path is not None:
                            result["path"] = path
                        if meta:
                            if isinstance(meta, dict):
                                result.update(meta)
                            elif isinstance(meta, list):
                                result["annotations"] = meta
                            else:
                                result["meta_raw"] = meta

                        yield result

                    for slot_id, cnt in consumed.items():
                        self._decrement_refcount_by(refs_mmap, slot_id, cnt)
            finally:
                os.close(fd)
                # Drop any mmap-aliasing locals before releasing the view, else
                # memoryview.release()/mmap.close() raise (live exported buffers).
                decoded = view = None
                data_view.release()
                data_mmap.close()
                refs_mmap.close()
