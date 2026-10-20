"""The body-field network's outputs kept in a workspace, so the network runs once per frame.

An analysis with a body model reads the network's fields in every stage: the
segment stage takes its masks (``mask_source="body_net"``), and the fit,
propagate and track stages and the region refits score their fits against
its A-P field and ends and start them from its trace.  Each stage used to
run the network again on its frames.  The store keeps, per workspace row,
the part of the prediction those readers use (:class:`body_proposal.FieldCrop`:
A-P and crossings on the mask grown by a margin, the end peaks)::

    <workspace>/fields/chunk_NNNNN.npz   FIELD_CHUNK_ROWS rows per file

Each chunk records the :func:`source_key` of the predictions in it: the
weights' digest, the flat-field switch, the dataset and the crossing
threshold.  A chunk with another key reads as empty and is replaced on the
next write, so predictions of another model are never mixed in.  Writes are
atomic (``os.replace``); two writers of one chunk at once can lose each
other's new rows, which only means predicting them again.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import hashlib
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .body_proposal import OVERLAP_THRESHOLD, FieldCrop

FIELDS_DIR = "fields"
FIELD_CHUNK_ROWS = 256
_CHUNK_CACHE_SIZE = 2


@lru_cache(maxsize=8)
def _digest(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def source_key(checkpoint: str | Path, flat_field: bool, dataset: str) -> str:
    """What the stored fields depend on besides the frames: the weights, the flat field, the dataset and the crossing threshold."""

    path = Path(checkpoint).resolve()
    if path.is_file():
        stat = path.stat()
        weights = _digest(str(path), stat.st_size, stat.st_mtime_ns)
    else:
        # Weights that are gone match no stored digest; their own key keeps a stub's fields apart.
        weights = f"missing:{path}"
    return f"{weights}|flat_field={bool(flat_field)}|{dataset}|overlap>={OVERLAP_THRESHOLD}"


def _pack(crops: dict[int, FieldCrop], key: str) -> dict[str, np.ndarray]:
    rows = np.array(sorted(crops), dtype=np.int64)
    ordered = [crops[int(r)] for r in rows]
    support = [np.packbits(c.support.ravel()) for c in ordered]
    ap = [c.ap[c.support] for c in ordered]
    crossing = [np.packbits(c.crossing[c.support]) for c in ordered]
    offsets = lambda parts: np.cumsum([0] + [len(p) for p in parts]).astype(np.int64)
    return {
        "key": np.array(key), "rows": rows,
        "shape": np.array([c.shape for c in ordered], dtype=np.int64).reshape(-1, 2),
        "box": np.array([c.box for c in ordered], dtype=np.int64).reshape(-1, 4),
        "peaks": np.array([(c.head, c.tail) for c in ordered], dtype=np.float64).reshape(-1, 2, 3),
        "support": np.concatenate(support) if ordered else np.zeros(0, np.uint8), "support_offsets": offsets(support),
        "ap": np.concatenate(ap) if ordered else np.zeros(0, np.float32), "ap_offsets": offsets(ap),
        "crossing": np.concatenate(crossing) if ordered else np.zeros(0, np.uint8), "crossing_offsets": offsets(crossing),
    }


def _unpack(chunk: dict[str, np.ndarray], k: int) -> FieldCrop:
    y0, y1, x0, x1 = (int(v) for v in chunk["box"][k])
    size = (y1 - y0) * (x1 - x0)
    a, b = chunk["support_offsets"][k : k + 2]
    support = np.unpackbits(chunk["support"][a:b], count=size).astype(bool).reshape(y1 - y0, x1 - x0)
    a, b = chunk["ap_offsets"][k : k + 2]
    ap = np.full(support.shape, np.nan, dtype=np.float32)
    ap[support] = chunk["ap"][a:b]
    a, b = chunk["crossing_offsets"][k : k + 2]
    crossing = np.zeros(support.shape, dtype=bool)
    crossing[support] = np.unpackbits(chunk["crossing"][a:b], count=int(support.sum())).astype(bool)
    head, tail = (tuple(float(v) for v in peak) for peak in chunk["peaks"][k])
    return FieldCrop(
        shape=(int(chunk["shape"][k][0]), int(chunk["shape"][k][1])), box=(y0, y1, x0, x1), support=support, ap=ap, crossing=crossing,
        head=head, tail=tail,
    )


class FieldStore:
    """The stored fields of one workspace under one source key; writes wait in memory until :meth:`flush`."""

    def __init__(self, workspace_path: str | Path, key: str) -> None:
        self.directory = Path(workspace_path) / FIELDS_DIR
        self.key = key
        self._chunks: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._pending: dict[int, dict[int, FieldCrop]] = {}

    def _path(self, chunk: int) -> Path:
        return self.directory / f"chunk_{chunk:05d}.npz"

    def _load(self, chunk: int) -> dict[str, np.ndarray]:
        """The chunk's arrays, empty when it is missing or holds another source's fields."""

        if chunk in self._chunks:
            self._chunks.move_to_end(chunk)
            return self._chunks[chunk]
        loaded: dict[str, np.ndarray] = {}
        path = self._path(chunk)
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["key"]) == self.key:
                    loaded = {name: archive[name] for name in archive.files}
        self._chunks[chunk] = loaded
        while len(self._chunks) > _CHUNK_CACHE_SIZE:
            self._chunks.popitem(last=False)
        return loaded

    def get(self, rows: Sequence[int]) -> dict[int, FieldCrop]:
        """The stored fields of these rows (the ones that have them)."""

        out: dict[int, FieldCrop] = {}
        for row in (int(r) for r in rows):
            chunk = row // FIELD_CHUNK_ROWS
            pending = self._pending.get(chunk, {})
            if row in pending:
                out[row] = pending[row]
                continue
            arrays = self._load(chunk)
            if arrays:
                hits = np.nonzero(arrays["rows"] == row)[0]
                if len(hits):
                    out[row] = _unpack(arrays, int(hits[0]))
        return out

    def put(self, crops: dict[int, FieldCrop]) -> None:
        """Keep these rows' fields; a chunk is written once a chunk's worth of rows waits, the rest at :meth:`flush`."""

        for row, crop in crops.items():
            self._pending.setdefault(int(row) // FIELD_CHUNK_ROWS, {})[int(row)] = crop
        if sum(len(rows) for rows in self._pending.values()) >= FIELD_CHUNK_ROWS:
            self.flush()

    def flush(self) -> None:
        for chunk, new in sorted(self._pending.items()):
            self._chunks.pop(chunk, None)
            existing = self._load(chunk)
            merged = {int(r): _unpack(existing, k) for k, r in enumerate(existing["rows"])} if existing else {}
            merged.update(new)
            payload = _pack(merged, self.key)
            self.directory.mkdir(parents=True, exist_ok=True)
            tmp = self.directory / f".{self._path(chunk).name}.tmp"
            with open(tmp, "wb") as handle:
                # Uncompressed: network outputs barely compress, and the support and crossings are bits already.
                np.savez(handle, **payload)
            os.replace(tmp, self._path(chunk))
            self._chunks[chunk] = payload
        self._pending.clear()

