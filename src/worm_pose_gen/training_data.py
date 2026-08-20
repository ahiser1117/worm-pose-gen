"""EXP-0004 training datasets; all proxy-HDF5 assumptions live here."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Collection, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset
import torch.nn.functional as F

from .geometry import in_fov_mask
from .renderer import render_worm
from .synthetic import (
    SyntheticConfig,
    anatomical_crop_transform,
    generate_synthetic_pose,
    original_to_render,
)


EXPECTED_RECORDS = ("2023-09-19-01", "2023-09-27-01", "2023-10-11-01")
_SHARED_PROXY_FILE: h5py.File | None = None
_SHARED_PROXY_PATH: Path | None = None
_SHARED_PROXY_PID: int | None = None
_SHARED_PROXY_USERS = 0


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_proxy_frame_exclusions(
    path: str | os.PathLike[str], *, expected_sha256: str
) -> dict[str, frozenset[int]]:
    """Load a hash-bound EXP-003 recording/frame exclusion manifest."""

    manifest_path = Path(path)
    actual_hash = sha256_file(manifest_path)
    if actual_hash != expected_sha256:
        raise RuntimeError(
            f"proxy exclusion sha256 mismatch: expected {expected_sha256}, got {actual_hash}"
        )
    payload = json.loads(manifest_path.read_text())
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment") != "EXP-003"
        or payload.get("protected_holdout_opened") is not False
    ):
        raise RuntimeError("invalid EXP-003 proxy exclusion manifest identity")
    recordings = payload.get("recordings")
    if not isinstance(recordings, dict) or set(recordings) != set(EXPECTED_RECORDS):
        raise RuntimeError("proxy exclusion recordings do not match development records")
    result: dict[str, frozenset[int]] = {}
    for record in EXPECTED_RECORDS:
        raw = recordings[record].get("excluded_frame_indices")
        if (
            not isinstance(raw, list)
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in raw)
            or raw != sorted(set(raw))
        ):
            raise RuntimeError(f"invalid proxy exclusion indices for {record}")
        result[record] = frozenset(raw)
    return result


def normalize_image(image: np.ndarray | Tensor, height: int = 192, width: int = 256) -> Tensor:
    """Deterministically convert one grayscale raster to float ``[1,H,W]``."""

    value = torch.as_tensor(np.asarray(image) if isinstance(image, np.ndarray) else image)
    if value.ndim != 2:
        raise ValueError("image must be a two-dimensional grayscale raster")
    if value.dtype == torch.uint8:
        value = value.to(torch.float32).div_(255.0)
    else:
        value = value.to(torch.float32)
        if bool(value.max() > 1):
            value = value / 255.0
    return F.interpolate(
        value[None, None], size=(height, width), mode="bilinear", align_corners=False
    )[0].clamp(0, 1)


class ProxyDataset(Dataset[dict[str, Tensor | str | int]]):
    """Lazy accepted-only proxy dataset backed by at most one read-only handle."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        expected_sha256: str,
        fold: int,
        split: str,
        excluded_frame_indices: Mapping[str, Collection[int]] | None = None,
    ) -> None:
        if fold not in range(len(EXPECTED_RECORDS)):
            raise ValueError("fold must be 0, 1, or 2")
        if split not in ("train", "validation"):
            raise ValueError("split must be 'train' or 'validation'")
        self.path = Path(path)
        self._file: h5py.File | None = None
        self._owner_pid: int | None = None
        self._registered_shared_user = False
        raw_exclusions = excluded_frame_indices or {}
        unknown_records = set(raw_exclusions) - set(EXPECTED_RECORDS)
        if unknown_records:
            raise ValueError(f"unknown proxy exclusion records: {sorted(unknown_records)}")
        self.excluded_frame_indices = {
            record: frozenset(int(value) for value in raw_exclusions.get(record, ()))
            for record in EXPECTED_RECORDS
        }
        if any(value < 0 for values in self.excluded_frame_indices.values() for value in values):
            raise ValueError("excluded proxy frame indices must be nonnegative")
        self.excluded_row_count = 0
        actual_hash = sha256_file(self.path)
        if actual_hash != expected_sha256:
            raise RuntimeError(f"proxy sha256 mismatch: expected {expected_sha256}, got {actual_hash}")
        held_out = EXPECTED_RECORDS[fold]
        self.records = tuple(
            record
            for record in EXPECTED_RECORDS
            if (record != held_out) == (split == "train")
        )
        self._rows: list[tuple[str, int, int]] = []
        handle = self._open()
        try:
            if int(handle.attrs.get("schema_version", -1)) != 1:
                raise RuntimeError("unsupported proxy schema_version")
            if not bool(handle.attrs.get("complete", False)):
                raise RuntimeError("proxy file is not marked complete")
            if tuple(sorted(handle.keys())) != tuple(sorted(EXPECTED_RECORDS)):
                raise RuntimeError("proxy recording groups do not match frozen development records")
            for record in self.records:
                group = handle[record]
                required = {
                    "accepted",
                    "accepted_sample_position",
                    "accepted_frame_index",
                    "accepted_image",
                    "centerline_xy",
                }
                if not required.issubset(group.keys()):
                    raise RuntimeError(f"proxy group {record} has an incomplete schema")
                positions = np.asarray(group["accepted_sample_position"], dtype=np.int64)
                frames = np.asarray(group["accepted_frame_index"], dtype=np.int64)
                images = group["accepted_image"]
                centerlines = group["centerline_xy"]
                if images.ndim != 3 or centerlines.shape[1:] != (100, 2):
                    raise RuntimeError(f"proxy group {record} has unexpected array shapes")
                if not (len(positions) == len(frames) == images.shape[0]):
                    raise RuntimeError(f"proxy group {record} accepted arrays are misaligned")
                accepted = np.asarray(group["accepted"], dtype=bool)
                if np.any(positions < 0) or np.any(positions >= len(accepted)) or not np.all(accepted[positions]):
                    raise RuntimeError(f"proxy group {record} accepted mapping is invalid")
                for position, frame in zip(positions, frames, strict=True):
                    if int(frame) in self.excluded_frame_indices[record]:
                        self.excluded_row_count += 1
                    else:
                        self._rows.append((record, int(position), int(frame)))
        except BaseException:
            self.close()
            raise

    def _open(self) -> h5py.File:
        global _SHARED_PROXY_FILE, _SHARED_PROXY_PATH, _SHARED_PROXY_PID, _SHARED_PROXY_USERS
        pid = os.getpid()
        if self._file is not None and self._owner_pid != pid:
            self._file = None
            self._owner_pid = None
            self._registered_shared_user = False
        if self._file is None:
            resolved = self.path.resolve(strict=True)
            shared_valid = (
                _SHARED_PROXY_FILE is not None
                and _SHARED_PROXY_PID == pid
                and bool(_SHARED_PROXY_FILE.id.valid)
            )
            if shared_valid and _SHARED_PROXY_PATH != resolved:
                raise RuntimeError("one process may open at most one proxy HDF5 file")
            if not shared_valid:
                # Never call close on a handle inherited across fork.
                _SHARED_PROXY_FILE = h5py.File(resolved, "r")
                _SHARED_PROXY_PATH = resolved
                _SHARED_PROXY_PID = pid
                _SHARED_PROXY_USERS = 0
            self._file = _SHARED_PROXY_FILE
            self._owner_pid = pid
            self._registered_shared_user = True
            _SHARED_PROXY_USERS += 1
        return self._file

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        record, sample_position, frame = self._rows[index]
        group = self._open()[record]
        # accepted_image row order matches accepted_sample_position row order.
        accepted_row = int(np.searchsorted(np.asarray(group["accepted_sample_position"]), sample_position))
        image_source = np.asarray(group["accepted_image"][accepted_row])
        centerline = torch.from_numpy(np.asarray(group["centerline_xy"][sample_position])).float()
        source_height, source_width = image_source.shape
        centerline = centerline * centerline.new_tensor((256 / source_width, 192 / source_height))
        return {
            "image": normalize_image(image_source),
            "centerline_xy": centerline,
            "image_support_target": in_fov_mask(centerline, 192, 256),
            "tier": "B_candidate_proxy",
            "record": record,
            "frame_index": frame,
            "sample_seed": -1,
        }

    def close(self) -> None:
        global _SHARED_PROXY_FILE, _SHARED_PROXY_PATH, _SHARED_PROXY_PID, _SHARED_PROXY_USERS
        if self._registered_shared_user and self._owner_pid == os.getpid():
            _SHARED_PROXY_USERS -= 1
            if _SHARED_PROXY_USERS == 0 and _SHARED_PROXY_FILE is not None:
                _SHARED_PROXY_FILE.close()
                _SHARED_PROXY_FILE = None
                _SHARED_PROXY_PATH = None
                _SHARED_PROXY_PID = None
        self._file = None
        self._owner_pid = None
        self._registered_shared_user = False

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_owner_pid"] = None
        state["_registered_shared_user"] = False
        return state


class SyntheticTierCDataset(Dataset[dict[str, Tensor | str | int]]):
    """Deterministic analytic Tier C examples rendered on demand."""

    def __init__(self, count: int, *, seed: int, profile: str) -> None:
        if count < 1:
            raise ValueError("count must be positive")
        if profile not in ("development", "held_out"):
            raise ValueError("profile must be development or held_out")
        self.count = count
        self.seed = seed
        self.profile = profile
        self.config = SyntheticConfig()

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        geometry = generate_tier_c_geometry(index, seed=self.seed, profile=self.profile)
        sample_seed = int(geometry["sample_seed"])
        pose = geometry["pose"]
        support = geometry["image_support_target"]
        centerline = geometry["centerline_xy"]
        width = pose["width_profile_render"].float()
        generator = torch.Generator().manual_seed(sample_seed + 10_000_000)
        noise = torch.randn((192, 256), generator=generator) * 0.015
        rendered = render_worm(
            centerline,
            width,
            192,
            256,
            foreground=0.18 + 0.06 * torch.rand((), generator=generator),
            background=0.72 + 0.12 * torch.rand((), generator=generator),
            noise=noise,
            image_support_target=support,
        )
        return {
            "image": rendered["image"].unsqueeze(0).float(),
            "centerline_xy": centerline,
            "image_support_target": support,
            "tier": "C",
            "record": "synthetic",
            "frame_index": -1,
            "sample_seed": sample_seed,
        }


class MaterializedPoseDataset(Dataset[dict[str, Tensor | str | int]]):
    """Immutable in-memory snapshot that prevents render/I/O stalls during training."""

    def __init__(self, source: Dataset) -> None:
        if len(source) < 1:
            raise ValueError("cannot materialize an empty pose dataset")
        samples: list[dict[str, Tensor | str | int]] = []
        for index in range(len(source)):
            raw = source[index]
            sample: dict[str, Tensor | str | int] = {}
            for name, value in raw.items():
                sample[name] = value.detach().cpu().clone() if isinstance(value, Tensor) else value
            samples.append(sample)
        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        return self.samples[index]


def materialized_dataset_sha256(dataset: Dataset) -> str:
    """Hash ordered sample identity and tensor bytes for paired-run verification."""

    digest = hashlib.sha256()
    digest.update(b"worm-pose-materialized-dataset-v1\0")
    for index in range(len(dataset)):
        digest.update(index.to_bytes(8, "little", signed=False))
        sample = dataset[index]
        for name in sorted(sample):
            encoded_name = name.encode("utf-8")
            digest.update(len(encoded_name).to_bytes(4, "little", signed=False))
            digest.update(encoded_name)
            value = sample[name]
            if isinstance(value, Tensor):
                tensor = value.detach().cpu().contiguous()
                metadata = json.dumps(
                    {"kind": "tensor", "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
                payload = tensor.numpy().tobytes(order="C")
            else:
                metadata = json.dumps(
                    {"kind": "scalar", "python_type": type(value).__name__},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
                payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
            digest.update(len(metadata).to_bytes(4, "little", signed=False))
            digest.update(metadata)
            digest.update(len(payload).to_bytes(8, "little", signed=False))
            digest.update(payload)
    return digest.hexdigest()


def generate_tier_c_geometry(
    index: int, *, seed: int, profile: str
) -> dict[str, Tensor | int | dict]:
    """Generate the exact geometry used by :class:`SyntheticTierCDataset`.

    This renderer-free path is also the executable EXP-0007 baseline source.
    """

    if index < 0:
        raise ValueError("index must be non-negative")
    config = SyntheticConfig()
    sample_seed = seed + int(index)
    pose = generate_synthetic_pose(sample_seed, config, profile=profile)
    original = pose["centerline_xy"]
    if index % 3:
        fraction = (0.05, 0.10, 0.20, 0.30, 0.40)[index % 5]
        end = "head" if index % 2 else "tail"
        _, camera, support = anatomical_crop_transform(original, fraction, end, config)
    else:
        camera = original
        support = torch.ones(config.num_points, dtype=torch.bool)
    return {
        "sample_seed": sample_seed,
        "pose": pose,
        "centerline_xy": original_to_render(camera, config).float(),
        "image_support_target": support,
    }


def make_datasets(
    proxy_path: str | os.PathLike[str],
    expected_sha256: str,
    *,
    fold: int,
    seed: int,
    synthetic_train_count: int,
    synthetic_validation_count: int = 128,
    excluded_frame_indices: Mapping[str, Collection[int]] | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    train = ConcatDataset(
        [
            ProxyDataset(
                proxy_path,
                expected_sha256=expected_sha256,
                fold=fold,
                split="train",
                excluded_frame_indices=excluded_frame_indices,
            ),
            SyntheticTierCDataset(synthetic_train_count, seed=seed + fold * 100_000, profile="development"),
        ]
    )
    proxy_validation = ProxyDataset(
        proxy_path,
        expected_sha256=expected_sha256,
        fold=fold,
        split="validation",
        excluded_frame_indices=excluded_frame_indices,
    )
    tier_c_validation = SyntheticTierCDataset(
        synthetic_validation_count,
        seed=seed + 5_000_000 + fold * 100_000,
        profile="held_out",
    )
    return train, proxy_validation, tier_c_validation
