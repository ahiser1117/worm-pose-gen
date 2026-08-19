"""Fail-closed exploratory inference for rejected worm-pose checkpoints.

This module deliberately does not expose a production/validated mode.  EXP-0007
did not accept a model, so callers must opt in to exploratory inference and all
written artifacts carry ``validation_status=exploratory_rejected_checkpoint``.

Frame APIs accept one ``[H,W]`` grayscale image or a batch ``[B,H,W]``.  Model
coordinates are mapped from the 192x256 training raster to the exact source
image dimensions before tangent, curvature and half-open FOV geometry are
computed.  The current checkpoints have no head/tail, calibrated uncertainty,
or quality heads: these are exported as explicitly documented conservative
sentinels (0.5, pi radians, and 0 respectively), never as measured confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
import yaml

from .data import HDF5FrameSource
from .geometry import curvature, in_fov_mask, tangent_angles
from .io import OutputProvenance, PoseHDF5Writer, SourceIdentity, sha256_file
from .model import WormProposalModule
from .training_data import normalize_image


VALIDATION_STATUS = "exploratory_rejected_checkpoint"
RENDER_HEIGHT = 192
RENDER_WIDTH = 256


class ExploratoryInferenceRequired(PermissionError):
    """Raised unless the caller explicitly accepts rejected-checkpoint use."""


class InferenceContractError(RuntimeError):
    """Raised when an input or prediction cannot satisfy the output contract."""


class TemporalInferenceUnsupported(InferenceContractError):
    """Raised because no temporal model/protocol has been validated."""

    validation_status = VALIDATION_STATUS


@dataclass(frozen=True)
class ExploratoryDeclaration:
    """Frozen declaration authorizing only rejected-checkpoint exploration."""

    config_path: Path
    config_sha256: str
    checkpoint_sha256: str
    model_variant: str
    encoder_pool_output: tuple[int, int]
    body_points: int
    input_height: int
    input_width: int
    validation_status: str


def load_exploratory_declaration(
    config_path: str | os.PathLike[str],
) -> ExploratoryDeclaration:
    """Load and strictly validate the exploratory declaration in ``final.yaml``."""

    path = Path(config_path)
    try:
        config_bytes = path.read_bytes()
        document = yaml.safe_load(config_bytes.decode("utf-8"))
        section = document["exploratory_inference"]
        deployment_authorized = document["deployment_authorized"]
    except (OSError, TypeError, KeyError, yaml.YAMLError) as error:
        raise InferenceContractError(f"invalid final inference config {path}: {error}") from error
    if deployment_authorized is not False:
        raise InferenceContractError("final config must explicitly deny deployment")
    if section.get("enabled_only_with_explicit_opt_in") is not True:
        raise InferenceContractError("final config must require explicit exploratory opt-in")
    validation_status = section.get("validation_status")
    if validation_status != VALIDATION_STATUS:
        raise InferenceContractError(
            f"final config validation_status must be {VALIDATION_STATUS!r}"
        )
    try:
        checkpoint_sha256 = str(section["checkpoint_sha256"])
        pool_values = tuple(int(value) for value in section["encoder_pool_output"])
        if len(pool_values) != 2:
            raise ValueError("encoder_pool_output must contain two values")
        declaration = ExploratoryDeclaration(
            config_path=path.resolve(strict=True),
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            checkpoint_sha256=checkpoint_sha256,
            model_variant=str(section["model_variant"]),
            encoder_pool_output=(pool_values[0], pool_values[1]),
            body_points=int(section["body_points"]),
            input_height=int(section["input_height"]),
            input_width=int(section["input_width"]),
            validation_status=validation_status,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise InferenceContractError(f"invalid exploratory model declaration: {error}") from error
    if len(checkpoint_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_sha256
    ):
        raise InferenceContractError("declared checkpoint_sha256 must be lowercase SHA-256")
    if any(value < 1 for value in declaration.encoder_pool_output):
        raise InferenceContractError("declared encoder_pool_output values must be positive")
    if min(declaration.body_points, declaration.input_height, declaration.input_width) < 1:
        raise InferenceContractError("declared body points and input dimensions must be positive")
    return declaration


def _validate_declared_model(
    model: WormProposalModule,
    declaration: ExploratoryDeclaration,
    checkpoint_sha256: str,
) -> None:
    actual: dict[str, object] = {
        "checkpoint_sha256": checkpoint_sha256,
        "model_variant": model.variant,
        "encoder_pool_output": tuple(int(value) for value in model.encoder.pool_output),
        "body_points": int(model.hparams.num_points),
        "input_height": int(model.hparams.image_height),
        "input_width": int(model.hparams.image_width),
    }
    expected: dict[str, object] = {
        "checkpoint_sha256": declaration.checkpoint_sha256,
        "model_variant": declaration.model_variant,
        "encoder_pool_output": declaration.encoder_pool_output,
        "body_points": declaration.body_points,
        "input_height": declaration.input_height,
        "input_width": declaration.input_width,
    }
    mismatches = [
        f"{name}: expected {expected_value!r}, got {actual[name]!r}"
        for name, expected_value in expected.items()
        if actual[name] != expected_value
    ]
    if mismatches:
        raise InferenceContractError(
            "checkpoint does not match final exploratory declaration: " + "; ".join(mismatches)
        )


def _load_declared_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    device: str | torch.device,
) -> tuple[WormProposalModule, ExploratoryDeclaration, str]:
    declaration = load_exploratory_declaration(config_path)
    checkpoint = Path(checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != declaration.checkpoint_sha256:
        raise InferenceContractError(
            "checkpoint does not match final exploratory declaration: "
            f"checkpoint_sha256: expected {declaration.checkpoint_sha256!r}, "
            f"got {checkpoint_sha256!r}"
        )
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but CUDA is unavailable")
    model = WormProposalModule.load_from_checkpoint(checkpoint, map_location=selected_device)
    if sha256_file(checkpoint) != checkpoint_sha256:
        raise InferenceContractError("checkpoint changed while its declaration was verified")
    _validate_declared_model(model, declaration, checkpoint_sha256)
    return model, declaration, checkpoint_sha256


@dataclass(frozen=True)
class TemporalInferenceStatus:
    supported: bool = False
    validation_status: str = VALIDATION_STATUS
    reason: str = (
        "temporal inference is unsupported: EXP-0007 tested only independent frames"
    )


@dataclass(frozen=True)
class PosePredictionBatch:
    """CPU tensors ready for ``PoseHDF5Writer`` plus explicit semantics."""

    centerline_xy: Tensor
    tangent_angle: Tensor
    curvature: Tensor
    in_fov_mask: Tensor
    image_support_probability: Tensor
    angle_uncertainty: Tensor
    head_tail_probability: Tensor
    quality_score: Tensor
    validation_status: str = VALIDATION_STATUS

    def __len__(self) -> int:
        return int(self.centerline_xy.shape[0])

    def writer_values(
        self, frame_index: NDArray[np.int64], timestamp: NDArray[np.float64]
    ) -> dict[str, np.ndarray]:
        """Convert one prediction batch to the exact writer batch contract."""

        if frame_index.shape != (len(self),) or timestamp.shape != (len(self),):
            raise ValueError("frame_index and timestamp must match prediction batch size")
        values: dict[str, np.ndarray] = {
            name: getattr(self, name).numpy()
            for name in (
                "centerline_xy",
                "tangent_angle",
                "curvature",
                "in_fov_mask",
                "image_support_probability",
                "angle_uncertainty",
                "head_tail_probability",
                "quality_score",
            )
        }
        values["frame_index"] = np.asarray(frame_index, dtype=np.int64)
        values["timestamp"] = np.asarray(timestamp, dtype=np.float64)
        return values


def require_exploratory_opt_in(allow_exploratory: bool) -> None:
    if not allow_exploratory:
        raise ExploratoryInferenceRequired(
            "no checkpoint passed EXP-0007; pass --allow-exploratory to use a rejected "
            "checkpoint and receive explicitly non-validated outputs"
        )


class ExploratoryPoseInference:
    """Independent-frame inference on CPU or a caller-selected CUDA device."""

    def __init__(
        self,
        model: WormProposalModule,
        *,
        original_height: int,
        original_width: int,
        device: str | torch.device = "cuda",
        allow_exploratory: bool = False,
        checkpoint_sha256: str,
        declaration: ExploratoryDeclaration,
    ) -> None:
        require_exploratory_opt_in(allow_exploratory)
        if original_height < 1 or original_width < 1:
            raise ValueError("original image dimensions must be positive")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA inference requested but CUDA is unavailable")
        self.original_height = int(original_height)
        self.original_width = int(original_width)
        _validate_declared_model(model, declaration, checkpoint_sha256)
        self.model = model.eval().to(self.device)
        self.checkpoint_sha256 = checkpoint_sha256
        self.declaration = declaration

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | os.PathLike[str],
        *,
        original_height: int,
        original_width: int,
        config_path: str | os.PathLike[str],
        device: str | torch.device = "cuda",
        allow_exploratory: bool = False,
    ) -> ExploratoryPoseInference:
        require_exploratory_opt_in(allow_exploratory)
        model, declaration, checkpoint_sha256 = _load_declared_checkpoint(
            checkpoint_path, config_path, device
        )
        return cls(
            model,
            original_height=original_height,
            original_width=original_width,
            device=device,
            allow_exploratory=True,
            checkpoint_sha256=checkpoint_sha256,
            declaration=declaration,
        )

    def predict_frame(self, frame: np.ndarray | Tensor) -> PosePredictionBatch:
        """Infer one frame; the returned batch has length one."""

        value = torch.as_tensor(frame)
        if value.ndim != 2:
            raise ValueError("one frame must have shape [H,W]")
        return self.predict_batch(value.unsqueeze(0))

    def predict_batch(self, frames: np.ndarray | Tensor) -> PosePredictionBatch:
        """Infer independent frames, using the configured GPU when requested."""

        value = torch.as_tensor(frames)
        expected = (self.original_height, self.original_width)
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(f"frames must have shape [B,{expected[0]},{expected[1]}]")
        if value.shape[0] < 1:
            raise ValueError("an inference batch cannot be empty")
        if value.dtype != torch.uint8:
            if not value.is_floating_point():
                raise TypeError("frames must be uint8 [0,255] or floating point [0,1]")
            if not bool(torch.isfinite(value).all()) or bool(
                torch.any((value < 0) | (value > 1))
            ):
                raise ValueError("floating-point frames must be finite and lie in [0,1]")
        images = torch.stack([normalize_image(frame) for frame in value], dim=0)
        images = images.to(self.device, non_blocking=self.device.type == "cuda")
        with torch.inference_mode():
            raw = self.model(images)
            centerline = raw["centerline_xy"].to(torch.float32)
            scale = centerline.new_tensor(
                (self.original_width / RENDER_WIDTH, self.original_height / RENDER_HEIGHT)
            )
            centerline = centerline * scale
            try:
                tangent = tangent_angles(centerline)
                bend = curvature(centerline)
            except ValueError as error:
                raise InferenceContractError(
                    f"checkpoint emitted degenerate centerline geometry: {error}"
                ) from error
            support = raw["image_support_probability"].to(torch.float32)
            geometric_fov = in_fov_mask(
                centerline, self.original_height, self.original_width
            )
            batch, points = support.shape
            uncertainty = torch.full_like(support, torch.pi)
            # Orientation is unvalidated.  Keep model order and encode an exact tie.
            head_tail = torch.full((batch,), 0.5, dtype=centerline.dtype, device=self.device)
            # Zero means "not quality-qualified", not a predicted accuracy score.
            quality = torch.zeros((batch,), dtype=centerline.dtype, device=self.device)
            tensors = (centerline, tangent, bend, support, uncertainty)
            if any(not bool(torch.isfinite(item).all()) for item in tensors):
                raise InferenceContractError("checkpoint emitted non-finite predictions")
            if centerline.shape != (batch, points, 2):
                raise InferenceContractError("model centerline/support shapes disagree")
        return PosePredictionBatch(
            centerline_xy=centerline.cpu(),
            tangent_angle=tangent.cpu(),
            curvature=bend.cpu(),
            in_fov_mask=geometric_fov.cpu(),
            image_support_probability=support.cpu(),
            angle_uncertainty=uncertainty.cpu(),
            head_tail_probability=head_tail.cpu(),
            quality_score=quality.cpu(),
        )

    def temporal_window_status(self) -> TemporalInferenceStatus:
        return TemporalInferenceStatus(validation_status=self.declaration.validation_status)

    def predict_temporal_window(self, frames: np.ndarray | Tensor) -> None:
        """Reject temporal inference instead of silently treating it as validated."""

        del frames
        raise TemporalInferenceUnsupported(self.temporal_window_status().reason)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("worm-pose-gen", "torch", "lightning", "h5py", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def _mark_exploratory(writer: PoseHDF5Writer) -> None:
    """Add schema-tolerated semantic metadata before any predictions are appended."""

    group = writer._group  # The existing writer intentionally owns the atomic file handle.
    if group is None:
        raise RuntimeError("writer must be open before inference metadata is added")
    group.attrs["validation_status"] = VALIDATION_STATUS
    group.attrs["inference_mode"] = "independent_frame_only"
    group.attrs["temporal_inference_supported"] = False
    group["centerline_xy"].attrs["mapping"] = (
        "x_original=x_render*original_width/256;"
        "y_original=y_render*original_height/192"
    )
    group["tangent_angle"].attrs["semantics"] = "recomputed from original-pixel centerline"
    group["curvature"].attrs["semantics"] = "recomputed from original-pixel centerline"
    group["in_fov_mask"].attrs["semantics"] = "geometric half-open point membership"
    group["image_support_probability"].attrs["semantics"] = (
        "rejected-checkpoint learned output; uncalibrated and distinct from geometric FOV"
    )
    group["head_tail_probability"].attrs["semantics"] = (
        "fixed 0.5 unknown orientation; model order retained"
    )
    group["angle_uncertainty"].attrs["semantics"] = (
        "fixed pi-radian uncalibrated sentinel; checkpoint has no uncertainty head"
    )
    group["quality_score"].attrs["semantics"] = (
        "fixed 0 rejected-checkpoint sentinel; not an estimated accuracy"
    )


def infer_hdf5(
    *,
    source_path: str | os.PathLike[str],
    dataset_path: str,
    checkpoint_path: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    allow_exploratory: bool = False,
    device: str | torch.device = "cuda",
    batch_size: int = 64,
    start: int = 0,
    stop: int | None = None,
    frame_rate: float | None = None,
) -> Path:
    """Stream a read-only frame dataset into a new, atomically published output."""

    require_exploratory_opt_in(allow_exploratory)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if frame_rate is not None and (not np.isfinite(frame_rate) or frame_rate <= 0):
        raise ValueError("frame_rate must be finite and positive")
    # Declaration and checkpoint checks intentionally precede even source stat/open.
    model, declaration, checkpoint_sha256 = _load_declared_checkpoint(
        checkpoint_path, config_path, device
    )
    source_identity = SourceIdentity.from_path(source_path, dataset_path)
    with HDF5FrameSource(
        source_path,
        dataset_path,
        expected_ndim=3,
        allowed_dtypes=(np.uint8, np.float32, np.float64),
        max_frames_per_read=batch_size,
    ) as source:
        height, width = source.info.frame_shape
        final_stop = len(source) if stop is None else stop
        if start < 0 or final_stop > len(source) or final_stop <= start:
            raise ValueError("requested frame interval must be non-empty and within the source")
        engine = ExploratoryPoseInference(
            model,
            original_height=height,
            original_width=width,
            device=device,
            allow_exploratory=True,
            checkpoint_sha256=checkpoint_sha256,
            declaration=declaration,
        )
        provenance = OutputProvenance(
            source=source_identity,
            checkpoint_sha256=engine.checkpoint_sha256,
            config_sha256=declaration.config_sha256,
            git_commit=_git_commit(),
            package_versions=_package_versions(),
            geometry_convention=(
                "pixel centers; x right, y down; original-image pixels; angles clockwise "
                "[-pi,pi); rejected checkpoint; independent frames only"
            ),
            image_height=height,
            image_width=width,
        )
        writer = PoseHDF5Writer(
            output_path,
            body_points=int(engine.model.hparams.num_points),
            provenance=provenance,
            chunk_frames=min(batch_size, 256),
            overwrite=False,
        )
        with writer:
            _mark_exploratory(writer)
            for batch_start in range(start, final_stop, batch_size):
                batch_stop = min(batch_start + batch_size, final_stop)
                frames = source.read_slice(batch_start, batch_stop)
                prediction = engine.predict_batch(frames)
                indices = np.arange(batch_start, batch_stop, dtype=np.int64)
                timestamps = (
                    indices.astype(np.float64) / frame_rate
                    if frame_rate is not None
                    else np.full(len(indices), np.nan, dtype=np.float64)
                )
                writer.append(**prediction.writer_values(indices, timestamps))
    return Path(output_path)
