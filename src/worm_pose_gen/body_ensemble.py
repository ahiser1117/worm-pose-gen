"""Body-field ensembles and flip test-time augmentation, as one loadable network.

A :class:`BodyFieldEnsemble` runs each member network on each flipped copy
of its input (:data:`FLIPS`), flips the outputs back, averages the
*probabilities* of all those views and returns their logits, so it is a
drop-in replacement for a :class:`body_net.BodyFieldModule`: every consumer
takes ``sigmoid(module(inputs))`` and reads ``.lags`` and ``.device``.

The difference channels flip with the frame (a flip of the whole input
stack), and every output is a per-pixel field that flips with it, A-P
included (arc position does not depend on the image orientation), so the
un-flipped views average directly.  Members must share their lags.

An ensemble is stored as one checkpoint file holding every member's
hyperparameters and weights (:func:`save_ensemble`), under
:data:`CHECKPOINT_KEY`; :func:`body_net.load_body_net` recognises it and
loads plain checkpoints as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import Tensor

from .body_net import BodyFieldModule


CHECKPOINT_KEY = "body_ensemble"
# Spatial dims of a [B,C,H,W] tensor each named flip reverses.
FLIPS: dict[str, tuple[int, ...]] = {"none": (), "h": (-1,), "v": (-2,), "hv": (-2, -1)}
# The averaged probability is clipped this far from 0 and 1 before its logit (float32 keeps 1 - 1e-6 apart from 1).
PROBABILITY_EPS = 1e-6


class BodyFieldEnsemble(torch.nn.Module):
    """The mean probability of ``members`` over the flipped views ``flips`` (names of :data:`FLIPS`), as logits."""

    def __init__(self, members: Sequence[BodyFieldModule], flips: Sequence[str] = ("none",)) -> None:
        super().__init__()
        if not members:
            raise ValueError("an ensemble needs at least one member")
        lags = {tuple(member.lags) for member in members}
        if len(lags) != 1:
            raise ValueError(f"ensemble members must share their lags; got {sorted(lags)}")
        unknown = [name for name in flips if name not in FLIPS]
        if unknown or not flips or len(set(flips)) != len(flips):
            raise ValueError(f"flips must be distinct names of {tuple(FLIPS)}; got {list(flips)}")
        self.members = torch.nn.ModuleList(members)
        self.flips = tuple(flips)
        self.lags = lags.pop()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def views(self) -> int:
        return len(self.members) * len(self.flips)

    def view_probabilities(self, images: Tensor) -> Iterator[Tensor]:
        """Each (member, flip) view's ``[B,5,H,W]`` float32 sigmoids, flipped back to the input's orientation."""

        for member in self.members:
            for name in self.flips:
                dims = FLIPS[name]
                logits = member(images.flip(dims) if dims else images).float()
                probability = torch.sigmoid(logits)
                yield probability.flip(dims) if dims else probability

    def forward(self, images: Tensor) -> Tensor:
        total = None
        for probability in self.view_probabilities(images):
            total = probability if total is None else total + probability
        with torch.autocast(device_type=images.device.type, enabled=False):
            mean = total / self.views
            return torch.logit(mean.clamp(PROBABILITY_EPS, 1.0 - PROBABILITY_EPS))


def _member_entry(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if CHECKPOINT_KEY in checkpoint:
        raise ValueError(f"{checkpoint_path} is itself an ensemble; list its members instead")
    return {"source": str(checkpoint_path), "hyper_parameters": dict(checkpoint["hyper_parameters"]),
            "state_dict": checkpoint["state_dict"]}


def save_ensemble(path: str | Path, member_checkpoints: Sequence[str | Path], flips: Sequence[str] = ("none",)) -> None:
    """Write one checkpoint holding every member's weights, checked by building the ensemble on the CPU."""

    entry = {"flips": list(flips), "members": [_member_entry(p) for p in member_checkpoints]}
    from_entry(entry)
    torch.save({CHECKPOINT_KEY: entry}, str(path))


def from_entry(entry: dict[str, Any]) -> BodyFieldEnsemble:
    """The ensemble a checkpoint's :data:`CHECKPOINT_KEY` entry describes, on the CPU."""

    members = []
    for member in entry["members"]:
        module = BodyFieldModule(**{**member["hyper_parameters"], "pretrained": False})
        module.load_state_dict(member["state_dict"])
        members.append(module)
    return BodyFieldEnsemble(members, entry["flips"])


def is_ensemble_checkpoint(checkpoint_path: str | Path) -> bool:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False, mmap=True)
    return isinstance(checkpoint, dict) and CHECKPOINT_KEY in checkpoint


def load_ensemble(checkpoint_path: str | Path) -> BodyFieldEnsemble:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    return from_entry(checkpoint[CHECKPOINT_KEY])
