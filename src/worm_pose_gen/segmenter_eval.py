"""Score segmenter checkpoints on a split, and decide promotions.

The training script uses this to compare a new run's best checkpoint with
the currently promoted one on the validation split; the evaluation script
uses the same per-sample scoring so the two never disagree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .run_records import checkpoint_fingerprint
from .segmentation_dataset import SegmentationStore, is_hand_labeled
from .segmenter import IGNORE_LABEL, SegmentationModule, load_segmenter, masked_binary_metrics


def score_split(module: SegmentationModule, store: SegmentationStore, split: str) -> list[dict[str, Any]]:
    """Per-sample IoU of ``module`` on every label in ``split``."""

    rows = []
    for record in store.records(split):
        image, mask, _ = store.load(record.sample_id)
        probability = module.predict_probability(image)
        valid = torch.as_tensor(mask != IGNORE_LABEL)[None]
        target = torch.as_tensor((mask == 1).astype(np.float32))[None]
        metrics = masked_binary_metrics(torch.as_tensor(probability)[None], target, valid)
        rows.append({
            "sample_id": record.sample_id,
            "label_source": record.label_source,
            "revision": record.revision,
            "label_pixels": int(((mask == 1) & (mask != IGNORE_LABEL)).sum()),
            "iou": float(metrics["iou"][0]),
        })
    return rows


def summarize_iou(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean, median, and minimum IoU over the hand-refined rows (all rows if none are)."""

    hand = [r for r in rows if is_hand_labeled(r["label_source"])] or rows
    values = np.asarray([r["iou"] for r in hand], dtype=np.float64)
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "min": None}
    return {"n": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)), "min": float(values.min())}


def compare_on_split(
    candidate: Path, incumbent: Path | None, store: SegmentationStore, split: str = "val", device: str | None = None,
) -> dict[str, Any]:
    """Score both checkpoints on ``split`` and say whether the candidate wins.

    The decision uses mean IoU over the hand-refined labels of the split, so
    a model that paints worm onto empty frames or drops a thin tail is
    penalized in proportion, where a median would hide it.  A missing or
    unreadable incumbent means the candidate wins by default.
    """

    candidate_rows = score_split(load_segmenter(candidate, device), store, split)
    result: dict[str, Any] = {
        "split": split,
        "metric": "mean IoU over hand-refined labels",
        "candidate": checkpoint_fingerprint(candidate),
        "candidate_score": summarize_iou(candidate_rows),
        "incumbent": None,
        "incumbent_score": None,
    }
    if incumbent is None or not Path(incumbent).exists():
        result["promote"] = True
        result["reason"] = "no promoted checkpoint yet"
        return result
    incumbent_rows = score_split(load_segmenter(incumbent, device), store, split)
    result["incumbent"] = checkpoint_fingerprint(incumbent)
    result["incumbent_score"] = summarize_iou(incumbent_rows)
    new, old = result["candidate_score"]["mean"], result["incumbent_score"]["mean"]
    result["promote"] = bool(new is not None and (old is None or new > old))
    result["reason"] = (
        f"candidate mean {new:.4f} {'>' if result['promote'] else '<='} incumbent mean {old:.4f} on {split}"
        if new is not None and old is not None else "incumbent could not be scored"
    )
    result["per_sample"] = [
        {"sample_id": c["sample_id"], "candidate": c["iou"], "incumbent": i["iou"]}
        for c, i in zip(candidate_rows, incumbent_rows)
    ]
    return result
