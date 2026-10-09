#!/usr/bin/env python3
"""Score the tube fitter with and without body-field evidence against reviewed held-out poses.

Ground truth is the body-field records of validation and test samples that a
person corrected or confirmed: ``traced`` records (midlines traced by hand,
mostly contacts and coils) and other ``accepted`` records (ordinary postures,
the regression check).  Every sample is fit to its hand mask four ways:

``baseline``  the standard starts (and the skeleton start reversed), oriented
              afterwards by the body's taper, as the pipeline does without a
              recording prior
``trace``     the same plus the network-proposed trace as a start
``fields``    the standard starts as the pipeline's fit stage builds them
              with the network: head first by the predicted ends where both
              are in view (``mask_fit.head_first``), else both orientations;
              scored against the network's A-P field and end points
              (``MaskFitConfig.field_*``)
``both``      ``fields`` plus the trace start, head first (the pipeline)

The trace start is laid from the head at the length of the sample's
recording (``body_fields.recording_length``, standing in for the recording
prior), and the ``fields`` and ``both`` fits keep that length, as the fit
stage does with a prior and the network; samples without one are fit with
a free length.

Per sample: the mean distance in body widths from each visible point of the
ground-truth midline to the fitted midline (blind to length and orientation,
so a body continued off camera is not penalized), the mean A-P error over
the mask against the ground truth's targets (wrong routes through a contact
and wrong orientations both show here), whether the head is at the true head
end, and the mask IoU.  The network sees each frame's stored context, as it
would in a recording.  Writes ``<out>/per_sample.csv`` and
``<out>/summary.json`` and prints a table per ground-truth group.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from worm_pose_gen import body_fields
from worm_pose_gen.batch_fit import PRESETS, fit_masks
from worm_pose_gen.body_net import load_body_net
from worm_pose_gen.body_proposal import FIELD_AP_WEIGHT, FIELD_END_WEIGHT, field_evidence, predict_fields, trace_start
from worm_pose_gen.body_targets import render_body_targets
from worm_pose_gen.classical import resample_centerline
from worm_pose_gen.mask_fit import default_width_template, head_first, orient_tail_last, orientation_pair
from worm_pose_gen.pipeline import initializations_for
from worm_pose_gen.segmentation_dataset import DEFAULT_DATASET_ROOT, SegmentationStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("baseline", "trace", "fields", "both")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "body_net" / "best.ckpt")
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--ap-weight", type=float, default=FIELD_AP_WEIGHT)
    parser.add_argument("--end-weight", type=float, default=FIELD_END_WEIGHT)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "checkpoints" / "body_net" / "evaluations" / "field_fitting")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def ground_truth(store: SegmentationStore, splits: list[str]) -> list[tuple]:
    cases = []
    for record in store.records():
        if record.split not in splits:
            continue
        path = body_fields.field_path(store.root, record.sample_id)
        if not path.exists():
            continue
        arrays, meta = body_fields.load(path, ("context", "context_valid", "centerline_xy", "width_profile", "ap"))
        if not meta.get("has_body") or body_fields.is_stale(meta, record):
            continue
        if meta.get("fit_method") in body_fields.TRACE_METHODS:
            group = "traced"
        elif body_fields.review_status(meta) == "accepted":
            group = "accepted"
        else:
            continue
        _, label, _ = store.load(record.sample_id)
        cases.append((record, group, label == 1, arrays))
    return cases


def score(result_curve, result_widths, iou, mask, truth) -> dict[str, float]:
    true_curve, true_widths, true_ap = truth["centerline_xy"], truth["width_profile"], truth["ap"].astype(np.float32)
    height, width = mask.shape
    inside = (true_curve[:, 0] >= 0) & (true_curve[:, 0] <= width - 1) & (true_curve[:, 1] >= 0) & (true_curve[:, 1] <= height - 1)
    scale = float(np.median(true_widths))
    # Each visible ground-truth point's distance to the fitted midline: blind to
    # length and orientation, so a body continued off camera is not penalized.
    dense = resample_centerline(np.asarray(result_curve, dtype=np.float64), 1000)
    nearest = np.linalg.norm(true_curve[inside][:, None, :] - dense[None, :, :], axis=-1).min(1)
    ap = render_body_targets(mask, result_curve, result_widths).ap
    known = np.isfinite(ap) & np.isfinite(true_ap)
    return {
        "distance_widths": float(nearest.mean()) / scale,
        "ap_error": float(np.abs(ap[known] - true_ap[known]).mean()) if known.any() else float("nan"),
        "head_correct": float(np.linalg.norm(result_curve[0] - true_curve[0]) < np.linalg.norm(result_curve[0] - true_curve[-1])),
        "iou": iou,
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    store = SegmentationStore(args.dataset_root)
    module = load_body_net(args.checkpoint, device)
    config = PRESETS[args.preset]
    field_config = replace(config, field_ap_weight=args.ap_weight, field_end_weight=args.end_weight)
    template = default_width_template(config.n_points)
    cases = ground_truth(store, args.splits)
    masks = [mask for _, _, mask, _ in cases]
    predictions = [predict_fields(module, arrays["context"], arrays["context_valid"]) for *_, arrays in cases]
    evidence = [field_evidence(p, m) for p, m in zip(predictions, masks)]
    lengths = [body_fields.recording_length(store, record) for record, *_ in cases]
    traces = [trace_start(p, m, config=config, length_px=length) for p, m, length in zip(predictions, masks, lengths)]
    standard = []
    for mask in masks:
        starts = initializations_for(mask, config)
        skeleton = next((s for s in starts if s.name == "skeleton_longest_path"), None)
        standard.append(starts + ([orientation_pair(skeleton, config=config)[1]] if skeleton is not None else []))

    def with_evidence(starts, frame_evidence):
        """The standard starts as ``pipeline.fit_frames`` builds them with the network."""

        base = [s for s in starts if not s.name.endswith("_reversed")]
        if frame_evidence.head_xy is not None and frame_evidence.tail_xy is not None:
            return [head_first(s, frame_evidence.head_xy, frame_evidence.tail_xy, config=config) for s in base]
        return [s for start in base for s in orientation_pair(start, config=config)]

    rows = []
    for variant in args.variants:
        with_trace = variant in ("trace", "both")
        uses_fields = variant in ("fields", "both")
        base = [with_evidence(s, e) for s, e in zip(standard, evidence)] if uses_fields else standard
        starts = [s + ([t] if with_trace and t is not None else []) for s, t in zip(base, traces)]
        if uses_fields:
            # One fit per recording length, held fixed (free where a sample has none).
            results = [None] * len(cases)
            for length in set(lengths):
                group = [k for k, other in enumerate(lengths) if other == length]
                group_config = field_config if length is None else replace(field_config, length_prior_px=length, length_fixed=True)
                fitted = fit_masks(
                    [masks[k] for k in group], [starts[k] for k in group], width_template=template, config=group_config,
                    device=device, fields=[evidence[k] for k in group],
                )
                for k, result in zip(group, fitted, strict=True):
                    results[k] = result
        else:
            results = fit_masks(masks, starts, width_template=template, config=config, device=device)
        for (record, group, mask, arrays), result in zip(cases, results):
            if not uses_fields:
                result, _ = orient_tail_last(result, config=config)
            iou = float(result.records[result.best_index]["final_iou"])
            row = {"variant": variant, "group": group, "sample_id": record.sample_id, "split": record.split,
                   "winner": str(result.initializations[result.best_index].name), "starts": len(result.initializations)}
            row.update(score(result.centerline_xy, result.width_profile, iou, mask, arrays))
            rows.append(row)
        print(f"{variant}: fitted {len(results)} samples", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "per_sample.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    print(f"{'group':9s} {'variant':9s} {'n':>3s} {'dist med':>8s} {'<0.5 w':>7s} {'A-P med':>8s} {'A-P mean':>8s} {'head':>6s} {'IoU med':>7s}")
    for group in ("traced", "accepted"):
        for variant in args.variants:
            sel = [r for r in rows if r["group"] == group and r["variant"] == variant]
            if not sel:
                continue
            d = np.array([r["distance_widths"] for r in sel]); ap = np.array([r["ap_error"] for r in sel])
            entry = {
                "n": len(sel), "distance_median": float(np.median(d)), "within_half_width": int((d < 0.5).sum()),
                "ap_error_median": float(np.nanmedian(ap)), "ap_error_mean": float(np.nanmean(ap)),
                "head_correct": float(np.mean([r["head_correct"] for r in sel])),
                "iou_median": float(np.median([r["iou"] for r in sel])),
                "trace_won": int(sum(r["winner"].startswith("network_trace") for r in sel)),
                "starts_mean": float(np.mean([r["starts"] for r in sel])),
            }
            summary[f"{group}/{variant}"] = entry
            print(f"{group:9s} {variant:9s} {entry['n']:3d} {entry['distance_median']:8.2f} {entry['within_half_width']:4d}/{entry['n']:<3d}"
                  f"{entry['ap_error_median']:8.3f} {entry['ap_error_mean']:8.3f} {entry['head_correct']:6.2f} {entry['iou_median']:7.3f}")
    (args.out / "summary.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "summary": summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
