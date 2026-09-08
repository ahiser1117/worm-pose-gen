#!/usr/bin/env python3
"""Build a labeling manifest of targeted frames for the segmenter (labeling round 2).

The round targets the frames where the segmenter merges adjacent body turns
or loses the body at the camera edge: coils, self-contact, enclosed holes,
fragments, and border contact.  Candidates come from the mask-only scans of
``scripts/find_sequence_clips.py`` (one ``<recording>_candidates.json`` per
recording); each candidate window contributes its peak frame and, for longer
windows, its first and last sampled frames.  Border-touching samples and a
spread of ordinary frames are added per recording so the labels also cover
the easy cases of every animal.

Recordings are assigned a split policy: ``auto`` (the store's balanced
per-frame assignment, as before) or one of ``train``, ``val``, ``test`` for
recordings whose animal should appear in that split only.  The manifest is
read by ``worm_pose_gen.label_app --queue``.

Example:

    scripts/project_env.sh uv run --no-sync --frozen python scripts/build_labeling_manifest.py \\
        --val-only 2023-09-07-13 --val-only 2024-02-01-07 --test-only 2024-06-18-12 --test-only 2024-05-28-02 \\
        --output docs/labeling_round_2/manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from worm_pose_gen.segmentation_dataset import DEFAULT_DATASET_ROOT, SegmentationStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDING_ROOT = Path("/store1/shared/all_data_raw/prj_aversion")
AMBIGUOUS_REASONS = ("short_skeleton", "holes", "fragments")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates-dir", type=Path, default=PROJECT_ROOT / "docs" / "pose_pipeline_step4" / "clip_candidates")
    parser.add_argument("--recording", action="append", dest="recordings", help="recording names to include (default: every scan in the directory)")
    parser.add_argument("--val-only", action="append", default=[], help="recording whose new labels are all validation")
    parser.add_argument("--test-only", action="append", default=[], help="recording whose new labels are all test")
    parser.add_argument("--train-only", action="append", default=[], help="recording whose new labels are all training")
    parser.add_argument("--window-frames", type=int, default=3, help="frames per candidate window (peak, first, last)")
    parser.add_argument("--max-window-frames", type=int, default=30, help="cap on window frames per recording")
    parser.add_argument("--border-frames", type=int, default=6, help="border-touching samples per recording")
    parser.add_argument("--ordinary-frames", type=int, default=6, help="evenly spread ordinary frames per recording")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--name", default="labeling_round_2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs" / "labeling_round_2" / "manifest.json")
    return parser.parse_args()


def recording_path(name: str) -> Path:
    return RECORDING_ROOT / name[:10] / f"{name}.h5"


def main() -> int:
    args = parse_args()
    store = SegmentationStore(args.dataset_root)
    rng = np.random.default_rng(args.seed)
    policy: dict[str, str] = {}
    for split, names in (("val", args.val_only), ("test", args.test_only), ("train", args.train_only)):
        for name in names:
            policy[name] = split
    scans = {p.name.replace("_candidates.json", ""): p for p in sorted(args.candidates_dir.glob("*_candidates.json"))}
    names = args.recordings or sorted(scans)
    recordings: dict[str, dict] = {}
    frames: list[dict] = []
    summary: dict[str, dict] = {}
    for name in names:
        if name not in scans:
            raise SystemExit(f"no candidate scan for {name} in {args.candidates_dir}")
        data = json.loads(scans[name].read_text())
        path = recording_path(name)
        if not path.exists():
            path = Path(data["recording"])
        recordings[name] = {"path": str(path), "split": policy.get(name, "auto")}
        chosen: dict[int, list[str]] = {}

        def add(frame: int, reason: str) -> None:
            if store.has(name, frame):
                return
            chosen.setdefault(int(frame), [])
            if reason not in chosen[int(frame)]:
                chosen[int(frame)].append(reason)

        # Candidate windows, most ambiguous first: coils (short skeleton), then holes, then fragments.
        windows = sorted(
            data.get("candidates", []),
            key=lambda w: (-(w["reasons"].get("short_skeleton", 0) + w["reasons"].get("holes", 0)), -w["samples"]),
        )
        budget = args.max_window_frames
        for window in windows:
            if budget <= 0:
                break
            reasons = ",".join(sorted(k for k in window["reasons"] if k in AMBIGUOUS_REASONS)) or "border"
            picks = [window["peak_frame"]]
            if args.window_frames >= 3 and window["end"] > window["start"]:
                picks += [window["start"], window["end"]]
            for frame in picks[: args.window_frames]:
                if budget <= 0:
                    break
                before = len(chosen)
                add(frame, f"window:{reasons}")
                budget -= int(len(chosen) > before)
        # Border-touching samples and ordinary frames from the per-sample rows when the scan kept them.
        samples = data.get("samples", [])
        border = [s["frame_index"] for s in samples if s.get("touches_border") and s["worm_pixels"] >= 500]
        ordinary = [s["frame_index"] for s in samples if not s.get("touches_border") and not s.get("ambiguous") and s["worm_pixels"] >= 500]
        total_frames = int(data.get("frames_scanned", 0)) * int(data.get("stride", 1))
        if border:
            for frame in rng.choice(border, size=min(args.border_frames, len(border)), replace=False):
                add(int(frame), "border")
        if ordinary:
            step = max(1, len(ordinary) // max(args.ordinary_frames, 1))
            for frame in ordinary[::step][: args.ordinary_frames]:
                add(int(frame), "ordinary")
        elif total_frames:
            for frame in np.linspace(0, total_frames - 1, args.ordinary_frames + 2)[1:-1]:
                add(int(frame), "ordinary")
        for frame in sorted(chosen):
            frames.append({"recording": name, "frame_index": int(frame), "reasons": chosen[frame]})
        summary[name] = {
            "split": recordings[name]["split"],
            "frames": len(chosen),
            "by_reason": {r: sum(1 for v in chosen.values() if any(x.startswith(r) for x in v)) for r in ("window", "border", "ordinary")},
            "already_labeled_skipped": sum(1 for w in windows for f in [w["peak_frame"]] if store.has(name, f)),
        }
    manifest = {
        "name": args.name,
        "description": (
            "Targeted labeling round for the segmenter: coils, self-contact, enclosed holes, fragments and camera-edge frames "
            "from the clip-candidate scans, plus border and ordinary frames per recording. Recordings marked val/test hold "
            "animals that appear in that split only; 'auto' keeps the store's balanced per-frame assignment."
        ),
        "candidates_dir": str(args.candidates_dir.relative_to(PROJECT_ROOT)) if args.candidates_dir.is_relative_to(PROJECT_ROOT) else str(args.candidates_dir),
        "recordings": recordings,
        "frames": frames,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=1))
    print(f"{len(frames)} frames over {len(recordings)} recordings -> {args.output}")
    for name, row in summary.items():
        print(f"  {name}: {row['frames']:3d} frames, split {row['split']}, {row['by_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
