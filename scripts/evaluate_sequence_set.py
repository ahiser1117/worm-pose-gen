#!/usr/bin/env python3
"""Fit and score the sequence evaluation set (plan step 4).

The set is a manifest of clips, each a recording and a frame range chosen
for coils, self-contact, enclosed holes, dropped fragments, or camera exits
(``docs/sequence_eval_set.json``, frame ranges only, no copies).  Each clip
is fit with ``scripts/fit_recording.py`` (video and residual images
included), the ambiguity signals are summarized per clip, and residual
images of the most ambiguous frames are rendered.  Results go to
``--output-dir/sequence_eval.json`` with the run directories recorded, so a
later step can compare against this one.

Example:

    scripts/project_env.sh uv run --no-sync --frozen python scripts/evaluate_sequence_set.py --clip coil_0623_a
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from worm_pose_gen.ambiguity import FLAG_NAMES
from worm_pose_gen.run_records import git_revision, utc_now


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "docs" / "sequence_eval_set.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "pose_pipeline_step4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip", action="append", dest="clips", help="clip names to run (default: all)")
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--prior", default="bootstrap", choices=("bootstrap", "none"))
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--ambiguous-frames", type=int, default=4, help="residual images for the most ambiguous frames per clip")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--extra", default="", help="extra arguments passed to fit_recording.py")
    parser.add_argument("--output-name", default="sequence_eval.json", help="result file inside --output-dir")
    return parser.parse_args()


def run_fit(clip: dict, args: argparse.Namespace) -> Path:
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "fit_recording.py"),
        "--recording", clip["recording"], "--start", str(clip["start"]), "--frames", str(clip["frames"]),
        "--preset", args.preset, "--prior", args.prior, "--video", "--scale", str(args.scale), "--residual-frames", "4",
        "--name", f"seq_{clip['name']}",
    ]
    if args.checkpoint is not None:
        command += ["--checkpoint", str(args.checkpoint)]
    if args.device is not None:
        command += ["--device", args.device]
    command += args.extra.split()
    completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        print(completed.stderr[-4000:], file=sys.stderr, flush=True)
        raise SystemExit(f"fit_recording failed on clip {clip['name']} (exit {completed.returncode}); its stderr tail is above")
    tail = completed.stdout[completed.stdout.rfind("\n{") :]
    summary = json.loads(tail)
    return Path(summary["outputs"]["poses"]).parent


def render_ambiguous(run_dir: Path, arrays: dict[str, np.ndarray], count: int, args: argparse.Namespace) -> list[int]:
    order = np.lexsort((arrays["iou"], -arrays["ambiguity_score"]))
    frames = [int(arrays["frame_index"][r]) for r in order if arrays["fitted"][r] and arrays["ambiguity_score"][r] > 0][:count]
    if frames:
        command = [sys.executable, str(PROJECT_ROOT / "scripts" / "render_pose_run.py"), str(run_dir), "--no-video", "--residual-frames", "0",
                   "--frames", ",".join(str(f) for f in frames)]
        if args.checkpoint is not None:
            command += ["--checkpoint", str(args.checkpoint)]
        if args.device is not None:
            command += ["--device", args.device]
        subprocess.run(command, check=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return frames


def clip_entry(clip: dict, run_dir: Path, seconds: float, ambiguous_frames: list[int]) -> dict:
    arrays = dict(np.load(run_dir / "poses.npz"))
    summary = json.loads((run_dir / "summary.json").read_text())
    fitted = arrays["fitted"]
    iou = arrays["iou"][fitted]
    score = arrays["ambiguity_score"]
    return {
        **{k: clip[k] for k in ("name", "recording", "start", "frames", "reasons")},
        "run": str(run_dir), "seconds": seconds,
        "frames_fitted": int(fitted.sum()),
        "iou_median": float(np.median(iou)), "iou_p10": float(np.percentile(iou, 10)), "iou_min": float(iou.min()),
        "frames_iou_below_0.9": int((iou < 0.9).sum()),
        "in_view_below_1": int((arrays["points_in_fov"][fitted] < arrays["centerline_xy"].shape[1]).sum()),
        "flag_counts": {name: int(arrays[f"flag_{name}"].sum()) for name in FLAG_NAMES},
        "frames_score_at_least_1": int((fitted & (score >= 1)).sum()),
        "frames_score_at_least_2": int((fitted & (score >= 2)).sum()),
        "iou_by_score": summary["ambiguity"]["iou_by_score"],
        "orientation": summary.get("orientation"),
        "orientation_flips": (summary.get("width_model") or {}).get("orientation_consistency"),
        "prior": None if summary.get("prior") is None else {k: summary["prior"][k] for k in ("length_px", "width_px", "frames_used")},
        "propagation": summary.get("propagation"),
        "iou_independent_median": float(np.median(arrays["iou_independent"][fitted])) if "iou_independent" in arrays else None,
        "frames_iou_independent_below_0.9": int((arrays["iou_independent"][fitted] < 0.9).sum()) if "iou_independent" in arrays else None,
        "fit_ms_per_frame": summary["ms_per_frame"]["fit"],
        "ambiguous_frames_rendered": ambiguous_frames,
        "video": summary["outputs"]["video"],
    }


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    clips = [c for c in manifest["clips"] if not args.clips or c["name"] in args.clips]
    if not clips:
        raise SystemExit("no clips selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    output = json.loads(output_path.read_text()) if output_path.exists() else {"clips": {}}
    output.update({"generated_at": utc_now(), "git": git_revision(PROJECT_ROOT), "manifest": str(args.manifest.relative_to(PROJECT_ROOT)), "preset": args.preset, "prior": args.prior})
    for clip in clips:
        started = time.perf_counter()
        run_dir = run_fit(clip, args)
        arrays = dict(np.load(run_dir / "poses.npz"))
        ambiguous = render_ambiguous(run_dir, arrays, args.ambiguous_frames, args)
        entry = clip_entry(clip, run_dir, time.perf_counter() - started, ambiguous)
        output["clips"][clip["name"]] = entry
        output_path.write_text(json.dumps(output, indent=1))
        print(
            f"{clip['name']:>16s}: {entry['frames_fitted']} frames, median IoU {entry['iou_median']:.3f}, p10 {entry['iou_p10']:.3f},"
            f" below 0.9: {entry['frames_iou_below_0.9']}, score>=1: {entry['frames_score_at_least_1']}, flags {entry['flag_counts']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
