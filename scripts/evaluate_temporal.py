#!/usr/bin/env python3
"""Label-free temporal consistency of a body-field network on the clips of eval_sets/temporal_eval_set.json.

See :mod:`worm_pose_gen.temporal_eval` for the clips and metrics.  Writes
``<output-dir>/temporal_eval.json`` (per-clip metrics, a summary over all
clips and per group) and ``<output-dir>/predictions/<clip>.npz`` (for
``scripts/render_field_comparison.py``).  ``--cache-only`` fills the clip
frame cache without a model.  Example::

    scripts/project_env.sh env CUDA_VISIBLE_DEVICES=3 uv run --no-sync --frozen python scripts/evaluate_temporal.py \\
        --model mine:on-food-finetune-329 --output-dir <data dir>/runs/temporal/on-food-finetune-329
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from worm_pose_gen import temporal_eval
from worm_pose_gen.run_records import git_revision, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", help="library model reference, e.g. mine:on-food-finetune-329")
    source.add_argument("--checkpoint", type=Path, help="body-field checkpoint (anything body_net.load_body_net reads)")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=temporal_eval.DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=temporal_eval.DEFAULT_CACHE)
    parser.add_argument("--clip", action="append", dest="clips", help="clip names (default: all)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-predictions", action="store_true", help="do not keep the per-clip predictions")
    parser.add_argument("--cache-only", action="store_true", help="only fill the clip frame cache")
    args = parser.parse_args()
    if not args.cache_only and (args.output_dir is None or (args.model is None and args.checkpoint is None)):
        parser.error("--output-dir and one of --model/--checkpoint are needed unless --cache-only")
    return args


def main() -> int:
    args = parse_args()
    chosen = temporal_eval.clips(args.manifest, args.clips)
    if args.cache_only:
        for clip in chosen:
            t = time.time()
            frames, first = temporal_eval.cached_clip(clip, args.cache_dir)
            print(f"{clip.name}: {len(frames)} frames from {first} ({time.time() - t:.0f} s)", flush=True)
        return 0

    import torch

    from worm_pose_gen.body_net import load_body_net

    checkpoint = args.checkpoint
    if args.model:
        from worm_pose_gen import library

        checkpoint = library.weights_path(library.Libraries.for_host(), args.model)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    module = load_body_net(checkpoint, device=device)
    per_clip: dict[str, dict] = {}
    for clip in chosen:
        t = time.time()
        frames, first = temporal_eval.cached_clip(clip, args.cache_dir)
        prediction = temporal_eval.predict_clip(module, frames, first, clip)
        if not args.no_predictions:
            temporal_eval.save_prediction(args.output_dir / "predictions" / f"{clip.name}.npz", prediction, clip)
        per_clip[clip.name] = temporal_eval.clip_metrics(prediction)
        m = per_clip[clip.name]
        print(
            f"{clip.name:>18}: flicker {100 * m['mask_flicker']:.2f}%  ap jitter {m['ap_jitter']:.4f}  ap flips {m['ap_flip_frames']}"
            f"  head jumps {m['head_jumps']} swaps {m['head_swaps']} jitter {m['head_jitter_px']:.1f}px  ({time.time() - t:.0f} s)", flush=True,
        )
    groups = {c.name: c.group for c in chosen}
    result = {
        "model": args.model, "checkpoint": str(checkpoint), "lags": list(module.lags), "generated_at": utc_now(),
        "git": git_revision(temporal_eval.PROJECT_ROOT), "manifest": str(args.manifest), "per_clip": per_clip, "summary": temporal_eval.summarize(per_clip, groups),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "temporal_eval.json").write_text(json.dumps(result, indent=1))
    for group, s in result["summary"].items():
        print(
            f"[{group}] flicker {100 * s['mask_flicker']:.2f}%  ap jitter {s['ap_jitter']:.4f}  ap flips {s['ap_flip_frames']:.0f}"
            f"  head jumps {s['head_jumps']:.0f} swaps {s['head_swaps']:.0f} jitter {s['head_jitter_px']:.1f}px  tail jumps {s['tail_jumps']:.0f}",
            flush=True,
        )
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
