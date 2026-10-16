#!/usr/bin/env python3
"""Run the region algorithms on the ambiguous stretches of the sequence set.

Each clip of ``docs/sequence_eval_set.json`` is analysed once into a
workspace under ``--root`` (the default stages: segment, prior, fit with
``--body-net``, ambiguity, propagate, track), then every algorithm of
``--algorithms`` runs on every stretch the propagate stage refit, with the
nearest good frames as anchors (``algorithms.propose_anchors``), as the
Workspace page's Refit fix runs them.  Nothing is installed, so the
workspaces stay as analysed and another checkout of the code (another
``PYTHONPATH``) can be scored on the same regions.  Per algorithm and clip
the result sums ``region_metrics`` over the stretches with each path
applied: frames below IoU 0.9, pose jumps over a body width, length jumps
over 3%, orientation flips, and seconds.

Example:

    scripts/project_env.sh uv run --no-sync --frozen python scripts/evaluate_region_algorithms.py \\
        --root /temp_data4/alex/external_artifacts/region_eval --body-net <library>/models/nir-body-lags3/weights.ckpt \\
        --output region_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from worm_pose_gen import algorithms
from worm_pose_gen.pipeline import read_summary, run_all
from worm_pose_gen.run_records import git_revision, utc_now
from worm_pose_gen.workspace import Workspace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "docs" / "sequence_eval_set.json"
DEFAULT_ALGORITHMS = "independent_multistart,beam_path,slow_refit,fixed_body_smoother,mirror"
COUNTS = ("frames", "frames_below_0_9", "pose_jumps_over_width", "length_jumps_over_3pct", "orientation_flips")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--clip", action="append", dest="clips", help="clip names to run (default: all)")
    parser.add_argument("--root", type=Path, required=True, help="directory of the clips' workspaces (made when missing)")
    parser.add_argument("--body-net", default=None, help="body-field network the workspaces are fit with")
    parser.add_argument("--checkpoint", default=None, help="segmenter checkpoint (default: the segment stage's)")
    parser.add_argument("--algorithms", default=DEFAULT_ALGORITHMS, help="comma-separated algorithm ids")
    parser.add_argument("--params", default="{}", help="JSON object: algorithm id -> its parameters")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True, help="result JSON (clips already in it are kept)")
    return parser.parse_args()


def clip_workspace(clip: dict, args: argparse.Namespace) -> Workspace:
    """The clip's analysed workspace, analysed now when it does not exist yet."""

    path = args.root / f"seq_{clip['name']}"
    if (path / "workspace.json").exists():
        return Workspace.open(path)
    workspace = Workspace.create(args.root, path.name, Path(clip["recording"]), int(clip["start"]), int(clip["start"]) + int(clip["frames"]) - 1)
    shared = {k: v for k, v in (("body_net", args.body_net), ("checkpoint", args.checkpoint)) if v is not None}
    started = time.perf_counter()
    run_all(workspace, {"*": shared}, device=args.device)
    print(f"{clip['name']}: analysed in {time.perf_counter() - started:.0f} s", flush=True)
    return workspace


def run_clip(workspace: Workspace, names: list[str], params: dict, args: argparse.Namespace) -> dict:
    state = workspace.load_state()
    stretches = [(int(a), int(b)) for a, b in (read_summary(workspace).get("propagation") or {}).get("stretches") or []]
    out: dict = {"stretches": stretches, "algorithms": {}}
    for name in names:
        totals = {"before": dict.fromkeys(COUNTS, 0), "after": dict.fromkeys(COUNTS, 0), "seconds": 0.0, "regions": 0, "errors": []}
        for first, last in stretches:
            before, after = algorithms.propose_anchors(state, first, last)
            try:
                result = algorithms.run_algorithm(workspace, name, first, last, params.get(name), anchor_before=before, anchor_after=after, device=args.device)
            except ValueError as error:
                totals["errors"].append(f"{first}-{last}: {error}")
                continue
            totals["regions"] += 1
            totals["seconds"] += float(result.metrics["seconds"])
            for key in COUNTS:
                totals["before"][key] += int(result.metrics_before[key])
                totals["after"][key] += int(result.metrics[key])
        out["algorithms"][name] = totals
        print(f"{workspace.info.name} {name}: {totals['regions']} regions, before {totals['before']}, after {totals['after']}, {totals['seconds']:.0f} s", flush=True)
    return out


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    clips = [c for c in manifest["clips"] if not args.clips or c["name"] in args.clips]
    if not clips:
        raise SystemExit("no clips selected")
    names = [n for n in args.algorithms.split(",") if n]
    params = json.loads(args.params)
    args.root.mkdir(parents=True, exist_ok=True)
    output = json.loads(args.output.read_text()) if args.output.exists() else {"clips": {}}
    output.update({"generated_at": utc_now(), "git": git_revision(PROJECT_ROOT), "code": algorithms.__file__, "params": params, "root": str(args.root)})
    for clip in clips:
        output["clips"][clip["name"]] = run_clip(clip_workspace(clip, args), names, params, args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
