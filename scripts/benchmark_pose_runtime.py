"""Reproducible CPU/CUDA pose-stage benchmark with saved numerical outputs.

Run each source revision in a separate process with the same arguments. The
default fixture is synthetic: it exercises workspace I/O, mask cleanup,
initialization, the unchanged fast fitting schedule, ambiguity and export.
It does not benchmark learned segmentation or recording-prior bootstrap.
Use --propagate to also exercise both temporal chains through interior frames.
--masks accepts an NPZ containing a boolean [frames, height, width] `masks`
array, for replay of previously saved development masks without source video.
--recording with --checkpoint measures actual learned segmentation and cached
flat-field correction; --prior-file also includes loading a recording prior.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--propagate", action="store_true")
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prior-file", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--fit-overrides", type=json.loads, default={}, help="JSON BatchFitConfig overrides (recorded in the report)")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if min(args.repeats, args.frames, args.threads) < 1:
        parser.error("repeats, frames and threads must be positive")
    if args.recording and (args.masks or not args.checkpoint or args.start < 0):
        parser.error("--recording requires --checkpoint and a nonnegative --start, and cannot be combined with --masks")
    # Bound BLAS as well as PyTorch; otherwise small initialization solves can
    # start hundreds of threads and contaminate the renderer measurements.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    sys.path.insert(0, str(args.source_root.resolve()))
    import h5py
    import numpy as np
    import torch
    from worm_pose_gen import batch_fit, pipeline
    from worm_pose_gen.workspace import Workspace

    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.recording:
        with h5py.File(args.recording, "r") as handle:
            source_shape = handle["/img_nir"].shape
        frame_count = min(args.frames, source_shape[0] - args.start)
        if frame_count < 1:
            parser.error("selected recording range is empty")
        shape = [frame_count, *source_shape[1:]]
        masks, overrides = None, {}
    elif args.masks:
        with np.load(args.masks, allow_pickle=False) as data:
            masks = np.asarray(data["masks"][: args.frames], dtype=bool)
        overrides = {}
    else:
        # Analytic varying-width tubes, independent of the renderer under test.
        yy, xx = np.mgrid[:128, :192]
        masks = []
        for i in range(args.frames):
            x = xx - (94 + 2 * np.sin(i))
            y = yy - (62 + 2 * np.cos(i))
            center_y = 12 * np.sin(x / 32 + i * 0.15)
            radius = 7 * np.sqrt(np.maximum(0, 1 - (x / 66) ** 2))
            masks.append((np.abs(y - center_y) < radius) & (np.abs(x) < 66))
        masks = np.asarray(masks)
        overrides = {
            "length_bounds_px": [60.0, 300.0], "width_bounds_px": [4.0, 30.0],
            "default_length_px": 140.0, "default_width_px": 12.0,
        }
    if masks is not None:
        if masks.ndim != 3 or not len(masks):
            parser.error("masks must be a nonempty [frames, height, width] array")
        frame_count, shape = len(masks), list(masks.shape)
    overrides.update(args.fit_overrides)
    params = {
        "checkpoint": str(args.checkpoint) if args.recording else None,
        "flat_field": bool(args.recording), "prior": "bootstrap" if args.prior_file else "none",
        "compile": args.compile, "init_workers": 0, "min_worm_pixels": 100,
        "overrides": overrides,
    }
    if args.prior_file:
        params["prior_file"] = str(args.prior_file)
    if args.dataset_root:
        params["dataset_root"] = str(args.dataset_root)
    stages = ["segment"] + (["prior"] if args.prior_file else []) + ["fit", "ambiguity"]
    if args.propagate:
        stages += ["propagate", "track"]
    stages += ["export"]
    source_hash = hashlib.sha256()
    for source in sorted((args.source_root / "worm_pose_gen").rglob("*.py")):
        source_hash.update(str(source.relative_to(args.source_root)).encode())
        source_hash.update(source.read_bytes())
    report = {
        "source_root": str(args.source_root.resolve()), "python": platform.python_version(),
        "source_sha256": source_hash.hexdigest(),
        "torch": torch.__version__, "device": args.device, "threads": args.threads,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "compile": args.compile, "shape": shape, "params": params,
        "stages": stages, "input": str(args.recording or args.masks or "synthetic"),
        "first_frame": args.start if args.recording else 0,
        "propagation_params": {"min_score": 0, "pad": 0, "beam": 1, "anchor_diversity": False, "jump_seeds": False} if args.propagate else None,
        "limitations": ("Learned segmentation and flat-field correction; no prior bootstrap or video export."
                        if args.recording else "Threshold segmentation; no learned network, flat-field estimation, prior bootstrap or video export."),
        "runs": [],
    }
    profiler = cProfile.Profile() if args.profile else None
    with tempfile.TemporaryDirectory(prefix="pose-runtime-") as directory:
        root = Path(directory)
        recording = args.recording or (root / "fixture.h5")
        first_frame = args.start if args.recording else 0
        if masks is not None:
            with h5py.File(recording, "w") as handle:
                handle.create_dataset("/img_nir", data=np.where(masks, 60, 200).astype(np.uint8))
        for repeat in range(args.repeats):
            workspace = Workspace.create(root, f"run-{repeat}", recording, first_frame, first_frame + frame_count - 1, settings=params)
            fallback_before = getattr(batch_fit, "_ENERGY_COMPILE_FALLBACKS", 0)
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            timings, summaries = {}, {}
            if profiler is not None:
                profiler.enable()
            for stage in stages:
                if stage == "propagate":
                    # A fixed interior stretch exercises both anchored chains
                    # regardless of tiny fit differences between revisions.
                    workspace.set_provenance([0, frame_count - 1], "benchmark_anchor", "benchmark", time.time())
                stage_params = {**params, "name": "benchmark"}
                if stage == "propagate":
                    stage_params.update({"min_score": 0, "pad": 0, "beam": 1, "anchor_diversity": False, "jump_seeds": False})
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                summary = pipeline.run_stage(workspace, stage, stage_params, device=args.device)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                timings[stage] = time.perf_counter() - start
                summaries[stage] = summary
                print(f"repeat {repeat + 1}: {stage} {timings[stage]:.3f}s", flush=True)
            if profiler is not None:
                profiler.disable()
            state = workspace.load_arrays()
            np.savez_compressed(args.output / f"outputs-{repeat}.npz", **state)
            fitted = state["fitted"].astype(bool)
            result = {
                "seconds": timings, "total_seconds": sum(timings.values()),
                "frames_fit": int(fitted.sum()),
                "median_iou": float(np.median(state["iou"][fitted])) if fitted.any() else None,
                "stage_summaries": summaries,
                "loss_compile_fallback_groups": getattr(batch_fit, "_ENERGY_COMPILE_FALLBACKS", 0) - fallback_before,
            }
            if args.device.startswith("cuda"):
                result["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
                report["gpu"] = torch.cuda.get_device_name()
            report["runs"].append(result)
            (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k != "stage_summaries"}), flush=True)
            if not fitted.all():
                raise RuntimeError("benchmark requires a successful fit for every input mask; see saved stage summaries")
    report["median_seconds"] = {s: float(np.median([r["seconds"][s] for r in report["runs"]])) for s in stages}
    report["median_total_seconds"] = float(np.median([r["total_seconds"] for r in report["runs"]]))
    if args.repeats > 1:
        report["warm_median_seconds"] = {s: float(np.median([r["seconds"][s] for r in report["runs"][1:]])) for s in stages}
        report["warm_median_total_seconds"] = float(np.median([r["total_seconds"] for r in report["runs"][1:]]))
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    if profiler is not None:
        profiler.dump_stats(str(args.output / "profile.pstats"))


if __name__ == "__main__":
    main()
