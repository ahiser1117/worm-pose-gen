"""Scoring a library model on a benchmark: the numbers the model picker shows.

One evaluation is one model on one frozen benchmark (docs/APP_SIMPLIFICATION.md,
section 1).  Each benchmark label is predicted from its stored context
(:mod:`library.inference`, lags converted to the benchmark setup's frame
rate), and scored:

``iou``
    mask IoU over the label's pixels that are not excluded from the loss
    (an empty label left empty scores 1), the same function training
    validates with (:func:`segmenter.masked_binary_metrics`, threshold 0.5).
``head_tail_correct``
    for a label whose body trains the body outputs (below) and whose two
    ends lie in the image: whether the predicted head peak is nearer the
    labeled head than the labeled tail.
``ap_error``
    the mean absolute A-P error (arc position, 0 head .. 1 tail) over the
    label's worm pixels where the target is defined (crossings are not).

The body numbers need the label's body targets (:mod:`library.targets`);
missing ones are built first, chain fits using the setup's mask default.  A
label's body counts only where it would train the body
(:func:`body_used`): not mask-only, and either settled by a person or
fitted at ``min_fit_iou`` or better.

The summary is ``iou_mean``, ``iou_worst5`` (the mean of the worst 5% of
labels, at least one: what an analyst ends up fixing by hand; with a small
benchmark ``worst5_count`` says how many labels that is), ``head_tail_correct``
(a fraction) and ``ap_error``, each with the number of labels behind it.
The evaluation file (``library.save_evaluation``) also holds one row per
label and the names of overlay PNGs of the worst frames, stored next to it in
``<evaluation stem>/`` (missed worm pixels magenta, extra ones green, the
predicted head red and tail blue).

``python -m worm_pose_gen.model_eval`` (and ``scripts/evaluate_model.py``)
runs one model on one or all of its setup's benchmarks; the app's
``evaluate`` jobs run exactly that command.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import signal
import sys
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from . import library
from .jobs import report_progress
from .library import Libraries, LabelRecord


WORST_FRAMES = 6
WORST_SHARE = 0.05
DEFAULT_MIN_FIT_IOU = 0.9
OVERLAY_MARGIN_PX = 40
OVERLAY_MIN_SIZE_PX = 160
MISSED_RGB = (255, 80, 163)
EXTRA_RGB = (87, 214, 141)
HEAD_RGB = (255, 70, 70)
TAIL_RGB = (108, 180, 255)

Progress = Callable[[float, str], None]


def _quiet(fraction: float, message: str) -> None:
    pass


# --------------------------------------------------------------------------- body targets


def body_used(record: LabelRecord, meta: dict[str, Any] | None, min_fit_iou: float) -> bool:
    """Whether a label's body targets train (and score) the body outputs.

    Today's rule: a mask-only label, a frame without a body and a body the
    fitter could not match (``fit_iou`` below ``min_fit_iou``) train the mask
    only, unless a person settled the body (a trace or a chosen head).
    """

    if meta is None or not meta.get("has_body") or record.mask_only:
        return False
    return record.status == "complete" or float(meta.get("fit_iou") or 0.0) >= float(min_fit_iou)


def mask_model_for(libraries: Libraries, setup_ref: str, device: Any) -> Any:
    """The setup's mask default, for chain fits while building targets; ``None`` when it has none."""

    from .library.inference import load_model

    setup = library.get_setup(libraries, setup_ref)
    ref = setup.defaults.get("mask")
    return None if ref is None else load_model(libraries, ref, device=device, fps=setup.fps)


def prepare_targets(
    libraries: Libraries, records: Sequence[LabelRecord], *, setup: str, device: Any = None, progress: Progress = _quiet,
) -> int:
    """Build the body targets the records lack; returns how many were built."""

    missing = [r for r in records if library.cached_meta(libraries, r) is None]
    if not missing:
        return 0
    segmenter = mask_model_for(libraries, setup, device)
    for count, record in enumerate(missing):
        progress(count / len(missing), f"preparing body targets {count + 1}/{len(missing)}")
        library.build_targets(libraries, record, segmenter=segmenter, device=device)
    return len(missing)


# --------------------------------------------------------------------------- scoring


def mask_iou(probability: NDArray[np.floating], mask: NDArray[np.uint8]) -> float:
    import torch

    from .segmenter import IGNORE_LABEL, masked_binary_metrics

    target = torch.as_tensor((mask == 1).astype(np.float32))[None]
    valid = torch.as_tensor(mask != IGNORE_LABEL)[None]
    return float(masked_binary_metrics(torch.as_tensor(probability)[None], target, valid)["iou"][0])


def peak_xy(values: NDArray[np.floating]) -> NDArray[np.float64]:
    y, x = np.unravel_index(int(np.argmax(values)), values.shape)
    return np.array([x, y], dtype=np.float64)


def _inside(xy: NDArray[np.floating], shape: tuple[int, int]) -> bool:
    return bool(np.all(np.isfinite(xy)) and 0 <= xy[0] <= shape[1] - 1 and 0 <= xy[1] <= shape[0] - 1)


def score_label(outputs: Any, label: Any, targets: tuple[dict[str, Any], dict[str, NDArray[Any]]] | None, use_body: bool) -> dict[str, Any]:
    """One label's row: IoU, and the body numbers where the label's body counts."""

    mask = label.mask
    record = label.record
    prediction = outputs.mask >= 0.5
    worm = mask == 1
    row: dict[str, Any] = {
        **record.identity, "split": record.split, "status": record.status,
        "iou": mask_iou(outputs.mask, mask),
        "missed_px": int((worm & ~prediction).sum()), "extra_px": int((prediction & (mask == 0)).sum()),
        "fit_iou": None if targets is None else targets[0].get("fit_iou"),
        "self_contact": None if targets is None else targets[0].get("self_contact"),
        "body_used": bool(use_body), "head_tail_correct": None, "ap_error": None,
    }
    if not use_body or outputs.ap is None or targets is None:
        return row
    _, arrays = targets
    defined = worm & np.isfinite(arrays["ap"])
    if defined.any():
        row["ap_error"] = float(np.abs(outputs.ap[defined] - arrays["ap"][defined]).mean())
    head, tail = np.asarray(arrays["head_xy"], float), np.asarray(arrays["tail_xy"], float)
    if outputs.head is not None and _inside(head, mask.shape) and _inside(tail, mask.shape):
        predicted = peak_xy(outputs.head)
        row["head_tail_correct"] = bool(np.linalg.norm(predicted - head) < np.linalg.norm(predicted - tail))
    return row


def worst_count(n: int) -> int:
    return max(1, math.ceil(WORST_SHARE * n)) if n else 0


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The picker's numbers from the per-label rows."""

    ious = np.sort(np.array([r["iou"] for r in rows], dtype=np.float64))
    k = worst_count(len(ious))
    oriented = [r["head_tail_correct"] for r in rows if r["head_tail_correct"] is not None]
    errors = [r["ap_error"] for r in rows if r["ap_error"] is not None]
    return {
        "labels": len(rows),
        "iou_mean": float(ious.mean()) if len(ious) else None,
        "iou_worst5": float(ious[:k].mean()) if k else None,
        "worst5_count": k,
        "head_tail_correct": float(np.mean(oriented)) if oriented else None,
        "head_tail_labels": len(oriented),
        "ap_error": float(np.mean(errors)) if errors else None,
        "ap_labels": len(errors),
    }


# --------------------------------------------------------------------------- overlays


def overlay_png(label: Any, outputs: Any, path: Path) -> None:
    """The label's frame cropped around the worm with the prediction's errors and ends drawn on it."""

    from PIL import Image, ImageDraw

    image, mask = label.image, label.mask
    prediction = outputs.mask >= 0.5
    worm = mask == 1
    rgb = np.repeat(image[..., None], 3, axis=2).astype(np.float32)
    for region, color in ((worm & ~prediction, MISSED_RGB), (prediction & (mask == 0), EXTRA_RGB)):
        rgb[region] = 0.35 * rgb[region] + 0.65 * np.array(color, np.float32)
    canvas = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    radius = 5
    for heatmap, color in ((outputs.head, HEAD_RGB), (outputs.tail, TAIL_RGB)):
        if heatmap is not None:
            x, y = peak_xy(heatmap)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
    present = np.argwhere(worm | prediction)
    height, width = mask.shape
    if len(present):
        (y0, x0), (y1, x1) = present.min(0), present.max(0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        size = max(OVERLAY_MIN_SIZE_PX, x1 - x0 + 2 * OVERLAY_MARGIN_PX, y1 - y0 + 2 * OVERLAY_MARGIN_PX)
        left = int(np.clip(cx - size / 2, 0, max(0, width - size)))
        top = int(np.clip(cy - size / 2, 0, max(0, height - size)))
        canvas = canvas.crop((left, top, min(width, left + int(size)), min(height, top + int(size))))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def overlay_dir(libraries: Libraries, model_ref: str, benchmark_ref: str) -> Path:
    """Where an evaluation's worst-frame overlays go: next to its JSON file, named like it."""

    return library.evaluation_path(libraries, model_ref, benchmark_ref).with_suffix("")


# --------------------------------------------------------------------------- one evaluation


def evaluate(
    libraries: Libraries, model_ref: str, benchmark_ref: str, *, device: Any = None, progress: Progress = _quiet,
    min_fit_iou: float = DEFAULT_MIN_FIT_IOU, worst_frames: int = WORST_FRAMES,
) -> dict[str, Any]:
    """Score ``model_ref`` on ``benchmark_ref``, store the evaluation and its overlays, and return it."""

    import shutil

    from .library.inference import load_model

    benchmark = library.get_benchmark(libraries, benchmark_ref)
    setup = library.get_setup(libraries, benchmark.setup)
    records = library.benchmark_labels(libraries, benchmark_ref)
    if not records:
        raise ValueError(f"{benchmark_ref} holds no labels")
    model = load_model(libraries, model_ref, device=device, fps=setup.fps)
    body = "ap" in model.outputs
    if body:
        prepare_targets(libraries, records, setup=benchmark.setup, device=device,
                        progress=lambda f, m: progress(0.5 * f, m))
    rows, kept = [], {}
    for count, record in enumerate(records):
        progress((0.5 if body else 0.0) + (0.5 if body else 0.95) * count / len(records), f"scoring label {count + 1}/{len(records)}")
        label = record.load()
        outputs = model.predict(label.context, label.context_valid)
        targets = library.load_targets(libraries, record) if body else None
        row = score_label(outputs, label, targets, body_used(record, None if targets is None else targets[0], min_fit_iou))
        rows.append(row)
        kept[len(rows) - 1] = (label, outputs)
        # Keep only what the worst-frame overlays can still need.
        worst = sorted(range(len(rows)), key=lambda i: rows[i]["iou"])[:worst_frames]
        kept = {i: kept[i] for i in worst if i in kept}
    directory = overlay_dir(libraries, model_ref, benchmark_ref)
    shutil.rmtree(directory, ignore_errors=True)
    overlays = []
    for rank, index in enumerate(sorted(kept, key=lambda i: rows[i]["iou"]), start=1):
        row = rows[index]
        name = f"worst{rank}-{row['recording']}-{row['frame']:06d}.png"
        overlay_png(*kept[index], directory / name)
        overlays.append({"file": name, "recording": row["recording"], "frame": row["frame"], "dataset": row["dataset"],
                         "iou": row["iou"], "head_tail_correct": row["head_tail_correct"], "ap_error": row["ap_error"]})
    result = {
        **summarize(rows), "lags_frames": list(model.lags), "min_fit_iou": float(min_fit_iou),
        "benchmark_setup": benchmark.setup, "worst": overlays, "rows": rows,
    }
    library.save_evaluation(libraries, model_ref, benchmark_ref, result)
    progress(1.0, "evaluated")
    return {"model": model_ref, "benchmark": benchmark_ref, **result}


def missing_evaluations(libraries: Libraries, setup_ref: str, benchmark_ref: str | None = None) -> list[tuple[str, str]]:
    """(model, benchmark) pairs of the setup that have no evaluation yet."""

    benchmarks = [benchmark_ref] if benchmark_ref else [b.ref for b in library.list_benchmarks(libraries, setup_ref)]
    pairs = []
    for card in library.list_models(libraries, setup_ref):
        done = library.evaluations(libraries, card.ref)
        pairs.extend((card.ref, b) for b in benchmarks if b not in done)
    return pairs


# --------------------------------------------------------------------------- command line


def add_library_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lab-library", type=Path, default=None, help="the lab library (default: by host name)")
    parser.add_argument("--library", type=Path, default=None, help="your personal library (default: by host name)")
    parser.add_argument("--device", default=None, help="torch device (default: CUDA when available)")


def libraries_from(args: argparse.Namespace) -> Libraries:
    return Libraries.for_host(args.lab_library, args.library)


def command(libraries: Libraries, model_ref: str, benchmark_ref: str) -> list[str]:
    """The process an ``evaluate`` job runs."""

    argv = [sys.executable, "-m", "worm_pose_gen.model_eval", "--library", str(libraries.personal), "--model", model_ref,
            "--benchmark", benchmark_ref]
    if libraries.lab is not None:
        argv += ["--lab-library", str(libraries.lab)]
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_library_arguments(parser)
    parser.add_argument("--model", required=True, help="model reference, e.g. lab:nir-body-lags3")
    parser.add_argument("--benchmark", action="append", default=None,
                        help="benchmark reference (repeatable; default: every benchmark of the model's setup)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    libraries = libraries_from(args)
    card = library.get_card(libraries, args.model)
    benchmarks = args.benchmark or [b.ref for b in library.list_benchmarks(libraries, card.setup)]
    if not benchmarks:
        raise SystemExit(f"{card.setup} has no benchmark; freeze one first")
    for number, benchmark in enumerate(benchmarks):
        def progress(fraction: float, message: str) -> None:
            report_progress((number + fraction) / len(benchmarks), f"{benchmark}: {message}")

        result = evaluate(libraries, args.model, benchmark, device=args.device, progress=progress)
        numbers = {k: result[k] for k in ("labels", "iou_mean", "iou_worst5", "worst5_count", "head_tail_correct", "ap_error")}
        print(f"{args.model} on {benchmark}: {numbers}", flush=True)
    report_progress(1.0, "evaluated", result={"model": args.model, "benchmarks": benchmarks})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
