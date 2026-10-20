#!/usr/bin/env python3
"""Calibrate the stage against the image and measure what substrate alignment does (docs/STAGE_ALIGNMENT.md).

For each ``--recording``: background shifts of frame pairs at lags 1, 4 and
16, the calibration fit to them (and to each quarter of the recording on
its own), written to the stage-calibration cache.  With ``--workspace``,
the same for the workspace's recording with the workspace's worm masks,
then the background difference energy ``frame[t+lag] - frame[t-lag]``
(worm masked out, flat-fielded) unaligned, stage-aligned and registered,
and a lag-16 figure of frames around ``--figure-frames``.  Writes
``<output-dir>/report.json`` and the figure.  Example::

    scripts/project_env.sh env OMP_NUM_THREADS=4 uv run --no-sync --frozen python scripts/stage_alignment_report.py \\
        --workspace <data dir>/workspaces/2023-03-30-01 --recording /storage/fs/data2/prj_aversion/data_raw/2023-08-22/2023-08-22-01.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import ndimage

from worm_pose_gen.recordings import RecordingSource
from worm_pose_gen.settings import CACHE_DIR, RUNS_DIR
from worm_pose_gen.stage_alignment import (
    CALIBRATION_LAGS,
    DEFAULT_CACHE,
    MASK_DILATION_PX,
    StageCalibration,
    bandpass,
    fit_calibration,
    measure_shifts,
    read_stage,
    register_shift,
    shift_image,
    substrate_shift_px,
)
from worm_pose_gen.workspace import Workspace


ENERGY_MODES = ("unaligned", "stage, offset 0", "stage, offset -0.5", "stage, fitted offset", "registered")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", action="append", default=[], help="recording to calibrate (dark-pixel worm masks)")
    parser.add_argument("--workspace", type=Path, help="workspace whose recording is calibrated with its masks and measured")
    parser.add_argument("--anchors", type=int, default=400, help="frames spread over each recording, each registered at lags 1, 4, 16")
    parser.add_argument("--energy-frames", type=int, default=150, help="frames spread over the workspace's recording for the energy table")
    parser.add_argument("--figure-frames", type=int, nargs=2, default=(4700, 4730), help="first and last frame t of the figure")
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "stage_alignment")
    return parser.parse_args()


def summarize(recording: Path, shifts: np.ndarray, calibration: StageCalibration) -> dict[str, Any]:
    """Calibration, its geometry, residuals per lag, and the per-quarter fits."""

    matrix = np.asarray(calibration.matrix)
    scales = np.linalg.svd(matrix, compute_uv=False)
    track = read_stage(recording)
    t, lag = shifts[:, 0], shifts[:, 1]
    predicted = (track.at(t + lag + calibration.offset_frames) - track.at(t + calibration.offset_frames)) @ matrix.T
    residual = np.linalg.norm(shifts[:, 2:4] - predicted, axis=1)
    finite = np.isfinite(residual)
    quarters = []
    for q in range(4):
        part = (t >= q * len(track.filled) / 4) & (t < (q + 1) * len(track.filled) / 4)
        fit = fit_calibration(track, shifts[part])
        quarters.append({"matrix": fit.matrix, "offset_frames": fit.offset_frames, "residual_px": fit.residual_px})
    return {
        "recording": str(recording),
        "calibration": calibration.to_dict(),
        "um_per_px": [float(0.1 / s) for s in scales],
        "rotation_deg": float(np.degrees(np.arctan2(matrix[1, 0] - matrix[0, 1], matrix[0, 0] + matrix[1, 1]))),
        "determinant_sign": float(np.sign(np.linalg.det(matrix))),
        "stage_rows_missing": float(np.mean(~np.isfinite(track.raw).all(axis=1))),
        "frames_unreliable": float(np.mean(~track.reliable)),
        "residual_by_lag": {
            int(l): {
                "median_px": float(np.median(residual[finite & (lag == l)])),
                "p90_px": float(np.percentile(residual[finite & (lag == l)], 90)),
                "over_3px": float(np.mean(residual[finite & (lag == l)] > 3.0)),
            }
            for l in CALIBRATION_LAGS
        },
        "quarters": quarters,
    }


def calibrate(recording: Path, anchors: int, mask_of: Callable[[int], np.ndarray | None] | None) -> dict[str, Any]:
    track = read_stage(recording)
    reliable = np.flatnonzero(track.reliable)
    frames = np.unique(np.linspace(reliable[0], reliable[-1] - max(CALIBRATION_LAGS), anchors).astype(np.int64))
    shifts = measure_shifts(recording, frames, CALIBRATION_LAGS, mask_of)
    calibration = fit_calibration(track, shifts)
    calibration.save(DEFAULT_CACHE / f"{recording.stem}.json")
    summary = summarize(recording, shifts, calibration)
    print(json.dumps({k: summary[k] for k in ("recording", "calibration", "um_per_px", "rotation_deg", "residual_by_lag")}), flush=True)
    return summary


class Pair:
    """Frames t-lag, t, t+lag of a workspace's recording, flat-fielded, with worm masks."""

    def __init__(self, source: RecordingSource, workspace: Workspace, t: int, lag: int) -> None:
        self.frames = {k: source.corrected(t + k)[1].astype(np.float64) for k in (-lag, 0, lag)}
        self.masks = {
            k: ndimage.binary_dilation(workspace.effective_mask(workspace.row_of(t + k)), iterations=MASK_DILATION_PX)
            for k in (-lag, 0, lag)
        }
        self.t, self.lag = t, lag

    def offsets(self, mode: str, recording: Path, calibration: StageCalibration) -> tuple[np.ndarray, np.ndarray] | None:
        """Whole-pixel moves that put frame t-lag's and t+lag's background where frame t has it."""

        lag = self.lag
        if mode == "unaligned":
            return np.zeros(2), np.zeros(2)
        if mode == "registered":
            moved = []
            for k in (-lag, lag):
                shift, _ = register_shift(bandpass(self.frames[0]), bandpass(self.frames[k]), self.masks[0] | self.masks[k])
                moved.append(shift)
            return np.round(moved[0]), np.round(moved[1])
        offset = {"stage, offset 0": 0.0, "stage, offset -0.5": -0.5}.get(mode, calibration.offset_frames)
        shifted = StageCalibration(calibration.matrix, offset, calibration.residual_px, calibration.pairs, calibration.outliers)
        substrate = substrate_shift_px(recording, [self.t - lag, self.t, self.t + lag], shifted)
        if not np.isfinite(substrate).all():
            return None
        return np.round(substrate[0] - substrate[1]), np.round(substrate[2] - substrate[1])

    def difference(self, before: np.ndarray, after: np.ndarray, filtered: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``frame[t+lag] - frame[t-lag]`` after the moves, the valid pixels off the worm in all three frames, and the valid worm pixels."""

        lag = self.lag
        a, b = (bandpass(self.frames[k]) if filtered else self.frames[k] for k in (-lag, lag))
        a, valid_a = shift_image(a, before)
        b, valid_b = shift_image(b, after)
        worm_a = shift_image(self.masks[-lag].astype(np.float64), before)[0] > 0.01
        worm_b = shift_image(self.masks[lag].astype(np.float64), after)[0] > 0.01
        worm = worm_a | worm_b | self.masks[0]
        return b - a, valid_a & valid_b & ~worm, valid_a & valid_b & worm


def energy_table(workspace: Workspace, calibration: StageCalibration, count: int) -> dict[str, Any]:
    recording = workspace.recording
    source = RecordingSource(recording, CACHE_DIR / "flat_fields")
    lags = CALIBRATION_LAGS
    anchors = np.linspace(max(lags) + 1, source.frame_count - max(lags) - 2, count).astype(np.int64)
    regions = ("background raw", "background band-passed", "worm raw")
    energy = {(lag, region, mode): [] for lag in lags for region in regions for mode in ENERGY_MODES}
    for t in anchors:
        for lag in lags:
            pair = Pair(source, workspace, int(t), lag)
            moves = {mode: pair.offsets(mode, recording, calibration) for mode in ENERGY_MODES}
            if any(move is None for move in moves.values()):
                continue
            for filtered in (False, True):
                differences = {mode: pair.difference(*moves[mode], filtered) for mode in ENERGY_MODES}
                # The same pixels for every mode: background off the worm everywhere, worm wherever any mode has it.
                background = np.logical_and.reduce([keep for _, keep, _ in differences.values()])
                worm = np.logical_or.reduce([on for _, _, on in differences.values()]) & np.logical_and.reduce(
                    [keep | on for _, keep, on in differences.values()]
                )
                if background.sum() < 10_000:
                    continue
                for mode, (difference, _, _) in differences.items():
                    region = "background band-passed" if filtered else "background raw"
                    energy[(lag, region, mode)].append(float(np.mean(difference[background] ** 2)))
                    if not filtered:
                        energy[(lag, "worm raw", mode)].append(float(np.mean(difference[worm] ** 2)))
    source.close()
    table: dict[str, Any] = {}
    for lag in lags:
        for region in regions:
            key = f"lag {lag}, {region}"
            table[key] = {mode: float(np.median(energy[(lag, region, mode)])) for mode in ENERGY_MODES}
            table[key]["pairs"] = len(energy[(lag, region, ENERGY_MODES[0])])
            print(key, json.dumps(table[key]), flush=True)
    return table


def figure(workspace: Workspace, calibration: StageCalibration, first: int, last: int, path: Path) -> list[int]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recording = workspace.recording
    usable = [t for t in range(first, last + 1) if np.isfinite(substrate_shift_px(recording, [t - 16, t, t + 16], calibration)).all()]
    if not usable:
        return []
    chosen = sorted({usable[0], usable[len(usable) // 2], usable[-1]})
    source = RecordingSource(recording, CACHE_DIR / "flat_fields")
    modes = ("unaligned", "stage, fitted offset", "registered")
    fig, axes = plt.subplots(2 * len(chosen), len(modes), figsize=(5.0 * len(modes), 3.9 * 2 * len(chosen)))
    for row, t in enumerate(chosen):
        pair = Pair(source, workspace, t, 16)
        for column, mode in enumerate(modes):
            for k, filtered in enumerate((False, True)):
                difference, keep, _ = pair.difference(*pair.offsets(mode, recording, calibration), filtered)
                ax = axes[2 * row + k, column]
                limit = 8.0 if filtered else 40.0
                ax.imshow(np.where(keep | pair.masks[0], difference, 0.0), cmap="gray", vmin=-limit, vmax=limit)
                ax.contour(pair.masks[0], levels=[0.5], colors="c", linewidths=0.6)
                kind = "band-passed" if filtered else "raw"
                ax.set_title(f"t={t}: frame[t+16] - frame[t-16], {kind}, {mode}\nbackground mean square {np.mean(difference[keep] ** 2):.1f}", fontsize=8)
                ax.axis("off")
    source.close()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return chosen


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {"recordings": []}
    if args.workspace is not None:
        workspace = Workspace.open(args.workspace)
        summary = calibrate(workspace.recording, args.anchors, lambda frame: workspace.effective_mask(workspace.row_of(frame)))
        report["recordings"].append(summary)
        calibration = StageCalibration.from_dict(summary["calibration"])
        name = workspace.recording.stem
        report["figure_frames"] = figure(
            workspace, calibration, *args.figure_frames, args.output_dir / f"{name}_lag16_aligned_vs_unaligned.png"
        )
        report["energy"] = energy_table(workspace, calibration, args.energy_frames)
    for recording in args.recording:
        report["recordings"].append(calibrate(Path(recording), args.anchors, None))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=1))
    print("done", args.output_dir)


if __name__ == "__main__":
    main()
