"""Figures of docs/FIT_LAYERS.md: every layer the fit stage reads and makes, on frames of a real workspace.

Two steps, so the GPU part runs once on a GPU host and the figures can be
redrawn anywhere:

    # on flv-c3 (pick an idle GPU first)
    scripts/project_env.sh env CUDA_VISIBLE_DEVICES=1 uv run --no-sync --frozen \
        python scripts/fit_layers_figures.py compute
    # anywhere
    scripts/project_env.sh uv run --no-sync --frozen python scripts/fit_layers_figures.py render

``compute`` re-runs, for each example frame, what the fit stage does
(``pipeline.fit_frames`` with the workspace's stored fit configuration):
the body-field prediction, the field evidence, the proposed trace and its
laid start, the standard starts as oriented, and a fit from every start on
its own (the stage keeps only the winner).  It also predicts the end
heatmap peaks over a window of consecutive frames.  ``render`` draws the
PNGs into ``docs/fit_layers/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = "2023-03-30-01"
# Examples: (frame, short name).  Chosen from the workspace's state: a clear
# in-view frame, a self-contact, a body leaving the camera, a frame whose
# orientation the evidence leaves open, and a contact the fit gets wrong.
EXAMPLES = [
    (8012, "clear"),
    (4722, "contact"),
    (12577, "edge"),
    (9583, "open_ends"),
    (12657, "contact_failure"),
]
# Consecutive frames whose end peaks and stored poses are plotted over time.
WINDOW = (4640, 4840)
# Frames of the window whose head heatmap peak alternates between two places.
FLIP = (4718, 4728)
DOCS = PROJECT / "docs" / "fit_layers"


def cache_path() -> Path:
    from worm_pose_gen.settings import RUNS_DIR

    return Path(RUNS_DIR) / "fit_layers" / f"{WORKSPACE}.npz"


def compute(device_name: str) -> None:
    from dataclasses import replace

    import torch

    from worm_pose_gen.body_net import OUTPUTS, load_body_net
    from worm_pose_gen.body_proposal import (
        RecordingFieldPredictor, crop_fields, field_evidence, propose_trace, trace_start,
    )
    from worm_pose_gen.batch_fit import fit_masks
    from worm_pose_gen.latent import decode_centerline
    from worm_pose_gen.mask_fit import head_first, lay_centerline, orientation_pair
    from worm_pose_gen.pipeline import SegmentParams, frame_starts, read_summary, workspace_frames, workspace_setup
    from worm_pose_gen.settings import DATA_DIR
    from worm_pose_gen.workspace import Workspace

    device = torch.device(device_name)
    workspace = Workspace.open(Path(DATA_DIR) / "workspaces" / WORKSPACE)
    summary = read_summary(workspace)
    setup = workspace_setup(workspace)
    config = replace(setup.config, compile_renderer=False)
    setup = replace(setup, config=config)
    frames = workspace_frames(workspace, SegmentParams())
    module = load_body_net(summary["fit_params"]["body_net"], device)
    predictor = RecordingFieldPredictor(module, frames)
    state = workspace.load_state()
    out: dict[str, np.ndarray] = {
        "lags": np.asarray(module.lags), "template": setup.template,
        "prior": np.array(json.dumps(setup.prior.to_dict())), "config": np.array(json.dumps({
            k: getattr(config, k) for k in ("field_ap_weight", "field_ap_sigma", "field_end_weight", "field_end_sigma_px",
                                           "length_prior_px", "width_prior_px", "width_shape_prior", "bend_weight",
                                           "stage_downsample", "stage_steps", "edge_softness")
        })),
    }
    examples = []
    for frame, name in EXAMPLES:
        row = workspace.row_of(frame)
        mask = workspace.effective_mask(row)
        prediction = predictor.predict_full([frame])[0]
        crop = crop_fields(prediction, mask)
        evidence = field_evidence(crop, mask)
        raw = frames.read([frame])[0]
        corrected, _, _ = frames.corrected([frame + o for o in (0, *module.lags, *(-lag for lag in module.lags))])
        differences = [
            (corrected[1 + k].astype(np.float32) - corrected[1 + len(module.lags) + k].astype(np.float32)) / 255.0
            for k in range(len(module.lags))
        ]
        trace_full = propose_trace(crop, mask)
        trace_headless = propose_trace(crop, mask, tail=False)
        width = setup.prior.width_px
        laid = None if trace_headless is None or evidence.head_xy is None else lay_centerline(trace_headless, mask, setup.start_length, width)
        standard, trace = frame_starts(mask, setup, crop)
        both = evidence.head_xy is not None and evidence.tail_xy is not None
        if both:
            standard = [head_first(s, evidence.head_xy, evidence.tail_xy, config=config) for s in standard]
        else:
            standard = [s for start in standard for s in orientation_pair(start, config=config)]
        starts = ([trace] if trace is not None else []) + standard
        # Each start fit on its own, as one frame each, so every start's outcome is kept.
        results = fit_masks(
            [mask] * len(starts), [[s] for s in starts], width_template=setup.template, config=config,
            device=device, fields=[evidence] * len(starts),
        )
        key = f"ex{len(examples)}"
        examples.append({"frame": frame, "name": name, "row": row, "starts": [s.name for s in starts], "both_ends": both})
        out[f"{key}_raw"] = raw
        out[f"{key}_corrected"] = corrected[0]
        out[f"{key}_differences"] = np.stack(differences)
        for channel in OUTPUTS:
            out[f"{key}_{channel}_net" if channel == "mask" else f"{key}_{channel}"] = getattr(prediction, channel).astype(np.float16)
        out[f"{key}_mask"] = mask
        out[f"{key}_evidence_ap"] = evidence.ap.astype(np.float16)
        out[f"{key}_peaks"] = np.array([prediction.peak("head"), prediction.peak("tail")])
        out[f"{key}_scored_ends"] = np.array([
            evidence.head_xy if evidence.head_xy is not None else [np.nan, np.nan],
            evidence.tail_xy if evidence.tail_xy is not None else [np.nan, np.nan],
        ])
        for label, value in (("trace_full", trace_full), ("trace_headless", trace_headless), ("laid", laid)):
            if value is not None:
                out[f"{key}_{label}"] = value
        out[f"{key}_start_curves"] = np.stack([decode_centerline(s.latent, config.coefficients) for s in starts])
        out[f"{key}_start_widths"] = np.array([s.width_px for s in starts])
        out[f"{key}_fit_curves"] = np.stack([r.centerline_xy for r in results])
        out[f"{key}_fit_profiles"] = np.stack([r.width_profile for r in results])
        out[f"{key}_fit_hard"] = np.stack([r.rendered_hard_mask for r in results])
        out[f"{key}_fit_crops"] = np.array([[r.crop.x0, r.crop.x1, r.crop.y0, r.crop.y1] for r in results])
        out[f"{key}_fit_history"] = np.stack([r.energy_history[:, 0] for r in results])
        out[f"{key}_fit_records"] = np.array(json.dumps([r.records[0] for r in results]))
        out[f"{key}_stored_curve"] = state["centerline_xy_independent"][row]
        out[f"{key}_stored_profile"] = state["width_profile_independent"][row]
        out[f"{key}_stored_best"] = np.array(str(state["best_start_independent"][row]))
        out[f"{key}_stored_iou"] = np.array(state["iou_independent"][row])
        print(frame, name, [(s, round(float(r.records[0]["final_energy"]), 4), round(float(r.records[0]["final_iou"]), 3)) for s, r in zip(examples[-1]["starts"], results)], flush=True)
    out["examples"] = np.array(json.dumps(examples))
    # Over a window of consecutive frames: the raw end peaks and the stored poses' ends.
    window = list(range(*WINDOW))
    rows = [workspace.row_of(f) for f in window]
    peaks = []
    scored = []
    for start in range(0, len(window), 32):
        chunk = window[start : start + 32]
        for f, fields in zip(chunk, predictor.predict(chunk)):
            peaks.append([fields.head, fields.tail])
            ev = field_evidence(fields, workspace.effective_mask(workspace.row_of(f)))
            scored.append([ev.head_xy is not None, ev.tail_xy is not None])
    out["window_frames"] = np.asarray(window)
    out["window_peaks"] = np.asarray(peaks)
    out["window_scored"] = np.asarray(scored)
    for name in ("centerline_xy", "centerline_xy_independent", "best_start_independent", "source", "iou_independent", "iou",
                 "orientation_gap", "ambiguity_score", "flag_self_contact"):
        out[f"window_{name}"] = state[name][rows]
    flip = list(range(*FLIP))
    flip_rows = [workspace.row_of(f) for f in flip]
    predictions = predictor.predict_full(flip)
    out["flip_frames"] = np.asarray(flip)
    out["flip_corrected"] = frames.corrected(flip)[0]
    out["flip_head"] = np.stack([p.head for p in predictions]).astype(np.float16)
    out["flip_mask"] = np.stack([workspace.effective_mask(r) for r in flip_rows])
    out["flip_curves"] = state["centerline_xy_independent"][flip_rows]
    out["flip_profiles"] = state["width_profile_independent"][flip_rows]
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    print(f"wrote {path}")
    frames.close()


# ---------------------------------------------------------------------------
# Rendering: the app's layer colours (app_ui/workspace_view.js, frame_canvas.js,
# workspace_draw.js) on dark panels, so the figures read like the Workspace page.

BG = "#14181c"
FG = "#d8dee4"
DIM = "#8b949e"
HEAD = "#57d68d"
TAIL = "#ff5d5d"
OUTLINE = "#ffd166"
MASK_RGB = (80, 220, 255)
OVERLAP_RGB = (255, 79, 163)
OFF_CAMERA = (0.16, 0.11, 0.11)
START_COLORS = {
    "network_trace": "#ff7ad9", "skeleton_longest_path": "#6cb4ff", "skeleton_longest_path_reversed": "#b878ff",
    "moments_straight": "#ffa94d", "moments_straight_reversed": "#e8590c",
}
SOURCE_COLORS = {0: "#6c757d", 1: "#6cb4ff", 2: "#b878ff"}
AP_STOPS = [(87, 214, 141), (108, 180, 255), (184, 120, 255), (255, 93, 93)]
END_THRESHOLD, END_MASK_PX, END_BORDER_PX = 0.3, 20.0, 20.0


def _style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
        "axes.labelcolor": FG, "axes.edgecolor": "#3a434c", "xtick.color": DIM, "ytick.color": DIM,
        "axes.titlecolor": FG, "font.size": 9, "axes.titlesize": 9.5, "legend.facecolor": "#1e242a",
        "legend.edgecolor": "#3a434c", "legend.fontsize": 8,
    })


def _ap_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("ap", [tuple(c / 255 for c in s) for s in AP_STOPS])


class Example:
    """One example frame's arrays from the cache (``ex<k>_<name>``)."""

    def __init__(self, z: np.lib.npyio.NpzFile, k: int, info: dict) -> None:
        self.z, self.key, self.info = z, f"ex{k}", info
        self.name, self.frame = info["name"], info["frame"]
        self.records = json.loads(str(z[f"{self.key}_fit_records"]))
        self.winner = int(np.argmin([r["final_energy"] for r in self.records]))

    def __getitem__(self, name: str) -> np.ndarray:
        return self.z[f"{self.key}_{name}"]

    def get(self, name: str) -> np.ndarray | None:
        key = f"{self.key}_{name}"
        return self.z[key] if key in self.z.files else None

    @property
    def shape(self) -> tuple[int, int]:
        return self["mask"].shape


def _box(mask: np.ndarray, *curves: np.ndarray | None, margin: int = 30, square: bool = False) -> tuple[int, int, int, int]:
    """(x0, x1, y0, y1) around the mask and the curves (which may leave the image), plus a margin."""

    ys, xs = np.nonzero(mask)
    x = [xs.min(), xs.max()]
    y = [ys.min(), ys.max()]
    for curve in curves:
        if curve is not None and len(curve):
            pts = np.asarray(curve).reshape(-1, 2)
            x += [pts[:, 0].min(), pts[:, 0].max()]
            y += [pts[:, 1].min(), pts[:, 1].max()]
    x0, x1, y0, y1 = int(min(x)) - margin, int(max(x)) + margin, int(min(y)) - margin, int(max(y)) + margin
    if square:
        side = max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        x0, x1, y0, y1 = cx - side // 2, cx + side // 2, cy - side // 2, cy + side // 2
    return x0, x1, y0, y1


def _show(ax, image: np.ndarray, box: tuple[int, int, int, int], *, cmap=None, vmin=None, vmax=None, dim: float = 1.0) -> None:
    """``image`` (gray or RGB(A), full frame) on ``box`` in image coordinates; outside the camera in OFF_CAMERA."""

    x0, x1, y0, y1 = box
    height, width = image.shape[:2]
    if image.ndim == 2:
        import matplotlib
        from matplotlib.colors import Normalize

        colormap = matplotlib.colormaps[cmap or "gray"]
        rgb = colormap(Normalize(vmin if vmin is not None else 0, vmax if vmax is not None else 255)(image.astype(np.float32)))[..., :3]
    else:
        rgb = image[..., :3].astype(np.float32)
    canvas = np.empty((y1 - y0, x1 - x0, 3), dtype=np.float32)
    canvas[:] = OFF_CAMERA
    sy0, sy1, sx0, sx1 = max(y0, 0), min(y1, height), max(x0, 0), min(x1, width)
    if sy0 < sy1 and sx0 < sx1:
        canvas[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = rgb[sy0:sy1, sx0:sx1] * dim
    ax.imshow(canvas, extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5), interpolation="nearest")
    ax.set_xlim(x0 - 0.5, x1 - 0.5)
    ax.set_ylim(y1 - 0.5, y0 - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _overlay(ax, values: np.ndarray, color: tuple[int, int, int], box, *, alpha: float = 0.9, where: np.ndarray | None = None) -> None:
    """A probability as one colour whose opacity is the value (the app's ``probabilityLut``)."""

    x0, x1, y0, y1 = box
    height, width = values.shape
    rgba = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.float32)
    rgba[..., :3] = np.asarray(color) / 255.0
    sy0, sy1, sx0, sx1 = max(y0, 0), min(y1, height), max(x0, 0), min(x1, width)
    v = np.asarray(values[sy0:sy1, sx0:sx1], dtype=np.float32)
    if where is not None:
        v = np.where(where[sy0:sy1, sx0:sx1], v, 0)
    rgba[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0, 3] = np.clip(v, 0, 1) * alpha
    ax.imshow(rgba, extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5), interpolation="nearest")


def _ap_layer(ax, ap: np.ndarray, defined: np.ndarray, box, *, alpha: float = 0.85, levels: int | None = None) -> None:
    x0, x1, y0, y1 = box
    height, width = ap.shape
    values = np.clip(np.asarray(ap, dtype=np.float32), 0, 1)
    if levels:
        values = (np.minimum(np.floor(values * levels), levels - 1) + 0.5) / levels
    colors = _ap_cmap()(values)
    colors[..., 3] = np.where(defined, alpha, 0)
    rgba = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.float32)
    sy0, sy1, sx0, sx1 = max(y0, 0), min(y1, height), max(x0, 0), min(x1, width)
    rgba[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = colors[sy0:sy1, sx0:sx1]
    ax.imshow(rgba, extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5), interpolation="nearest")


def _contour(ax, mask: np.ndarray, color: str, *, width: float = 0.8, origin: tuple[int, int] = (0, 0), style: str = "-") -> None:
    ox, oy = origin
    ys = np.arange(mask.shape[0]) + oy
    xs = np.arange(mask.shape[1]) + ox
    if mask.any() and not mask.all():
        ax.contour(xs, ys, mask.astype(np.float32), levels=[0.5], colors=[color], linewidths=width, linestyles=style)


def _outline(curve: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """The app's outline (``drawOutline``): both sides offset by half the width along the normals, as one closed polygon."""

    n = len(curve)
    a = curve[np.maximum(np.arange(n) - 1, 0)]
    b = curve[np.minimum(np.arange(n) + 1, n - 1)]
    d = b - a
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    r = np.asarray(widths)[:, None] / 2
    left = np.column_stack((curve[:, 0] - d[:, 1] * r[:, 0], curve[:, 1] + d[:, 0] * r[:, 0]))
    right = np.column_stack((curve[:, 0] + d[:, 1] * r[:, 0], curve[:, 1] - d[:, 0] * r[:, 0]))
    polygon = np.vstack((left, right[::-1], left[:1]))
    return polygon


def _midline(ax, curve: np.ndarray, *, color: str = HEAD, width: float = 1.6, style: str = "-", ends: bool = True, size: float = 6) -> None:
    ax.plot(curve[:, 0], curve[:, 1], style, color=color, lw=width, solid_capstyle="round")
    if ends:
        ax.plot(*curve[0], "s", ms=size, mfc=HEAD, mec="black", mew=0.8)
        ax.plot(*curve[-1], "o", ms=size, mfc=TAIL, mec="black", mew=0.8)


def _start_profile(width_px: float, template: np.ndarray, shape: np.ndarray) -> np.ndarray:
    from worm_pose_gen.latent import cubic_bspline_basis

    correction = cubic_bspline_basis(len(template), len(shape)) @ np.asarray(shape)
    return width_px * template * np.exp(correction - correction.mean())


def _end_verdict(peak: np.ndarray, mask: np.ndarray) -> tuple[bool, str]:
    """Whether an end heatmap's peak scores fits (``body_proposal.field_evidence``), and why not."""

    from scipy import ndimage

    x, y, value = float(peak[0]), float(peak[1]), float(peak[2])
    if value < END_THRESHOLD:
        return False, f"{value:.2f} < {END_THRESHOLD}"
    distance = float(ndimage.distance_transform_edt(~mask)[int(y), int(x)])
    if distance > END_MASK_PX:
        return False, f"{value:.2f}, {distance:.0f} px off mask"
    height, width = mask.shape
    if min(x, y, width - 1 - x, height - 1 - y) <= END_BORDER_PX:
        return False, f"{value:.2f}, at image edge"
    return True, f"{value:.2f}"


def _save(fig, path: Path) -> None:
    """The figure as a 256-colour PNG: a fifth of the bytes of a full-colour one on these noisy frames."""

    import io

    from PIL import Image

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight", pad_inches=0.08)
    image = Image.open(buffer).convert("RGB")
    image.quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(path, optimize=True)
    print(f"wrote {path}")


def fig_inputs(ex: Example, lags: np.ndarray, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    box = _box(ex["mask"], margin=40)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6))
    full = ex["corrected"]
    _show(axes[0, 0], full, (0, full.shape[1], 0, full.shape[0]))
    axes[0, 0].add_patch(Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, ec=OUTLINE, lw=1.2))
    axes[0, 0].set_title(f"flat-fielded frame {ex.frame} ({full.shape[1]}×{full.shape[0]})")
    _show(axes[0, 1], ex["raw"], box)
    axes[0, 1].set_title("raw NIR frame (crop)")
    _show(axes[0, 2], ex["corrected"], box)
    axes[0, 2].set_title("flat-fielded (network channel 0)")
    differences = ex["differences"]
    limit = float(np.percentile(np.abs(differences), 99.8))
    for k, lag in enumerate(lags):
        ax = axes[1, k]
        _show(ax, differences[k], box, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(f"frame[t+{lag}] − frame[t−{lag}]  ({lag / 20:.2f} s)")
    fig.suptitle("Network input: the frame plus one symmetric difference per lag", color=FG, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    _save(fig, out)
    plt.close(fig)


def fig_outputs(examples: list[Example], out: Path) -> None:
    import matplotlib.pyplot as plt

    titles = ["frame", "mask probability", "cleaned mask (fit target)", "A-P field (mask > 0.5)", "head heatmap", "tail heatmap", "overlap (on the mask)"]
    fig, axes = plt.subplots(len(examples), 7, figsize=(17, 2.75 * len(examples) + 0.4))
    for row, ex in enumerate(examples):
        box = _box(ex["mask"], margin=20, square=True)
        image = ex["corrected"]
        net_mask = ex["mask"]
        for col in range(7):
            ax = axes[row, col]
            _show(ax, image, box, dim=1.0 if col == 0 else 0.45)
            if row == 0:
                ax.set_title(titles[col])
        _overlay(axes[row, 1], ex["mask_net"].astype(np.float32), MASK_RGB, box)
        raw = ex["mask_net"].astype(np.float32) >= 0.5
        _overlay(axes[row, 2], net_mask.astype(np.float32), MASK_RGB, box, alpha=0.55)
        _overlay(axes[row, 2], (net_mask & ~raw).astype(np.float32), (255, 209, 102), box, alpha=1.0)
        filled = int((net_mask & ~raw).sum())
        axes[row, 2].text(0.02, 0.02, f"{filled} px filled", transform=axes[row, 2].transAxes, color=OUTLINE, fontsize=8)
        _ap_layer(axes[row, 3], ex["ap"], ex["mask_net"].astype(np.float32) > 0.5, box)
        peaks = ex["peaks"]
        _overlay(axes[row, 4], ex["head"].astype(np.float32), tuple(int(HEAD[i : i + 2], 16) for i in (1, 3, 5)), box)
        axes[row, 4].text(0.02, 0.02, f"peak {peaks[0, 2]:.2f}", transform=axes[row, 4].transAxes, color=HEAD, fontsize=8)
        _overlay(axes[row, 5], ex["tail"].astype(np.float32), tuple(int(TAIL[i : i + 2], 16) for i in (1, 3, 5)), box)
        axes[row, 5].text(0.02, 0.02, f"peak {peaks[1, 2]:.2f}", transform=axes[row, 5].transAxes, color=TAIL, fontsize=8)
        _overlay(axes[row, 6], ex["overlap"].astype(np.float32), OVERLAP_RGB, box, where=net_mask)
        crossing = int(((ex["overlap"].astype(np.float32) >= 0.5) & net_mask).sum())
        axes[row, 6].text(0.02, 0.02, f"{crossing} px ≥ 0.5", transform=axes[row, 6].transAxes, color="#ff4fa3", fontsize=8)
        axes[row, 0].set_ylabel(f"{ex.name}\nframe {ex.frame}", color=FG)
    fig.suptitle("The body-field network's five outputs (sigmoids), and the mask the fit is run against", color=FG, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    _save(fig, out)
    plt.close(fig)


def fig_ends(examples: list[Example], out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(examples), figsize=(4.1 * len(examples), 4.6))
    for ax, ex in zip(axes, examples):
        mask = ex["mask"]
        peaks = ex["peaks"]
        box = _box(mask, peaks[:, :2], margin=35, square=True)
        _show(ax, ex["corrected"], box, dim=0.55)
        _overlay(ax, mask.astype(np.float32), MASK_RGB, box, alpha=0.22)
        _overlay(ax, ex["head"].astype(np.float32), tuple(int(HEAD[i : i + 2], 16) for i in (1, 3, 5)), box)
        _overlay(ax, ex["tail"].astype(np.float32), tuple(int(TAIL[i : i + 2], 16) for i in (1, 3, 5)), box)
        height, width = mask.shape
        # The image-edge band in which an end does not score.
        from matplotlib.patches import Rectangle

        ax.add_patch(Rectangle((-0.5, -0.5), width, height, fill=False, ec="#ffffff", lw=0.6, ls=":"))
        ax.add_patch(Rectangle((END_BORDER_PX - 0.5, END_BORDER_PX - 0.5), width - 2 * END_BORDER_PX, height - 2 * END_BORDER_PX, fill=False, ec=DIM, lw=0.6, ls="--"))
        lines = []
        for k, (label, color, marker) in enumerate((("head", HEAD, "s"), ("tail", TAIL, "o"))):
            ok, why = _end_verdict(peaks[k], mask)
            ax.plot(peaks[k, 0], peaks[k, 1], marker, ms=9, mfc=color if ok else "none", mec=color if not ok else "black", mew=1.6)
            ax.add_patch(plt.Circle((peaks[k, 0], peaks[k, 1]), END_MASK_PX, fill=False, ec=color, lw=0.6, ls=":"))
            lines.append((f"{label}: {why} → {'scores' if ok else 'not used'}", color))
        for j, (text, color) in enumerate(lines):
            ax.text(0.02, 0.97 - 0.07 * j, text, transform=ax.transAxes, color=color, fontsize=8.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="none", alpha=0.75))
        ax.set_title(f"{ex.name} (frame {ex.frame})")
    fig.suptitle("Head and tail: the heatmap peak, kept only if ≥ 0.3, within 20 px of the mask and > 20 px from the image edge (dashed)",
                 color=FG, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    _save(fig, out)
    plt.close(fig)


def fig_trace(examples: list[Example], prior: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(examples), 3, figsize=(13, 4.2 * len(examples)))
    for row, ex in enumerate(examples):
        mask = ex["mask"]
        evidence = ex["evidence_ap"].astype(np.float32)
        defined = np.isfinite(evidence)
        crossing = mask & ~defined
        full = ex.get("trace_full")
        headless = ex.get("trace_headless")
        laid = ex.get("laid")
        starts = ex.info["starts"]
        start_curve = ex["start_curves"][starts.index("network_trace")] if "network_trace" in starts else None
        box = _box(mask, start_curve, margin=25, square=True)
        ax = axes[row, 0]
        _show(ax, ex["corrected"], box, dim=0.45)
        _ap_layer(ax, np.nan_to_num(evidence), defined, box, levels=20, alpha=0.9)
        _overlay(ax, crossing.astype(np.float32), (150, 150, 150), box, alpha=0.9)
        ax.set_title("A-P evidence in 20 bands (grey: predicted crossing, left out)")
        ax = axes[row, 1]
        _show(ax, ex["corrected"], box, dim=0.45)
        _ap_layer(ax, np.nan_to_num(evidence), defined, box, levels=20, alpha=0.35)
        if full is not None:
            ax.plot(full[:, 0], full[:, 1], "-", color="white", lw=1.0)
            ax.plot(full[:, 0], full[:, 1], "o", ms=4, mfc="white", mec="black", mew=0.5)
            has_head = bool(np.isfinite(ex["scored_ends"][0]).all())
            ax.plot(*full[0], "s", ms=8, mfc=HEAD if has_head else "none", mec="black" if has_head else "white")
            ax.plot(*full[-1], "o", ms=8, mfc=TAIL, mec="black")
        ax.set_title(f"proposed trace: {'head, ' if np.isfinite(ex['scored_ends'][0]).all() else ''}one point per band, tail ({0 if full is None else len(full)} points)")
        ax = axes[row, 2]
        _show(ax, ex["corrected"], box, dim=0.45)
        _overlay(ax, mask.astype(np.float32), MASK_RGB, box, alpha=0.18)
        if headless is not None:
            ax.plot(headless[:, 0], headless[:, 1], "o", ms=3.5, mfc="white", mec="black", mew=0.5, label="band points (no tail)")
        if laid is not None:
            arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(laid, axis=0), axis=1))))
            band_arc = 0.0 if headless is None else float(np.sum(np.linalg.norm(np.diff(headless, axis=0), axis=1)))
            ax.plot(laid[:, 0], laid[:, 1], "-", color=START_COLORS["network_trace"], lw=1.6, label=f"laid from the head to {prior['length_px']:.0f} px")
            beyond = arc >= min(band_arc, arc[-1])
            if beyond.sum() > 1:
                ax.plot(laid[beyond, 0], laid[beyond, 1], "-", color="#ffffff", lw=2.6, alpha=0.5, label="continued along the mask")
            ax.plot(*laid[0], "s", ms=8, mfc=HEAD, mec="black")
            ax.plot(*laid[-1], "o", ms=8, mfc=TAIL, mec="black")
            ax.set_title("trace start: laid from the head at the prior's length")
        elif start_curve is not None:
            ax.plot(start_curve[:, 0], start_curve[:, 1], "-", color=START_COLORS["network_trace"], lw=1.6, label="trace extended to the prior's length")
            ax.plot(*start_curve[0], "s", ms=8, mfc=HEAD, mec="black")
            ax.plot(*start_curve[-1], "o", ms=8, mfc=TAIL, mec="black")
            ax.set_title("no head in view: the trace runs to the tail instead")
        ax.legend(loc="lower left")
        axes[row, 0].set_ylabel(f"{ex.name}\nframe {ex.frame}", color=FG)
    fig.suptitle("From the A-P field to a starting midline (body_proposal.propose_trace, trace_start)", color=FG, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    _save(fig, out)
    plt.close(fig)


def fig_starts(examples: list[Example], template: np.ndarray, prior: dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, len(examples), figsize=(4.6 * len(examples), 4.9))
    for ax, ex in zip(axes, examples):
        curves = ex["start_curves"]
        box = _box(ex["mask"], curves, margin=25, square=True)
        _show(ax, ex["corrected"], box, dim=0.45)
        _overlay(ax, ex["mask"].astype(np.float32), MASK_RGB, box, alpha=0.2)
        handles = []
        for name, curve, width in zip(ex.info["starts"], curves, ex["start_widths"]):
            color = START_COLORS.get(name, "#cccccc")
            polygon = _outline(curve, _start_profile(width, template, prior["width_shape"]))
            ax.plot(polygon[:, 0], polygon[:, 1], "-", color=color, lw=0.7, alpha=0.8)
            ax.plot(curve[:, 0], curve[:, 1], "-", color=color, lw=1.5)
            ax.plot(*curve[0], "s", ms=7, mfc=color, mec="black")
            handles.append(Line2D([], [], color=color, marker="s", mec="black", label=name.replace("_longest_path", "")))
        ax.legend(handles=handles, loc="lower left")
        verdict = "both ends scored → each start fit head first" if ex.info["both_ends"] else "ends not both scored → standard starts fit both ways"
        ax.set_title(f"{ex.name} (frame {ex.frame})\n{verdict}", fontsize=8.8)
    fig.suptitle("The starts of the fit (square = the start's first point, the model's head)", color=FG, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    _save(fig, out)
    plt.close(fig)


def fig_model(ex: Example, contact: Example, template: np.ndarray, prior: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    from worm_pose_gen.latent import cubic_bspline_basis, encode_centerline

    w = ex.winner
    curve = ex["fit_curves"][w]
    profile = ex["fit_profiles"][w]
    fig = plt.figure(figsize=(15, 8.4))
    grid = fig.add_gridspec(2, 6)
    # (a) tangent angle
    ax = fig.add_subplot(grid[0, 0:2])
    latent = encode_centerline(curve, 16)
    basis = cubic_bspline_basis(99, 16)
    s = np.linspace(0, 1, 99)
    for j in range(16):
        ax.plot(s, basis[:, j] * latent[j] + latent[16], color=DIM, lw=0.6, alpha=0.6)
    ax.plot(s, basis @ latent[:16] + latent[16], color=HEAD, lw=2, label="θ(s) = φ + Σ c_j B_j(s)")
    ax.axhline(latent[16], color=OUTLINE, lw=0.8, ls="--", label=f"φ (rotation) = {latent[16]:.2f} rad")
    ax.set_xlabel("position along the body s (0 head, 1 tail)")
    ax.set_ylabel("tangent angle (rad)")
    ax.set_title("(a) shape: 16 cubic B-spline coefficients of the tangent angle")
    ax.legend(loc="best")
    # (b) width profile
    ax = fig.add_subplot(grid[0, 2:4])
    position = np.linspace(0, 1, len(profile))
    ax.plot(position, template * prior["width_px"], color=DIM, lw=1.2, ls="--", label=f"template × prior width ({prior['width_px']:.1f} px)")
    ax.plot(position, _start_profile(prior["width_px"], template, prior["width_shape"]), color=OUTLINE, lw=1.2, ls=":", label="prior's mean profile (tail last)")
    ax.plot(position, profile, color=OUTLINE, lw=2, label="fitted diameter")
    ax.set_xlabel("position along the body s")
    ax.set_ylabel("full width (px)")
    ax.set_title("(b) width: scale × template × exp(6-coefficient correction)")
    ax.legend(loc="lower center")
    # (c) the numbers
    ax = fig.add_subplot(grid[0, 4:6])
    ax.axis("off")
    rows = [
        ("latent (20 values)", ""),
        ("  shape c_1..c_16", "tangent angle B-spline"),
        ("  rotation φ", f"{latent[16]:.3f} rad"),
        ("  length L", f"{latent[17]:.1f} px (fixed: prior {prior['length_px']:.1f})"),
        ("  centroid (x, y)", f"({latent[18]:.1f}, {latent[19]:.1f})"),
        ("width", f"{np.median(profile[20:80]):.1f} px (midbody median)"),
        ("width shape (6)", "log correction, mean-centred"),
        ("midline", "100 points, L/99 apart"),
        ("point 0 / point 99", "head / tail"),
    ]
    for j, (k, v) in enumerate(rows):
        ax.text(0.0, 0.95 - j * 0.1, k, color=FG, fontsize=9.5, family="monospace", transform=ax.transAxes)
        ax.text(0.45, 0.95 - j * 0.1, v, color=DIM, fontsize=9.5, transform=ax.transAxes)
    ax.set_title("(c) one pose")
    # (d) midline and outline
    box = _box(ex["mask"], curve, margin=20, square=True)
    ax = fig.add_subplot(grid[1, 0:2])
    _show(ax, ex["corrected"], box, dim=0.6)
    polygon = _outline(curve, profile)
    ax.plot(polygon[:, 0], polygon[:, 1], color=OUTLINE, lw=1.2)
    ax.plot(curve[:, 0], curve[:, 1], ".", color=HEAD, ms=2.5)
    _midline(ax, curve, width=1.0)
    ax.set_title("(d) midline (100 points) and outline as the app draws it")
    # (e) rendered tube vs mask
    ax = fig.add_subplot(grid[1, 2:4])
    _render_vs_mask(ax, ex, w, box)
    ax.set_title(f"(e) rendered tube vs mask: IoU {ex.records[w]['final_iou']:.3f}")
    # (f) what the A-P term compares, on the contact
    from scipy import ndimage

    ax = fig.add_subplot(grid[1, 4:6])
    cw = contact.winner
    ccurve = contact["fit_curves"][cw]
    evidence = contact["evidence_ap"].astype(np.float32)
    cbox = _box(contact["mask"], ccurve, margin=15, square=True)
    _show(ax, contact["corrected"], cbox, dim=0.4)
    _ap_layer(ax, np.nan_to_num(evidence), np.isfinite(evidence), cbox, alpha=0.3)
    s_points = np.linspace(0, 1, len(ccurve))
    sampled = ndimage.map_coordinates(np.nan_to_num(evidence), [ccurve[:, 1], ccurve[:, 0]], order=1)
    front = s_points <= 0.5
    ax.plot(ccurve[~front, 0], ccurve[~front, 1], ".", color=DIM, ms=3)
    points = ax.scatter(ccurve[front, 0], ccurve[front, 1], c=sampled[front] - s_points[front], cmap="RdBu_r", vmin=-0.1, vmax=0.1, s=10, zorder=3)
    ax.plot(*ccurve[0], "s", ms=7, mfc=HEAD, mec="black", zorder=4)
    colorbar = fig.colorbar(points, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("A-P(point) − s", color=FG)
    ax.set_title(f"(f) A-P term on {contact.name}: front half only (evidence {contact.records[cw]['final_field_energy']:.4f})")
    fig.suptitle(f"The tube model: how a midline and an outline come out of the fit ({ex.name}, frame {ex.frame})", color=FG, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    _save(fig, out)
    plt.close(fig)


def _render_vs_mask(ax, ex: Example, k: int, box) -> None:
    """Pixels in both (white), tube only (orange) and mask only (cyan)."""

    mask = ex["mask"]
    hard = np.zeros_like(mask)
    x0, x1, y0, y1 = ex["fit_crops"][k]
    hard[y0:y1, x0:x1] = ex["fit_hard"][k][: y1 - y0, : x1 - x0]
    rgb = np.zeros(mask.shape + (3,), dtype=np.float32)
    rgb[mask & hard] = (0.85, 0.87, 0.89)
    rgb[hard & ~mask] = (1.0, 0.55, 0.26)
    rgb[mask & ~hard] = (0.31, 0.86, 1.0)
    base = np.repeat((ex["corrected"].astype(np.float32) / 255 * 0.35)[..., None], 3, axis=2)
    either = mask | hard
    base[either] = rgb[either]
    _show(ax, base, box)


def fig_fits(examples: list[Example], out: Path) -> None:
    import matplotlib.pyplot as plt

    columns = max(len(ex.info["starts"]) for ex in examples)
    fig, axes = plt.subplots(len(examples), columns + 1, figsize=(3.9 * (columns + 1), 3.9 * len(examples)))
    for row, ex in enumerate(examples):
        curves = ex["fit_curves"]
        box = _box(ex["mask"], curves, ex["start_curves"], margin=20, square=True)
        for col in range(columns):
            ax = axes[row, col]
            if col >= len(ex.info["starts"]):
                ax.axis("off")
                continue
            name = ex.info["starts"][col]
            color = START_COLORS.get(name, "#cccccc")
            _render_vs_mask(ax, ex, col, box)
            start = ex["start_curves"][col]
            ax.plot(start[:, 0], start[:, 1], "--", color="white", lw=0.9, alpha=0.8)
            ax.plot(*start[0], "s", ms=5, mfc="none", mec="white")
            _midline(ax, curves[col], color=color, width=1.6)
            r = ex.records[col]
            text = (f"overlap (1 − Dice) {r['final_soft_dice_energy']:.4f}\nevidence {r['final_field_energy']:.4f}\n"
                    f"total {r['final_energy']:.4f}   IoU {r['final_iou']:.3f}")
            ax.text(0.02, 0.02, text, transform=ax.transAxes, color=FG, fontsize=7.8, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="none", alpha=0.8))
            won = col == ex.winner
            ax.set_title(("★ " if won else "") + name.replace("_longest_path", ""), color=OUTLINE if won else FG)
            for spine in ax.spines.values():
                spine.set_edgecolor(OUTLINE if won else "#3a434c")
                spine.set_linewidth(2.0 if won else 0.8)
        ax = axes[row, columns]
        history = ex["fit_history"]
        for name, h in zip(ex.info["starts"], history):
            ax.plot(h, color=START_COLORS.get(name, "#cccccc"), lw=1.3, label=name.replace("_longest_path", ""))
        ax.axvline(60, color=DIM, lw=0.7, ls=":")
        ax.text(30, 0.98, "÷4", color=DIM, ha="center", va="top", transform=ax.get_xaxis_transform())
        ax.text(110, 0.98, "÷2", color=DIM, ha="center", va="top", transform=ax.get_xaxis_transform())
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("1 − soft Dice")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("overlap energy during the fit")
        axes[row, 0].set_ylabel(f"{ex.name}\nframe {ex.frame}", color=FG)
    fig.suptitle("Every start fit on its own (dashed: start, solid: fit; white = tube and mask, orange = tube only, cyan = mask only)",
                 color=FG, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    _save(fig, out)
    plt.close(fig)


def fig_final(examples: list[Example], out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(examples), figsize=(3.6 * len(examples), 4.0))
    for ax, ex in zip(axes, examples):
        curve, profile = ex["stored_curve"], ex["stored_profile"]
        box = _box(ex["mask"], curve, margin=15, square=True)
        _show(ax, ex["corrected"], box)
        _overlay(ax, ex["mask"].astype(np.float32), (255, 255, 255), box, alpha=0.12)
        polygon = _outline(curve, profile)
        ax.plot(polygon[:, 0], polygon[:, 1], color=OUTLINE, lw=1.1)
        _midline(ax, curve)
        best = str(ex["stored_best"]).replace("_longest_path", "")
        ax.set_title(f"{ex.name} (frame {ex.frame})\nwinner {best}, IoU {float(ex['stored_iou']):.3f}", fontsize=8.8)
    fig.suptitle("The fit stage's output as the Workspace page draws it: mask, outline, midline, head ■ and tail ●", color=FG, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    _save(fig, out)
    plt.close(fig)


def fig_window(z: np.lib.npyio.NpzFile, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    frames = z["window_frames"]
    peaks = z["window_peaks"]
    scored = z["window_scored"]
    independent = z["window_centerline_xy_independent"]
    final = z["window_centerline_xy"]
    best = z["window_best_start_independent"]
    source = z["window_source"]
    fig, axes = plt.subplots(5, 1, figsize=(13, 9.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 2.2, 2.2, 0.45, 0.45]})
    ax = axes[0]
    ax.plot(frames, peaks[:, 0, 2], color=HEAD, lw=1.4, label="head peak")
    ax.plot(frames, peaks[:, 1, 2], color=TAIL, lw=1.4, label="tail peak")
    ax.axhline(END_THRESHOLD, color=DIM, lw=0.8, ls="--")
    ax.text(frames[0], END_THRESHOLD + 0.02, " 0.3 threshold", color=DIM, fontsize=8)
    for k, (color, label) in enumerate(((HEAD, "head"), (TAIL, "tail"))):
        missing = ~scored[:, k]
        ax.plot(frames[missing], np.full(missing.sum(), -0.06 - 0.06 * k), "|", color=color, ms=7)
    ax.text(frames[-1] + 1, -0.06, "head not used", color=HEAD, fontsize=7.5, va="center")
    ax.text(frames[-1] + 1, -0.12, "tail not used", color=TAIL, fontsize=7.5, va="center")
    ax.set_ylim(-0.16, 1.05)
    ax.set_ylabel("heatmap peak")
    ax.legend(loc="center left", ncol=2)
    ax.set_title("Network ends over 200 consecutive frames (10 s) of 2023-03-30-01")
    ax = axes[1]
    step = lambda curves, i: np.r_[np.nan, np.linalg.norm(np.diff(curves[:, i], axis=0), axis=1)]
    ax.semilogy(frames, step(independent, 0), color=HEAD, lw=1.2, label="head (point 0)")
    ax.semilogy(frames, step(independent, -1), color=TAIL, lw=1.2, alpha=0.85, label="tail (point 99)")
    ax.axhline(44, color=DIM, lw=0.8, ls="--")
    ax.text(frames[0], 50, " one body width", color=DIM, fontsize=8)
    ax.set_ylabel("move from the\nframe before (px)")
    ax.set_title("Independent fits (the fit stage's output): where each end moves between frames")
    ax.legend(loc="upper left", ncol=2)
    ax = axes[2]
    distance = np.linalg.norm(independent[:, 0] - peaks[:, 0, :2], axis=1)
    peak_step = np.r_[np.nan, np.linalg.norm(np.diff(peaks[:, 0, :2], axis=0), axis=1)]
    used_both = scored[:, 0] & np.r_[False, scored[:-1, 0]]
    ax.semilogy(frames, np.where(used_both, np.maximum(peak_step, 0.05), np.nan), "x", ms=4, color=OUTLINE, label="network head peak: move from the frame before (both used)")
    ax.semilogy(frames, np.where(scored[:, 0], distance, np.nan), "o", ms=2.5, color=HEAD, label="fitted head − network head (head used)")
    ax.semilogy(frames, np.where(~scored[:, 0], distance, np.nan), "o", ms=2.5, color=DIM, label="fitted head − head peak (not used)")
    ax.semilogy(frames, step(final, 0), color="#6cb4ff", lw=0.9, alpha=0.8, label="head move after propagation (later stage)")
    ax.set_ylabel("px")
    ax.set_ylim(0.03, 1e4)
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    ax.set_title("The fitted head sits on the network's head peak, so it moves when the peak moves")
    ax = axes[3]
    names = sorted(set(best.tolist()))
    for name in names:
        hit = best == name
        ax.bar(frames[hit], 1, width=1.0, color=START_COLORS.get(name, "#cccccc"))
    ax.set_yticks([])
    ax.set_ylabel("winner", rotation=0, ha="right", va="center")
    ax.legend(handles=[Patch(color=START_COLORS.get(n, "#ccc"), label=n.replace("_longest_path", "")) for n in names],
              loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=len(names), fontsize=7.5)
    ax = axes[4]
    for code, label in ((0, "independent"), (1, "forward chain"), (2, "backward chain")):
        hit = source == code
        ax.bar(frames[hit], 1, width=1.0, color=SOURCE_COLORS[code], label=label)
    ax.set_yticks([])
    ax.set_ylabel("final pose\nfrom", rotation=0, ha="right", va="center")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=3, fontsize=7.5)
    ax.set_xlabel("frame")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    _save(fig, out)
    plt.close(fig)


def fig_flip(z: np.lib.npyio.NpzFile, out: Path) -> None:
    import matplotlib.pyplot as plt

    frames = z["flip_frames"]
    heads = z["flip_head"].astype(np.float32)
    masks = z["flip_mask"]
    curves = z["flip_curves"]
    box = _box(masks[len(masks) // 2], margin=15, square=True)
    columns = (len(frames) + 1) // 2
    fig, axes = plt.subplots(2, columns, figsize=(3.0 * columns, 6.9))
    for ax, frame, head, mask, curve, image in zip(axes.flat, frames, heads, masks, curves, z["flip_corrected"]):
        _show(ax, image, box, dim=0.5)
        _overlay(ax, mask.astype(np.float32), MASK_RGB, box, alpha=0.15)
        _overlay(ax, np.clip(head / max(float(head.max()), 1e-6), 0, 1), tuple(int(HEAD[i : i + 2], 16) for i in (1, 3, 5)), box)
        _midline(ax, curve, color="#ff7ad9", width=1.2, size=5)
        y, x = np.unravel_index(int(np.argmax(head)), head.shape)
        ax.plot(x, y, "+", color="white", ms=12, mew=1.5)
        ax.set_title(f"frame {frame}: peak {head.max():.2f}", fontsize=8.5)
    fig.suptitle("Head heatmap (scaled to its own peak; + = argmax) and the fitted midline: the argmax alternates between two places",
                 color=FG, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    _save(fig, out)
    plt.close(fig)


def render(cache: Path, docs: Path) -> None:
    _style()
    docs.mkdir(parents=True, exist_ok=True)
    z = np.load(cache)
    info = json.loads(str(z["examples"]))
    examples = {e["name"]: Example(z, k, e) for k, e in enumerate(info)}
    prior = json.loads(str(z["prior"]))
    prior["width_shape"] = np.asarray(prior["width_shape"])
    template = z["template"]
    clear, contact, edge, open_ends, failure = (examples[n] for n in ("clear", "contact", "edge", "open_ends", "contact_failure"))
    fig_inputs(clear, z["lags"], docs / "01_inputs.png")
    fig_outputs([clear, contact], docs / "02_outputs.png")
    fig_ends([clear, edge, open_ends, failure], docs / "03_ends.png")
    fig_trace([clear, contact, failure], prior, docs / "04_trace.png")
    fig_starts([clear, edge, open_ends], template, prior, docs / "05_starts.png")
    fig_model(clear, contact, template, prior, docs / "06_model.png")
    fig_fits([contact, edge, open_ends, failure], docs / "07_fits.png")
    fig_final([clear, contact, edge, open_ends, failure], docs / "08_final.png")
    fig_window(z, docs / "09_window.png")
    fig_flip(z, docs / "10_head_flip.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("compute")
    run.add_argument("--device", default="cuda")
    sub.add_parser("render")
    args = parser.parse_args()
    if args.command == "compute":
        compute(args.device)
    else:
        render(cache_path(), DOCS)


if __name__ == "__main__":
    main()
