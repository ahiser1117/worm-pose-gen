#!/usr/bin/env python3
"""Build self-training pseudo-label items from analysed workspaces, and a contact sheet to audit them.

    build_pseudo_labels.py --workspaces <ws dir>... --output <items dir> [--clear-limit N] [--jobs J]
    build_pseudo_labels.py --sheet <items dir> --sheet-count 48 [--sheet-kind bridged]

Each workspace must have been analysed through the propagate stage
(``pipeline.main --stages segment,prior,fit,ambiguity,propagate,track``).
Selection and targets are :mod:`worm_pose_gen.pseudo_labels`'; the output
directory holds one ``.npz`` per item and ``index.json`` (the items, the
selection parameters and every workspace's selection counts).  The sheet
shows each sampled item twice: the frame as is, and with the A-P target
(head red to tail blue), the ignored band (yellow) and the head (green ring).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
import json
from pathlib import Path
import sys

import numpy as np


def build_one(args: tuple[str, str, dict, int | None, int]) -> tuple[str, list[dict], dict]:
    from worm_pose_gen.pseudo_labels import SelectionParams, build_items

    workspace, output, params, limit, seed = args
    entries, counts = build_items(Path(workspace), Path(output), SelectionParams(**params), limit=limit, seed=seed)
    return workspace, entries, counts


def build(args: argparse.Namespace) -> None:
    from worm_pose_gen.pseudo_labels import SelectionParams, write_index

    params = {f.name: getattr(args, f.name) for f in fields(SelectionParams) if getattr(args, f.name, None) is not None}
    jobs = [(str(w), str(args.output), params, args.clear_limit, args.seed) for w in args.workspaces]
    entries: list[dict] = []
    counts: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for workspace, found, workspace_counts in pool.map(build_one, jobs):
            entries += found
            counts[Path(workspace).name] = {**workspace_counts, "items": len(found)}
            print(f"{Path(workspace).name}: {len(found)} items {workspace_counts}", flush=True)
    write_index(args.output, entries, SelectionParams(**params), extra={"workspaces": counts})
    kinds = [e["kind"] for e in entries]
    print(json.dumps({"items": len(entries), "clear": kinds.count("clear"), "bridged": kinds.count("bridged"),
                      "recordings": len({e["recording"] for e in entries})}), flush=True)


def tile(item: dict, size: int) -> np.ndarray:
    from PIL import Image

    from worm_pose_gen.pseudo_labels import IGNORE

    offsets = list(item["offsets"])
    frame = item["frames"][offsets.index(0)].astype(np.float32)
    mask = item["mask"]
    ys, xs = np.nonzero(mask != 0)
    pad = 30
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)
    side = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0, x0 = max(0, cy - side // 2), max(0, cx - side // 2)
    y1, x1 = min(mask.shape[0], y0 + side), min(mask.shape[1], x0 + side)
    gray = np.repeat(frame[..., None], 3, axis=2)
    overlay = gray.copy()
    ap = item["ap"].astype(np.float32)
    body = (mask == 1) & np.isfinite(ap)
    colour = np.stack((255 * (1 - ap), 60 + 0 * ap, 255 * ap), axis=-1)
    overlay[body] = 0.45 * overlay[body] + 0.55 * colour[body]
    overlay[mask == IGNORE] = 0.4 * overlay[mask == IGNORE] + 0.6 * np.array([255, 230, 0])
    overlay[(mask == 1) & item["overlap"].astype(bool)] = np.array([255, 255, 255])
    hx, hy = item["head_xy"]
    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
    ring = np.abs(np.hypot(xx - hx, yy - hy) - 14) < 2.5
    overlay[ring] = np.array([0, 255, 0])
    panels = [Image.fromarray(np.clip(a[y0:y1, x0:x1], 0, 255).astype(np.uint8)).resize((size, size)) for a in (gray, overlay)]
    joined = np.concatenate([np.asarray(p) for p in panels], axis=1)
    return joined


def sheet(args: argparse.Namespace) -> None:
    from PIL import Image, ImageDraw

    from worm_pose_gen.pseudo_labels import read_index

    index = read_index(args.sheet)
    entries = [e for e in index["items"] if args.sheet_kind in (None, e["kind"])]
    rng = np.random.default_rng(args.seed)
    chosen = [entries[k] for k in sorted(rng.choice(len(entries), size=min(args.sheet_count, len(entries)), replace=False))]
    size, columns = args.tile, args.columns
    rows = (len(chosen) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * 2 * size, rows * (size + 16)), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    listing = []
    for k, entry in enumerate(chosen):
        with np.load(Path(args.sheet) / entry["file"]) as archive:
            item = {name: archive[name] for name in archive.files}
        x, y = (k % columns) * 2 * size, (k // columns) * (size + 16)
        canvas.paste(Image.fromarray(tile(item, size)), (x, y + 16))
        caption = f"{k}: {entry['workspace'][:18]} f{entry['frame']} {entry['kind'][0]} {entry['iou']:.3f}"
        draw.text((x + 2, y + 2), caption, fill=(255, 255, 255))
        listing.append({"tile": k, **entry})
    output = Path(args.sheet_output or Path(args.sheet).parent / f"audit_{args.sheet_kind or 'all'}.jpg")
    canvas.save(output, quality=88)
    output.with_suffix(".json").write_text(json.dumps(listing, indent=1))
    print(output)


def main(argv: list[str] | None = None) -> int:
    from worm_pose_gen.pseudo_labels import SelectionParams

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspaces", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--clear-limit", type=int, default=None, help="at most this many clear items per workspace (random)")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    for f in fields(SelectionParams):
        parser.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name, type=type(f.default), default=None)
    parser.add_argument("--sheet", type=Path, default=None, help="an items directory to audit")
    parser.add_argument("--sheet-count", type=int, default=48)
    parser.add_argument("--sheet-kind", choices=("clear", "bridged"), default=None)
    parser.add_argument("--sheet-output", type=Path, default=None)
    parser.add_argument("--tile", type=int, default=220)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args(argv)
    if args.sheet is not None:
        sheet(args)
    else:
        if not args.workspaces or args.output is None:
            parser.error("--workspaces and --output are required to build")
        build(args)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
