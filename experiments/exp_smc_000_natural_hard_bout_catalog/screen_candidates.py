#!/usr/bin/env python3
"""Development-only visual/classical screening for natural hard-bout candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from worm_pose_gen.classical import ClassicalConfig, extract_centerline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOWED = {
    "2023-09-19-01": PROJECT_ROOT / "nir_videos/2023-09-19-01.h5",
    "2023-09-27-01": PROJECT_ROOT / "nir_videos/2023-09-27-01.h5",
    "2023-10-11-01": PROJECT_ROOT / "nir_videos/2023-10-11-01.h5",
}
SEEDS = {
    "2023-09-19-01": [5559, 10262, 14538, 19242],
    "2023-09-27-01": [3424, 8133, 12413, 14554],
    "2023-10-11-01": [12, 4308, 13785, 16801],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=40)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--classical", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = ClassicalConfig()
    evidence: list[dict[str, object]] = []
    for recording, path in ALLOWED.items():
        resolved = path.resolve(strict=True)
        if "2025-03-06" in str(resolved):
            raise RuntimeError("protected holdout path encountered")
        with h5py.File(path, "r") as source:
            images = source["img_nir"]
            for seed in SEEDS[recording]:
                indices = np.arange(max(0, seed - args.radius), min(len(images), seed + args.radius + 1), args.stride)
                rows = int(np.ceil(len(indices) / 5))
                fig, axes = plt.subplots(rows, 5, figsize=(15, 2.35 * rows), squeeze=False)
                for ax in axes.flat:
                    ax.axis("off")
                for ax, frame_index in zip(axes.flat, indices):
                    image = images[int(frame_index)]
                    p1, p99 = np.percentile(image, [1, 99])
                    ax.imshow(image, cmap="gray", vmin=p1, vmax=p99)
                    item: dict[str, object] = {
                        "recording": recording,
                        "frame_index": int(frame_index),
                        "seed_frame": seed,
                    }
                    title = f"f{frame_index}"
                    if args.classical:
                        result = extract_centerline(image, config)
                        item.update(
                            accepted=bool(result.accepted),
                            quality_score=float(result.quality_score),
                            rejection_reasons=list(result.rejection_reasons),
                            qc={key: (value.item() if hasattr(value, "item") else value) for key, value in result.qc.items()},
                        )
                        title += " A" if result.accepted else " R"
                        if result.centerline_xy is not None:
                            ax.plot(result.centerline_xy[:, 0], result.centerline_xy[:, 1], color="#00ffff", linewidth=0.8)
                    ax.set_title(title, fontsize=8)
                    evidence.append(item)
                fig.suptitle(f"{recording} seed f{seed}; raw development frames" + (" + classical screen" if args.classical else ""))
                fig.tight_layout()
                fig.savefig(args.output_dir / f"{recording}_f{seed:06d}.png", dpi=120)
                plt.close(fig)
    (args.output_dir / "screen_evidence.json").write_text(json.dumps({
        "protected_holdout_opened": False,
        "source_dataset_path": "/img_nir",
        "radius": args.radius,
        "stride": args.stride,
        "classical_evaluated": args.classical,
        "frames": evidence,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
