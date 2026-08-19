#!/usr/bin/env python3
"""Stream explicitly exploratory independent-frame inference to a new HDF5 file."""

from __future__ import annotations

import argparse
from pathlib import Path

from worm_pose_gen.inference import infer_hdf5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="explicit absolute HDF5 dataset path")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="required final declaration, normally configs/final.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--frame-rate", type=float)
    parser.add_argument(
        "--allow-exploratory",
        action="store_true",
        help="required acknowledgement that the checkpoint failed EXP-0007 validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = infer_hdf5(
        source_path=args.source,
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_path=args.output,
        allow_exploratory=args.allow_exploratory,
        device=args.device,
        batch_size=args.batch_size,
        start=args.start,
        stop=args.stop,
        frame_rate=args.frame_rate,
    )
    print(output)


if __name__ == "__main__":
    main()
