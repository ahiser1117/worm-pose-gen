#!/usr/bin/env python3
"""Train a model on library datasets: prepare body targets, train, evaluate on every benchmark, write the card.

The same command the app's ``train`` jobs run; see
:mod:`worm_pose_gen.model_training` for what it does and ``--help`` for the
settings.  Example, fine-tuning the lab body-field net on a personal
dataset::

    scripts/train.py --setup lab:nir-flv --dataset mine:nir-copper --start-from lab:nir-body-lags3 --max-epochs 50
"""

from worm_pose_gen.model_training import main


if __name__ == "__main__":
    raise SystemExit(main())
