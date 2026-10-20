#!/usr/bin/env python3
"""Score a library model on benchmarks and store the evaluations the model picker shows.

The same command the app's ``evaluate`` jobs run; see
:mod:`worm_pose_gen.model_eval` for the metrics.  Without ``--benchmark``
the model is scored on every benchmark of its setup::

    scripts/evaluate_model.py --model lab:nir-body-lags3 --benchmark lab:nir-v1
"""

from worm_pose_gen.model_eval import main


if __name__ == "__main__":
    raise SystemExit(main())
