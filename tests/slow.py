"""The marker for tests that run the pose optimizer for ten seconds or more.

``WORM_POSE_FAST_TESTS=1`` (``scripts/run_tests.py --fast``) skips them, so an
edit can be checked in about a minute; the whole suite runs before a commit.
"""

from __future__ import annotations

import os
import unittest


slow = unittest.skipIf(os.environ.get("WORM_POSE_FAST_TESTS") == "1", "slow test, skipped by WORM_POSE_FAST_TESTS=1")
