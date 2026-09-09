"""Cross-component contracts for mask editing, training and model selection."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from worm_pose_gen.app import AppConfig, config_from_args, parse_args
from worm_pose_gen.app.routers.jobs import stage_job
from worm_pose_gen.pose_viewer import Segmenters
from worm_pose_gen.workspace import Workspace


class Phase4ConfigurationTests(unittest.TestCase):
    def test_user_artifacts_default_to_workspace_root(self):
        config = AppConfig(workspaces_root=Path("/tmp/example-workspaces"))
        self.assertEqual(config.corpus_root, Path("/tmp/example-workspaces/corpus"))
        self.assertEqual(config.checkpoints_root, Path("/tmp/example-workspaces/checkpoints"))
        config = config_from_args(parse_args([
            "--corpus-root", "/tmp/user-labels", "--checkpoints-root", "/tmp/user-models", "--gpus", "",
        ]))
        self.assertEqual(config.corpus_root, Path("/tmp/user-labels"))
        self.assertEqual(config.checkpoints_root, Path("/tmp/user-models"))

    def test_segment_and_bootstrap_jobs_use_selected_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = Workspace.create(root, "test", root / "video.h5", 0, 1)
            workspace.info.settings["checkpoint"] = "/tmp/selected.ckpt"
            app = Mock()
            app.workspace.return_value = workspace
            app.config = AppConfig(workspaces_root=root, checkpoint=Path("/tmp/base.ckpt"), dataset_root=root / "cache")
            for stage in ("segment", "prior", "fit"):
                spec, command = stage_job(app, {"workspace": "test"}, stage)
                self.assertEqual(spec.params["params"]["checkpoint"], "/tmp/selected.ckpt")
                self.assertEqual(spec.params["params"]["dataset_root"], str(root / "cache"))
                self.assertIn("/tmp/selected.ckpt", " ".join(command))
            spec, _ = stage_job(app, {"workspace": "test", "params": {"checkpoint": None}}, "segment")
            self.assertIsNone(spec.params["params"]["checkpoint"])


class CheckpointCacheTests(unittest.TestCase):
    def test_replaced_checkpoint_reloads_model(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.ckpt"
            checkpoint.write_bytes(b"first")
            first, second = Mock(), Mock()
            first.predict_probability_batch.return_value = np.zeros((1, 8, 8), dtype=np.float32)
            second.predict_probability_batch.return_value = np.ones((1, 8, 8), dtype=np.float32)
            segmenters = Segmenters(torch.device("cpu"), checkpoint)
            frame = np.zeros((8, 8), dtype=np.uint8)
            with patch("worm_pose_gen.pose_viewer.load_segmenter", side_effect=[first, second]) as load:
                before = segmenters.signature(None)
                self.assertEqual(float(segmenters.probability(None, frame)[0].sum()), 0)
                segmenters.probability(None, frame)
                self.assertEqual(load.call_count, 1)
                replacement = checkpoint.with_suffix(".new")
                replacement.write_bytes(b"replacement")
                replacement.replace(checkpoint)
                self.assertNotEqual(segmenters.signature(None), before)
                self.assertEqual(float(segmenters.probability(None, frame)[0].sum()), 64)
                self.assertEqual(load.call_count, 2)


if __name__ == "__main__":
    unittest.main()
