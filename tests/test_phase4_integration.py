"""Cross-component contracts: the app's command line, stage jobs on the workspace's model, and reloading a replaced checkpoint."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from worm_pose_gen.app import AppConfig, config_from_args, parse_args
from worm_pose_gen.app.frame_view import Segmenters
from worm_pose_gen.app.routers.jobs import stage_job
from worm_pose_gen.workspace import Workspace


class Phase4ConfigurationTests(unittest.TestCase):
    def test_command_line(self):
        args = parse_args(["--workspaces-root", "/tmp/example-workspaces", "--gpus", "", "--dev", "--library", "/tmp/mine", "--queue", "/tmp/manifest.json"])
        config = config_from_args(args)
        self.assertEqual(config.workspaces_root, Path("/tmp/example-workspaces"))
        self.assertEqual(config.jobs_root, Path("/tmp/example-workspaces"))
        self.assertEqual(config.gpus, ())
        self.assertTrue(config.dev)
        self.assertEqual(config.library, Path("/tmp/mine"))
        self.assertEqual(args.queue, Path("/tmp/manifest.json"))
        self.assertIsNone(config.server_device)

    def test_segment_and_fit_jobs_use_the_workspace_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = Workspace.create(root, "test", root / "video.h5", 0, 1)
            workspace.info.settings["checkpoint"] = "/tmp/selected.ckpt"
            app = Mock()
            app.workspace.return_value = workspace
            app.config = AppConfig(workspaces_root=root, dataset_root=root / "cache")
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
            checkpoint = Path(directory) / "weights.ckpt"
            checkpoint.write_bytes(b"first")
            first, second = Mock(), Mock()
            first.predict_probability_batch.return_value = np.zeros((1, 8, 8), dtype=np.float32)
            second.predict_probability_batch.return_value = np.ones((1, 8, 8), dtype=np.float32)
            segmenters = Segmenters(torch.device("cpu"))
            frame = np.zeros((8, 8), dtype=np.uint8)
            path = str(checkpoint)
            self.assertEqual(segmenters.probability(None, frame), (None, None))
            with patch("worm_pose_gen.app.frame_view.load_segmenter", side_effect=[first, second]) as load:
                before = segmenters.signature(path)
                self.assertEqual(float(segmenters.probability(path, frame)[0].sum()), 0)
                segmenters.probability(path, frame)
                self.assertEqual(load.call_count, 1)
                replacement = checkpoint.with_suffix(".new")
                replacement.write_bytes(b"replacement")
                replacement.replace(checkpoint)
                self.assertNotEqual(segmenters.signature(path), before)
                self.assertEqual(float(segmenters.probability(path, frame)[0].sum()), 64)
                self.assertEqual(load.call_count, 2)


if __name__ == "__main__":
    unittest.main()
