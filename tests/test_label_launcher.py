"""The legacy console command routes normal labeling to the unified app."""
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from worm_pose_gen.label_app import unified_main


class LabelLauncherTests(unittest.TestCase):
    def test_old_flags_map_to_unified_config_and_recording_registry(self):
        state = mock.Mock()
        app = SimpleNamespace(state=SimpleNamespace(app_state=state))
        with mock.patch("worm_pose_gen.app.create_app", return_value=app) as create, mock.patch("uvicorn.run") as run:
            with contextlib.redirect_stdout(io.StringIO()):
                unified_main(["--dataset-root", "/tmp/existing-labels", "--recording", "/tmp/a.h5",
                              "--recording", "/tmp/b.h5", "--checkpoint", "/tmp/worm.ckpt", "--device", "cpu",
                              "--host", "127.0.0.2", "--port", "9001"])
        config = create.call_args.args[0]
        self.assertEqual(config.corpus_root, Path("/tmp/existing-labels"))
        self.assertEqual(config.dataset_root, Path("/tmp/existing-labels"))
        self.assertEqual(config.checkpoint, Path("/tmp/worm.ckpt"))
        self.assertEqual((config.host, config.port, config.device, config.gpus), ("127.0.0.2", 9001, "cpu", ()))
        self.assertEqual(state.register_recording.call_args_list, [
            mock.call({"path": "/tmp/a.h5", "dataset": "/img_nir"}),
            mock.call({"path": "/tmp/b.h5", "dataset": "/img_nir"}),
        ])
        run.assert_called_once_with(app, host="127.0.0.2", port=9001, log_level="info")
        state.close.assert_called_once()

    def test_queue_explicitly_falls_back_without_losing_manifest_semantics(self):
        argv = ["--queue", "/tmp/queue.json", "--dataset-root", "/tmp/labels", "--port", "9002"]
        stderr = io.StringIO()
        with mock.patch("worm_pose_gen.label_app.main") as legacy, mock.patch("worm_pose_gen.app.create_app") as create:
            with contextlib.redirect_stderr(stderr):
                unified_main(argv)
        legacy.assert_called_once_with(argv)
        create.assert_not_called()
        self.assertIn("Deprecated standalone labeler", stderr.getvalue())
        self.assertIn("manifest order and split pledges", stderr.getvalue())

    def test_registration_failure_closes_state_and_does_not_launch(self):
        state = mock.Mock()
        state.register_recording.side_effect = ValueError("recording is unavailable")
        app = SimpleNamespace(state=SimpleNamespace(app_state=state))
        with mock.patch("worm_pose_gen.app.create_app", return_value=app), mock.patch("uvicorn.run") as run:
            with self.assertRaisesRegex(ValueError, "unavailable"):
                unified_main(["--recording", "/tmp/missing.h5"])
        run.assert_not_called()
        state.close.assert_called_once()
