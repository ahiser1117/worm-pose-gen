"""Snapshot/export consistency and safe immutable download coverage."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pyarrow.parquet as pq

from worm_pose_gen import pipeline
from worm_pose_gen.app.exporting import export_workspace, exported_file
from worm_pose_gen.workspace import Workspace
from tests.test_pose_viewer import _write_recording, _write_run


class ExportWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        recording = root / 'rec.h5'
        _write_recording(recording)
        _write_run(root / 'run', recording)
        self.workspace = Workspace.import_run(root / 'workspaces', root / 'run', 'test')
        self.app = SimpleNamespace(workspace=lambda name: self.workspace, check_writable=Mock(), device='cpu')

    def test_named_export_snapshot_and_no_overwrite(self):
        summary_path = self.workspace.path / 'summary.json'
        summary_path.write_text('{"source": "test"}')
        result = export_workspace(self.app, 'test', 'review-v1')
        self.assertEqual(summary_path.read_text(), '{"source": "test"}')
        snapshot = Path(result['snapshot_path'])
        self.assertEqual((snapshot / 'state.npz').read_bytes(), self.workspace.state_path.read_bytes())
        path = exported_file(self.workspace, snapshot.name, 'review-v1.parquet')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), result['sha256'])
        self.assertEqual(pq.read_table(path).num_rows, result['rows'])
        with self.assertRaises(ValueError):
            export_workspace(self.app, 'test', 'review-v1')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), result['sha256'])

    def test_busy_and_unsafe_names(self):
        with pipeline.workspace_lock(self.workspace):
            with self.assertRaises(pipeline.WorkspaceBusy):
                export_workspace(self.app, 'test', 'busy')
        for name in ('../escape', '/tmp/escape', '', 'bad/name'):
            with self.assertRaises(ValueError):
                export_workspace(self.app, 'test', name)
        with self.assertRaises(ValueError):
            exported_file(self.workspace, '..', 'a.parquet')
        self.assertFalse((self.workspace.path / 'exports' / 'busy.parquet').exists())

    def test_failed_export_cleans_owned_artifacts(self):
        with patch.object(pipeline, 'run_export', side_effect=RuntimeError('failed write')):
            with self.assertRaises(RuntimeError):
                export_workspace(self.app, 'test', 'retry')
        self.assertEqual(self.workspace.snapshots(), [])
        self.assertFalse((self.workspace.path / 'exports' / 'retry.parquet').exists())
        self.assertEqual(export_workspace(self.app, 'test', 'retry')['name'], 'retry')
