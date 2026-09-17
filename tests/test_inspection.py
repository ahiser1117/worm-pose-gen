"""Review persistence and stale review rejection over a synthetic workspace."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from types import SimpleNamespace
from worm_pose_gen.pipeline import workspace_lock, WorkspaceBusy
from worm_pose_gen.app.routers.inspection import get_inspection, review, ReviewRequest
from pydantic import ValidationError

from worm_pose_gen.app.inspection import inspection, mark_reviewed
from worm_pose_gen.app.workspace_view import WorkspaceView
from worm_pose_gen.workspace import Workspace
from tests.test_pose_viewer import _write_recording, _write_run


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        recording = root / 'recording.h5'
        _write_recording(recording)
        _write_run(root / 'run', recording)
        self.workspace = Workspace.import_run(root / 'workspaces', root / 'run')
        self.view = WorkspaceView(self.workspace, None, None)

    def test_review_is_persistent_and_does_not_clear_flags(self):
        before = inspection(self.view)
        after = mark_reviewed(self.view, 0, self.workspace.n - 1, before['revision'])
        self.assertEqual(after['summary']['reviewed_frames'], self.workspace.n)
        self.assertEqual(after['summary']['flagged_frames'], before['summary']['flagged_frames'])
        self.assertEqual(after['summary']['attention_segments'], 0)
        reopened = WorkspaceView(Workspace.open(self.workspace.path), None, None)
        self.assertEqual(inspection(reopened)['reviewed_rows'], list(range(self.workspace.n)))

    def test_changed_pose_requires_new_review_and_rejects_old_token(self):
        before = inspection(self.view)
        mark_reviewed(self.view, 0, 1, before['revision'])
        arrays = self.workspace.load_state()
        arrays['iou'][0] = .123
        self.workspace.save_state(arrays)
        self.assertEqual(inspection(self.view)['reviewed_rows'], [])
        with self.assertRaises(HTTPException) as error:
            mark_reviewed(self.view, 0, 1, before['revision'])
        self.assertEqual(error.exception.status_code, 409)

    def test_mask_change_requires_review(self):
        before = inspection(self.view)
        mark_reviewed(self.view, 0, 0, before['revision'])
        self.workspace.set_override_mask(0, np.zeros(self.workspace.image_shape, dtype=np.uint8))
        self.assertEqual(inspection(self.view)['reviewed_rows'], [])

    def test_invalid_bounds_are_rejected(self):
        token = inspection(self.view)['revision']
        for first, last in [(-1, 0), (1, 0), (0, self.workspace.n)]:
            with self.assertRaises(ValueError):
                mark_reviewed(self.view, first, last, token)

    def test_pipeline_lock_prevents_reviewing_partial_changes(self):
        token = inspection(self.view)['revision']
        with workspace_lock(self.workspace):
            with self.assertRaises(WorkspaceBusy):
                inspection(self.view)
            with self.assertRaises(WorkspaceBusy):
                mark_reviewed(self.view, 0, 0, token)

    def test_router_contract_and_stale_token(self):
        app = SimpleNamespace(view=lambda name: self.view, check_writable=lambda name: None)
        before = get_inspection('example', app)
        token = before['revision']
        response = review('example', ReviewRequest(first=0, last=1, revision=token), app)
        self.assertEqual(response['reviewed_rows'], [0, 1])
        with self.assertRaises(HTTPException) as error:
            review('example', ReviewRequest(first=0, last=1, revision='stale'), app)
        self.assertEqual(error.exception.status_code, 409)
        with self.assertRaises(ValidationError):
            ReviewRequest(first=0.5, last=1, revision=token)

    def test_disjoint_edit_preserves_review_but_local_edit_invalidates(self):
        from worm_pose_gen.edits import flip_orientation
        before = inspection(self.view)
        mark_reviewed(self.view, 0, 0, before['revision'])
        flip_orientation(self.workspace, [self.workspace.n - 1])
        after = inspection(self.view)
        self.assertNotEqual(before['revision'], after['revision'])
        self.assertEqual(after['reviewed_rows'], [0])
        reopened = WorkspaceView(Workspace.open(self.workspace.path), None, None)
        self.assertEqual(inspection(reopened)['reviewed_rows'], [0])
        flip_orientation(self.workspace, [0])
        self.assertEqual(inspection(self.view)['reviewed_rows'], [])

    def test_neighbour_change_invalidates_quality_context(self):
        token = inspection(self.view)['revision']
        mark_reviewed(self.view, 0, 0, token)
        arrays = self.workspace.load_state()
        arrays['centerline_xy'][1] += 2
        self.workspace.save_state(arrays)
        self.assertEqual(inspection(self.view)['reviewed_rows'], [])

    def test_global_settings_and_recording_change_invalidate_all(self):
        import json
        import os
        token = inspection(self.view)['revision']
        mark_reviewed(self.view, 0, self.workspace.n - 1, token)
        info = json.loads(self.workspace.info_path.read_text())
        info['settings']['threshold'] = .75
        self.workspace.info_path.write_text(json.dumps(info))
        changed = inspection(self.view)
        self.assertEqual(changed['reviewed_rows'], [])
        mark_reviewed(self.view, 0, self.workspace.n - 1, changed['revision'])
        recording = self.workspace.recording
        stat = recording.stat()
        os.utime(recording, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        self.assertEqual(inspection(self.view)['reviewed_rows'], [])

    def test_unchanged_revision_reuses_row_fingerprints(self):
        from unittest.mock import patch
        token = inspection(self.view)['revision']
        mark_reviewed(self.view, 0, 0, token)
        with patch.object(Workspace, 'mask_revision', side_effect=AssertionError('cached fingerprint should be reused')):
            self.assertEqual(inspection(self.view)['reviewed_rows'], [0])

    def test_disjoint_mask_edit_preserves_review(self):
        from worm_pose_gen.edits import set_mask
        row = self.workspace.n - 1
        token = inspection(self.view)['revision']
        mark_reviewed(self.view, row, row, token)
        set_mask(self.workspace, 0, np.zeros(self.workspace.image_shape, dtype=np.uint8))
        self.assertEqual(inspection(self.view)['reviewed_rows'], [row])
        reopened = WorkspaceView(Workspace.open(self.workspace.path), None, None)
        self.assertEqual(inspection(reopened)['reviewed_rows'], [row])
        set_mask(self.workspace, row, np.zeros(self.workspace.image_shape, dtype=np.uint8))
        self.assertEqual(inspection(self.view)['reviewed_rows'], [])
