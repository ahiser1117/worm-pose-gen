"""The Issues panel's review (Looks OK): persistence, fingerprints per row, and stale review rejection over a synthetic workspace.

The workspace's frames start at 0 with step 1, so frames and rows coincide.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from types import SimpleNamespace
from worm_pose_gen.pipeline import workspace_lock, WorkspaceBusy
from pydantic import ValidationError

from worm_pose_gen.app import inspection as review_state
from worm_pose_gen.app.inspection import issues, review_issue
from worm_pose_gen.app.routers import fixes as fix_routes
from worm_pose_gen import edits
from worm_pose_gen.app.workspace_view import WorkspaceView
from worm_pose_gen.workspace import Workspace
from tests.test_frame_view import _write_recording, _write_workspace


def reviewed_rows(view: WorkspaceView) -> list[int]:
    """The rows whose review still holds."""

    return sorted(review_state._reviewed(view))


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        recording = root / 'recording.h5'
        _write_recording(recording)
        self.workspace = _write_workspace(root / 'workspaces', 'demo', recording)
        self.view = WorkspaceView(self.workspace, None, None)

    def test_review_is_persistent_and_does_not_clear_flags(self):
        before = issues(self.view)
        after = review_issue(self.view, 0, self.workspace.n - 1, before['revision'])
        self.assertEqual(len(after['issues']), len(before['issues']))
        self.assertEqual([i['reasons'] for i in after['issues']], [i['reasons'] for i in before['issues']])
        self.assertEqual({i['state'] for i in after['issues']}, {'reviewed'})
        reopened = WorkspaceView(Workspace.open(self.workspace.path), None, None)
        self.assertEqual(reviewed_rows(reopened), list(range(self.workspace.n)))

    def test_changed_pose_requires_new_review_and_rejects_old_token(self):
        before = issues(self.view)
        review_issue(self.view, 0, 1, before['revision'])
        arrays = self.workspace.load_state()
        arrays['iou'][0] = .123
        self.workspace.save_state(arrays)
        self.assertEqual(reviewed_rows(self.view), [])
        with self.assertRaises(HTTPException) as error:
            review_issue(self.view, 0, 1, before['revision'])
        self.assertEqual(error.exception.status_code, 409)

    def test_mask_change_requires_review(self):
        before = issues(self.view)
        review_issue(self.view, 0, 0, before['revision'])
        self.workspace.set_override_mask(0, np.zeros(self.workspace.image_shape, dtype=np.uint8))
        self.assertEqual(reviewed_rows(self.view), [])

    def test_invalid_bounds_are_rejected(self):
        token = issues(self.view)['revision']
        for first, last in [(-1, 0), (1, 0), (0, self.workspace.n)]:
            with self.assertRaises(ValueError):
                review_issue(self.view, first, last, token)

    def test_pipeline_lock_prevents_reviewing_partial_changes(self):
        token = issues(self.view)['revision']
        with workspace_lock(self.workspace):
            with self.assertRaises(WorkspaceBusy):
                issues(self.view)
            with self.assertRaises(WorkspaceBusy):
                review_issue(self.view, 0, 0, token)

    def test_disjoint_edit_preserves_review_but_local_edit_invalidates(self):
        from worm_pose_gen.edits import flip_orientation
        before = issues(self.view)
        review_issue(self.view, 0, 0, before['revision'])
        flip_orientation(self.workspace, [self.workspace.n - 1])
        after = issues(self.view)
        self.assertNotEqual(before['revision'], after['revision'])
        self.assertEqual(reviewed_rows(self.view), [0])
        reopened = WorkspaceView(Workspace.open(self.workspace.path), None, None)
        self.assertEqual(reviewed_rows(reopened), [0])
        flip_orientation(self.workspace, [0])
        self.assertEqual(reviewed_rows(self.view), [])

    def test_neighbour_change_invalidates_quality_context(self):
        token = issues(self.view)['revision']
        review_issue(self.view, 0, 0, token)
        arrays = self.workspace.load_state()
        arrays['centerline_xy'][1] += 2
        self.workspace.save_state(arrays)
        self.assertEqual(reviewed_rows(self.view), [])

    def test_global_settings_and_recording_change_invalidate_all(self):
        import json
        import os
        token = issues(self.view)['revision']
        review_issue(self.view, 0, self.workspace.n - 1, token)
        info = json.loads(self.workspace.info_path.read_text())
        info['settings']['threshold'] = .75
        self.workspace.info_path.write_text(json.dumps(info))
        changed = issues(self.view)
        self.assertEqual(reviewed_rows(self.view), [])
        review_issue(self.view, 0, self.workspace.n - 1, changed['revision'])
        recording = self.workspace.recording
        stat = recording.stat()
        os.utime(recording, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        self.assertEqual(reviewed_rows(self.view), [])

    def test_unchanged_revision_reuses_row_fingerprints(self):
        from unittest.mock import patch
        token = issues(self.view)['revision']
        review_issue(self.view, 0, 0, token)
        with patch.object(Workspace, 'mask_revision', side_effect=AssertionError('cached fingerprint should be reused')):
            self.assertEqual(reviewed_rows(self.view), [0])

    def test_disjoint_mask_edit_preserves_review(self):
        from worm_pose_gen.edits import set_mask
        row = self.workspace.n - 1
        token = issues(self.view)['revision']
        review_issue(self.view, row, row, token)
        set_mask(self.workspace, 0, np.zeros(self.workspace.image_shape, dtype=np.uint8))
        self.assertEqual(reviewed_rows(self.view), [row])
        reopened = WorkspaceView(Workspace.open(self.workspace.path), None, None)
        self.assertEqual(reviewed_rows(reopened), [row])
        set_mask(self.workspace, row, np.zeros(self.workspace.image_shape, dtype=np.uint8))
        self.assertEqual(reviewed_rows(self.view), [])

    def test_issues_have_plain_reasons_and_follow_review_and_fixes(self):
        payload = issues(self.view)
        self.assertEqual(payload['summary']['issues'], 1)
        issue = payload['issues'][0]
        last = self.workspace.n - 1
        self.assertEqual((issue['rows'], issue['reasons'], issue['state']), ([last, last], ['coiled', 'mask fits poorly'], 'unreviewed'))
        self.assertEqual(issue['refit']['algorithm'], 'beam_path')
        with self.assertRaises(HTTPException) as error:
            review_issue(self.view, issue['frames'][0], issue['frames'][1], 'stale')
        self.assertEqual(error.exception.status_code, 409)
        reviewed = review_issue(self.view, issue['frames'][0], issue['frames'][1], payload['revision'])
        self.assertEqual((reviewed['issues'][0]['state'], reviewed['summary']['done']), ('reviewed', 1))
        self.assertEqual(reviewed_rows(self.view), [last])
        # A fix next to the issue drops its review (the neighbour's pose is part of the fingerprint) and is an issue of its own, fixed.
        edits.flip_orientation(self.workspace, [last - 1])
        # (The ambiguity refresh around the flip may flag the frame before it as well.)
        after = issues(self.view)
        self.assertEqual(len(after['issues']), 1)
        merged = after['issues'][0]
        self.assertEqual((merged['rows'][1], merged['state']), (last, 'unreviewed'))
        self.assertLessEqual(merged['rows'][0], last - 1)
        self.assertIn('head/tail uncertain', merged['reasons'])
        review_issue(self.view, merged['frames'][0], merged['frames'][1], after['revision'])
        self.assertEqual(issues(self.view)['issues'][0]['state'], 'fixed')
        with self.assertRaises(ValueError):
            review_issue(self.view, 0, 99, issues(self.view)['revision'])

    def test_issue_router_contract(self):
        app = SimpleNamespace(view=lambda name: self.view, check_writable=lambda name: None)
        payload = fix_routes.get_issues('example', app)
        frames = payload['issues'][0]['frames']
        response = fix_routes.review('example', fix_routes.ReviewRequest(first=frames[0], last=frames[1], revision=payload['revision']), app)
        self.assertEqual(response['issues'][0]['state'], 'reviewed')
        with self.assertRaises(ValidationError):
            fix_routes.ReviewRequest(first=0.5, last=1, revision='x')
