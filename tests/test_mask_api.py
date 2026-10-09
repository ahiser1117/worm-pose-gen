"""Edit mask on the Workspace page: saving an override and undoing it, across fitted, empty and custom-dataset workspaces."""
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from fastapi.testclient import TestClient

from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.app.images import data_url, decode_mask_data_url, mask_to_png_values
from worm_pose_gen.workspace import Workspace
from tests.test_frame_view import HEIGHT, WIDTH, FRAMES, _write_recording, _write_workspace


class MaskAPITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.root = root
        self.recording = root / 'recordings' / 'rec.h5'
        self.recording.parent.mkdir()
        _write_recording(self.recording)
        config = AppConfig(workspaces_root=root / 'workspaces', dataset_root=root / 'dataset', gpus=(0,), device='cpu', job_interval=0.1,
                           lab_library=root / 'lab', library=root / 'mine')
        self.app = create_app(config)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.labels = np.zeros((HEIGHT, WIDTH), np.uint8)
        self.labels[20:35, 30:80] = 1
        self.labels[22:25, 40:44] = 255

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.directory.cleanup()

    def check(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def exercise(self, workspace, frame):
        endpoint = f'/api/workspaces/{workspace.info.name}'
        before = workspace.load_arrays()
        old_mask = self.check(self.client.get(endpoint + '/mask', params={'frame': frame}))
        self.assertEqual((old_mask['height'], old_mask['width']), self.labels.shape)
        encoded = data_url(mask_to_png_values(self.labels))
        result = self.check(self.client.post(endpoint + '/mask', json={'frame': frame, 'mask': encoded, 'revision': old_mask['revision']}))
        self.assertTrue(result['mask']['has_override'])
        self.assertTrue(result['mask']['stale'])
        self.assertFalse(result['frame']['stats']['fitted'])
        np.testing.assert_array_equal(decode_mask_data_url(result['mask']['mask'], self.labels.shape), self.labels)
        revision = result['mask']['revision']
        # A save against a revision that is no longer current is refused.
        self.assertEqual(self.client.post(endpoint + '/mask', json={'frame': frame, 'mask': encoded, 'revision': old_mask['revision']}).status_code, 400)
        preview = self.check(self.client.get(endpoint + '/frame', params={'frame': frame}))
        self.assertEqual(preview['mask_final_source'], 'override')
        self.assertIn('mask_override', preview['layers'])
        self.assertTrue(preview['has_override'])
        self.assertEqual(preview['mask_revision'], revision)
        light = self.check(self.client.get(endpoint + '/frame', params={'frame': frame, 'detail': 'light'}))
        self.assertEqual(list(light['layers']), ['image'])
        self.assertTrue(light['details_deferred'])
        self.assertNotIn('has_override', light)
        self.assertNotIn('mask_revision', light)
        # The edit is in the Fixes list, and its Undo removes the override and restores the arrays.
        fixes = self.check(self.client.get(endpoint + '/fixes'))['fixes']
        self.assertEqual([(f['kind'], f['frames']) for f in fixes], [('mask', [frame, frame])])
        self.check(self.client.post(f"{endpoint}/fixes/{fixes[0]['id']}/undo", json={'frame': frame}))
        restored = self.check(self.client.get(endpoint + '/mask', params={'frame': frame}))
        self.assertFalse(restored['has_override'])
        self.assertIsNone(workspace.get_override_mask(workspace.row_of(frame)))
        current = workspace.load_arrays()
        for key, value in before.items():
            np.testing.assert_array_equal(current[key], value, err_msg=key)

    def test_fitted_workspace_round_trip(self):
        workspace = _write_workspace(self.root / 'workspaces', 'fitted', self.recording)
        self.exercise(workspace, int(workspace.frame_index[-1]))

    def test_unfitted_workspace_round_trip(self):
        workspace = Workspace.create(self.root / 'workspaces', 'empty', self.recording, 0, FRAMES - 1)
        self.exercise(workspace, 0)

    def test_custom_dataset_round_trip(self):
        recording = self.root / 'recordings' / 'custom.h5'
        with h5py.File(self.recording, 'r') as original, h5py.File(recording, 'w') as custom:
            custom.create_dataset('/camera/video', data=original['/img_nir'][:])
        workspace = Workspace.create(self.root / 'workspaces', 'custom', recording, 0, FRAMES - 1, settings={'dataset': '/camera/video'})
        self.exercise(workspace, 0)


if __name__ == '__main__':
    unittest.main()
