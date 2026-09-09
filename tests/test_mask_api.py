"""Mask API transactions across imported, empty, and custom-dataset workspaces."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
from fastapi.testclient import TestClient
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.workspace import Workspace
from worm_pose_gen.label_app import data_url, mask_to_png_values, decode_mask_data_url
from tests.test_pose_viewer import HEIGHT, WIDTH, FRAMES, _write_recording, _write_run


class MaskAPITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.root = root
        self.recording = root / 'recordings' / 'rec.h5'
        self.recording.parent.mkdir()
        _write_recording(self.recording)
        self.run = root / 'runs' / 'demo'
        _write_run(self.run, self.recording)
        config = AppConfig(workspaces_root=root/'workspaces', recording_roots=(root/'recordings',),
            poses_root=root/'runs', dataset_root=root/'dataset', checkpoint=None, prior_cache=None,
            notes=root/'notes.json', gpus=(0,), device='cpu', job_interval=0.1)
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
        self.workspace = workspace
        before = workspace.load_arrays()
        old_mask = self.check(self.client.get(endpoint+'/mask', params={'frame':frame}))
        self.assertEqual((old_mask['height'],old_mask['width']), self.labels.shape)
        encoded = data_url(mask_to_png_values(self.labels))
        result = self.check(self.client.post(endpoint+'/mask',json={'frame':frame,'mask':encoded,'revision':old_mask['revision']}))
        self.assertTrue(result['mask']['has_override'])
        self.assertTrue(result['mask']['stale'])
        self.assertFalse(result['frame']['stats']['fitted'])
        np.testing.assert_array_equal(decode_mask_data_url(result['mask']['mask'],self.labels.shape),self.labels)
        revision = result['mask']['revision']
        self.assertEqual(self.client.post(endpoint+'/mask',json={'frame':frame,'mask':encoded,'revision':old_mask['revision']}).status_code,400)
        preview = self.check(self.client.get(endpoint+'/frame',params={'frame':frame,'threshold':0.9}))
        self.assertEqual(preview['mask_final_source'],'override')
        self.assertIn('mask_override',preview['layers'])
        cleared = self.check(self.client.delete(endpoint+'/mask',params={'frame':frame,'revision':revision}))
        self.assertFalse(cleared['mask']['has_override'])
        self.check(self.client.post(endpoint+'/edits',json={'kind':'undo','frame':frame}))
        restored = self.check(self.client.get(endpoint+'/mask',params={'frame':frame}))
        self.assertTrue(restored['has_override'])
        np.testing.assert_array_equal(decode_mask_data_url(restored['mask'],self.labels.shape),self.labels)
        self.check(self.client.post(endpoint+'/edits',json={'kind':'undo','frame':frame}))
        self.assertIsNone(workspace.get_override_mask(workspace.row_of(frame)))
        current = workspace.load_arrays()
        for key, value in before.items():
            np.testing.assert_array_equal(current[key],value,err_msg=key)

    def test_imported_pose_round_trip(self):
        workspace = Workspace.import_run(self.root/'workspaces',self.run,'imported')
        self.exercise(workspace,int(workspace.frame_index[-1]))

    def test_unfitted_workspace_round_trip(self):
        workspace = Workspace.create(self.root/'workspaces','empty',self.recording,0,FRAMES-1)
        self.exercise(workspace,0)

    def test_custom_dataset_round_trip(self):
        recording = self.root/'recordings'/'custom.h5'
        with h5py.File(self.recording,'r') as original, h5py.File(recording,'w') as custom:
            custom.create_dataset('/camera/video',data=original['/img_nir'][:])
        workspace = Workspace.create(self.root/'workspaces','custom',recording,0,FRAMES-1,settings={'dataset':'/camera/video'})
        self.exercise(workspace,0)

    def test_threshold_preview_does_not_change_override_fitter_starts(self):
        workspace = Workspace.import_run(self.root/'workspaces', self.run, 'starts')
        workspace.set_override_mask(0, self.labels)
        segmenters = self.app.state.app_state.viewer.segmenters
        with patch.object(segmenters, 'probability', return_value=(np.zeros(self.labels.shape), 'preview')):
            with patch('worm_pose_gen.pose_viewer.standard_initializations', return_value=[]) as initialize:
                self.check(self.client.get('/api/workspaces/starts/starts', params={'frame': 0, 'threshold': .8}))
        initialize.assert_called_once()
        np.testing.assert_array_equal(initialize.call_args.args[0], self.labels == 1)
