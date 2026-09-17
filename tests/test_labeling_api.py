"""Canonical independent-label contracts and traversal regression fixtures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient
import h5py
import numpy as np

from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.corpus import CorpusStore
from worm_pose_gen.label_app import data_url, mask_to_png_values, decode_mask_data_url
from worm_pose_gen.workspace import Workspace


class LabelingApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.config = AppConfig(workspaces_root=self.root/'workspaces', recording_roots=(self.root,),
            poses_root=self.root/'poses', corpus_root=self.root/'corpus', checkpoint=None,
            prior_cache=None, notes=self.root/'notes.json', device='cpu', gpus=(), dataset_root=self.root/'cache')
        self.app = create_app(self.config)
        self.state = self.app.state.app_state
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.image = np.full((64,64), 100, np.uint8)
        self.mask = np.zeros((64,64), np.uint8)
        self.mask[20:40,20:40] = 1
        self.mask[0,0] = 255
        self.png = data_url(mask_to_png_values(self.mask))
        self.path = self.root/'movie.h5'
        with h5py.File(self.path, 'w') as f:
            for dataset in ('/img_nir','/other'):
                f.create_dataset(dataset, data=np.stack([self.image]*6))
        self.target = {'recording': str(self.path), 'dataset':'/img_nir','frame':0}
        self.pool = {'recordings':[{'recording':str(self.path),'dataset':'/img_nir'}]}
        self.patcher = mock.patch('worm_pose_gen.label_app.RecordingSource.corrected', return_value=(self.image,self.image))
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.state.close()
        self.client.close()
        self.directory.cleanup()

    def post(self, route, payload, status=200):
        response = self.client.post('/api/labeling/'+route,json=payload)
        self.assertEqual(response.status_code,status,response.text)
        return response.json()

    def save(self, target=None, **extra):
        return self.post('save',{'target':target or self.target,'mask':self.png,'revision':0,**extra})

    def test_direct_save_cas_dataset_identity_and_no_workspace(self):
        first = self.save(split='val')['sample']
        self.assertEqual(first['revision'],1)
        self.post('save',{'target':self.target,'mask':self.png,'revision':0},400)
        second = self.save(revision=1,split='train')['sample']
        self.assertEqual((second['revision'],second['split']),(2,'val'))
        other = self.save({**self.target,'dataset':'/other'})['sample']
        self.assertNotEqual(first['sample_id'],other['sample_id'])
        self.assertEqual(self.state.workspace_names(),[])
        self.assertTrue((self.config.corpus_root/'revisions'/first['sample_id']/'00000001.npz').exists())
        read = self.post('frame',{'target':self.target})
        self.assertEqual(read['corpus_revision'],2)
        self.assertFalse(read['capabilities']['network'])
        np.testing.assert_array_equal(decode_mask_data_url(read['mask'],self.mask.shape),self.mask)

    def test_saved_label_missing_source_and_proposal_refinement(self):
        sample = self.save()['sample']
        self.path.unlink()
        target = {'sample_id':sample['sample_id']}
        self.post('frame',{'target':target})
        proposal = self.post('proposals',{'target':target,'source':'saved_corpus','request_id':'one'})
        self.assertEqual(proposal['request_id'],'one')
        np.testing.assert_array_equal(decode_mask_data_url(proposal['mask'],self.mask.shape),self.mask)
        refined = self.post('refine',{'target':target,'mask':self.png,'method':'dilate','draft_generation':3})
        decoded = decode_mask_data_url(refined['mask'],self.mask.shape)
        self.assertEqual(decoded[0,0],255)
        self.assertGreater((decoded==1).sum(),(self.mask==1).sum())
        self.assertEqual(refined['draft_generation'],3)
        self.post('save',{'target':target,'mask':refined['mask'],'revision':1})
        self.post('proposals',{'target':target,'source':'network'},400)
        self.post('refine',{'target':target,'mask':'data:image/png;base64,garbage','method':'dilate'},400)

    def test_workspace_frame_preserves_independent_revisions_and_frozen_corpus_save(self):
        ws = Workspace.create(self.config.workspaces_root,'demo',self.path,0,4,2)
        ws.set_override_mask(0,self.mask)
        target = {'workspace':'demo','frame':0}
        read = self.post('frame',{'target':target})
        self.assertEqual(read['target']['workspace'],'demo')
        self.assertTrue(read['has_override'])
        self.assertEqual(read['corpus_revision'],0)
        frozen = self.mask.copy(); frozen[1,1]=1
        self.post('save',{'target':target,'mask':data_url(mask_to_png_values(frozen)),'revision':0})
        np.testing.assert_array_equal(ws.get_override_mask(0),self.mask)
        saved = self.post('proposals',{'target':target,'source':'saved_corpus'})
        np.testing.assert_array_equal(decode_mask_data_url(saved['mask'],self.mask.shape),frozen)
        self.post('next',{'mode':'sequential','pool':{'workspace':'demo','frames':[0,1]},'current':target},400)
        next_frame = self.post('next',{'mode':'sequential','pool':{'workspace':'demo'},'current':target})
        self.assertEqual(next_frame['target']['frame'],2)

    def test_traversal_exhaustion_stride_no_network_and_distinct_pool(self):
        answer=self.post('next',{'mode':'sequential','pool':self.pool,'current':self.target,'stride':2})
        self.assertEqual(answer['target']['frame'],2)
        answer=self.post('next',{'mode':'sequential','pool':self.pool,'current':{**self.target,'frame':5},'stride':1})
        self.assertTrue(answer['exhausted'])
        self.post('next',{'mode':'uncertain','pool':self.pool},400)
        self.post('next',{'mode':'random'},400)
        for frame in range(6): self.save({**self.target,'frame':frame})
        self.assertTrue(self.post('next',{'mode':'random','pool':self.pool})['exhausted'])

    def test_manifest_aliases_pledges_progress_and_workspace_bounds(self):
        path=self.root/'queue.json'
        path.write_text(json.dumps({'name':'queue','recordings':{'alias':{'path':'movie.h5','dataset':'/other','split':'val'}},
                                   'frames':[{'recording':'alias','frame_index':i} for i in (0,2,4)]}))
        manifest=self.post('manifests',{'path':str(path)})
        self.assertEqual(manifest['progress']['remaining'],3)
        answer=self.post('next',{'mode':'queue','manifest_id':manifest['id']})
        self.assertEqual(answer['target']['dataset'],'/other')
        saved=self.save(answer['target'],manifest_id=manifest['id'],split='train')['sample']
        self.assertEqual(saved['split'],'val')
        self.client.delete('/api/corpus/labels/'+saved['sample_id'])
        frame=self.post('frame',{'target':answer['target']})
        self.assertEqual(frame['pledged_split'],'val')
        restored=self.save(answer['target'],split='train')['sample']
        self.assertEqual(restored['split'],'val')
        Workspace.create(self.config.workspaces_root,'other',self.path,0,2,2,settings={'dataset':'/other'})
        answer=self.post('next',{'mode':'queue','manifest_id':manifest['id'],'pool':{'workspace':'other','frames':[2]}})
        self.assertEqual(answer['target']['workspace'],'other')
        self.assertEqual(answer['target']['frame'],2)
        self.assertTrue(self.post('next',{'mode':'queue','manifest_id':manifest['id'],
            'pool':{'workspace':'other','frames':[2]},'current':answer['target']})['exhausted'])

    def test_browse_cursor_survives_deleted_current_and_filters(self):
        for frame in (0,2,4): self.save({**self.target,'frame':frame},split='val')
        first=self.post('next',{'mode':'browse','filters':{'split':'val'}})
        self.client.delete('/api/corpus/labels/'+first['target']['sample_id'])
        second=self.post('next',{'mode':'browse','filters':{'split':'val'},'cursor':first['cursor'],'current':first['target']})
        self.assertEqual(second['target']['frame'],2)
        response=self.client.get('/api/corpus',params={'split':'val','source':'manual:corpus'})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['filtered_counts']['val'],2)
        self.assertEqual(response.json()['facets']['recordings'][0]['dataset'],'/img_nir')

    def test_browse_workspace_mapping_and_recording_pool(self):
        for frame in (0,2,4): self.save({**self.target,'frame':frame})
        self.save({**self.target,'dataset':'/other','frame':1})
        Workspace.create(self.config.workspaces_root,'browse',self.path,0,4,2)
        answer=self.post('next',{'mode':'browse','pool':{'workspace':'browse','frames':[2,4]}})
        self.assertEqual(answer['target']['workspace'],'browse')
        self.assertEqual(answer['target']['frame'],2)
        answer=self.post('next',{'mode':'browse','pool':{'recordings':[{'recording':str(self.path),'dataset':'/other'}]}})
        self.assertEqual(answer['target']['dataset'],'/other')

    def test_network_threshold_and_uncertainty_evaluate_unique_candidates(self):
        probability=np.full(self.image.shape,0.6,np.float32)
        with mock.patch.object(self.state.viewer.segmenters,'resolve',return_value=self.root/'fake.ckpt'), \
             mock.patch.object(self.state.viewer.segmenters,'probability',return_value=(probability,'fake.ckpt')) as predict:
            proposed=self.post('proposals',{'target':self.target,'source':'network','threshold':0.7})
            self.assertFalse(decode_mask_data_url(proposed['mask'],self.mask.shape).any())
            self.assertIn('probability',proposed)
            predict.reset_mock()
            answer=self.post('next',{'mode':'uncertain','pool':self.pool,'candidates':100})
            self.assertIn(answer['target']['frame'],range(6))
            self.assertEqual(predict.call_count,6)
        with mock.patch('worm_pose_gen.app.labeling.Proposer.classical',return_value=(self.mask==1,self.mask==0)):
            for source in ('classical','raw_threshold'):
                self.post('proposals',{'target':self.target,'source':source})

    def test_manifest_missing_source_duplicate_identity_and_conflicting_pledge(self):
        path=self.root/'missing.json'
        data={'recordings':{'missing':{'path':'missing.h5','split':'test'}},'frames':[{'recording':'missing','frame_index':0}]}
        path.write_text(json.dumps(data))
        manifest=self.post('manifests',{'path':str(path)})
        self.assertEqual(len(manifest['errors']),1)
        answer=self.post('next',{'mode':'queue','manifest_id':manifest['id']})
        self.assertTrue(answer['blocked'])
        self.assertFalse(answer['exhausted'])
        data={'recordings':{'a':{'path':'movie.h5','split':'test'},'b':{'path':'movie.h5','split':'val'}},'frames':[]}
        path.write_text(json.dumps(data))
        self.post('manifests',{'path':str(path)},400)
        data['recordings']['b']['split']='test'
        data['frames']=[{'recording':'a','frame_index':0},{'recording':'b','frame_index':0}]
        path.write_text(json.dumps(data))
        self.post('manifests',{'path':str(path)},400)

    def test_legacy_adoption_canonical_basename_collision_and_direct_alias(self):
        store=CorpusStore(self.config.corpus_root)
        old=store.save('movie',0,self.image,self.mask,source_path=str(self.path),label_source='legacy',split='test')
        response=self.client.post('/api/corpus/labels',json={'target':self.target,'mask':self.png,'revision':1})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['sample']['sample_id'],old.sample_id)
        self.assertEqual(response.json()['sample']['split'],'test')
        folder=self.root/'elsewhere'; folder.mkdir()
        other=folder/'movie.h5'
        with h5py.File(other,'w') as f: f.create_dataset('/img_nir',data=np.stack([self.image]))
        saved=self.save({**self.target,'recording':str(other)})['sample']
        self.assertNotEqual(saved['sample_id'],old.sample_id)


if __name__=='__main__': unittest.main()
