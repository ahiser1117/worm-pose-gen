"""Canonical independent-label contracts and the label groups Paint walks."""
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

    def open_group(self, payload, status=200):
        response = self.client.post('/api/labeling/groups', json=payload)
        self.assertEqual(response.status_code, status, response.text)
        return response.json()

    def group(self, group_id):
        response = self.client.get('/api/labeling/groups/'+group_id)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_manifest_group_starts_at_first_unlabeled_and_keeps_pledges(self):
        path=self.root/'queue.json'
        path.write_text(json.dumps({'name':'queue','recordings':{'alias':{'path':'movie.h5','dataset':'/other','split':'val'}},
                                   'frames':[{'recording':'alias','frame_index':i,'reasons':['contact']} for i in (0,2,4)]}))
        self.save({**self.target,'dataset':'/other'})
        group=self.open_group({'kind':'manifest','path':str(path)})
        self.assertEqual((group['kind'],group['name']),('manifest','queue'))
        self.assertEqual(group['progress'],{'total':3,'labeled':1,'remaining':2})
        self.assertEqual([e['labeled'] for e in group['entries']],[True,False,False])
        self.assertEqual(group['first_unlabeled'],1)
        entry=group['entries'][1]
        self.assertEqual((entry['target']['dataset'],entry['split'],entry['reasons']),('/other','val',['contact']))
        frame=self.post('frame',{'target':entry['target'],'group_id':group['id']})
        self.assertEqual((frame['pledged_split'],frame['entry']['position']),('val',2))
        saved=self.save(entry['target'],group_id=group['id'],split='train')['sample']
        self.assertEqual(saved['split'],'val')
        self.assertEqual(self.group(group['id'])['first_unlabeled'],2)
        self.client.delete('/api/corpus/labels/'+saved['sample_id'])
        self.assertEqual(self.post('frame',{'target':entry['target']})['pledged_split'],'val')
        self.assertEqual(self.save(entry['target'],split='train')['sample']['split'],'val')
        self.post('frame',{'target':{**self.target,'frame':5},'group_id':group['id']},400)
        listing=self.client.get('/api/labeling/groups').json()
        self.assertEqual([g['id'] for g in listing['groups']],[group['id']])
        self.assertEqual(listing['groups'][0]['progress']['labeled'],2)
        self.assertNotIn('entries',listing['groups'][0])
        discovered={Path(m['path']).parent.name for m in listing['discovered']}
        self.assertIn('labeling_round_3_contact',discovered)
        self.assertEqual(self.client.delete('/api/labeling/groups/'+group['id']).status_code,200)
        self.assertEqual(self.client.get('/api/labeling/groups/'+group['id']).status_code,404)

    def test_recording_sections_persist_and_come_from_workspace_ranges(self):
        section=self.open_group({'kind':'section','recording':str(self.path),'first':1,'last':5,'step':2})
        self.assertEqual([e['target']['frame'] for e in section['entries']],[1,3,5])
        self.assertEqual((section['kind'],section['name'],section['first_unlabeled']),('section','movie frames 1-5',0))
        self.open_group({'kind':'section','recording':str(self.path),'first':4,'last':6},400)
        self.open_group({'kind':'section','recording':str(self.path),'first':0,'last':2,'step':0},400)
        Workspace.create(self.config.workspaces_root,'demo',self.path,0,4,2,settings={'dataset':'/other'})
        ranged=self.open_group({'kind':'section','workspace':'demo','first':2,'last':4,'origin':{'workspace':'demo','reasons':['holes']}})
        self.assertEqual([e['target'] for e in ranged['entries']],
                         [{'recording':str(self.path.resolve()),'dataset':'/other','frame':f} for f in (2,4)])
        self.assertEqual((ranged['name'],ranged['origin']['reasons']),('demo frames 2-4',['holes']))
        self.save(ranged['entries'][0]['target'],group_id=ranged['id'])
        # Sections outlive the server: a new app lists them from label_sections.json.
        stored=json.loads((self.config.workspaces_root/'label_sections.json').read_text())
        self.assertEqual(set(stored),{section['id'],ranged['id']})
        other=create_app(self.config)
        try:
            client=TestClient(other)
            listed={g['id']:g for g in client.get('/api/labeling/groups').json()['groups']}
            self.assertEqual(listed[ranged['id']]['progress'],{'total':2,'labeled':1,'remaining':1})
            self.assertEqual(client.get('/api/labeling/groups/'+ranged['id']).json()['first_unlabeled'],1)
            self.assertEqual(client.delete('/api/labeling/groups/'+section['id']).status_code,200)
            client.close()
        finally:
            other.state.app_state.close()
        self.assertEqual(set(json.loads((self.config.workspaces_root/'label_sections.json').read_text())),{ranged['id']})

    def test_samples_group_edits_saved_labels(self):
        first=self.save()['sample']
        second=self.save({**self.target,'frame':3})['sample']
        group=self.open_group({'kind':'samples','sample_ids':[second['sample_id'],first['sample_id']],'name':'filtered'})
        self.assertEqual([e['target']['sample_id'] for e in group['entries']],[second['sample_id'],first['sample_id']])
        self.assertEqual((group['progress']['remaining'],group['first_unlabeled']),(0,None))
        updated=self.save(group['entries'][0]['target'],group_id=group['id'],revision=1)['sample']
        self.assertEqual((updated['sample_id'],updated['revision']),(second['sample_id'],2))
        self.open_group({'kind':'samples','sample_ids':[]},400)
        self.open_group({'kind':'samples','sample_ids':['nope']},404)

    def test_corpus_listing_filters(self):
        for frame in (0,2,4): self.save({**self.target,'frame':frame},split='val')
        response=self.client.get('/api/corpus',params={'split':'val','source':'manual:corpus'})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['filtered_counts']['val'],3)
        self.assertEqual(response.json()['facets']['recordings'][0]['dataset'],'/img_nir')

    def test_network_threshold_and_other_proposals(self):
        probability=np.full(self.image.shape,0.6,np.float32)
        with mock.patch.object(self.state.viewer.segmenters,'resolve',return_value=self.root/'fake.ckpt'), \
             mock.patch.object(self.state.viewer.segmenters,'probability',return_value=(probability,'fake.ckpt')):
            proposed=self.post('proposals',{'target':self.target,'source':'network','threshold':0.7})
            self.assertFalse(decode_mask_data_url(proposed['mask'],self.mask.shape).any())
            self.assertIn('probability',proposed)
        with mock.patch('worm_pose_gen.app.labeling.Proposer.classical',return_value=(self.mask==1,self.mask==0)):
            for source in ('classical','raw_threshold'):
                self.post('proposals',{'target':self.target,'source':source})

    def test_manifest_missing_source_duplicate_identity_and_conflicting_pledge(self):
        path=self.root/'missing.json'
        data={'recordings':{'missing':{'path':'missing.h5','split':'test'}},'frames':[{'recording':'missing','frame_index':0}]}
        path.write_text(json.dumps(data))
        group=self.open_group({'kind':'manifest','path':str(path)})
        self.assertEqual(len(group['errors']),1)
        self.assertIsNotNone(group['entries'][0]['error'])
        data={'recordings':{'a':{'path':'movie.h5','split':'test'},'b':{'path':'movie.h5','split':'val'}},'frames':[]}
        path.write_text(json.dumps(data))
        self.open_group({'kind':'manifest','path':str(path)},400)
        data['recordings']['b']['split']='test'
        data['frames']=[{'recording':'a','frame_index':0},{'recording':'b','frame_index':0}]
        path.write_text(json.dumps(data))
        self.open_group({'kind':'manifest','path':str(path)},400)
        self.open_group({'kind':'manifest','path':str(self.root/'absent.json')},404)
        self.open_group({'kind':'other'},400)

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
