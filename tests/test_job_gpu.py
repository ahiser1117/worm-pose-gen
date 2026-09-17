"""Requested GPU scheduling, persistence and explicit-device retries."""
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.jobs import JobRunner, JobSpec, LocalGPUBackend
from tests.test_jobs import _finishing, _waiting, _wait_until


class JobGpuTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.backend = LocalGPUBackend([0, 3], cwd=self.root, terminate_grace=.2)
        self.runner = JobRunner(self.root, self.backend)

    def tearDown(self):
        for r in self.runner.list('running'):
            self.runner.cancel(r.id)
        self.temp.cleanup()

    def test_physical_device_reaches_child_and_survives_reload(self):
        record = self.runner.submit(JobSpec('command', gpu=3), _finishing())
        self.runner = JobRunner(self.root, self.backend)
        result = _wait_until(self.runner, record.id, ('done', 'failed'))
        self.assertEqual(result.state, 'done')
        self.assertEqual(result.spec.gpu, 3)
        self.assertEqual(result.gpu, 3)
        self.assertEqual(result.result['gpu'], '3')

    def test_busy_gpu_waits_without_workspace_overtaking(self):
        holder = self.runner.submit(JobSpec('command', workspace='other', gpu=3), _waiting(self.root/'hold'))
        first = self.runner.submit(JobSpec('command', workspace='a', gpu=3), _waiting(self.root/'first'))
        later = self.runner.submit(JobSpec('command', workspace='a', gpu=0), _finishing())
        unrelated = self.runner.submit(JobSpec('command', workspace='b', gpu=0), _finishing())
        self.runner.tick()
        self.assertEqual([r.state for r in [holder, first, later, unrelated]], ['running','queued','queued','running'])
        (self.root/'hold').touch()
        _wait_until(self.runner, holder.id, ('done',))
        self.assertEqual(first.state, 'running')
        self.assertEqual(later.state, 'queued')
        (self.root/'first').touch()
        _wait_until(self.runner, later.id, ('done',))
        self.assertEqual(later.gpu, 0)

    def test_reject_invalid_or_disabled_device_without_creating_job(self):
        for gpu in [True, -1, '3', 3.0, 99]:
            with self.subTest(gpu=gpu), self.assertRaises(ValueError):
                self.runner.submit(JobSpec('command', gpu=gpu), _finishing())
        with self.assertRaises(ValueError):
            self.runner.submit(JobSpec('command', gpus=0, gpu=3), _finishing())
        self.assertEqual(self.runner.list(), [])

    def test_removed_gpu_after_restart_fails_instead_of_switching(self):
        record = self.runner.submit(JobSpec('command', gpu=3), _finishing())
        self.runner = JobRunner(self.root, LocalGPUBackend([0], cwd=self.root))
        self.runner.tick()
        restored = self.runner.get(record.id)
        self.assertEqual(restored.state, 'failed')
        self.assertIn('GPU 3', restored.error)
        self.assertIsNone(restored.pid)

    def test_submission_and_retry_api_device_choices(self):
        config = AppConfig(workspaces_root=self.root/'ws', recording_roots=(), poses_root=self.root/'poses',
                           dataset_root=self.root/'dataset', notes=self.root/'notes.json', checkpoint=None,
                           prior_cache=None, gpus=(0, 3), device='cpu')
        app = create_app(config)
        runner = app.state.app_state.runner
        client = TestClient(app)
        self.addCleanup(app.state.app_state.close)
        self.addCleanup(client.close)
        response = client.post('/api/jobs', json={'kind':'command', 'gpu':3, 'command':_finishing()})
        self.assertEqual(response.status_code, 200, response.text)
        first = response.json()
        self.assertEqual(first['spec']['gpu'], 3)
        self.assertEqual(client.post(f"/api/jobs/{first['id']}/retry", json={}).status_code, 400)
        self.assertEqual(_wait_until(runner, first['id'], ('done',)).result['gpu'], '3')
        for payload, expected in [({}, 3), ({'gpu':0}, 0), ({'gpu':None}, 0)]:
            response = client.post(f"/api/jobs/{first['id']}/retry", json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            retry = response.json()
            self.assertNotEqual(retry['id'], first['id'])
            self.assertEqual(retry['command'], first['command'])
            done = _wait_until(runner, retry['id'], ('done',))
            self.assertEqual(done.gpu, expected)
        self.assertEqual(client.get(f"/api/jobs/{first['id']}").json()['spec']['gpu'], 3)
        for invalid in [True, '3', 99]:
            self.assertEqual(client.post('/api/jobs', json={'kind':'command','gpu':invalid,'command':_finishing()}).status_code, 400)


if __name__ == '__main__':
    unittest.main()
