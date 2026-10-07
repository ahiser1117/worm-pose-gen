"""Compute detection (local GPUs, SLURM, per-host defaults, node-local paths) and ``GET /api/compute`` with job placement."""

from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from worm_pose_gen import compute
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.compute import (
    CONSERVATIVE_SLURM_DEFAULTS, SlurmPartition, detect_compute, detect_slurm, local_gpus, node_local, slurm_defaults,
)
from tests.test_jobs import _finishing, _wait_until
from tests.test_slurm_backend import FakeSlurm, shared_tempdir


class HostDefaultsTest(unittest.TestCase):
    def test_engaging_hosts_and_unknown_hosts(self) -> None:
        for host in ("orcd-login001", "orcd-login003.mit.edu", "node2433", "node2433.inband"):
            with self.subTest(host=host):
                defaults = slurm_defaults(host)
                self.assertEqual((defaults.partition, defaults.gres), ("ou_bcs_normal", "gpu:1"))
        for host in ("flv-c3.mit.edu", "laptop", "nodeA"):
            with self.subTest(host=host):
                self.assertEqual(slurm_defaults(host), CONSERVATIVE_SLURM_DEFAULTS)
        self.assertIsNone(CONSERVATIVE_SLURM_DEFAULTS.partition)

    def test_node_local(self) -> None:
        with mock.patch.dict(os.environ, {"TMPDIR": "/state/partition1/job-7"}):
            self.assertEqual(node_local("/tmp/x/jobs"), "/tmp")
            self.assertEqual(node_local("/tmp"), "/tmp")
            self.assertEqual(node_local("/scratch/me/ws"), "/scratch")
            self.assertEqual(node_local("/dev/shm/a"), "/dev/shm")
            self.assertEqual(node_local("/var/tmp/a"), "/var/tmp")
            self.assertEqual(node_local("/state/partition1/job-7/ws"), "/state/partition1/job-7")
            self.assertIsNone(node_local("/tmpfiles/a"))
            self.assertIsNone(node_local("/orcd/scratch/bcs/001/me"))
            self.assertIsNone(node_local("/home/me/worm-pose-library"))
        # A link from shared storage into /tmp is node-local too.
        with shared_tempdir() as shared, mock.patch.dict(os.environ, {"TMPDIR": ""}):
            link = Path(shared) / "jobs"
            link.symlink_to("/tmp")
            self.assertEqual(node_local(link / "j1.log"), "/tmp")
            self.assertIsNone(node_local(Path(shared) / "real"))


class DetectTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = shared_tempdir()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_no_slurm(self) -> None:
        with mock.patch.dict(os.environ, {"PATH": str(self.root)}):
            info = detect_slurm("flv-c3")
        self.assertFalse(info.available)
        self.assertEqual(info.reason, "sbatch, squeue, sacct, scancel not on PATH")
        self.assertEqual(info.partitions, ())
        self.assertEqual(info.defaults, CONSERVATIVE_SLURM_DEFAULTS)

    def test_partial_slurm_is_not_available(self) -> None:
        fake = FakeSlurm(self.root, commands=("sbatch", "squeue"))
        try:
            with mock.patch.dict(os.environ, {"PATH": str(fake.bin)}):
                info = detect_slurm("node1234")
        finally:
            fake.close()
        self.assertFalse(info.available)
        self.assertEqual(info.reason, "sacct, scancel not on PATH")

    def test_slurm_with_partitions(self) -> None:
        fake = FakeSlurm(self.root)
        try:
            info = detect_slurm("node1234")
            self.assertTrue(info.available)
            self.assertIsNone(info.reason)
            self.assertEqual(info.partitions, (
                SlurmPartition(name="gpu", default=True, available=True, time_limit="1-00:00:00", gres=("gpu:a100:4", "gpu:h100:4")),
                SlurmPartition(name="cpu", default=False, available=True, time_limit="12:00:00", gres=()),
            ))
            self.assertEqual(info.defaults.partition, "ou_bcs_normal")
            # Without sinfo SLURM still works; only the partition list is empty.
            (fake.bin / "sinfo").unlink()
            self.assertEqual(detect_slurm("node1234").partitions, ())
        finally:
            fake.close()

    def test_local_gpus_without_cuda(self) -> None:
        # The suite runs with CUDA_VISIBLE_DEVICES empty: no GPU, and no failure.
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
            self.assertEqual(local_gpus(), ())
        with mock.patch.dict("sys.modules", {"torch": None}):
            self.assertEqual(local_gpus(), ())
        found = detect_compute()
        self.assertEqual(found.host, compute.hostname())
        self.assertEqual(found.to_dict()["slurm"]["defaults"]["time"], slurm_defaults().time)


class ComputeApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = shared_tempdir()
        self.root = Path(self._tmp.name)
        self.fake: FakeSlurm | None = None

    def tearDown(self) -> None:
        if self.fake is not None:
            self.fake.close()
        self._tmp.cleanup()

    def _client(self, gpus: tuple[int, ...]) -> TestClient:
        root = self.root / f"app{len(list(self.root.glob('app*')))}"
        config = AppConfig(
            workspaces_root=root / "ws", recording_roots=(), poses_root=root / "poses", dataset_root=root / "dataset",
            notes=root / "notes.json", checkpoint=None, prior_cache=None, gpus=gpus, device="cpu",
        )
        app = create_app(config)
        state = app.state.app_state
        # SLURM jobs run from this test's shared directory (the repository may sit anywhere).
        if "slurm" in state.runner.backends:
            state.runner.backends["slurm"].cwd = root
            state.runner.backends["slurm"].poll_seconds = 0.0
        client = TestClient(app)
        self.addCleanup(state.close)
        self.addCleanup(client.close)
        self.runner = state.runner
        return client

    def test_nothing_to_run_on(self) -> None:
        with mock.patch.dict(os.environ, {"PATH": str(self.root)}):
            client = self._client(gpus=())
        payload = client.get("/api/compute").json()
        self.assertFalse(payload["can_run"])
        self.assertIsNone(payload["default_run_on"])
        self.assertEqual(payload["local"], {"available": False, "reason": f"no GPU on {payload['host']} is enabled for jobs",
                                            "gpus": [], "max_concurrent": 1})
        self.assertFalse(payload["slurm"]["available"])
        self.assertIn("sbatch, squeue, sacct, scancel not on PATH", payload["slurm"]["reason"])
        self.assertEqual(payload["reason"], f"{payload['local']['reason']}, and {payload['slurm']['reason']}")
        response = client.post("/api/jobs", json={"kind": "command", "run_on": "slurm", "command": _finishing()})
        self.assertEqual(response.status_code, 400)
        self.assertIn("SLURM is not available", response.json()["error"])

    def test_local_and_slurm(self) -> None:
        self.fake = FakeSlurm(self.root)
        client = self._client(gpus=(0, 3))
        payload = client.get("/api/compute").json()
        self.assertTrue(payload["can_run"])
        self.assertIsNone(payload["reason"])
        self.assertEqual(payload["default_run_on"], "local")
        self.assertEqual(payload["local"]["available"], True)
        self.assertEqual([gpu["index"] for gpu in payload["local"]["gpus"]], [0, 3])
        self.assertEqual(payload["local"]["max_concurrent"], 2)
        self.assertEqual(payload["slurm"]["available"], True)
        self.assertEqual([p["name"] for p in payload["slurm"]["partitions"]], ["gpu", "cpu"])
        self.assertEqual(payload["slurm"]["partitions"][0]["gres"], ["gpu:a100:4", "gpu:h100:4"])
        self.assertEqual(set(payload["slurm"]["defaults"]), {"partition", "time", "gres", "cpus_per_task", "mem"})

        # A command job through SLURM with the host's time limit and a chosen partition; then retries.
        response = client.post("/api/jobs", json={"kind": "command", "run_on": "slurm", "slurm": {"partition": "cpu"}, "command": _finishing()})
        self.assertEqual(response.status_code, 200, response.text)
        job = response.json()
        self.assertEqual(job["spec"]["run_on"], "slurm")
        self.assertEqual(job["spec"]["slurm"], {"partition": "cpu", "time": payload["slurm"]["defaults"]["time"]})
        self.assertEqual(_wait_until(self.runner, job["id"], ("done", "failed")).state, "done")
        self.assertEqual(self.runner.get(job["id"]).slurm_job_id, "1000")
        retry = client.post(f"/api/jobs/{job['id']}/retry", json={}).json()
        self.assertEqual((retry["spec"]["run_on"], retry["spec"]["slurm"]), ("slurm", job["spec"]["slurm"]))
        self.assertEqual(_wait_until(self.runner, retry["id"], ("done", "failed")).state, "done")
        moved = client.post(f"/api/jobs/{job['id']}/retry", json={"run_on": "local"}).json()
        self.assertEqual((moved["spec"]["run_on"], moved["spec"]["slurm"], moved["spec"]["gpu"]), ("local", None, None))
        self.assertEqual(_wait_until(self.runner, moved["id"], ("done", "failed")).state, "done")
        back = client.post(f"/api/jobs/{moved['id']}/retry", json={"run_on": "slurm"}).json()
        self.assertEqual((back["spec"]["slurm"]["partition"], back["spec"]["gpu"]), (payload["slurm"]["defaults"]["partition"], None))
        _wait_until(self.runner, back["id"], ("done", "failed"))

        # Omitted run_on: this machine, as default_run_on says.
        local = client.post("/api/jobs", json={"kind": "command", "command": _finishing()}).json()
        self.assertEqual((local["spec"]["run_on"], local["spec"]["slurm"]), ("local", None))
        _wait_until(self.runner, local["id"], ("done", "failed"))
        for bad in ({"run_on": "slurm", "gpu": 3}, {"run_on": "slurm", "slurm": "fast"}, {"run_on": "slurm", "slurm": {"time": "soon"}},
                    {"run_on": "local", "slurm": {"partition": "gpu"}}, {"run_on": "elsewhere"}):
            with self.subTest(payload=bad):
                self.assertEqual(client.post("/api/jobs", json={"kind": "command", "command": _finishing(), **bad}).status_code, 400)

    def test_slurm_only_host_defaults_to_slurm(self) -> None:
        self.fake = FakeSlurm(self.root)
        client = self._client(gpus=())
        payload = client.get("/api/compute").json()
        self.assertTrue(payload["can_run"])
        self.assertFalse(payload["local"]["available"])
        self.assertEqual(payload["default_run_on"], "slurm")
        job = client.post("/api/jobs", json={"kind": "command", "command": _finishing()}).json()
        self.assertEqual(job["spec"]["run_on"], "slurm")
        self.assertEqual(_wait_until(self.runner, job["id"], ("done", "failed")).state, "done")


if __name__ == "__main__":
    unittest.main()
