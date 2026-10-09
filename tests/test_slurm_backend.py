"""The SLURM backend against a fake scheduler: submit, progress, done, failed, cancel, restart, node-local paths.

``FakeSlurm`` puts ``sbatch``, ``squeue``, ``sacct``, ``scancel`` and
``sinfo`` scripts on the ``PATH``.  Its ``sbatch`` reads the ``#SBATCH``
lines of the batch script and really runs it in the background (from
``--chdir``, appending to ``--output``), so the job's progress file, log and
exit code are the real ones.  Marker files in its state directory steer it:
``hold`` keeps new jobs pending, ``purge`` makes ``squeue`` forget finished
jobs (so the outcome must come from ``sacct``), ``no_accounting`` empties
``sacct``, ``down`` makes ``squeue`` time out and ``reject`` makes ``sbatch``
refuse.

The test roots live under the repository's ``.cache``: the backend refuses
anything under ``/tmp``, where ``tempfile`` would put them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest

from worm_pose_gen.compute import SlurmDefaults
from worm_pose_gen.jobs import JobRecord, JobRunner, JobSpec, LocalGPUBackend, SlurmBackend, pid_alive
from tests.test_jobs import _failing, _finishing, _waiting, _wait_until


SHARED_TMP = Path(__file__).resolve().parents[1] / ".cache" / "tests"
DEFAULTS = SlurmDefaults(partition="gpu", time="02:00:00", gres="gpu:1", cpus_per_task=2, mem="4G")

_COMMON = """\
import json, os, shlex, signal, subprocess, sys
from pathlib import Path
state = Path(os.environ["FAKE_SLURM_DIR"])

def option(args, name):
    for arg in args:
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None

def status(job):
    if (state / f"{job}.cancelled").exists():
        return "CANCELLED"
    if (state / f"{job}.exit").exists():
        return "COMPLETED" if int((state / f"{job}.exit").read_text()) == 0 else "FAILED"
    return "RUNNING" if (state / f"{job}.started").exists() else "PENDING"
"""

FAKE_COMMANDS = {
    "sbatch": _COMMON + """
script = Path(sys.argv[-1])
if (state / "reject").exists():
    print("sbatch: error: invalid partition specified: gpu", file=sys.stderr)
    sys.exit(1)
options = {}
for line in script.read_text().splitlines():
    if line.startswith("#SBATCH "):
        key, _, value = line[len("#SBATCH "):].partition("=")
        options[key] = shlex.split(value)[0] if value else ""
counter = state / "counter"
job = int(counter.read_text()) + 1 if counter.exists() else 1000
counter.write_text(str(job))
(state / f"{job}.json").write_text(json.dumps({
    "argv": sys.argv[1:], "options": options,
    "slurm_env": sorted(k for k in os.environ if k.startswith("SLURM_")),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}))
q = shlex.quote
run = (
    f"while [ -e {q(str(state / 'hold'))} ]; do sleep 0.02; done; touch {q(str(state / f'{job}.started'))}; "
    f"cd {q(options['--chdir'])} && SLURM_JOB_ID={job} bash {q(str(script))} >> {q(options['--output'])} 2>&1; "
    f"echo $? > {q(str(state / f'{job}.exit'))}"
)
process = subprocess.Popen(["bash", "-c", run], start_new_session=True, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
(state / f"{job}.pid").write_text(str(process.pid))
print(f"{job};fakecluster")
""",
    "squeue": _COMMON + """
job = option(sys.argv[1:], "--jobs")
if (state / "down").exists():
    print("slurm_load_jobs error: Socket timed out on send/recv operation", file=sys.stderr)
    sys.exit(1)
known = (state / f"{job}.json").exists()
if not known or ((state / "purge").exists() and status(job) in ("COMPLETED", "FAILED", "CANCELLED")):
    print("slurm_load_jobs error: Invalid job id specified", file=sys.stderr)
    sys.exit(1)
reason = {"PENDING": "Resources", "FAILED": "NonZeroExitCode"}.get(status(job), "None")
print(f"{status(job)}|{reason}")
""",
    "sacct": _COMMON + """
job = option(sys.argv[1:], "--jobs")
if (state / "no_accounting").exists() or not (state / f"{job}.json").exists():
    sys.exit(0)
code = int((state / f"{job}.exit").read_text()) if (state / f"{job}.exit").exists() else 0
print({"CANCELLED": "CANCELLED by 1014|0:15"}.get(status(job), f"{status(job)}|{code}:0"))
""",
    "scancel": _COMMON + """
job = sys.argv[-1]
if status(job) in ("COMPLETED", "FAILED", "CANCELLED"):
    print(f"scancel: error: Kill job error on job id {job}: Job/step already completing or completed", file=sys.stderr)
    sys.exit(1)
(state / f"{job}.cancelled").touch()
try:
    os.killpg(int((state / f"{job}.pid").read_text()), signal.SIGTERM)
except ProcessLookupError:
    pass
""",
    "sinfo": """\
print("gpu*|up|1-00:00:00|gpu:a100:4")
print("gpu*|up|1-00:00:00|gpu:h100:4")
print("cpu|up|12:00:00|(null)")
""",
}


class FakeSlurm:
    """The fake scheduler's commands on the ``PATH`` (all, or ``commands``) and its state directory, undone by ``close``."""

    def __init__(self, root: Path, commands: tuple[str, ...] = tuple(FAKE_COMMANDS)) -> None:
        self.bin = root / "fake-slurm-bin"
        self.state = root / "fake-slurm-state"
        self.bin.mkdir(parents=True)
        self.state.mkdir(parents=True)
        for name in commands:
            path = self.bin / name
            path.write_text(f"#!{sys.executable}\n" + FAKE_COMMANDS[name])
            path.chmod(0o755)
        self._saved = {key: os.environ.get(key) for key in ("PATH", "FAKE_SLURM_DIR")}
        os.environ["PATH"] = f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}"
        os.environ["FAKE_SLURM_DIR"] = str(self.state)

    def flag(self, name: str, on: bool = True) -> None:
        marker = self.state / name
        marker.touch() if on else marker.unlink(missing_ok=True)

    def submitted(self, job_id: str) -> dict:
        return json.loads((self.state / f"{job_id}.json").read_text())

    def pid(self, job_id: str) -> int:
        return int((self.state / f"{job_id}.pid").read_text())

    def close(self) -> None:
        self.flag("hold", False)
        for pid_file in self.state.glob("*.pid"):
            try:
                os.killpg(int(pid_file.read_text()), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def shared_tempdir() -> tempfile.TemporaryDirectory:
    """A temporary directory on storage the backend accepts (not under /tmp)."""

    SHARED_TMP.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=SHARED_TMP)


class SlurmBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = shared_tempdir()
        self.root = Path(self._tmp.name)
        self.slurm = FakeSlurm(self.root)
        self.backend = SlurmBackend(DEFAULTS, cwd=self.root, poll_seconds=0.0, terminate_grace=5.0)
        self.runner = JobRunner(self.root, LocalGPUBackend([7], cwd=self.root, terminate_grace=2.0), self.backend)

    def tearDown(self) -> None:
        self.runner.stop()
        self.slurm.close()
        self._tmp.cleanup()

    def _spec(self, label: str = "", workspace: str | None = "ws", **slurm) -> JobSpec:
        return JobSpec(kind="command", workspace=workspace, label=label, run_on="slurm", slurm=slurm or None)

    def _wait_progress(self, runner: JobRunner, job_id: str, progress: float) -> JobRecord:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            runner.tick()
            record = runner.get(job_id)
            if record.progress >= progress:
                return record
            time.sleep(0.02)
        raise AssertionError(f"job {job_id} did not reach progress {progress}: {runner.get(job_id)}")

    def test_submit_runs_the_command_and_reports_done(self) -> None:
        saved = os.environ.get("SLURM_JOB_ID")
        os.environ["SLURM_JOB_ID"] = "4242"  # the app itself runs inside an interactive allocation
        try:
            record = self.runner.submit(self._spec("first"), _finishing("all done"))
            self.assertEqual(record.spec.slurm, {"partition": "gpu", "time": "02:00:00"})
            self.runner.tick()
        finally:
            if saved is None:
                os.environ.pop("SLURM_JOB_ID", None)
            else:
                os.environ["SLURM_JOB_ID"] = saved
        started = self.runner.get(record.id)
        self.assertEqual(started.state, "running")
        self.assertEqual(started.slurm_job_id, "1000")
        self.assertIsNone(started.gpu)
        self.assertIsNone(started.pid)
        submitted = self.slurm.submitted("1000")
        self.assertEqual(submitted["argv"], ["--parsable", str(self.root / "jobs" / f"{record.id}.sbatch")])
        self.assertEqual(submitted["options"], {
            "--job-name": f"worm-pose-{record.id}", "--output": record.log_path, "--open-mode": "append",
            "--chdir": str(self.root), "--time": "02:00:00", "--partition": "gpu", "--gres": "gpu:1",
            "--cpus-per-task": "2", "--mem": "4G",
        })
        # The new job does not inherit the app's own allocation or GPU assignment.
        self.assertEqual(submitted["slurm_env"], [])
        self.assertIsNone(submitted["cuda_visible_devices"])
        done = _wait_until(self.runner, record.id, ("done", "failed", "cancelled"))
        self.assertEqual(done.state, "done", done.error)
        self.assertEqual(done.slurm_state, "COMPLETED")
        self.assertEqual(done.progress, 1.0)
        self.assertEqual(done.message, "all done")
        self.assertEqual(done.result, {"gpu": None})
        self.assertIsNone(done.error)
        log = self.runner.log(record.id)
        self.assertIn("hello from the job", log)
        self.assertIn(f"worm-pose job {record.id}: SLURM job 1000 on", log)
        stored = json.loads((self.root / "jobs" / f"{record.id}.json").read_text())
        self.assertEqual((stored["state"], stored["slurm_job_id"], stored["spec"]["run_on"]), ("done", "1000", "slurm"))

    def test_progress_while_running(self) -> None:
        stop = self.root / "stop"
        record = self.runner.submit(self._spec(), _waiting(stop))
        running = self._wait_progress(self.runner, record.id, 0.1)
        self.assertEqual((running.state, running.slurm_state, running.message), ("running", "RUNNING", "waiting"))
        stop.touch()
        self.assertEqual(_wait_until(self.runner, record.id, ("done", "failed")).state, "done")

    def test_failed_job_carries_state_exit_code_and_log_tail(self) -> None:
        record = self.runner.submit(self._spec(), _failing())
        failed = _wait_until(self.runner, record.id, ("done", "failed"))
        self.assertEqual(failed.state, "failed")
        self.assertIn("SLURM job 1000 ended FAILED", failed.error or "")
        self.assertIn("boom happened", failed.error or "")
        self.assertEqual((failed.progress, failed.message), (0.25, "about to fail"))

    def test_outcome_from_accounting_once_the_queue_forgets_the_job(self) -> None:
        self.slurm.flag("purge")
        good = self.runner.submit(self._spec(workspace="a"), _finishing())
        bad = self.runner.submit(self._spec(workspace="b"), _failing())
        self.assertEqual(_wait_until(self.runner, good.id, ("done", "failed")).state, "done")
        failed = _wait_until(self.runner, bad.id, ("done", "failed"))
        self.assertIn("ended FAILED (exit code 1:0)", failed.error or "")

    def test_job_unknown_to_slurm_finishes_from_its_progress_file(self) -> None:
        self.slurm.flag("purge")
        self.slurm.flag("no_accounting")
        good = self.runner.submit(self._spec(workspace="a"), _finishing())
        bad = self.runner.submit(self._spec(workspace="b"), _failing())
        self.assertEqual(_wait_until(self.runner, good.id, ("done", "failed")).state, "done")
        failed = _wait_until(self.runner, bad.id, ("done", "failed"))
        self.assertIn("SLURM has no record of job 1001", failed.error or "")

    def test_unanswered_scheduler_leaves_the_job_running(self) -> None:
        stop = self.root / "stop"
        record = self.runner.submit(self._spec(), _waiting(stop))
        self._wait_progress(self.runner, record.id, 0.1)
        self.slurm.flag("down")
        stop.touch()
        exit_file = self.slurm.state / "1000.exit"
        deadline = time.monotonic() + 15
        while not exit_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.runner.tick()
        self.assertEqual((self.runner.get(record.id).state, self.runner.get(record.id).slurm_state), ("running", "RUNNING"))
        self.slurm.flag("down", False)
        self.assertEqual(_wait_until(self.runner, record.id, ("done", "failed")).state, "done")

    def test_pending_job_and_cancel(self) -> None:
        self.slurm.flag("hold")
        record = self.runner.submit(self._spec(), _finishing())
        self.runner.tick()
        self.runner.tick()
        pending = self.runner.get(record.id)
        self.assertEqual((pending.state, pending.slurm_state), ("running", "PENDING"))
        self.assertEqual(pending.message, "waiting in the SLURM queue (Resources)")
        cancelled = self.runner.cancel(record.id)
        self.assertEqual(cancelled.state, "cancelled")
        self.assertEqual(cancelled.slurm_state, "CANCELLED")
        self.assertTrue(cancelled.cancel_requested)
        self.assertNotIn("hello from the job", self.runner.log(record.id))

    def test_cancel_running_job(self) -> None:
        record = self.runner.submit(self._spec(), _waiting(self.root / "never"))
        self._wait_progress(self.runner, record.id, 0.1)
        pid = self.slurm.pid("1000")
        cancelled = self.runner.cancel(record.id)
        self.assertEqual(cancelled.state, "cancelled")
        self.assertIsNotNone(cancelled.finished_at)
        deadline = time.monotonic() + 5
        while pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(pid_alive(pid))
        # A job that finished before the cancel keeps its outcome (scancel is never called).
        done = self.runner.submit(self._spec(), _finishing())
        self.runner.tick()
        exit_file = self.slurm.state / "1001.exit"
        deadline = time.monotonic() + 15
        while not exit_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(self.runner.cancel(done.id).state, "done")

    def test_restart_reattaches_by_slurm_job_id(self) -> None:
        stop = self.root / "stop"
        record = self.runner.submit(self._spec(), _waiting(stop))
        queued = self.runner.submit(self._spec("second"), _finishing())
        self._wait_progress(self.runner, record.id, 0.1)
        self.runner.stop()
        del self.runner

        fresh = JobRunner(self.root, LocalGPUBackend([7], cwd=self.root), SlurmBackend(DEFAULTS, cwd=self.root, poll_seconds=0.0))
        self.runner = fresh
        fresh.recover()
        fresh.tick()
        reattached = fresh.get(record.id)
        self.assertEqual((reattached.state, reattached.slurm_job_id, reattached.progress), ("running", "1000", 0.1))
        # The second job on the workspace still waits for the first.
        self.assertEqual(fresh.get(queued.id).state, "queued")
        stop.touch()
        self.assertEqual(_wait_until(fresh, record.id, ("done", "failed")).state, "done")
        self.assertEqual(_wait_until(fresh, queued.id, ("done", "failed")).slurm_job_id, "1001")

    def test_restart_on_a_host_without_slurm(self) -> None:
        record = self.runner.submit(self._spec(), _waiting(self.root / "stop"))
        queued = self.runner.submit(self._spec(), _finishing())
        self.runner.tick()
        local_only = JobRunner(self.root, LocalGPUBackend([7], cwd=self.root))
        local_only.recover()
        lost = local_only.get(record.id)
        self.assertEqual(lost.state, "failed")
        self.assertIn("SLURM is not available on this host", lost.error or "")
        self.assertIn("SLURM job 1000", lost.error or "")
        local_only.tick()
        self.assertEqual(local_only.get(queued.id).state, "failed")
        with self.assertRaisesRegex(ValueError, "SLURM is not available"):
            local_only.submit(self._spec(), _finishing())
        (self.root / "stop").touch()

    def test_node_local_paths_are_refused(self) -> None:
        with tempfile.TemporaryDirectory(prefix="node-local-") as tmp:
            self.assertTrue(tmp.startswith(("/tmp", os.environ.get("TMPDIR") or "/tmp")))
            runner = JobRunner(Path(tmp), LocalGPUBackend([7]), SlurmBackend(DEFAULTS, cwd=self.root))
            with self.assertRaisesRegex(ValueError, "node-local") as caught:
                runner.submit(self._spec(), _finishing())
            self.assertIn("job log", str(caught.exception))
            self.assertIn("progress file", str(caught.exception))
            self.assertEqual(runner.list(), [])
            # Local jobs on the same root are fine.
            runner.submit(JobSpec(kind="command"), _finishing())
            # The working directory (the repository) is checked too.
            runner = JobRunner(self.root, LocalGPUBackend([7]), SlurmBackend(DEFAULTS, cwd=Path(tmp)))
            with self.assertRaisesRegex(ValueError, "working directory"):
                runner.submit(self._spec(), _finishing())
        # Paths in the command, also inside a JSON argument (a stage's --params).
        for command in (["python", "-m", "x", "--workspace", "/scratch/me/ws"],
                        ["python", "--params", json.dumps({"checkpoint": "/dev/shm/model.ckpt", "x": [1, "/var/tmp/a"]})]):
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, "command path /(scratch|dev/shm)"):
                self.runner.submit(self._spec(), command)
        self.assertEqual(self.runner.list(), [])

    def test_settings_validation_and_overrides(self) -> None:
        for slurm, message in (({"time": "soon"}, "time limit"), ({"partition": "gpu\n#SBATCH --x"}, "partition"),
                               ({"account": "lab"}, "unknown slurm settings")):
            with self.subTest(slurm=slurm), self.assertRaisesRegex(ValueError, message):
                self.runner.submit(self._spec(**slurm), _finishing())
        with self.assertRaisesRegex(ValueError, "SLURM assigns one"):
            self.runner.submit(JobSpec(kind="command", run_on="slurm", gpu=7), _finishing())
        with self.assertRaisesRegex(ValueError, "run_on 'slurm'"):
            self.runner.submit(JobSpec(kind="command", slurm={"time": "10"}), _finishing())
        with self.assertRaisesRegex(ValueError, "unknown run_on"):
            self.runner.submit(JobSpec(kind="command", run_on="cloud"), _finishing())
        self.assertEqual(self.runner.list(), [])
        record = self.runner.submit(JobSpec(kind="command", run_on="slurm", gpus=0, slurm={"partition": "cpu", "time": "1-00:00:00"}), _finishing())
        self.assertEqual(record.spec.slurm, {"partition": "cpu", "time": "1-00:00:00"})
        self.assertEqual(_wait_until(self.runner, record.id, ("done", "failed")).state, "done")
        options = self.slurm.submitted("1000")["options"]
        self.assertEqual((options["--partition"], options["--time"]), ("cpu", "1-00:00:00"))
        self.assertNotIn("--gres", options)
        self.assertIn("export CUDA_VISIBLE_DEVICES=\n", (self.root / "jobs" / f"{record.id}.sbatch").read_text())

    def test_sbatch_refusal_fails_the_job(self) -> None:
        self.slurm.flag("reject")
        record = self.runner.submit(self._spec(), _finishing())
        self.runner.tick()
        failed = self.runner.get(record.id)
        self.assertEqual(failed.state, "failed")
        self.assertIn("sbatch failed: sbatch: error: invalid partition specified", failed.error or "")

    def test_slurm_jobs_take_no_local_slot_and_workspaces_stay_serial(self) -> None:
        stops = {name: self.root / f"stop-{name}" for name in ("local", "slurm", "other")}
        self.runner.max_concurrent = 1
        local = self.runner.submit(JobSpec(kind="command", workspace="alpha"), _waiting(stops["local"]))
        same = self.runner.submit(self._spec("same workspace", workspace="alpha"), _finishing())
        other = self.runner.submit(self._spec("other workspace", workspace="beta"), _waiting(stops["slurm"]))
        local_other = self.runner.submit(JobSpec(kind="command", workspace="gamma"), _waiting(stops["other"]))
        self.runner.tick()
        states = [self.runner.get(r.id).state for r in (local, same, other, local_other)]
        # The SLURM job on beta runs beside the local one; the local slot is full, alpha is busy.
        self.assertEqual(states, ["running", "queued", "running", "queued"])
        self.assertEqual(self.runner.get(local.id).gpu, 7)
        stops["local"].touch()
        _wait_until(self.runner, local.id, ("done",))
        self.assertEqual(self.runner.get(same.id).state, "running")
        self.assertEqual(self.runner.get(local_other.id).state, "running")
        # A SLURM job writing a workspace blocks a local job on it too.
        after = self.runner.submit(JobSpec(kind="command", workspace="beta"), _finishing())
        stops["other"].touch()
        _wait_until(self.runner, local_other.id, ("done",))
        self.assertEqual(self.runner.get(after.id).state, "queued")
        stops["slurm"].touch()
        _wait_until(self.runner, other.id, ("done",))
        self.assertEqual(_wait_until(self.runner, after.id, ("done", "failed")).state, "done")
        self.assertEqual(_wait_until(self.runner, same.id, ("done", "failed")).state, "done")


if __name__ == "__main__":
    unittest.main()
