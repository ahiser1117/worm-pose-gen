from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from worm_pose_gen.jobs import (
    JobRecord,
    JobRunner,
    JobSpec,
    LocalGPUBackend,
    STATES,
    log_tail,
    pid_alive,
    process_matches,
    process_start_ticks,
    report_progress,
)


GPUS = [7, 8]
SRC = str(Path(__file__).resolve().parents[1] / "src")

# Every child imports report_progress from the package under test.
_PRELUDE = f"import sys, os, time; sys.path.insert(0, {SRC!r}); from worm_pose_gen.jobs import report_progress\n"


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", _PRELUDE + code]


def _finishing(message: str = "finished") -> list[str]:
    """Reports half way, then a result with the assigned gpu, exits 0."""

    return _python(
        "report_progress(0.5, 'half way')\n"
        "print('hello from the job')\n"
        f"report_progress(1.0, {message!r}, {{'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}})\n"
    )


def _failing() -> list[str]:
    return _python("report_progress(0.25, 'about to fail')\nprint('boom happened', file=sys.stderr)\nsys.exit(1)\n")


def _waiting(stop_file: Path) -> list[str]:
    """Reports the gpu it got, then sleeps in a loop until ``stop_file`` exists (10 s cap)."""

    return _python(
        f"report_progress(0.1, 'waiting', {{'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}})\n"
        f"deadline = time.monotonic() + 10\n"
        f"while not os.path.exists({str(stop_file)!r}) and time.monotonic() < deadline: time.sleep(0.02)\n"
    )


def _stubborn(stop_file: Path) -> list[str]:
    """Ignores SIGTERM and waits for ``stop_file`` (10 s cap): only SIGKILL ends it early."""

    return _python(
        "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "report_progress(0.1, 'stubborn')\n"
        "deadline = time.monotonic() + 10\n"
        f"while not os.path.exists({str(stop_file)!r}) and time.monotonic() < deadline: time.sleep(0.02)\n"
    )


def _wait_until(runner: JobRunner, job_id: str, states: tuple[str, ...], timeout: float = 15.0) -> JobRecord:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        runner.tick()
        record = runner.get(job_id)
        if record.state in states:
            return record
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} stayed {runner.get(job_id).state}, wanted {states}")


class JobsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.backend = LocalGPUBackend(GPUS, cwd=self.root, terminate_grace=2.0)
        self.runner = JobRunner(self.root, self.backend)

    def tearDown(self) -> None:
        for record in self.runner.list("running"):
            self.runner.cancel(record.id)
        self.runner.stop()
        self._tmp.cleanup()

    def _spec(self, label: str = "", workspace: str = "ws") -> JobSpec:
        return JobSpec(kind="command", params={"x": 1}, workspace=workspace, frames=[0, 5], label=label)

    def test_submit_persists_record_and_command(self) -> None:
        record = self.runner.submit(self._spec("first"), _finishing())
        self.assertEqual(record.state, "queued")
        self.assertRegex(record.id, r"^j[0-9a-f]{8}$")
        stored = json.loads((self.root / "jobs" / f"{record.id}.json").read_text())
        self.assertEqual(stored["command"], _finishing())
        self.assertEqual(stored["spec"]["label"], "first")
        self.assertEqual(stored["spec"]["frames"], [0, 5])
        self.assertIn("queued", STATES)
        self.assertEqual(self.runner.max_concurrent, len(GPUS))

    def test_progress_result_and_done(self) -> None:
        record = self.runner.submit(self._spec(), _finishing("all done"))
        self.runner.tick()
        started = self.runner.get(record.id)
        self.assertEqual(started.state, "running")
        self.assertIsNotNone(started.pid)
        self.assertIn(started.gpu, GPUS)
        done = _wait_until(self.runner, record.id, ("done", "failed"))
        self.assertEqual(done.state, "done")
        self.assertEqual(done.progress, 1.0)
        self.assertEqual(done.message, "all done")
        self.assertEqual(done.result, {"gpu": str(done.gpu)})
        self.assertIsNotNone(done.started_at)
        self.assertIsNotNone(done.finished_at)
        self.assertIsNone(done.error)
        self.assertIn("hello from the job", self.runner.log(record.id))
        self.assertEqual(self.runner.get(record.id).gpu, done.gpu)
        self.assertEqual(self.runner._free_gpus(), GPUS)
        self.assertIsNotNone(started.pid_start)
        self.assertFalse(done.cancel_requested)

    def test_failed_job_carries_log_tail(self) -> None:
        record = self.runner.submit(self._spec(), _failing())
        failed = _wait_until(self.runner, record.id, ("done", "failed"))
        self.assertEqual(failed.state, "failed")
        self.assertIn("boom happened", failed.error or "")
        self.assertEqual(failed.progress, 0.25)
        self.assertEqual(failed.message, "about to fail")
        self.assertEqual(self.runner.log(record.id, tail=1).strip(), "boom happened")
        self.assertIn("boom happened", self.runner.log(record.id))

    def test_fifo_with_max_concurrent_and_gpu_pool(self) -> None:
        stops = [self.root / f"stop{i}" for i in range(3)]
        records = [self.runner.submit(self._spec(f"job{i}", workspace=f"ws{i}"), _waiting(stops[i])) for i in range(3)]
        self.runner.tick()
        states = [self.runner.get(r.id).state for r in records]
        self.assertEqual(states, ["running", "running", "queued"])
        gpus = [self.runner.get(r.id).gpu for r in records[:2]]
        self.assertEqual(sorted(gpus), GPUS)
        self.assertIsNone(records[2].gpu)
        self.runner.tick()
        self.assertEqual(self.runner.get(records[2].id).state, "queued")
        # The children saw the gpu they were assigned.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and any(self.runner.get(r.id).result is None for r in records[:2]):
            self.runner.tick()
            time.sleep(0.02)
        for record, gpu in zip(records[:2], gpus):
            self.assertEqual(self.runner.get(record.id).result, {"gpu": str(gpu)})
        # Releasing the first job frees its gpu for the third.
        stops[0].touch()
        _wait_until(self.runner, records[0].id, ("done",))
        third = self.runner.get(records[2].id)
        self.assertEqual(third.state, "running")
        self.assertEqual(third.gpu, gpus[0])
        for stop in stops[1:]:
            stop.touch()
        for record in records[1:]:
            self.assertEqual(_wait_until(self.runner, record.id, ("done", "failed")).state, "done")

    def test_jobs_on_one_workspace_run_one_at_a_time(self) -> None:
        # Two stages on one workspace would overwrite each other's arrays: the second waits, another workspace's job does not.
        stops = [self.root / f"stop{i}" for i in range(3)]
        first = self.runner.submit(self._spec("first", workspace="alpha"), _waiting(stops[0]))
        second = self.runner.submit(self._spec("second", workspace="alpha"), _waiting(stops[1]))
        other = self.runner.submit(self._spec("other", workspace="beta"), _waiting(stops[2]))
        self.runner.tick()
        self.assertEqual([self.runner.get(r.id).state for r in (first, second, other)], ["running", "queued", "running"])
        self.runner.tick()
        self.assertEqual(self.runner.get(second.id).state, "queued")
        stops[0].touch()
        _wait_until(self.runner, first.id, ("done",))
        self.assertEqual(self.runner.get(second.id).state, "running")
        for stop in stops[1:]:
            stop.touch()
        for record in (second, other):
            self.assertEqual(_wait_until(self.runner, record.id, ("done", "failed")).state, "done")
        # Jobs without a workspace are never held back.
        free = self.runner.submit(JobSpec(kind="command"), _finishing())
        self.assertEqual(_wait_until(self.runner, free.id, ("done", "failed")).state, "done")

    def test_max_concurrent_below_gpu_count(self) -> None:
        runner = JobRunner(self.root, self.backend, max_concurrent=1)
        stops = [self.root / f"stop{i}" for i in range(2)]
        records = [runner.submit(self._spec(), _waiting(stops[i])) for i in range(2)]
        runner.tick()
        self.assertEqual([runner.get(r.id).state for r in records], ["running", "queued"])
        stops[0].touch()
        _wait_until(runner, records[0].id, ("done",))
        self.assertEqual(runner.get(records[1].id).state, "running")
        stops[1].touch()
        _wait_until(runner, records[1].id, ("done",))
        self.runner = runner

    def test_cancel_running_and_queued(self) -> None:
        stop = self.root / "never"
        running = self.runner.submit(self._spec(), _waiting(stop))
        queued_a = self.runner.submit(self._spec(), _waiting(stop))
        queued_b = self.runner.submit(self._spec(), _waiting(stop))
        self.runner.tick()
        self.assertEqual(self.runner.get(queued_b.id).state, "queued")
        pid = self.runner.get(running.id).pid
        self.assertTrue(pid_alive(pid))
        cancelled = self.runner.cancel(running.id)
        self.assertEqual(cancelled.state, "cancelled")
        self.assertIsNotNone(cancelled.finished_at)
        deadline = time.monotonic() + 5
        while pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(pid_alive(pid))
        # A queued job cancels without ever starting; the other queued one takes the freed gpu.
        self.assertEqual(self.runner.cancel(queued_b.id).state, "cancelled")
        self.assertIsNone(self.runner.get(queued_b.id).pid)
        self.runner.tick()
        self.assertEqual(self.runner.get(queued_a.id).state, "running")
        self.assertEqual(self.runner.get(running.id).state, "cancelled")
        # Cancelling a finished job is a no-op.
        self.assertEqual(self.runner.cancel(queued_b.id).state, "cancelled")
        stored = json.loads((self.root / "jobs" / f"{running.id}.json").read_text())
        self.assertEqual(stored["state"], "cancelled")

    def test_persistence_recover_and_id_counter(self) -> None:
        done = self.runner.submit(self._spec("done"), _finishing())
        _wait_until(self.runner, done.id, ("done",))
        orphan = self.runner.submit(self._spec("orphan"), _finishing())
        queued = self.runner.submit(self._spec("queued"), _finishing())
        # Pretend the server died while ``orphan`` ran under a pid that no longer exists.
        path = self.root / "jobs" / f"{orphan.id}.json"
        stored = json.loads(path.read_text())
        stored.update(state="running", pid=2**22 + 12345, started_at="2026-09-08T00:00:00+00:00", gpu=GPUS[0])
        path.write_text(json.dumps(stored))
        self.runner.stop()

        fresh = JobRunner(self.root, LocalGPUBackend(GPUS, cwd=self.root))
        fresh.recover()
        self.assertEqual(fresh.get(done.id).state, "done")
        self.assertEqual(fresh.get(done.id).result, {"gpu": str(self.runner.get(done.id).gpu)})
        recovered = fresh.get(orphan.id)
        self.assertEqual(recovered.state, "failed")
        self.assertEqual(recovered.error, "server restarted")
        self.assertIsNotNone(recovered.finished_at)
        self.assertEqual(json.loads(path.read_text())["state"], "failed")
        self.assertEqual(fresh.get(queued.id).state, "queued")
        # A job that finished while the server was down (progress 1.0 on disk) is done, not failed.
        finished_path = self.root / "jobs" / f"{done.id}.json"
        stored = json.loads(finished_path.read_text())
        stored.update(state="running", pid=2**22 + 12346, finished_at=None, error=None)
        finished_path.write_text(json.dumps(stored))
        again = JobRunner(self.root, LocalGPUBackend(GPUS, cwd=self.root))
        again.recover()
        self.assertEqual(again.get(done.id).state, "done")
        self.assertEqual(again.get(done.id).result, {"gpu": str(self.runner.get(done.id).gpu)})
        self.assertIsNone(again.get(done.id).error)
        later = fresh.submit(self._spec("later"), _finishing())
        ids = [done.id, orphan.id, queued.id, later.id]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(set(ids)), 4)
        self.assertEqual(int(later.id[1:], 16), int(queued.id[1:], 16) + 1)
        # The queued job survives the restart and runs on the fresh runner.
        self.assertEqual(_wait_until(fresh, queued.id, ("done", "failed")).state, "done")
        fresh.stop()

    def test_reused_pid_is_not_the_job(self) -> None:
        # After a reboot a stale record's pid may belong to any process: the start time tells them apart.
        stranger = subprocess.Popen(["sleep", "30"])
        try:
            record = self.runner.submit(self._spec("stale"), _finishing())
            path = self.root / "jobs" / f"{record.id}.json"
            stored = json.loads(path.read_text())
            ticks = process_start_ticks(stranger.pid)
            self.assertIsNotNone(ticks)
            stored.update(state="running", pid=stranger.pid, pid_start=ticks + 1, gpu=GPUS[0])
            path.write_text(json.dumps(stored))
            fresh = JobRunner(self.root, LocalGPUBackend(GPUS, cwd=self.root))
            self.assertFalse(process_matches(fresh.get(record.id)))
            fresh.recover()
            self.assertEqual(fresh.get(record.id).state, "failed")
            self.assertEqual(fresh._free_gpus(), GPUS)
            self.assertEqual(fresh.cancel(record.id).state, "failed")
            self.assertIsNone(stranger.poll())  # the stranger was neither killed nor waited on
            # The same pid with the right start time is the job.
            stored.update(pid_start=ticks)
            path.write_text(json.dumps(stored))
            fresh = JobRunner(self.root, LocalGPUBackend(GPUS, cwd=self.root))
            self.assertTrue(process_matches(fresh.get(record.id)))
            fresh.recover()
            fresh.tick()
            self.assertEqual(fresh.get(record.id).state, "running")
            self.assertEqual(fresh._free_gpus(), GPUS[1:])
            self.assertTrue(pid_alive(stranger.pid))
        finally:
            stranger.kill()
            stranger.wait()

    def test_cancel_after_exit_keeps_the_real_outcome(self) -> None:
        record = self.runner.submit(self._spec(), _finishing("all done"))
        self.runner.tick()
        process = self.backend._processes[record.id]
        self.assertEqual(process.wait(timeout=15), 0)
        # Not polled yet: the record still says running, but the job finished on its own.
        self.assertEqual(self.runner.get(record.id).state, "running")
        cancelled = self.runner.cancel(record.id)
        self.assertEqual(cancelled.state, "done")
        self.assertEqual(cancelled.result, {"gpu": str(cancelled.gpu)})
        self.assertEqual(cancelled.message, "all done")
        self.assertEqual(json.loads((self.root / "jobs" / f"{record.id}.json").read_text())["state"], "done")

    def test_cancel_waits_outside_the_lock(self) -> None:
        stop = self.root / "never"
        record = self.runner.submit(self._spec(), _stubborn(stop))
        _wait_until(self.runner, record.id, ("running",))
        deadline = time.monotonic() + 10
        while self.runner.get(record.id).progress < 0.1 and time.monotonic() < deadline:
            self.runner.tick()
            time.sleep(0.02)
        pid = self.runner.get(record.id).pid
        outcome: list[JobRecord] = []
        thread = threading.Thread(target=lambda: outcome.append(self.runner.cancel(record.id)))
        started = time.monotonic()
        thread.start()
        time.sleep(0.2)
        # While the cancel waits out the grace period the runner still answers.
        t0 = time.monotonic()
        listed = self.runner.list("running")
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertEqual([r.id for r in listed], [record.id])
        self.assertTrue(self.runner.get(record.id).cancel_requested)
        thread.join(timeout=15)
        self.assertFalse(thread.is_alive())
        self.assertGreaterEqual(time.monotonic() - started, self.backend.terminate_grace)
        self.assertEqual(outcome[0].state, "cancelled")
        self.assertFalse(pid_alive(pid))
        self.assertEqual(self.runner.get(record.id).state, "cancelled")

    def test_unstartable_command_fails_without_blocking_the_queue(self) -> None:
        bad = self.runner.submit(self._spec("bad", workspace="a"), [sys.executable, "-c", "bad\x00byte"])
        good = self.runner.submit(self._spec("good", workspace="b"), _finishing())
        self.runner.tick()
        failed = self.runner.get(bad.id)
        self.assertEqual(failed.state, "failed")
        self.assertIn("could not start", failed.error or "")
        self.assertEqual(_wait_until(self.runner, good.id, ("done", "failed")).state, "done")

    def test_list_ordering_and_filter(self) -> None:
        first = self.runner.submit(self._spec("a"), _finishing())
        second = self.runner.submit(self._spec("b"), _finishing())
        third = self.runner.submit(self._spec("c"), _failing())
        self.assertEqual([r.id for r in self.runner.list()], [third.id, second.id, first.id])
        self.assertEqual([r.id for r in self.runner.list("queued")], [third.id, second.id, first.id])
        for record in (first, second, third):
            _wait_until(self.runner, record.id, ("done", "failed"))
        self.assertEqual([r.id for r in self.runner.list("failed")], [third.id])
        self.assertEqual([r.id for r in self.runner.list("done")], [second.id, first.id])
        self.assertEqual(self.runner.list("running"), [])
        with self.assertRaises(KeyError):
            self.runner.get("j00000000")

    def test_background_thread_runs_jobs(self) -> None:
        record = self.runner.submit(self._spec(), _finishing())
        self.runner.start(interval=0.05)
        deadline = time.monotonic() + 15
        while self.runner.get(record.id).state not in ("done", "failed") and time.monotonic() < deadline:
            time.sleep(0.02)
        self.runner.stop()
        self.assertEqual(self.runner.get(record.id).state, "done")

    def test_record_round_trips_through_json(self) -> None:
        record = self.runner.submit(self._spec("rt"), _finishing())
        loaded = JobRecord.from_dict(json.loads(json.dumps(record.to_dict())))
        self.assertEqual(loaded, record)
        self.assertEqual(JobSpec.from_dict({"kind": "stage", "unknown": 1}).kind, "stage")
        # Records written before the identity fields existed still load.
        old = {k: v for k, v in record.to_dict().items() if k not in ("pid_start", "cancel_requested")}
        self.assertEqual(JobRecord.from_dict(old), record)

    def test_process_start_ticks(self) -> None:
        self.assertIsNone(process_start_ticks(None))
        self.assertIsNone(process_start_ticks(2**22 + 12345))
        own = process_start_ticks(os.getpid())
        self.assertIsInstance(own, int)
        self.assertGreater(own, 0)


class ReportProgressTest(unittest.TestCase):
    def test_writes_atomically_when_env_set_and_noop_otherwise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "progress.json"
            saved = os.environ.pop("WORM_POSE_PROGRESS_FILE", None)
            try:
                report_progress(0.5, "no target")
                self.assertFalse(target.exists())
                os.environ["WORM_POSE_PROGRESS_FILE"] = str(target)
                report_progress(1.5, "clamped", {"n": 3})
                payload = json.loads(target.read_text())
                self.assertEqual(payload, {"progress": 1.0, "message": "clamped", "result": {"n": 3}})
                self.assertFalse(target.with_name("progress.json.tmp").exists())
            finally:
                if saved is None:
                    os.environ.pop("WORM_POSE_PROGRESS_FILE", None)
                else:
                    os.environ["WORM_POSE_PROGRESS_FILE"] = saved

    def test_log_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log"
            path.write_text("a\nb\nc\n")
            self.assertEqual(log_tail(path), "a\nb\nc\n")
            self.assertEqual(log_tail(path, 2), "b\nc")
            self.assertEqual(log_tail(Path(tmp) / "missing"), "")


if __name__ == "__main__":
    unittest.main()
