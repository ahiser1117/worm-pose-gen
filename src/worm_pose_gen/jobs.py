"""Job runner for long computations, with a local-GPU backend.

Anything longer than a frame runs as a job: a subprocess started by a
backend, reporting progress through a small JSON file, surviving a server
restart and cancellable.  Every job is a file ``jobs/<id>.json`` under the
runner's root so the state is inspectable on disk and so a Slurm backend can
be added later without changing the runner: the backend only starts, polls
and cancels a record, the runner owns the queue, the GPU pool and the files.

A job process reports progress by calling :func:`report_progress`, which
rewrites the file named by the environment variable
``WORM_POSE_PROGRESS_FILE`` atomically; the backend reads it on every poll.
Job ids come from a counter file under the root, so ids are never reused
across restarts.

Jobs on one workspace run one at a time, first in first out: every stage
rewrites the workspace's arrays whole, so two at once would overwrite each
other.  A job inherited from a previous server process is identified by its
pid and the process start time, so a pid reused after a reboot is not
mistaken for the job.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Any, Protocol


STATES = ("queued", "running", "done", "failed", "cancelled")
FINISHED_STATES = ("done", "failed", "cancelled")

PROGRESS_FILE_ENV = "WORM_POSE_PROGRESS_FILE"
JOB_ID_ENV = "WORM_POSE_JOB_ID"

REPO_ROOT = Path(__file__).resolve().parents[2]

ERROR_TAIL_LINES = 20
CANCEL_POLL_SECONDS = 0.05


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- records


@dataclass
class JobSpec:
    """What a job is: a stage, an export or a bare command, with its parameters."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    workspace: str | None = None
    frames: list[int] | None = None
    gpus: int = 1
    label: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobSpec":
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass
class JobRecord:
    """The persisted state of one job; ``asdict`` gives the JSON on disk."""

    id: str
    spec: JobSpec
    state: str = "queued"
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    progress: float = 0.0
    message: str = ""
    log_path: str = ""
    pid: int | None = None
    # Start time of the process (clock ticks since boot): with the pid it identifies the process.
    pid_start: int | None = None
    gpu: int | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    command: list[str] = field(default_factory=list)
    progress_path: str = ""
    cancel_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        known = {name for name in cls.__dataclass_fields__}
        fields = {key: value for key, value in data.items() if key in known}
        fields["spec"] = JobSpec.from_dict(fields.get("spec") or {"kind": "command"})
        return cls(**fields)

    @property
    def finished(self) -> bool:
        return self.state in FINISHED_STATES


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def _read_json(path: Path) -> Any | None:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- progress


def report_progress(progress: float, message: str = "", result: dict[str, Any] | None = None) -> None:
    """Called from inside a job: publish progress (0..1), a message and an optional result.

    Writes atomically to the file named by ``WORM_POSE_PROGRESS_FILE``; a
    no-op when the variable is unset so stage code also runs outside a job.
    """

    target = os.environ.get(PROGRESS_FILE_ENV)
    if not target:
        return
    payload = {"progress": float(min(max(progress, 0.0), 1.0)), "message": str(message), "result": result}
    _write_json_atomic(Path(target), payload)


def read_progress(path: str | Path) -> dict[str, Any] | None:
    """The last progress report of a job, or None when it has not reported yet."""

    if not path:
        return None
    payload = _read_json(Path(path))
    return payload if isinstance(payload, dict) else None


def log_tail(path: str | Path, lines: int | None = None) -> str:
    """The log text of a job, or its last ``lines`` lines."""

    if not path:
        return ""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return ""
    if lines is None:
        return text
    return "\n".join(text.splitlines()[-lines:]) if lines > 0 else ""


def pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, OverflowError):
        return False
    return True


def process_start_ticks(pid: int | None) -> int | None:
    """When process ``pid`` started, in clock ticks since boot (field 22 of ``/proc/<pid>/stat``); None when unknown."""

    if pid is None or pid <= 0:
        return None
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # The command name (field 2) may hold spaces and parentheses; the fields
    # after its closing parenthesis start at field 3.
    rest = text.rpartition(")")[2].split()
    try:
        return int(rest[19])
    except (IndexError, ValueError):
        return None


def process_matches(record: JobRecord) -> bool:
    """The job's process is still running and is the one that was started (not a reuse of its pid)."""

    if not pid_alive(record.pid):
        return False
    if record.pid_start is None:
        return True
    return process_start_ticks(record.pid) == record.pid_start


# --------------------------------------------------------------------------- backends


class JobBackend(Protocol):
    """Starts, polls and cancels one job; the runner owns the queue and the files.

    ``cancel`` asks the job to stop and returns at once; a later ``poll``
    reaps it as ``cancelled``.  A job whose process already exited is
    finished with its real outcome instead.
    """

    def start(self, record: JobRecord, command: list[str], env: dict[str, str]) -> JobRecord: ...

    def poll(self, record: JobRecord) -> JobRecord: ...

    def cancel(self, record: JobRecord) -> JobRecord: ...


def _apply_progress(record: JobRecord) -> None:
    payload = read_progress(record.progress_path)
    if payload is None:
        return
    try:
        record.progress = float(min(max(payload.get("progress", record.progress), 0.0), 1.0))
    except (TypeError, ValueError):
        pass
    record.message = str(payload.get("message", record.message) or "")
    result = payload.get("result")
    if isinstance(result, dict):
        record.result = result


def _finish(record: JobRecord, state: str, error: str | None = None) -> JobRecord:
    record.state = state
    record.finished_at = utc_now()
    record.error = error
    if state == "done":
        record.progress = 1.0
    return record


def _finish_gone(record: JobRecord, error: str) -> JobRecord:
    """Finish a job whose process is gone with no exit code: its progress file tells whether it finished."""

    _apply_progress(record)
    if record.cancel_requested:
        return _finish(record, "cancelled")
    if record.progress >= 1.0:
        return _finish(record, "done")
    return _finish(record, "failed", error)


class LocalGPUBackend:
    """Runs jobs as subprocesses on this machine, one GPU each.

    ``CUDA_VISIBLE_DEVICES`` is set to the GPU the runner assigned, the
    working directory is the repository root and stdout and stderr go to the
    record's log file.  Processes started by this backend are reaped through
    their ``Popen`` handle; a job inherited from a previous server process is
    followed by pid and start time only, so its exit code is unknown and its
    outcome is read from the progress file.
    """

    def __init__(self, gpus: list[int], cwd: Path | None = None, terminate_grace: float = 5.0) -> None:
        self.gpus = [int(gpu) for gpu in gpus]
        self.cwd = Path(cwd) if cwd is not None else REPO_ROOT
        self.terminate_grace = float(terminate_grace)
        self._processes: dict[str, subprocess.Popen] = {}
        self._logs: dict[str, Any] = {}

    def start(self, record: JobRecord, command: list[str], env: dict[str, str]) -> JobRecord:
        job_env = dict(env)
        if record.gpu is not None:
            job_env["CUDA_VISIBLE_DEVICES"] = str(record.gpu)
        Path(record.log_path).parent.mkdir(parents=True, exist_ok=True)
        log = open(record.log_path, "ab")
        try:
            process = subprocess.Popen(
                list(command), cwd=str(self.cwd), env=job_env, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        except Exception:
            log.close()
            raise
        self._processes[record.id] = process
        self._logs[record.id] = log
        record.pid = process.pid
        record.pid_start = process_start_ticks(process.pid)
        record.started_at = utc_now()
        record.state = "running"
        return record

    def poll(self, record: JobRecord) -> JobRecord:
        if record.state != "running":
            return record
        process = self._processes.get(record.id)
        if process is not None:
            code = process.poll()
            # Read the progress file after the exit check: the last report lands just before the exit.
            _apply_progress(record)
            if code is None:
                return record
            self._release(record.id)
            if record.cancel_requested:
                return _finish(record, "cancelled")
            return self._finish_with_code(record, code)
        if process_matches(record):
            _apply_progress(record)
            return record
        return _finish_gone(record, "process exited without finishing (exit code unknown)")

    def cancel(self, record: JobRecord) -> JobRecord:
        """Send SIGTERM and return; ``poll`` reaps the job.  A job that already exited keeps its real outcome."""

        if record.state != "running":
            return record
        process = self._processes.get(record.id)
        if process is not None and process.poll() is not None:
            return self.poll(record)
        if process is None and not process_matches(record):
            return self.poll(record)
        record.cancel_requested = True
        _kill_pid(record.pid, signal.SIGTERM)
        return record

    def kill(self, record: JobRecord) -> None:
        """SIGKILL a job that ignored SIGTERM (only when the process is still the job's)."""

        if record.state != "running":
            return
        if record.id in self._processes or process_matches(record):
            _kill_pid(record.pid, signal.SIGKILL)

    def _finish_with_code(self, record: JobRecord, code: int) -> JobRecord:
        if code == 0:
            return _finish(record, "done")
        tail = log_tail(record.log_path, ERROR_TAIL_LINES)
        return _finish(record, "failed", tail or f"exit code {code}")

    def _release(self, job_id: str) -> None:
        self._processes.pop(job_id, None)
        log = self._logs.pop(job_id, None)
        if log is not None:
            log.close()


def _kill_pid(pid: int | None, sig: int) -> None:
    """Signal a job's whole process group (it was started in its own session)."""

    if pid is None:
        return
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


# --------------------------------------------------------------------------- runner


class JobRunner:
    """The queue: persists records under ``root/jobs``, hands GPUs out, ticks.

    ``tick`` polls running jobs, then starts queued jobs first in, first out
    while a GPU is free, fewer than ``max_concurrent`` run, and no running
    job targets the same workspace.  ``start`` runs ``tick`` from a
    background thread once a second; ``recover`` finishes jobs left running
    by a dead server from their progress files.
    """

    def __init__(self, root: Path, backend: JobBackend, max_concurrent: int | None = None) -> None:
        self.root = Path(root)
        self.backend = backend
        self.jobs_dir = self.root / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.gpus: list[int] | None = list(getattr(backend, "gpus", None) or []) or None
        if max_concurrent is None:
            max_concurrent = len(self.gpus) if self.gpus else 1
        self.max_concurrent = max(1, int(max_concurrent))
        self._lock = threading.RLock()
        self._records: dict[str, JobRecord] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._load()

    # ----- persistence

    def _load(self) -> None:
        for path in sorted(self.jobs_dir.glob("j*.json")):
            data = _read_json(path)
            if isinstance(data, dict) and "id" in data:
                self._records[data["id"]] = JobRecord.from_dict(data)

    def _record_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _save(self, record: JobRecord) -> None:
        self._records[record.id] = record
        _write_json_atomic(self._record_path(record.id), record.to_dict())

    def _next_id(self) -> str:
        counter = self.jobs_dir / "counter"
        current = 0
        text = counter.read_text().strip() if counter.exists() else ""
        if text.isdigit():
            current = int(text)
        current += 1
        tmp = counter.with_name("counter.tmp")
        tmp.write_text(f"{current}\n")
        os.replace(tmp, counter)
        return f"j{current:08x}"

    # ----- public api

    def submit(self, spec: JobSpec, command: list[str]) -> JobRecord:
        with self._lock:
            job_id = self._next_id()
            record = JobRecord(
                id=job_id,
                spec=spec,
                command=[str(part) for part in command],
                log_path=str(self.jobs_dir / f"{job_id}.log"),
                progress_path=str(self.jobs_dir / f"{job_id}.progress.json"),
            )
            self._save(record)
            return record

    def tick(self) -> None:
        with self._lock:
            self._poll_running()
            self._start_queued()

    def list(self, state: str | None = None) -> list[JobRecord]:
        with self._lock:
            records = [r for r in self._records.values() if state is None or r.state == state]
            return sorted(records, key=lambda r: r.id, reverse=True)

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._records:
                raise KeyError(job_id)
            return self._records[job_id]

    def cancel(self, job_id: str) -> JobRecord:
        """Stop a job.  Returns once it has finished (its real outcome when it exited on its own)."""

        with self._lock:
            record = self.get(job_id)
            if record.finished:
                return record
            if record.state != "running":
                self._save(_finish(record, "cancelled"))
                return record
            record = self.backend.cancel(record)
            self._save(record)
            if record.finished:
                return record
        # The wait happens outside the lock so the tick thread and the API keep going.
        grace = float(getattr(self.backend, "terminate_grace", 5.0))
        if not self._wait_finished(job_id, grace):
            kill = getattr(self.backend, "kill", None)
            if kill is not None:
                with self._lock:
                    kill(self.get(job_id))
                self._wait_finished(job_id, grace)
        with self._lock:
            record = self.get(job_id)
            if not record.finished:
                self._save(_finish(record, "cancelled"))
            return record

    def _wait_finished(self, job_id: str, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if self._poll_record(self.get(job_id)).finished:
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(CANCEL_POLL_SECONDS)

    def log(self, job_id: str, tail: int | None = None) -> str:
        record = self.get(job_id)
        return log_tail(record.log_path, tail)

    def recover(self) -> None:
        """After a restart: a running job whose process is gone finished with the server (done at progress 1, else failed)."""

        with self._lock:
            for record in list(self._records.values()):
                if record.state == "running" and not process_matches(record):
                    self._save(_finish_gone(record, "server restarted"))

    def start(self, interval: float = 1.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(float(interval),), name="job-runner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    # ----- scheduling

    def _loop(self, interval: float) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - the loop must survive a bad tick
                print(f"job runner tick failed: {exc!r}", flush=True)
            self._stop.wait(interval)

    def _running(self) -> list[JobRecord]:
        return [r for r in self._records.values() if r.state == "running"]

    def _poll_record(self, record: JobRecord) -> JobRecord:
        before = (record.state, record.progress, record.message, record.result)
        updated = self.backend.poll(record)
        if (updated.state, updated.progress, updated.message, updated.result) != before or updated is not record:
            self._save(updated)
        return updated

    def _poll_running(self) -> None:
        for record in self._running():
            self._poll_record(record)

    def _free_gpus(self) -> list[int]:
        if self.gpus is None:
            return []
        busy = {r.gpu for r in self._running() if r.gpu is not None}
        return [gpu for gpu in self.gpus if gpu not in busy]

    def _busy_workspaces(self) -> set[str]:
        return {r.spec.workspace for r in self._running() if r.spec.workspace}

    def _start_queued(self) -> None:
        queued = sorted((r for r in self._records.values() if r.state == "queued"), key=lambda r: r.id)
        free = self._free_gpus()
        busy = self._busy_workspaces()
        for record in queued:
            if len(self._running()) >= self.max_concurrent:
                break
            if self.gpus is not None and not free:
                break
            # One job per workspace at a time; later jobs on it wait their turn.
            if record.spec.workspace and record.spec.workspace in busy:
                continue
            record.gpu = free.pop(0) if self.gpus is not None else None
            self._start(record)
            if record.state == "running" and record.spec.workspace:
                busy.add(record.spec.workspace)

    def _start(self, record: JobRecord) -> None:
        env = self._job_env(record)
        try:
            started = self.backend.start(record, record.command, env)
        except Exception as exc:  # noqa: BLE001 - a job that cannot start fails; it must not stop the queue
            started = _finish(record, "failed", f"could not start: {exc!r}")
        self._save(started)

    def _job_env(self, record: JobRecord) -> dict[str, str]:
        env = dict(os.environ)
        env[PROGRESS_FILE_ENV] = record.progress_path
        env[JOB_ID_ENV] = record.id
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env
