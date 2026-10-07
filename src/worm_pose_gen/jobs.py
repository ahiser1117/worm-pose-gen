"""Job runner for long computations, on this machine's GPUs or through SLURM.

Anything longer than a frame runs as a job: a process started by a backend,
reporting progress through a small JSON file, surviving a server restart and
cancellable.  Every job is a file ``jobs/<id>.json`` under the runner's root
so the state is inspectable on disk.  A job's ``run_on`` picks its backend:

- ``local`` (:class:`LocalGPUBackend`): a subprocess on this machine, one GPU
  of the runner's pool each, at most ``max_concurrent`` at once.
- ``slurm`` (:class:`SlurmBackend`): a batch script that runs the same
  command, submitted with ``sbatch``; SLURM queues it, so it takes no local
  GPU and does not count against ``max_concurrent``.

A backend only checks, starts, polls and cancels a record; the runner owns
the queue, the GPU pool and the files.

A job process reports progress by calling :func:`report_progress`, which
rewrites the file named by the environment variable
``WORM_POSE_PROGRESS_FILE`` atomically; the backend reads it on every poll
(for a SLURM job the file is on shared storage the compute node writes to).
Job ids come from a counter file under the root, so ids are never reused
across restarts.

Jobs on one workspace run one at a time, first in first out, whichever
backend runs them: every stage rewrites the workspace's arrays whole, so two
at once would overwrite each other.  A local job inherited from a previous
server process is identified by its pid and the process start time, so a pid
reused after a reboot is not mistaken for the job; a SLURM job by its SLURM
job id, which the scheduler still answers for after the restart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import threading
import time
from typing import Any, Iterator, Protocol

from .compute import CONSERVATIVE_SLURM_DEFAULTS, SlurmDefaults, node_local


STATES = ("queued", "running", "done", "failed", "cancelled")
FINISHED_STATES = ("done", "failed", "cancelled")
RUN_ON = ("local", "slurm")

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
    # Physical GPU requested by the user; None lets the queue choose (local jobs only).
    gpu: int | None = None
    label: str = ""
    # Which backend runs the job: "local" or "slurm".
    run_on: str = "local"
    # A SLURM job's {"partition", "time"}; the backend fills in its defaults at submit.
    slurm: dict[str, Any] | None = None

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
    # A SLURM job: its id, and its last state there (PENDING, RUNNING, COMPLETED, ...).
    slurm_job_id: str | None = None
    slurm_state: str | None = None

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
    """Checks, starts, polls and cancels one job; the runner owns the queue and the files.

    ``prepare`` runs at submit, before the record is saved: it raises
    ``ValueError`` for a job this backend cannot run and fills in the
    backend's defaults on the spec.  ``cancel`` asks the job to stop and
    returns at once; a later ``poll`` reaps it as ``cancelled``.  A job whose
    process already exited is finished with its real outcome instead.
    """

    def prepare(self, record: JobRecord) -> None: ...

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

    def prepare(self, record: JobRecord) -> None:
        """A requested GPU must be one of the pool's; SLURM settings do not apply here."""

        spec = record.spec
        if spec.slurm is not None:
            raise ValueError("slurm settings apply only to a job with run_on 'slurm'")
        if spec.gpu is None:
            return
        if isinstance(spec.gpu, bool) or not isinstance(spec.gpu, int) or spec.gpu < 0:
            raise ValueError("gpu must be a non-negative integer or null for automatic selection")
        if spec.gpus == 0:
            raise ValueError("a CPU-only job cannot request a GPU")
        if spec.gpu not in self.gpus:
            raise ValueError(f"GPU {spec.gpu} is not enabled for jobs; available GPUs: {self.gpus}")

    def start(self, record: JobRecord, command: list[str], env: dict[str, str]) -> JobRecord:
        job_env = dict(env)
        if record.gpu is not None:
            job_env["CUDA_VISIBLE_DEVICES"] = str(record.gpu)
        elif record.spec.gpus == 0:
            job_env["CUDA_VISIBLE_DEVICES"] = ""
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


# A time limit as sbatch takes it: minutes, MM:SS, HH:MM:SS, D-HH, D-HH:MM or D-HH:MM:SS.
SLURM_TIME = re.compile(r"^(\d+-)?\d+(:\d{1,2}){0,2}$")
# A partition name, or a comma-separated list of them (SLURM runs the job on the first that can).
SLURM_PARTITION = re.compile(r"^[A-Za-z0-9_.-]+(,[A-Za-z0-9_.-]+)*$")
SLURM_SETTINGS = ("partition", "time")
# Job states after which SLURM will not run the job again; any other state is still in the queue or running.
SLURM_FINISHED_STATES = frozenset({
    "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED",
    "SPECIAL_EXIT",
})
SLURM_COMMAND_TIMEOUT = 30.0
# What a SLURM job's environment must not inherit from the app's: the app's own allocation (in an
# interactive session) and its GPU assignment; SLURM sets both for the new job.
SLURM_INHERITED_SKIP = ("SLURM_", "CUDA_VISIBLE_DEVICES")


def _slurm_command(argv: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess | None:
    """Run one SLURM command; None when it could not run or did not answer in time."""

    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=SLURM_COMMAND_TIMEOUT, env=env, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None


def _last_fields(output: str) -> list[str] | None:
    """The ``|``-separated fields of the last non-empty line of a SLURM listing, or None when it is empty."""

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1].split("|") if lines else None


def _command_paths(command: list[str]) -> Iterator[str]:
    """The absolute paths a command names: its arguments, and the strings inside a JSON-object argument (a stage's ``--params``)."""

    def walk(value: Any) -> Iterator[str]:
        if isinstance(value, str) and value.startswith("/"):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from walk(item)
        elif isinstance(value, list):
            for item in value:
                yield from walk(item)

    for part in command:
        if part.startswith("/"):
            yield part
        elif part.startswith("{"):
            try:
                yield from walk(json.loads(part))
            except ValueError:
                continue


class SlurmBackend:
    """Runs jobs through SLURM: a batch script per job, submitted with ``sbatch``.

    The script ``jobs/<id>.sbatch`` holds the SLURM request as ``#SBATCH``
    lines (partition and time from the spec, the GPU request, CPUs and
    memory from the host's defaults; output appended to the record's log
    file) and runs the job's command from the repository root with the
    progress variables exported.  ``sbatch --parsable`` gives the SLURM job
    id, which the record keeps; ``poll`` asks ``squeue`` for the state and,
    once the job has left the queue, ``sacct`` for the outcome and exit
    code.  Nothing lives in memory but a query throttle, so a fresh backend
    after a restart picks up where the old one stopped.

    The scheduler is asked about a job at most once per ``poll_seconds``;
    progress comes from the progress file on every poll.  ``prepare``
    refuses a job whose log, progress file, working directory or any path
    in its command is on node-local storage (``compute.node_local``).
    """

    def __init__(
        self, defaults: SlurmDefaults = CONSERVATIVE_SLURM_DEFAULTS, cwd: Path | None = None, poll_seconds: float = 5.0,
        terminate_grace: float = 15.0,
    ) -> None:
        self.defaults = defaults
        self.cwd = Path(cwd) if cwd is not None else REPO_ROOT
        self.poll_seconds = float(poll_seconds)
        self.terminate_grace = float(terminate_grace)
        self._queried: dict[str, float] = {}

    def settings(self, requested: dict[str, Any] | None) -> dict[str, Any]:
        """``{"partition", "time"}`` of a job: the request over the host's defaults, validated."""

        requested = dict(requested or {})
        unknown = sorted(set(requested) - set(SLURM_SETTINGS))
        if unknown:
            raise ValueError(f"unknown slurm settings {unknown}; expected {list(SLURM_SETTINGS)}")
        partition = requested.get("partition") or self.defaults.partition
        limit = requested.get("time") or self.defaults.time
        if partition is not None and (not isinstance(partition, str) or not SLURM_PARTITION.match(partition)):
            raise ValueError(f"invalid SLURM partition {partition!r}")
        if not isinstance(limit, str) or not SLURM_TIME.match(limit):
            raise ValueError(f"invalid SLURM time limit {limit!r}; use minutes, HH:MM:SS or D-HH:MM:SS")
        return {"partition": partition, "time": limit}

    def prepare(self, record: JobRecord) -> None:
        spec = record.spec
        if spec.gpu is not None:
            raise ValueError("a SLURM job cannot ask for a GPU of this machine; SLURM assigns one (leave gpu null)")
        spec.slurm = self.settings(spec.slurm)
        named = [("job log", record.log_path), ("progress file", record.progress_path), ("working directory", str(self.cwd))]
        named += [("command path", path) for path in _command_paths(record.command)]
        local = [f"{what} {path} (under {root})" for what, path in named if path and (root := node_local(path)) is not None]
        if local:
            raise ValueError(
                "cannot run on SLURM: a compute node cannot read node-local storage: " + "; ".join(local)
                + ". Keep workspaces, job records and models on shared storage."
            )

    def script_path(self, record: JobRecord) -> Path:
        return Path(record.log_path).with_suffix(".sbatch")

    def batch_script(self, record: JobRecord, command: list[str]) -> str:
        settings = self.settings(record.spec.slurm)
        directives = [
            f"--job-name=worm-pose-{record.id}",
            f"--output={shlex.quote(record.log_path)}",
            "--open-mode=append",
            f"--chdir={shlex.quote(str(self.cwd))}",
            f"--time={settings['time']}",
        ]
        if settings["partition"]:
            directives.append(f"--partition={settings['partition']}")
        if record.spec.gpus > 0:
            directives.append(f"--gres={self.defaults.gres}")
        directives += [f"--cpus-per-task={self.defaults.cpus_per_task}", f"--mem={self.defaults.mem}"]
        lines = ["#!/bin/bash", f"# worm-pose job {record.id}: {record.spec.label or record.spec.kind}"]
        lines += [f"#SBATCH {directive}" for directive in directives]
        lines += [
            "",
            f'echo "worm-pose job {record.id}: SLURM job $SLURM_JOB_ID on $(hostname)"',
            f"export {PROGRESS_FILE_ENV}={shlex.quote(record.progress_path)}",
            f"export {JOB_ID_ENV}={shlex.quote(record.id)}",
            "export PYTHONUNBUFFERED=1",
        ]
        if record.spec.gpus == 0:
            lines.append("export CUDA_VISIBLE_DEVICES=")
        lines.append("exec " + shlex.join(command))
        return "\n".join(lines) + "\n"

    def start(self, record: JobRecord, command: list[str], env: dict[str, str]) -> JobRecord:
        script = self.script_path(record)
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(self.batch_script(record, command))
        submit_env = {key: value for key, value in env.items() if not key.startswith(SLURM_INHERITED_SKIP)}
        result = _slurm_command(["sbatch", "--parsable", str(script)], env=submit_env)
        if result is None:
            raise RuntimeError("sbatch did not answer")
        job_id = result.stdout.strip().split(";")[0]
        if result.returncode != 0 or not job_id:
            raise RuntimeError(f"sbatch failed: {(result.stderr or result.stdout).strip()}")
        record.slurm_job_id = job_id
        record.slurm_state = "PENDING"
        record.message = f"submitted to SLURM as job {job_id}"
        record.started_at = utc_now()
        record.state = "running"
        return record

    def query(self, job_id: str) -> tuple[str, str] | None:
        """``(state, detail)`` of a SLURM job, or None when the scheduler did not answer.

        ``detail`` is the reason of a pending job and the exit code of a
        finished one.  The state is ``UNKNOWN`` when neither ``squeue`` nor
        ``sacct`` knows the job (purged from the queue, no accounting).
        """

        queue = _slurm_command(["squeue", "--noheader", "--states=all", f"--jobs={job_id}", "--format=%T|%r"])
        if queue is None:
            return None
        if queue.returncode != 0 and "invalid job id" not in queue.stderr.lower():
            return None
        queued = _last_fields(queue.stdout) if queue.returncode == 0 else None
        if queued and queued[0] not in SLURM_FINISHED_STATES:
            return queued[0], queued[1] if len(queued) > 1 else ""
        accounting = _slurm_command(["sacct", "--noheader", "--allocations", "--parsable2", f"--jobs={job_id}", "--format=State,ExitCode"])
        answered = accounting is not None and accounting.returncode == 0
        accounted = _last_fields(accounting.stdout) if answered else None
        if accounted:
            # "CANCELLED by <uid>" -> CANCELLED
            state = accounted[0].split()[0] if accounted[0].strip() else "UNKNOWN"
            if state in SLURM_FINISHED_STATES or not queued:
                return state, accounted[1] if len(accounted) > 1 else ""
        if queued:
            # Finished in the queue, not in accounting yet.
            return queued[0], ""
        return ("UNKNOWN", "") if answered else None

    def poll(self, record: JobRecord) -> JobRecord:
        if record.state != "running":
            return record
        now = time.monotonic()
        if now - self._queried.get(record.id, -self.poll_seconds) < self.poll_seconds:
            _apply_progress(record)
            return record
        self._queried[record.id] = now
        status = self.query(str(record.slurm_job_id))
        if status is None:
            _apply_progress(record)
            return record
        state, detail = status
        if state == "UNKNOWN":
            self._queried.pop(record.id, None)
            return _finish_gone(record, f"SLURM has no record of job {record.slurm_job_id} (outcome unknown)")
        record.slurm_state = state
        if read_progress(record.progress_path) is None:
            record.message = f"waiting in the SLURM queue ({detail})" if state == "PENDING" else f"SLURM job {state.lower()}"
        # Read the progress file after the state: the last report lands just before the exit.
        _apply_progress(record)
        if state not in SLURM_FINISHED_STATES:
            return record
        self._queried.pop(record.id, None)
        if state == "COMPLETED":
            return _finish(record, "done")
        if state == "CANCELLED":
            return _finish(record, "cancelled")
        tail = log_tail(record.log_path, ERROR_TAIL_LINES)
        return _finish(record, "failed", f"SLURM job {record.slurm_job_id} ended {state} (exit code {detail or 'unknown'})" + (f"\n{tail}" if tail else ""))

    def cancel(self, record: JobRecord) -> JobRecord:
        """``scancel`` the job and return; ``poll`` reaps it.  A job that already finished keeps its real outcome."""

        if record.state != "running":
            return record
        self._queried.pop(record.id, None)
        record = self.poll(record)
        if record.finished:
            return record
        result = _slurm_command(["scancel", str(record.slurm_job_id)])
        if result is None or result.returncode != 0:
            raise RuntimeError(f"scancel {record.slurm_job_id} failed: {'' if result is None else (result.stderr or result.stdout).strip()}")
        record.cancel_requested = True
        self._queried.pop(record.id, None)
        return record


# --------------------------------------------------------------------------- runner


class JobRunner:
    """The queue: persists records under ``root/jobs``, hands GPUs out, ticks.

    ``tick`` polls running jobs, then starts queued jobs first in, first out
    while no running job targets the same workspace; a local job also waits
    for a free GPU and for fewer than ``max_concurrent`` local jobs to run,
    a SLURM job is submitted at once (SLURM queues it).  ``start`` runs
    ``tick`` from a background thread once a second; ``recover`` finishes
    local jobs left running by a dead server from their progress files
    (SLURM jobs are simply polled again).  ``slurm`` is None where SLURM is
    not available, and a SLURM job is then refused.
    """

    def __init__(
        self, root: Path, local: LocalGPUBackend, slurm: SlurmBackend | None = None, max_concurrent: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.backends: dict[str, JobBackend] = {"local": local}
        if slurm is not None:
            self.backends["slurm"] = slurm
        self.jobs_dir = self.root / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.gpus: list[int] | None = list(local.gpus) or None
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
            backend = self._backend(spec.run_on)
            job_id = self._next_id()
            record = JobRecord(
                id=job_id,
                spec=spec,
                command=[str(part) for part in command],
                log_path=str(self.jobs_dir / f"{job_id}.log"),
                progress_path=str(self.jobs_dir / f"{job_id}.progress.json"),
            )
            backend.prepare(record)
            self._save(record)
            return record

    def _backend(self, run_on: str) -> JobBackend:
        if run_on not in RUN_ON:
            raise ValueError(f"unknown run_on {run_on!r}; expected one of {RUN_ON}")
        if run_on not in self.backends:
            raise ValueError("SLURM is not available on this host (sbatch, squeue, sacct and scancel must be on PATH)")
        return self.backends[run_on]

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
            backend = self.backends.get(record.spec.run_on)
            if backend is None:
                self._save(_unavailable(record))
                return record
            record = backend.cancel(record)
            self._save(record)
            if record.finished:
                return record
        # The wait happens outside the lock so the tick thread and the API keep going.
        grace = float(getattr(backend, "terminate_grace", 5.0))
        if not self._wait_finished(job_id, grace):
            kill = getattr(backend, "kill", None)
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
        """After a restart: a local job whose process is gone finished with the server (done at progress 1, else failed).

        A SLURM job ran on without the server; the next tick asks SLURM about
        it.  A SLURM job on a host without SLURM can no longer be followed and
        fails.
        """

        with self._lock:
            for record in list(self._records.values()):
                if record.state != "running":
                    continue
                if record.spec.run_on not in self.backends:
                    self._save(_unavailable(record))
                elif record.spec.run_on == "local" and not process_matches(record):
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

    def _running(self, run_on: str | None = None) -> list[JobRecord]:
        return [r for r in self._records.values() if r.state == "running" and (run_on is None or r.spec.run_on == run_on)]

    def _poll_record(self, record: JobRecord) -> JobRecord:
        backend = self.backends.get(record.spec.run_on)
        if backend is None:
            self._save(_unavailable(record))
            return record
        before = (record.state, record.progress, record.message, record.result, record.slurm_state)
        updated = backend.poll(record)
        if (updated.state, updated.progress, updated.message, updated.result, updated.slurm_state) != before or updated is not record:
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
            local = record.spec.run_on == "local"
            if record.spec.run_on not in self.backends:
                self._save(_unavailable(record))
                continue
            needs_gpu = local and record.spec.gpus > 0 and self.gpus is not None
            requested = record.spec.gpu if local else None
            if requested is not None and (self.gpus is None or requested not in self.gpus):
                self._save(_finish(record, "failed", f"requested GPU {requested} is no longer enabled for jobs"))
                continue
            # Reserve this workspace's place even when the job has to wait for
            # a GPU, so a later job on the workspace cannot overtake it.
            if record.spec.workspace and record.spec.workspace in busy:
                continue
            if record.spec.workspace:
                busy.add(record.spec.workspace)
            if local and len(self._running("local")) >= self.max_concurrent:
                continue
            if needs_gpu and not free:
                continue
            if requested is not None and requested not in free:
                continue
            record.gpu = (requested if requested is not None else free[0]) if needs_gpu else None
            if record.gpu is not None:
                free.remove(record.gpu)
            self._start(record)

    def _start(self, record: JobRecord) -> None:
        env = self._job_env(record)
        try:
            started = self.backends[record.spec.run_on].start(record, record.command, env)
        except Exception as exc:  # noqa: BLE001 - a job that cannot start fails; it must not stop the queue
            started = _finish(record, "failed", f"could not start: {exc!r}")
        self._save(started)

    def _job_env(self, record: JobRecord) -> dict[str, str]:
        env = dict(os.environ)
        env[PROGRESS_FILE_ENV] = record.progress_path
        env[JOB_ID_ENV] = record.id
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env


def _unavailable(record: JobRecord) -> JobRecord:
    """Fail a SLURM job on a server without SLURM (the job records were made on another host)."""

    error = "SLURM is not available on this host"
    if record.slurm_job_id:
        error += f"; the job's outcome is unknown here (SLURM job {record.slurm_job_id})"
    return _finish(record, "failed", error)
