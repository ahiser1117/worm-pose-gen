"""Where jobs can run: the GPUs of this machine and SLURM.

The app runs on an flv-c machine (local GPUs, no SLURM) or in an
interactive session on MIT's Engaging cluster (the session's GPUs, and
SLURM for everything bigger).  :func:`detect_compute` looks once, at
startup: the GPUs torch sees on this node, and whether the SLURM commands
the job runner needs (``sbatch``, ``squeue``, ``sacct``, ``scancel``) are
on the ``PATH``, with the partitions ``sinfo`` lists.

A SLURM job's partition, time limit and resources come from a per-host
table, looked up by hostname as the lab's other code does
(``SLURM_DEFAULTS_BY_HOST``); a host the table does not know gets
conservative defaults, and the user can set partition and time per job.

A SLURM job runs on a compute node, so everything it reads or writes must be
on storage that node can see.  :func:`node_local` names the paths that are
not: ``/tmp``, ``/var/tmp``, ``/dev/shm``, ``/scratch`` and ``$TMPDIR``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import fnmatch
import os
from pathlib import Path
import shutil
import socket
import subprocess
from typing import Any


SLURM_COMMANDS = ("sbatch", "squeue", "sacct", "scancel")
SINFO_TIMEOUT_SECONDS = 10.0

NODE_LOCAL_ROOTS = ("/tmp", "/var/tmp", "/dev/shm", "/scratch")


@dataclass(frozen=True)
class LocalGPU:
    """One GPU of this machine; ``index`` is what a job's ``CUDA_VISIBLE_DEVICES`` is set to."""

    index: int
    name: str
    memory_gb: float


@dataclass(frozen=True)
class SlurmDefaults:
    """What a job asks SLURM for unless the user sets partition or time.

    ``partition`` None leaves the choice to the cluster's default partition;
    ``gres`` is the request of a GPU job (a CPU-only job asks for none).
    """

    partition: str | None
    time: str
    gres: str
    cpus_per_task: int
    mem: str


# Hostname glob -> defaults; the first match wins.  Engaging: the BCS
# partition the lab's fits use, without preemption (ou_bcs_low requeues
# preempted jobs, and a pipeline stage or a training run would restart from
# the beginning).  Login nodes are orcd-login*, compute nodes node<NNNN>.
SLURM_DEFAULTS_BY_HOST: tuple[tuple[str, SlurmDefaults], ...] = (
    ("orcd-*", SlurmDefaults(partition="ou_bcs_normal", time="12:00:00", gres="gpu:1", cpus_per_task=8, mem="32G")),
    ("node[0-9]*", SlurmDefaults(partition="ou_bcs_normal", time="12:00:00", gres="gpu:1", cpus_per_task=8, mem="32G")),
)

# A cluster this table does not know: its default partition, one hour, one GPU.
CONSERVATIVE_SLURM_DEFAULTS = SlurmDefaults(partition=None, time="01:00:00", gres="gpu:1", cpus_per_task=4, mem="16G")


@dataclass(frozen=True)
class SlurmPartition:
    """One partition as ``sinfo`` reports it (``gres`` merged over its nodes)."""

    name: str
    default: bool
    available: bool
    time_limit: str
    gres: tuple[str, ...]


@dataclass(frozen=True)
class SlurmInfo:
    """Whether SLURM can run jobs from here; ``reason`` says why not."""

    available: bool
    reason: str | None
    partitions: tuple[SlurmPartition, ...]
    defaults: SlurmDefaults


@dataclass(frozen=True)
class Compute:
    """What :func:`detect_compute` found on this host."""

    host: str
    gpus: tuple[LocalGPU, ...]
    slurm: SlurmInfo

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hostname() -> str:
    return socket.gethostname()


def slurm_defaults(host: str | None = None) -> SlurmDefaults:
    """The SLURM defaults of ``host`` (this machine by default)."""

    name = (host or hostname()).split(".")[0]
    for pattern, defaults in SLURM_DEFAULTS_BY_HOST:
        if fnmatch.fnmatchcase(name, pattern):
            return defaults
    return CONSERVATIVE_SLURM_DEFAULTS


def local_gpus() -> tuple[LocalGPU, ...]:
    """The GPUs torch sees on this machine (none without CUDA or with ``CUDA_VISIBLE_DEVICES`` empty)."""

    try:
        import torch
    except ImportError:
        return ()
    if not torch.cuda.is_available():
        return ()
    gpus = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpus.append(LocalGPU(index=index, name=str(properties.name), memory_gb=round(properties.total_memory / 2**30, 1)))
    return tuple(gpus)


def slurm_partitions() -> tuple[SlurmPartition, ...]:
    """The partitions ``sinfo`` lists, in its order; empty when ``sinfo`` is missing or fails."""

    if shutil.which("sinfo") is None:
        return ()
    try:
        result = subprocess.run(
            ["sinfo", "--noheader", "--format=%P|%a|%l|%G"], capture_output=True, text=True, timeout=SINFO_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if result.returncode != 0:
        return ()
    # sinfo prints one line per distinct node configuration, so a partition can repeat.
    merged: dict[str, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) != 4 or not fields[0]:
            continue
        name, avail, limit, gres = fields
        entry = merged.setdefault(name.rstrip("*"), {
            "default": name.endswith("*"), "available": avail == "up", "time_limit": limit, "gres": [],
        })
        for item in gres.split(","):
            if item and item != "(null)" and item not in entry["gres"]:
                entry["gres"].append(item)
    return tuple(
        SlurmPartition(name=name, default=entry["default"], available=entry["available"], time_limit=entry["time_limit"], gres=tuple(entry["gres"]))
        for name, entry in merged.items()
    )


def detect_slurm(host: str | None = None) -> SlurmInfo:
    defaults = slurm_defaults(host)
    missing = [command for command in SLURM_COMMANDS if shutil.which(command) is None]
    if missing:
        return SlurmInfo(available=False, reason=f"{', '.join(missing)} not on PATH", partitions=(), defaults=defaults)
    return SlurmInfo(available=True, reason=None, partitions=slurm_partitions(), defaults=defaults)


def detect_compute() -> Compute:
    """The local GPUs and SLURM, as found now; the app calls it once at startup."""

    host = hostname()
    return Compute(host=host, gpus=local_gpus(), slurm=detect_slurm(host))


def node_local(path: str | Path) -> str | None:
    """The node-local root ``path`` lies under (a compute node cannot read it), or None."""

    roots = list(NODE_LOCAL_ROOTS)
    if os.environ.get("TMPDIR"):
        roots.append(os.environ["TMPDIR"])
    resolved = Path(path).resolve()
    for root in roots:
        for candidate in {Path(root), Path(root).resolve()}:
            if resolved == candidate or candidate in resolved.parents:
                return str(root)
    return None
