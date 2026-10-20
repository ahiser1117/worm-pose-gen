"""Where the two libraries live, and how items in them are named.

The lab library is a shared directory the developer publishes into
(``scripts/publish.py``); the app only reads it.  It is looked up by host
name, as the lab's other code finds its data: the flv-c machines share one
on store1, and Engaging (login nodes ``orcd-*``, compute nodes ``node<NNNN>``)
has a copy in the lab's project storage, kept in step by
``scripts/sync_lab_library.sh``.  The personal library is the user's own and
holds everything the app creates; it is ``library/`` in the data directory
(:mod:`worm_pose_gen.settings`).  The app's ``--lab-library`` / ``--library``
override either.

An item is named by a reference string, ``lab:<id>`` or ``mine:<id>``; the
scope says which library holds it.  Ids are plain names (letters, digits,
``-`` and ``_``), so they are safe as directory and file names.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import fnmatch
import json
import os
from pathlib import Path
import re
import socket
import tempfile
from typing import Any, Iterator

import numpy as np

from ..settings import LIBRARY_DIR


# Hostname glob -> the lab library; the first match wins.  On flv the NFS
# path itself: flv-c2 has no /store1 symlink.
LAB_LIBRARY_FLV = Path("/storage/fs/store1/shared/worm-pose-models")
LAB_LIBRARY_ENGAGING = Path("/orcd/data/flavell/001/shared/worm-pose-models")
LAB_LIBRARY_BY_HOST: tuple[tuple[str, Path], ...] = (
    ("flv-c*", LAB_LIBRARY_FLV),
    ("orcd-*", LAB_LIBRARY_ENGAGING),
    ("node[0-9]*", LAB_LIBRARY_ENGAGING),
)

# The process's umask, read once at import (reading it means setting it, which is not thread-safe later).
UMASK = os.umask(0)
os.umask(UMASK)

SCOPES = ("lab", "mine")
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def short_hostname() -> str:
    return socket.gethostname().split(".")[0]


def default_lab_root(host: str | None = None) -> Path | None:
    """The lab library of this host, or ``None`` where the lab has none."""

    name = host or short_hostname()
    return next((root for pattern, root in LAB_LIBRARY_BY_HOST if fnmatch.fnmatchcase(name, pattern)), None)


def check_id(item_id: str) -> str:
    if not isinstance(item_id, str) or not ID_PATTERN.fullmatch(item_id):
        raise ValueError(f"invalid library id {item_id!r}: use letters, digits, '-' and '_'")
    return item_id


def parse_ref(ref: str) -> tuple[str, str]:
    """``"lab:nir-labels"`` -> ``("lab", "nir-labels")``."""

    scope, separator, item_id = str(ref).partition(":")
    if not separator or scope not in SCOPES:
        raise ValueError(f"invalid library reference {ref!r}: expected lab:<id> or mine:<id>")
    return scope, check_id(item_id)


def make_ref(scope: str, item_id: str) -> str:
    if scope not in SCOPES:
        raise ValueError(f"unknown library scope {scope!r}")
    return f"{scope}:{check_id(item_id)}"


def ref_filename(ref: str) -> str:
    """A reference as a file name stem, ``lab.nir-v1``; ids hold no dots, so it is unambiguous."""

    scope, item_id = parse_ref(ref)
    return f"{scope}.{item_id}"


@dataclass(frozen=True)
class Libraries:
    """The lab library (``None`` where there is none) and the personal library."""

    lab: Path | None
    personal: Path

    @classmethod
    def for_host(cls, lab: Path | None = None, personal: Path | None = None) -> "Libraries":
        """The host's libraries, with either root overridden."""

        return cls(Path(lab) if lab is not None else default_lab_root(), Path(personal) if personal is not None else LIBRARY_DIR)

    def root(self, scope: str) -> Path:
        if scope == "mine":
            return self.personal
        if scope == "lab":
            if self.lab is None:
                raise LookupError("no lab library is configured on this host")
            return self.lab
        raise ValueError(f"unknown library scope {scope!r}")

    def scopes(self) -> tuple[str, ...]:
        """The scopes to list: the lab's only when its directory exists."""

        return ("lab", "mine") if self.lab is not None and self.lab.is_dir() else ("mine",)

    def writable_root(self, ref: str) -> Path:
        """The root of a personal reference; a lab reference is refused, since the app never writes there."""

        scope, _ = parse_ref(ref)
        if scope != "mine":
            raise PermissionError(f"{ref} is in the lab library, which is read-only")
        return self.personal


# --------------------------------------------------------------------------- files


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write_json(path: Path, value: Any) -> None:
    """Write JSON atomically: a unique temporary in the same directory replaces ``path``.

    The temporary is created private (0600); it gets the permissions an
    ordinary new file would, so the lab library stays readable to the lab.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", suffix=".partial", delete=False) as handle:
        json.dump(value, handle, indent=1, sort_keys=True, default=_plain)
    os.chmod(handle.name, 0o666 & ~UMASK)
    os.replace(handle.name, path)


def _plain(value: Any) -> Any:
    """NumPy scalars (a fit's flags and scores often are) as Python values."""

    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except FileNotFoundError:
        return []


@contextmanager
def locked(directory: Path) -> Iterator[None]:
    """Hold ``<directory>/.lock`` exclusively (between processes too) for a read-modify-write."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
