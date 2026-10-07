"""Where the two libraries live, and how items in them are named.

The lab library is a shared directory the developer publishes into
(``scripts/publish.py``); the app only reads it.  The personal library is
the user's own and holds everything the app creates.  Both are looked up by
host name, as the lab's other code finds its data, and the app's
``--lab-library`` / ``--library`` override them.

An item is named by a reference string, ``lab:<id>`` or ``mine:<id>``; the
scope says which library holds it.  Ids are plain names (letters, digits,
``-`` and ``_``), so they are safe as directory and file names.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import socket
import tempfile
from typing import Any, Iterator

import numpy as np


LAB_ROOT_FLV = Path("/store1/shared/worm-pose-models")
LAB_LIBRARY_BY_HOST: dict[str, Path] = {
    "flv-c2": LAB_ROOT_FLV,
    "flv-c3": LAB_ROOT_FLV,
    "flv-c4": LAB_ROOT_FLV,
}
# Personal libraries on the flv machines go on the large local-network store,
# not the size-limited home directory; ``{user}`` is the login name.
PERSONAL_LIBRARY_BY_HOST: dict[str, str] = {
    "flv-c2": "/temp_data4/{user}/worm-pose-library",
    "flv-c3": "/temp_data4/{user}/worm-pose-library",
    "flv-c4": "/temp_data4/{user}/worm-pose-library",
}
PERSONAL_LIBRARY_ELSEWHERE = "~/worm-pose-library"

SCOPES = ("lab", "mine")
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def short_hostname() -> str:
    return socket.gethostname().split(".")[0]


def default_lab_root(host: str | None = None) -> Path | None:
    """The lab library of this host, or ``None`` where the lab has none."""

    return LAB_LIBRARY_BY_HOST.get(host or short_hostname())


def default_personal_root(host: str | None = None, user: str | None = None) -> Path:
    template = PERSONAL_LIBRARY_BY_HOST.get(host or short_hostname(), PERSONAL_LIBRARY_ELSEWHERE)
    return Path(template.format(user=user or getpass.getuser())).expanduser()


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

        return cls(Path(lab) if lab is not None else default_lab_root(), Path(personal) if personal is not None else default_personal_root())

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
    """Write JSON atomically: a unique temporary in the same directory replaces ``path``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", suffix=".partial", delete=False) as handle:
        json.dump(value, handle, indent=1, sort_keys=True, default=_plain)
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
