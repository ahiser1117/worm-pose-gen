"""Setups (microscopes), their default models, and which setup a recording belongs to.

``setups/<id>.json`` holds what the pipeline needs to read and normalize a
microscope's videos and which models to use on them::

    {"name": ..., "description": ...,
     "video": {"dataset_path": "/img_nir", "flat_field": true},
     "pixel_size_um": 1.0, "fps": 20.0,
     "recording_roots": ["/store1/..."],
     "defaults": {"mask": "lab:nir-hand284", "body": "lab:nir-body-lags3"}}

``defaults`` is keyed by role: ``mask`` is the model whose mask the pipeline
uses, ``body`` the one whose body fields (A-P, head, tail) it uses.  A user
overrides the defaults of a lab setup in the personal
``setups/<id>.override.json`` (``{"defaults": {...}}``); every change of a
default is appended to the library's ``setups/defaults_log.jsonl`` with who
made it, when, and why, as the promotion log did.

A recording belongs to the setup it is registered to in the personal
``recordings.json`` (recordings added by hand), else to the setup whose
``recording_roots`` contain it.  A recording's id is its file name without
the extension (``2023-06-23-01``): the lab's acquisition names files by date
and index, so the stem is unique within a setup and survives the file being
mirrored to another store.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import getpass
from pathlib import Path
from typing import Any

from .roots import (
    Libraries, append_jsonl, check_id, locked, make_ref, parse_ref, read_json, read_jsonl, write_json,
)
from ..workspace import utc_now


SETUPS_DIR = "setups"
DEFAULTS_LOG = "defaults_log.jsonl"
RECORDINGS_FILE = "recordings.json"
# What a model must output to fill each default role.
ROLE_OUTPUTS = {"mask": ("mask",), "body": ("ap", "head", "tail")}
DEFAULT_VIDEO = {"dataset_path": "/img_nir", "flat_field": True}


@dataclass(frozen=True)
class Setup:
    ref: str
    name: str
    description: str = ""
    video: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_VIDEO))
    pixel_size_um: float | None = None
    fps: float | None = None
    recording_roots: tuple[str, ...] = ()
    # The effective defaults: the setup's own, overridden by the personal override of a lab setup.
    defaults: dict[str, str] = field(default_factory=dict)
    overridden: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "recording_roots": list(self.recording_roots), "overridden": list(self.overridden)}


def setup_path(root: Path, setup_id: str) -> Path:
    return Path(root) / SETUPS_DIR / f"{check_id(setup_id)}.json"


def override_path(libraries: Libraries, setup_id: str) -> Path:
    return libraries.personal / SETUPS_DIR / f"{check_id(setup_id)}.override.json"


def write_setup(
    root: Path, setup_id: str, *, name: str, description: str = "", video: dict[str, Any] | None = None,
    pixel_size_um: float | None = None, fps: float | None = None, recording_roots: list[str] | tuple[str, ...] = (),
    defaults: dict[str, str] | None = None,
) -> None:
    """Write a setup file into a library root (the personal one from the app; the lab one only from scripts)."""

    video = {**DEFAULT_VIDEO, **(video or {})}
    for value, label in ((pixel_size_um, "pixel_size_um"), (fps, "fps")):
        if value is not None and not float(value) > 0:
            raise ValueError(f"{label} must be positive")
    for role in defaults or {}:
        if role not in ROLE_OUTPUTS:
            raise ValueError(f"unknown default role {role!r}; expected one of {tuple(ROLE_OUTPUTS)}")
    write_json(setup_path(root, setup_id), {
        "name": str(name), "description": str(description), "video": video,
        "pixel_size_um": None if pixel_size_um is None else float(pixel_size_um),
        "fps": None if fps is None else float(fps),
        "recording_roots": [str(p) for p in recording_roots], "defaults": dict(defaults or {}),
    })


def create_setup(libraries: Libraries, setup_id: str, **fields: Any) -> Setup:
    """A new personal setup; an existing id is refused."""

    ref = make_ref("mine", setup_id)
    path = setup_path(libraries.personal, setup_id)
    if path.exists():
        raise FileExistsError(f"setup {ref} already exists")
    write_setup(libraries.personal, setup_id, **fields)
    return get_setup(libraries, ref)


def get_setup(libraries: Libraries, ref: str) -> Setup:
    scope, setup_id = parse_ref(ref)
    data = read_json(setup_path(libraries.root(scope), setup_id))
    if data is None:
        raise LookupError(f"unknown setup {ref}")
    defaults = dict(data.get("defaults") or {})
    overridden: tuple[str, ...] = ()
    if scope == "lab":
        override = (read_json(override_path(libraries, setup_id)) or {}).get("defaults") or {}
        defaults.update(override)
        overridden = tuple(sorted(override))
    return Setup(
        ref=ref, name=str(data.get("name") or setup_id), description=str(data.get("description") or ""),
        video={**DEFAULT_VIDEO, **(data.get("video") or {})}, pixel_size_um=data.get("pixel_size_um"),
        fps=data.get("fps"), recording_roots=tuple(data.get("recording_roots") or ()), defaults=defaults,
        overridden=overridden,
    )


def list_setups(libraries: Libraries) -> list[Setup]:
    setups = []
    for scope in libraries.scopes():
        directory = libraries.root(scope) / SETUPS_DIR
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else ():
            if path.name.endswith(".override.json"):
                continue
            setups.append(get_setup(libraries, make_ref(scope, path.stem)))
    return setups


def set_default(libraries: Libraries, setup_ref: str, role: str, model_ref: str, *, reason: str, who: str | None = None) -> Setup:
    """Make ``model_ref`` the setup's default for ``role``, logging who and why.

    A personal setup changes in place; a lab setup gets (or updates) its
    personal override.  The model must exist and give the role's outputs.
    """

    from .models import get_card

    if role not in ROLE_OUTPUTS:
        raise ValueError(f"unknown default role {role!r}; expected one of {tuple(ROLE_OUTPUTS)}")
    if not str(reason or "").strip():
        raise ValueError("a default change needs a reason")
    setup = get_setup(libraries, setup_ref)
    card = get_card(libraries, model_ref)
    missing = [name for name in ROLE_OUTPUTS[role] if name not in card.outputs]
    if missing:
        raise ValueError(f"{model_ref} cannot be the {role} default: it has no {', '.join(missing)} output")
    scope, setup_id = parse_ref(setup_ref)
    with locked(libraries.personal / SETUPS_DIR):
        if scope == "mine":
            path = setup_path(libraries.personal, setup_id)
            data = read_json(path)
            data["defaults"] = {**(data.get("defaults") or {}), role: model_ref}
        else:
            path = override_path(libraries, setup_id)
            data = read_json(path) or {}
            data["defaults"] = {**(data.get("defaults") or {}), role: model_ref}
        write_json(path, data)
        log_default(libraries.personal, setup_ref, role, model_ref, previous=setup.defaults.get(role), reason=reason, who=who)
    return get_setup(libraries, setup_ref)


def log_default(root: Path, setup_ref: str, role: str, model_ref: str, *, previous: str | None, reason: str, who: str | None = None) -> None:
    append_jsonl(Path(root) / SETUPS_DIR / DEFAULTS_LOG, {
        "setup": setup_ref, "role": role, "model": model_ref, "previous": previous,
        "who": who or getpass.getuser(), "at": utc_now(), "reason": str(reason).strip(),
    })


def defaults_log(libraries: Libraries, setup_ref: str) -> list[dict[str, Any]]:
    """Every default change of the setup, from the lab's log and the personal one, oldest first."""

    entries = []
    for scope in libraries.scopes():
        entries.extend(e for e in read_jsonl(libraries.root(scope) / SETUPS_DIR / DEFAULTS_LOG) if e.get("setup") == setup_ref)
    return sorted(entries, key=lambda e: str(e.get("at")))


# --------------------------------------------------------------------------- recordings


def recording_id(path: str | Path) -> str:
    """A recording's id: its file name without the extension."""

    name = Path(path).stem
    if not name or "/" in name:
        raise ValueError(f"no recording id for {path!r}")
    return name


def registered_recordings(libraries: Libraries) -> dict[str, dict[str, Any]]:
    """The personal ``recordings.json``: absolute path -> ``{"setup": ref, "registered_at": ...}``."""

    return read_json(libraries.personal / RECORDINGS_FILE, {}) or {}


def register_recording(libraries: Libraries, path: str | Path, setup_ref: str) -> dict[str, Any]:
    """Assign a recording outside every setup's roots (or move it to another setup) in the personal library."""

    get_setup(libraries, setup_ref)
    resolved = str(Path(path).expanduser().resolve())
    if not Path(resolved).is_file():
        raise FileNotFoundError(f"{resolved} is not a file")
    with locked(libraries.personal):
        registry = registered_recordings(libraries)
        registry[resolved] = {"setup": setup_ref, "registered_at": utc_now()}
        write_json(libraries.personal / RECORDINGS_FILE, registry)
    return {"id": recording_id(resolved), "path": resolved, "setup": setup_ref}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def setup_for_recording(libraries: Libraries, path: str | Path) -> str | None:
    """The setup a recording belongs to: its registration, else the setup whose root contains it."""

    resolved = Path(path).expanduser().resolve()
    entry = registered_recordings(libraries).get(str(resolved))
    if entry is not None:
        return str(entry["setup"])
    for setup in list_setups(libraries):
        for root in setup.recording_roots:
            if _inside(resolved, Path(root).expanduser().resolve()):
                return setup.ref
    return None


def recording_sources(libraries: Libraries, setup_ref: str) -> list[Path]:
    """The setup's recording roots plus the recording files registered to it (``recordings.find_recordings`` takes both)."""

    setup = get_setup(libraries, setup_ref)
    registered = [Path(p) for p, entry in registered_recordings(libraries).items() if entry.get("setup") == setup_ref]
    return [Path(root) for root in setup.recording_roots] + registered
