"""Named, immutable exports captured under the workspace writer lock."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from .. import pipeline
from ..workspace import Workspace


def export_workspace(app, name: str, label: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", label):
        raise ValueError("Export name must be 1–80 letters, digits, dots, underscores or hyphens, starting with a letter or digit")
    workspace = app.workspace(name)
    app.check_writable(name)
    with pipeline.workspace_lock(workspace, timeout=0), workspace._lock:
        app.check_writable(name)
        destination = workspace.path / "exports" / f"{label}.parquet"
        if destination.exists():
            raise ValueError("That export name already exists; choose a new name")
        snapshot = workspace.snapshot(label)
        created_destination = False
        try:
            # Export the captured workspace itself; run_export's summary update
            # must never mutate the live workspace or invalidate human review.
            for filename in ("workspace.json", "summary.json", "imported_summary.json", "human_review.json", "mask_edits.json", "recording_prior.json", "edits.jsonl"):
                source = workspace.path / filename
                if source.is_file():
                    shutil.copyfile(source, snapshot / filename)
            for directory in ("masks", "overrides"):
                source = workspace.path / directory
                if source.is_dir():
                    shutil.copytree(source, snapshot / directory)
            captured = Workspace.open(snapshot)
            result = pipeline.run_export(captured, pipeline.ExportParams(name=label), device=app.device, progress=None, job=f"export:{label}")
            frozen = snapshot / destination.name
            Path(result["path"]).replace(frozen)
            with frozen.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest = {**result, "path": str(destination), "name": label, "workspace": name, "snapshot": snapshot.name, "snapshot_path": str(snapshot), "sha256": digest, "scope": "whole_workspace"}
            (snapshot / "export.json").write_text(json.dumps(manifest, indent=2))
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output, frozen.open("rb") as source:
                created_destination = True
                shutil.copyfileobj(source, output)
            return manifest
        except Exception:
            if created_destination:
                destination.unlink(missing_ok=True)
            shutil.rmtree(snapshot)
            raise



def exported_file(workspace, snapshot: str, filename: str) -> Path:
    if snapshot in (".", "..") or Path(snapshot).name != snapshot or Path(filename).name != filename or not filename.endswith(".parquet"):
        raise ValueError("Invalid export path")
    root = workspace.snapshots_dir.resolve()
    path = (root / snapshot / filename).resolve()
    if not path.is_relative_to(root) or not path.is_file() or not (path.parent / "export.json").is_file():
        raise FileNotFoundError("Export not found")
    manifest = json.loads((path.parent / "export.json").read_text())
    if filename != manifest["name"] + ".parquet":
        raise FileNotFoundError("Export not found")
    return path
