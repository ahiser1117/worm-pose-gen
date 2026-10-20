"""Exports from the app: one per request, named automatically, written under the workspace lock.

``pipeline.run_export`` does the work (``worm_pose_gen.export_table``); this
module takes the workspace lock without waiting, so an export never mixes
the arrays of a stage that is rewriting them, and finds exports again for
listing and download.  An export is a directory ``exports/<name>/`` holding
``<name>.parquet`` and ``export.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import pipeline
from ..export_table import METADATA_FILE


def export_workspace(app, name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Export workspace ``name`` now; ``params`` are ``pipeline.ExportParams`` (pixel size, fps, setup)."""

    workspace = app.workspace(name)
    app.check_writable(name)
    with pipeline.workspace_lock(workspace, timeout=0):
        return pipeline.run_export(workspace, pipeline.ExportParams.from_dict(params), device=app.device, progress=None, job=f"export:{name}")


def list_exports(workspace) -> list[dict[str, Any]]:
    """The ``export.json`` of every export of the workspace, newest first."""

    root = Path(workspace.path) / "exports"
    found = [json.loads(path.read_text()) for path in root.glob(f"*/{METADATA_FILE}")] if root.is_dir() else []
    return sorted(found, key=lambda metadata: metadata["created_at"], reverse=True)


def exported_file(workspace, export: str, filename: str) -> Path:
    """The path of ``filename`` (the table or ``export.json``) of export ``export``; ``FileNotFoundError`` otherwise."""

    root = (Path(workspace.path) / "exports").resolve()
    directory = (root / export).resolve()
    if export.startswith(".") or directory.parent != root or not (directory / METADATA_FILE).is_file():
        raise FileNotFoundError("Export not found")
    if filename not in (f"{directory.name}.parquet", METADATA_FILE):
        raise FileNotFoundError("Export not found")
    return directory / filename
