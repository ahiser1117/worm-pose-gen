"""Pseudo-label sets: self-training items from the pipeline's confident fits, named like any other library item.

``pseudo_labels/<id>/`` holds::

    pseudo_labels.json   {"setup": ref, "name", "description", "author", "created_at"}
    index.json           the items, their selection parameters and every workspace's counts (:mod:`..pseudo_labels`)
    <workspace>/         one ``.npz`` per item, and the workspace's own ``index.json``

``scripts/build_pseudo_labels.py --set <id>`` builds a personal set from
analysed workspaces; ``scripts/publish.py pseudo-labels`` copies one into
the lab library.  A training run names the sets it mixes in by reference
(``lab:self-train-r1``), and so does its model card, so a recipe that uses
them can be repeated by anyone who can read the library.
"""

from __future__ import annotations

import getpass
from pathlib import Path
from typing import Any

from .roots import Libraries, check_id, make_ref, parse_ref, read_json, write_json
from ..workspace import utc_now


PSEUDO_DIR = "pseudo_labels"
CARD = "pseudo_labels.json"


def pseudo_dir(root: Path, set_id: str) -> Path:
    return Path(root) / PSEUDO_DIR / check_id(set_id)


class PseudoLabelSet:
    """One set as a library holds it: its card and the item directory training reads."""

    def __init__(self, libraries: Libraries, ref: str) -> None:
        self.ref = ref
        self.scope, self.id = parse_ref(ref)
        self.root = pseudo_dir(libraries.root(self.scope), self.id)
        self.info = read_json(self.root / CARD)
        if self.info is None:
            raise LookupError(f"unknown pseudo-label set {ref}")

    @property
    def setup(self) -> str:
        return str(self.info["setup"])

    def summary(self) -> dict[str, Any]:
        index = read_json(self.root / "index.json", {}) or {}
        kinds = [entry["kind"] for entry in index.get("items", [])]
        return {
            "ref": self.ref, "setup": self.setup, "name": self.info.get("name"), "description": self.info.get("description"),
            "author": self.info.get("author"), "created_at": self.info.get("created_at"), "items": len(kinds),
            "counts": {kind: kinds.count(kind) for kind in sorted(set(kinds))},
            "recordings": len({entry["recording"] for entry in index.get("items", [])}),
        }


def create_pseudo_labels(
    libraries: Libraries, set_id: str, *, setup: str, name: str = "", description: str = "", author: str | None = None,
    scope: str = "mine",
) -> PseudoLabelSet:
    """Write the card of a set whose items are (or will be) built in its directory.

    It is personal unless a script passes ``scope="lab"`` with a writable lab root.
    """

    from .setups import get_setup

    get_setup(libraries, setup)
    path = pseudo_dir(libraries.root(scope), set_id) / CARD
    if path.exists():
        raise FileExistsError(f"pseudo-label set {make_ref(scope, set_id)} already exists")
    write_json(path, {"setup": setup, "name": name or set_id, "description": description,
                      "author": author or getpass.getuser(), "created_at": utc_now()})
    return PseudoLabelSet(libraries, make_ref(scope, set_id))


def get_pseudo_labels(libraries: Libraries, ref: str) -> PseudoLabelSet:
    return PseudoLabelSet(libraries, ref)


def list_pseudo_labels(libraries: Libraries, setup: str | None = None) -> list[PseudoLabelSet]:
    found = []
    for scope in libraries.scopes():
        directory = libraries.root(scope) / PSEUDO_DIR
        for path in sorted(directory.glob(f"*/{CARD}")) if directory.is_dir() else ():
            pseudo = PseudoLabelSet(libraries, make_ref(scope, path.parent.name))
            if setup is None or pseudo.setup == setup:
                found.append(pseudo)
    return found
