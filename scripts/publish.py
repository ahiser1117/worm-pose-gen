#!/usr/bin/env python3
"""Publish a personal setup, dataset, model or labels into the lab library (the developer's step; the app never writes there).

    scripts/publish.py setup mine:rig2
    scripts/publish.py labels lab:nir-flv
    scripts/publish.py dataset mine:copper-plates
    scripts/publish.py model mine:copper-ft --default body --reason "fixes heads on copper plates"

The item is copied, never moved, under the same id (``--as`` renames it),
and an id the lab already has is refused: a published item is frozen, and a
new version gets a new id.  References inside it (``setup``, ``parent``,
``trained_on``) that name personal items become lab references, so those
items must be published first; the script says which.

``labels`` takes a setup and copies the current personal label of each
frame of its collection into the lab's part (of the published setup, for a
personal one) as the frame's next lab revision, unless the lab has that
revision's file already.  The file is copied as it is, so its ``sha256``
and built targets stay valid, and its ``saved_at`` makes it the frame's
current label for everyone.  A dataset holds only its splits, so publish
the labels of its recordings too.  A model
keeps its training records and its evaluations on lab benchmarks (stored
in the personal library, since the app cannot write the lab's), not those on
personal benchmarks.  ``--default ROLE`` also makes a published model the lab
setup's default for that role, logged with ``--reason``.

The lab root defaults to the host's (``library.LAB_LIBRARY_BY_HOST``) and
the personal one to the host default; ``--lab`` and ``--library`` override
them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any

from worm_pose_gen.library import Collection, Libraries, get_card, get_setup
from worm_pose_gen.library.collection import ENTRY_FIELDS, collection_dir
from worm_pose_gen.library.datasets import Dataset, dataset_dir
from worm_pose_gen.library.labels import label_key
from worm_pose_gen.library.models import model_dir
from worm_pose_gen.library.roots import check_id, locked, make_ref, parse_ref, read_json, write_json
from worm_pose_gen.library.setups import ROLE_OUTPUTS, log_default, setup_path


KINDS = ("setup", "labels", "dataset", "model")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=KINDS)
    parser.add_argument("ref", help="the personal item, mine:<id> (for labels: the setup, lab:<id> or mine:<id>)")
    parser.add_argument("--as", dest="lab_id", default=None, help="its id in the lab library (default: the same)")
    parser.add_argument("--lab", type=Path, default=None, help="the lab library root")
    parser.add_argument("--library", type=Path, default=None, help="the personal library root")
    parser.add_argument("--default", choices=tuple(ROLE_OUTPUTS), default=None, help="(model) make it the lab setup's default for this role")
    parser.add_argument("--reason", default="", help="why the default changes (required with --default)")
    return parser.parse_args(argv)


def lab_ref(libraries: Libraries, ref: str | None) -> str | None:
    """A reference as the lab sees it: a personal item must already be in the lab under the same id."""

    if ref is None:
        return None
    scope, item_id = parse_ref(ref)
    if scope == "lab":
        return ref
    translated = make_ref("lab", item_id)
    kind_paths = (setup_path(libraries.root("lab"), item_id), dataset_dir(libraries.root("lab"), item_id) / "dataset.json",
                  model_dir(libraries.root("lab"), item_id) / "model.json")
    if not any(path.exists() for path in kind_paths):
        raise SystemExit(f"{ref} is not in the lab library; publish it first")
    return translated


def staged_copy(source: Path, target: Path) -> Path:
    """Copy a directory to a hidden staging path beside ``target``; the caller renames it into place."""

    staging = target.with_name(f".{target.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(source, staging, ignore=shutil.ignore_patterns(".lock", "*.partial"))
    return staging


def publish_setup(libraries: Libraries, ref: str, lab_id: str) -> str:
    _, setup_id = parse_ref(ref)
    data = read_json(setup_path(libraries.personal, setup_id))
    data["defaults"] = {role: lab_ref(libraries, model) for role, model in (data.get("defaults") or {}).items()}
    write_json(setup_path(libraries.root("lab"), lab_id), data)
    return make_ref("lab", lab_id)


def publish_labels(libraries: Libraries, setup_ref: str) -> int:
    """Copy the setup's current personal labels into the lab's part of its collection; returns how many were copied."""

    target_setup = lab_ref(libraries, setup_ref)
    target = Collection(libraries, target_setup)
    root = collection_dir(libraries.root("lab"), target_setup)
    mine = [r for r in Collection(libraries, setup_ref).labels() if r.scope == "mine"]
    copied = 0
    with locked(root):
        index = read_json(root / "index.json", {}) or {}
        for record in mine:
            key = label_key(record.recording, record.frame)
            entries = index.get(key, [])
            if any(entry["sha256"] == record.sha256 for entry in entries):
                continue
            revision = (entries[-1]["revision"] if entries else 0) + 1
            path = target.label_path("lab", record.recording, record.frame, revision)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record.path, path)
            index[key] = [*entries, {**{name: getattr(record, name) for name in ENTRY_FIELDS}, "revision": revision}]
            copied += 1
        write_json(root / "index.json", index)
    return copied


def publish_dataset(libraries: Libraries, ref: str, lab_id: str) -> str:
    dataset = Dataset(libraries, ref)
    target = dataset_dir(libraries.root("lab"), lab_id)
    info = {**dataset.info, "setup": lab_ref(libraries, dataset.setup), "published_from": ref}
    staging = target.with_name(f".{target.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    write_json(staging / "dataset.json", info)
    write_json(staging / "splits.json", dataset.splits())
    staging.rename(target)
    return make_ref("lab", lab_id)


def publish_model(libraries: Libraries, ref: str, lab_id: str) -> str:
    card = get_card(libraries, ref)
    _, model_id = parse_ref(ref)
    target = model_dir(libraries.root("lab"), lab_id)
    data: dict[str, Any] = read_json(model_dir(libraries.personal, model_id) / "model.json")
    data.update(
        setup=lab_ref(libraries, card.setup), parent=lab_ref(libraries, card.parent),
        trained_on=[{**entry, "dataset": lab_ref(libraries, entry["dataset"])} for entry in card.trained_on],
        published_from=ref,
    )
    staging = staged_copy(model_dir(libraries.personal, model_id), target)
    evaluations = staging / "evaluations"
    for path in sorted(evaluations.glob("*.json")) if evaluations.is_dir() else ():
        if not path.name.startswith("lab."):  # scores on personal benchmarks stay personal
            path.unlink()
    write_json(staging / "model.json", data)
    staging.rename(target)
    return make_ref("lab", lab_id)


def set_lab_default(libraries: Libraries, model: str, role: str, reason: str) -> None:
    card = get_card(libraries, model)
    missing = [name for name in ROLE_OUTPUTS[role] if name not in card.outputs]
    if missing:
        raise SystemExit(f"{model} has no {', '.join(missing)} output for the {role} default")
    _, setup_id = parse_ref(card.setup)
    path = setup_path(libraries.root("lab"), setup_id)
    data = read_json(path)
    previous = (data.get("defaults") or {}).get(role)
    data["defaults"] = {**(data.get("defaults") or {}), role: model}
    write_json(path, data)
    log_default(libraries.root("lab"), card.setup, role, model, previous=previous, reason=reason)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    libraries = Libraries.for_host(lab=args.lab, personal=args.library)
    if libraries.lab is None or not libraries.lab.is_dir():
        sys.exit(f"no lab library at {libraries.lab}; pass --lab")
    scope, item_id = parse_ref(args.ref)
    if args.kind == "labels":
        get_setup(libraries, args.ref)
        print(f"published {publish_labels(libraries, args.ref)} label(s) of {args.ref} into {libraries.lab}")
        return
    if scope != "mine":
        sys.exit(f"{args.ref} is already in the lab library")
    lab_id = check_id(args.lab_id or item_id)
    if args.default and args.kind != "model":
        sys.exit("--default applies to models")
    if args.default and not args.reason.strip():
        sys.exit("--default needs --reason")
    existing = {"setup": setup_path(libraries.lab, lab_id), "dataset": dataset_dir(libraries.lab, lab_id),
                "model": model_dir(libraries.lab, lab_id)}[args.kind]
    if existing.exists():
        sys.exit(f"the lab library already has {args.kind} {lab_id}; publish a new version under a new id (--as)")
    if args.kind == "setup":
        get_setup(libraries, args.ref)
        published = publish_setup(libraries, args.ref, lab_id)
    elif args.kind == "dataset":
        published = publish_dataset(libraries, args.ref, lab_id)
    else:
        published = publish_model(libraries, args.ref, lab_id)
        if args.default:
            set_lab_default(libraries, published, args.default, args.reason)
    print(f"published {args.ref} as {published} in {libraries.lab}")


if __name__ == "__main__":
    main()
