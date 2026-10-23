#!/usr/bin/env python3
"""Publish a personal setup, dataset, benchmark, pseudo-label set, model or labels into the lab library (the developer's step; the app never writes there).

    scripts/publish.py setup mine:rig2
    scripts/publish.py labels lab:nir-flv
    scripts/publish.py dataset mine:copper-plates
    scripts/publish.py benchmark mine:copper-plates-b1
    scripts/publish.py pseudo-labels mine:copper-self-train
    scripts/publish.py model mine:copper-ft --default body --default mask --reason "fixes heads on copper plates"

The item is copied, never moved, under the same id (``--as`` renames it),
and an id the lab already has is refused: a published item is frozen, and a
new version gets a new id.  References inside it (``setup``, ``parent``,
``trained_on``) that name personal items become lab references, so those
items must be published first; the script says which: a model's
pseudo-label sets as well as its setup, parent and datasets.

``labels`` takes a setup and copies the current personal label of each
frame of its collection into the lab's part (of the published setup, for a
personal one) as the frame's next lab revision, unless the lab has that
revision's file already.  The file is copied as it is, so its ``sha256``
and built targets stay valid, and its ``saved_at`` makes it the frame's
current label for everyone.  A dataset holds only its splits, so publish
the labels of its recordings too.  A benchmark's personal label revisions
become the lab revisions with the same ``sha256``, so publish the labels
first; ``--dataset`` names the lab dataset it scores when its own is not
published.  A model
keeps its training records and its evaluations on lab benchmarks (stored
in the personal library, since the app cannot write the lab's), not those on
personal benchmarks.  ``--default ROLE`` (repeatable) also makes a published
model the lab setup's default for that role, logged with ``--reason``.

The lab library is written on one host; ``scripts/sync_lab_library.sh``
copies it to the other system's.

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

from worm_pose_gen.library import Collection, Libraries, get_benchmark, get_card, get_pseudo_labels, get_setup
from worm_pose_gen.library.benchmarks import benchmark_path
from worm_pose_gen.library.collection import ENTRY_FIELDS, collection_dir
from worm_pose_gen.library.datasets import Dataset, dataset_dir
from worm_pose_gen.library.labels import label_key
from worm_pose_gen.library.models import model_dir
from worm_pose_gen.library.pseudo_sets import CARD as PSEUDO_CARD, pseudo_dir
from worm_pose_gen.library.roots import check_id, locked, make_ref, parse_ref, read_json, write_json
from worm_pose_gen.library.setups import ROLE_OUTPUTS, log_default, setup_path


KINDS = ("setup", "labels", "dataset", "benchmark", "pseudo-labels", "model")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=KINDS)
    parser.add_argument("ref", help="the personal item, mine:<id> (for labels: the setup, lab:<id> or mine:<id>)")
    parser.add_argument("--as", dest="lab_id", default=None, help="its id in the lab library (default: the same)")
    parser.add_argument("--lab", type=Path, default=None, help="the lab library root")
    parser.add_argument("--library", type=Path, default=None, help="the personal library root")
    parser.add_argument("--default", choices=tuple(ROLE_OUTPUTS), action="append", default=[], help="(model) make it the lab setup's default for this role (repeatable)")
    parser.add_argument("--reason", default="", help="why the default changes (required with --default)")
    parser.add_argument("--dataset", default=None, help="(benchmark) the lab dataset it scores, in place of its own")
    return parser.parse_args(argv)


def lab_ref(libraries: Libraries, ref: str | None) -> str | None:
    """A reference as the lab sees it: a personal item must already be in the lab under the same id."""

    if ref is None:
        return None
    scope, item_id = parse_ref(ref)
    if scope == "lab":
        return ref
    translated = make_ref("lab", item_id)
    lab = libraries.root("lab")
    kind_paths = (setup_path(lab, item_id), dataset_dir(lab, item_id) / "dataset.json", model_dir(lab, item_id) / "model.json",
                  benchmark_path(lab, item_id), pseudo_dir(lab, item_id) / PSEUDO_CARD)
    if not any(path.exists() for path in kind_paths):
        raise SystemExit(f"{ref} is not in the lab library; publish it first")
    return translated


def staged_copy(source: Path, target: Path) -> Path:
    """Copy a directory to a hidden staging path beside ``target``; the caller renames it into place.

    Everything in the copy is made readable by everyone: the lab reads it,
    and a personal file may be private (``write_json`` wrote 0600 files
    before it set the permissions of an ordinary new file).
    """

    staging = target.with_name(f".{target.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(source, staging, ignore=shutil.ignore_patterns(".lock", "*.partial"))
    for path in (staging, *staging.rglob("*")):
        path.chmod(path.stat().st_mode | (0o555 if path.is_dir() else 0o444))
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


def publish_benchmark(libraries: Libraries, ref: str, lab_id: str, dataset: str | None) -> str:
    """Copy a benchmark, its personal label revisions replaced by the lab revisions with the same ``sha256``."""

    found = get_benchmark(libraries, ref)
    lab_setup = lab_ref(libraries, found.setup)
    lab_index = read_json(collection_dir(libraries.root("lab"), lab_setup) / "index.json", {}) or {}
    entries = []
    for entry in found.entries:
        if entry["scope"] == "mine":
            same = [e for e in lab_index.get(label_key(entry["recording"], entry["frame"]), []) if e["sha256"] == entry["sha256"]]
            if not same:
                raise SystemExit(f"{ref}: {label_key(entry['recording'], entry['frame'])} is not in the lab library; publish the labels first")
            entry = {**entry, "scope": "lab", "revision": same[0]["revision"]}
        entries.append({**entry, "setup": lab_setup})
    data = read_json(benchmark_path(libraries.personal, parse_ref(ref)[1]))
    lab_dataset = lab_ref(libraries, dataset or found.dataset)
    data.pop("datasets", None)  # written by the migration's benchmark only
    data.update(setup=lab_setup, dataset=lab_dataset, labels=entries, published_from=ref)
    write_json(benchmark_path(libraries.root("lab"), lab_id), data)
    return make_ref("lab", lab_id)


def publish_pseudo_labels(libraries: Libraries, ref: str, lab_id: str) -> str:
    pseudo = get_pseudo_labels(libraries, ref)
    target = pseudo_dir(libraries.root("lab"), lab_id)
    staging = staged_copy(pseudo.root, target)
    write_json(staging / PSEUDO_CARD, {**pseudo.info, "setup": lab_ref(libraries, pseudo.setup), "published_from": ref})
    staging.rename(target)
    return make_ref("lab", lab_id)


def lab_trained_on(libraries: Libraries, entry: dict[str, Any]) -> dict[str, Any]:
    if "pseudo_labels" in entry:
        return {**entry, "pseudo_labels": [lab_ref(libraries, r) for r in entry["pseudo_labels"]]}
    return {**entry, "dataset": lab_ref(libraries, entry["dataset"])}


def publish_model(libraries: Libraries, ref: str, lab_id: str) -> str:
    card = get_card(libraries, ref)
    _, model_id = parse_ref(ref)
    target = model_dir(libraries.root("lab"), lab_id)
    data: dict[str, Any] = read_json(model_dir(libraries.personal, model_id) / "model.json")
    hparams = dict(card.hparams)
    if hparams.get("pseudo_labels"):
        hparams["pseudo_labels"] = [lab_ref(libraries, r) for r in hparams["pseudo_labels"]]
    data.update(
        setup=lab_ref(libraries, card.setup), parent=lab_ref(libraries, card.parent),
        trained_on=[lab_trained_on(libraries, entry) for entry in card.trained_on], hparams=hparams, published_from=ref,
    )
    staging = staged_copy(model_dir(libraries.personal, model_id), target)
    evaluations = staging / "evaluations"
    for path in sorted(evaluations.iterdir()) if evaluations.is_dir() else ():
        if not path.name.startswith("lab."):  # scores (and their overlays) on personal benchmarks stay personal
            shutil.rmtree(path) if path.is_dir() else path.unlink()
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
    if args.dataset and args.kind != "benchmark":
        sys.exit("--dataset applies to benchmarks")
    if args.default and not args.reason.strip():
        sys.exit("--default needs --reason")
    existing = {"setup": setup_path(libraries.lab, lab_id), "dataset": dataset_dir(libraries.lab, lab_id),
                "benchmark": benchmark_path(libraries.lab, lab_id), "pseudo-labels": pseudo_dir(libraries.lab, lab_id),
                "model": model_dir(libraries.lab, lab_id)}[args.kind]
    if existing.exists():
        sys.exit(f"the lab library already has {args.kind} {lab_id}; publish a new version under a new id (--as)")
    if args.kind == "setup":
        get_setup(libraries, args.ref)
        published = publish_setup(libraries, args.ref, lab_id)
    elif args.kind == "dataset":
        published = publish_dataset(libraries, args.ref, lab_id)
    elif args.kind == "benchmark":
        published = publish_benchmark(libraries, args.ref, lab_id, args.dataset)
    elif args.kind == "pseudo-labels":
        published = publish_pseudo_labels(libraries, args.ref, lab_id)
    else:
        published = publish_model(libraries, args.ref, lab_id)
        for role in args.default:
            set_lab_default(libraries, published, role, args.reason)
    print(f"published {args.ref} as {published} in {libraries.lab}")


if __name__ == "__main__":
    main()
