#!/usr/bin/env python3
"""Make a personal-library body-field model that averages several members and/or flipped inputs.

The members are library body-field models with the same setup and lags; the
new model's weights file holds all of them (:mod:`worm_pose_gen.body_ensemble`)
and loads through ``body_net.load_body_net`` like any other, so the pipeline,
the app and the evaluations use it unchanged.  Score it afterwards with
``scripts/evaluate_model.py --model mine:<name>``.  Example, three seeds with
four flips::

    scripts/make_body_ensemble.py --name arch-ens3-tta4 --member mine:on-food-finetune-329 \\
        --member mine:arch-base-s1 --member mine:arch-base-s2 --flips none,h,v,hv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from worm_pose_gen import library
from worm_pose_gen.body_ensemble import FLIPS, save_ensemble
from worm_pose_gen.model_eval import add_library_arguments, libraries_from


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_library_arguments(parser)
    parser.add_argument("--name", required=True, help="the new model's id")
    parser.add_argument("--member", action="append", required=True, help="member model reference (repeatable)")
    parser.add_argument("--flips", default="none", help=f"comma-separated views of each member, of {', '.join(FLIPS)}")
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)
    libraries = libraries_from(args)
    flips = [name.strip() for name in args.flips.split(",") if name.strip()]

    cards = [library.get_card(libraries, ref) for ref in args.member]
    for card in cards:
        if card.kind != "body_net":
            raise SystemExit(f"{card.ref} is a {card.kind}, not a body_net")
        if card.setup != cards[0].setup or card.inputs != cards[0].inputs or card.outputs != cards[0].outputs:
            raise SystemExit(f"{card.ref} differs from {cards[0].ref} in setup, inputs or outputs")
    trained_on = []
    for card in cards:
        trained_on += [entry for entry in card.trained_on if entry not in trained_on]
    description = f"{len(cards)} member{'s' if len(cards) > 1 else ''} ({', '.join(args.member)}) x flips {'/'.join(flips)}"
    card = {
        "name": args.name, "kind": "body_net", "setup": cards[0].setup, "inputs": cards[0].inputs,
        "outputs": list(cards[0].outputs), "trained_on": trained_on, "parent": cards[0].ref,
        "hparams": {"ensemble_members": list(args.member), "tta_flips": flips, "lags": cards[0].inputs.get("lags_frames"),
                    "member_hparams": {c.ref: c.hparams for c in cards}},
        "notes": f"Mean probability of {description}. {args.notes}".strip(),
    }
    with tempfile.TemporaryDirectory(dir=libraries.personal) as scratch:
        weights = Path(scratch) / "weights.ckpt"
        save_ensemble(weights, [library.weights_path(libraries, ref) for ref in args.member], flips)
        created = library.create_model(
            libraries, args.name, card, weights,
            training_records={"ensemble.json": {"members": list(args.member), "flips": flips,
                                                "member_cards": [c.to_dict() for c in cards]}},
        )
    print(json.dumps({"model": created.ref, "members": list(args.member), "flips": flips}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
