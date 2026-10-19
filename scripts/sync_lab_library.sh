#!/usr/bin/env bash
# Copy the lab library from the flv machines to Engaging (the developer's step, after scripts/publish.py).
#
#   scripts/sync_lab_library.sh USER@orcd-login.mit.edu
#
# The flv copy (library.roots.LAB_LIBRARY_FLV) is the one publish.py writes;
# Engaging's (LAB_LIBRARY_ENGAGING) only receives it. Nothing is deleted on
# Engaging: published items are frozen, and the files that do change (setups,
# the defaults log, label indexes) are replaced. Files are made readable by
# everyone who can reach the directory, as the app needs to read them.

set -euo pipefail

FLV=/storage/fs/store1/shared/worm-pose-models
ENGAGING=/orcd/data/flavell/001/shared/worm-pose-models

if (( $# != 1 )); then
  printf 'usage: %s USER@ENGAGING-LOGIN-HOST\n' "$0" >&2
  exit 64
fi
ssh "$1" mkdir -p "$ENGAGING"
rsync -a --chmod=Da+rx,Fa+r --exclude=.lock --exclude='*.partial' --info=stats1 "$FLV/" "$1:$ENGAGING/"
printf 'synced %s to %s:%s\n' "$FLV" "$1" "$ENGAGING"
