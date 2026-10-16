"""Producing the next item of a sequence on a background thread while the caller works on the current one."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Iterator, TypeVar

T = TypeVar("T")


def prefetched(items: Iterable[T]) -> Iterator[T]:
    """``items`` in order, the next one produced on a background thread while the caller works on the current one.

    A stage's preparation (reading, the network's pass, collecting cleaned
    masks, submitting starts, a fit's host-side targets) then runs while the
    GPU fits the item before it; torch, h5py, NumPy and SciPy release the GIL
    for most of their work.  The producer must not touch what the caller's
    work on an item changes.
    """

    items = iter(items)
    with ThreadPoolExecutor(max_workers=1) as thread:
        following = thread.submit(next, items, _END)
        while (current := following.result()) is not _END:
            following = thread.submit(next, items, _END)
            yield current


_END: Any = object()
