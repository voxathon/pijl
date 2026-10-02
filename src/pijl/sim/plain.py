"""The plain stepper (engine option dirty=off): every part runs and every net is
carried, every tick. The reference the other steppers are tested against, tick for
tick. Bound onto a Circuit as its step / run_until_stable (see sim/config.py)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from ..parts import Ctx
from .circuit import _FAILED, _evaluate

if TYPE_CHECKING:
    from .circuit import Circuit

TAKES_POKES = False  # (doesn't need to know what was written from outside)


def step(c: Circuit) -> None:
    if c._nets_dirty:
        c._rebuild_nets()
    states = c._pins.states

    # Phase 1: every part computes outputs from current inputs, one kind at a time.
    # Compute all first, then write, so evaluation order doesn't matter.
    now = time.monotonic()
    results = []
    for batch in c._eval_batches():
        t = batch.type
        if t.kind in c.faults:
            continue
        ctx = Ctx(batch.parts, c.tick, now)
        outs = c._guard(t, "eval", lambda: _evaluate(t, ctx, [states[idx] for idx in batch.ins]))
        if outs is not _FAILED:
            results.append((batch, outs))
    for batch, outs in results:
        if c._settling:  # settling parts take their new outputs only half the time
            keep = (c._settle[batch.slots] <= 0) | (c.rng.random(len(batch.parts)) < 0.5)
            for idx, values in zip(batch.outs, outs):
                states[idx[keep]] = values[keep]
        else:
            for idx, values in zip(batch.outs, outs):
                states[idx] = values

    # Phase 2: every net resolves its drivers and hands the value to its readers.
    c._carry()
    c.tick += 1
    if c._settling:
        s = c._settle[: c._n_part_slots]
        s[s > 0] -= 1
        c._settling = int(np.count_nonzero(s))


def run_until_stable(c: Circuit, limit: int) -> int | None:
    for n in range(limit + 1):
        before = c._pins.states.copy()
        c.step()
        if not c._settling and np.array_equal(before, c._pins.states):
            return n
    return None
