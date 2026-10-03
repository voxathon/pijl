"""Evaluation a kind at a time (engine option eval=batches): every kind's eval runs on
arrays of its instances' inputs. The steppers (sim/plain.py, sim/dirty.py) call it
through the functions a Circuit binds when it's built (see Circuit._bind):

  run_all(c)                     every part runs; outputs written
  run_all_tracked(c) -> bool     the same, and did any output change?
  run_some(c, dirty, bid) -> []  just these pure parts (their batch numbers: bid),
                                 plus every part that isn't pure; the output pins
                                 that changed
  COSTS                          its cost model, for sim/dirty.py

Settling (Circuit.settle_ticks) only happens on run_all / run_all_tracked: a
stepper runs everything while anything is settling.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from ..parts import Ctx
from .circuit import _FAILED, _evaluate, _Handles

if TYPE_CHECKING:
    from .circuit import Circuit


class COSTS:
    """What a step costs, roughly (microseconds; measured with bogobips on an
    i5-9600K), for sim/dirty.py to choose between running every pure part and only
    the dirty ones. Only the ratios matter. Picking parts out costs much more per part
    than running a whole kind at once, and every kind touched costs its share of numpy
    calls."""

    FULL_FIXED, FULL_PER_PART = 75.0, 0.019
    PICK_FIXED, PICK_PER_KIND, PICK_PER_PART = 20.0, 43.0, 0.1


def run_all(c: Circuit) -> None:
    states = c._pins.states
    for batch, outs in _results(c, c._eval_batches()):
        if c._settling:  # settling parts take their new outputs only half the time
            keep = (c._settle[batch.slots] <= 0) | (c.rng.random(len(batch.parts)) < 0.5)
            for idx, values in zip(batch.outs, outs):
                states[idx[keep]] = values[keep]
        else:
            for idx, values in zip(batch.outs, outs):
                states[idx] = values


def run_all_tracked(c: Circuit) -> bool:
    states = c._pins.states
    moved = False
    for batch, outs in _results(c, c._eval_batches()):
        keep = None
        if c._settling:
            keep = (c._settle[batch.slots] <= 0) | (c.rng.random(len(batch.parts)) < 0.5)
        for idx, values in zip(batch.outs, outs):
            if keep is not None:
                idx, values = idx[keep], values[keep]
            moved = moved or not np.array_equal(states[idx], values)
            states[idx] = values
    return moved


def run_some(c: Circuit, dirty: np.ndarray, bid: np.ndarray) -> list[np.ndarray]:
    return some(c, dirty, bid, enumerate(c._eval_batches()))


def some(c: Circuit, dirty: np.ndarray, bid: np.ndarray, numbered) -> list[np.ndarray]:
    """run_some over just these (batch number, batch) pairs."""
    states = c._pins.states
    now = time.monotonic()
    moved: list[np.ndarray] = []
    results = []
    for b, batch in numbered:
        t = batch.type
        if t.kind in c.faults:
            continue
        if t.pure:
            pos = c._batch_pos[dirty[bid == b]]
            if not pos.size:
                continue
            parts, ins = _Handles(c, batch.slots[pos]), [idx[pos] for idx in batch.ins]
            outs_at = [idx[pos] for idx in batch.outs]
        else:  # (never skipped)
            parts, ins, outs_at = batch.parts, batch.ins, batch.outs
        ctx = Ctx(parts, c.tick, now)
        outs = c._guard(t, "eval", lambda: _evaluate(t, ctx, [states[idx] for idx in ins], batch.out_widths))
        if outs is not _FAILED:
            results.append((outs_at, outs))
    for outs_at, outs in results:
        for idx, values in zip(outs_at, outs):
            diff = states[idx] != values
            if diff.any():
                idx = idx[diff]
                moved.append(idx)
                states[idx] = values[diff]
    return moved


def _results(c: Circuit, batches):
    """Every batch's eval, all computed before any is written (so the order kinds run
    in doesn't matter): (batch, output arrays), failed ones left out."""
    states = c._pins.states
    now = time.monotonic()
    results = []
    for batch in batches:
        t = batch.type
        if t.kind in c.faults:
            continue
        ctx = Ctx(batch.parts, c.tick, now)
        outs = c._guard(t, "eval", lambda: _evaluate(t, ctx, [states[idx] for idx in batch.ins], batch.out_widths))
        if outs is not _FAILED:
            results.append((batch, outs))
    return results
