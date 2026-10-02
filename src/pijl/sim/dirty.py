"""The dirty-set stepper (engine option dirty=adaptive). Bound onto a Circuit as its
step / run_until_stable (see sim/config.py).

A pure part whose inputs didn't change since it last ran would only say the same
again, so a step runs just the dirty ones: those with an input that changed in the
last carry, or a pin written from outside (Pin.state, Circuit.write_pins: "pokes").
Only nets with a changed driver are carried. When picking parts out would cost more
than running everything (a rough cost model: the evaluator's COSTS), the step runs
everything, and when a carry changes many readers, the next step does without making
a list at all.

Tick for tick the same as sim/plain.py: tests/test_circuit.py checks it against it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..logic import CODE, Z, fights, resolve

if TYPE_CHECKING:
    from .circuit import Circuit

TAKES_POKES = True


def step(c: Circuit) -> None:
    if c._nets_dirty:
        c._rebuild_nets()
    batches = c._eval_batches()
    store = c._pins
    states = store.states
    poked = _take_poked(c)
    faults = len(c.faults)
    full = c._full or bool(c._settling)
    if not full:
        dirty = c._dirty
        if poked.size:
            dirty = _distinct(np.concatenate((dirty, store.part[poked])), c._n_part_slots)
        bid = c._batch_of[dirty]
        dirty, bid = dirty[bid >= 0], bid[bid >= 0]
        if dirty.size:  # (none: next to free) is picking them out worth it?
            cost = c._costs
            kinds = np.zeros(len(batches), bool)
            kinds[bid] = True
            pick = cost.PICK_FIXED + cost.PICK_PER_KIND * np.count_nonzero(kinds)
            pick += cost.PICK_PER_PART * dirty.size
            full = pick > cost.FULL_FIXED + cost.FULL_PER_PART * c._n_pure

    # Phase 1: parts compute outputs from current inputs (sim/batches.py, sim/lut.py).
    if full:
        outs_moved, moved = c._run_all_tracked(), []
    else:
        outs_moved, moved = False, c._run_some(dirty, bid)

    # Phase 2: nets resolve their drivers and hand the value to their readers. Many
    # changed readers (past `most`, where picking out even one kind's parts would cost
    # more than running everything): the next step runs everything, no list.
    cost = c._costs
    most = int(
        (cost.FULL_FIXED + cost.FULL_PER_PART * c._n_pure - cost.PICK_FIXED - cost.PICK_PER_KIND)
        / cost.PICK_PER_PART
    )
    most = max(most, 0)  # (nothing changed: idle next, never a full run)
    if full or c._full:  # (a hook that failed just now set its outputs to X)
        n, changed = _carry_all(c, most, poked)
    else:
        touched = np.concatenate([poked, *moved]) if moved else poked
        nets = c._pin_net[touched]
        n, changed = _carry_some(c, _distinct(nets[nets >= 0], len(c.net_value)), most)
    c._quiet = not (outs_moved or moved or n) and len(c.faults) == faults
    c._full = changed is None
    if changed is not None:
        c._dirty = _distinct(changed, c._n_part_slots)
        c._full = len(c._dirty) > most
    c.tick += 1
    if c._settling:
        s = c._settle[: c._n_part_slots]
        s[s > 0] -= 1
        c._settling = int(np.count_nonzero(s))
        if not c._settling:
            c._full = True  # (some kept their old outputs: everyone runs once)


def run_until_stable(c: Circuit, limit: int) -> int | None:
    for n in range(limit + 1):
        c.step()
        if not c._settling and c._quiet:
            return n
    return None


def _take_poked(c: Circuit) -> np.ndarray:
    """The pin slots written from outside step() since the last call."""
    poked = c._poked
    if not poked:
        return np.empty(0, np.intp)
    poked = np.concatenate([np.atleast_1d(np.asarray(p, np.intp)) for p in poked])
    c._poked.clear()
    return _distinct(poked, c._pins.n)


def _carry_all(c: Circuit, most: int, poked: np.ndarray) -> tuple[int, np.ndarray | None]:
    """Circuit._carry, keeping track. After a carry every reader holds its net's value,
    and only a poke can change that, so only the readers of nets whose value changed,
    or that were poked, are handed it again. Returns how many readers changed and
    their parts (None: more than `most`)."""
    states = c._pins.states
    value, conflict = c._resolve()
    old = c.net_value
    c.net_value, c.net_conflict = value, conflict
    moved = value != old
    n = int(np.dot(moved, c._rd_count))  # readers whose state changes
    if n > most:  # (many: all of them at once is cheaper than picking them out)
        states[c._readers] = value[c._reader_net]
        return n, None
    if poked.size:
        more = c._pin_net[poked]
        moved[more[more >= 0]] = True
    nets = np.flatnonzero(moved)
    return _hand_out(c, nets, value[nets], most)


def _carry_some(c: Circuit, nets: np.ndarray, most: int) -> tuple[int, np.ndarray | None]:
    """_carry_all for just these nets (no repeats): no other net's drivers changed."""
    if not nets.size:
        return 0, nets
    states = c._pins.states
    value = np.zeros(len(nets), CODE)
    conflict = np.zeros(len(nets), bool)
    solo = c._solo_of[nets]
    m = solo >= 0
    value[m] = states[solo[m]]
    g = c._multi_at[nets]
    m = g >= 0
    if m.any():
        d, starts = _gather(c._drivers, c._drv_starts[g[m]], c._drv_count[g[m]], states)
        value[m] = resolve(d, starts)
        conflict[m] = fights(d, starts)
    g = c._weak_at[nets]
    m = (g >= 0) & (value == Z)
    if m.any():
        w, starts = _gather(c._weak, c._weak_starts[g[m]], c._weak_count[g[m]], states)
        value[m] = resolve(w, starts)
        conflict[m] = fights(w, starts)
    c.net_value[nets] = value
    c.net_conflict[nets] = conflict
    return _hand_out(c, nets, value, most)


def _hand_out(c: Circuit, nets: np.ndarray, value: np.ndarray, most: int) -> tuple[int, np.ndarray | None]:
    """Give these nets' readers their net's new value. How many readers changed, and
    their parts (None: more than `most`)."""
    states = c._pins.states
    first = c._rd_start[nets]
    count = c._rd_start[nets + 1] - first
    readers = c._rd_sorted[_segments(first, count)]
    new = np.repeat(value, count)
    diff = states[readers] != new
    states[readers[diff]] = new[diff]
    n = int(np.count_nonzero(diff))
    return n, (None if n > most else c._pins.part[readers[diff]])


def _distinct(a: np.ndarray, size: int) -> np.ndarray:
    """The values in `a` (all in range(size)), once each, ascending: by sorting when
    they're few next to `size`, else by marking them in a mask. (np.unique is slow at
    both ends: it hashes, and has a lot of overhead for a few.)"""
    if len(a) < 2:
        return a
    if len(a) * 16 < size:
        a = np.sort(a)
        return a[np.concatenate(([True], a[1:] != a[:-1]))]
    mask = np.zeros(size, bool)
    mask[a] = True
    return np.flatnonzero(mask)


def _segments(first: np.ndarray, count: np.ndarray) -> np.ndarray:
    """first[0], first[0] + 1, ... (count[0] of them), then the same for [1], ..."""
    ends = np.cumsum(count)
    total = int(ends[-1]) if len(ends) else 0
    return np.repeat(first - (ends - count), count) + np.arange(total)


def _gather(
    slots: np.ndarray, first: np.ndarray, count: np.ndarray, states: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Some groups of a grouped array: their pins' states, and where each group starts."""
    return states[slots[_segments(first, count)]], np.cumsum(count) - count
