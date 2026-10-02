"""Evaluation by lookup table (engine option eval=lut). Same interface as
sim/batches.py: run_all, run_all_tracked, run_some, COSTS.

A pure part's outputs depend only on its inputs, and with four levels per pin a kind
with k inputs has 4^k input combinations. For k <= KMAX that's a small table, made
once by calling the kind's eval on every combination at once (tabulate). Every
output of every such part on the board is then one *unit*: a table to look in, the
pins that pick the row, the pin it writes. A step looks up every unit at once:

    row = table_start + in0 + 4 * in1 + 16 * in2 + 64 * in3    (the pins' codes)
    out = tables[row]

That's a few numpy calls for the whole board, however many kinds it has: no eval
call per kind, no Logic arrays. Kinds that can't be tabulated (not pure, more than
KMAX inputs, or an eval that fails on some combination) still run a kind at a time,
through sim/batches.py, in the same step.

Tick for tick the same as eval=batches (tests/test_circuit.py checks it), settling's
random draws included: those go batch by batch, in the same order.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import numpy as np

from ..logic import CODE
from ..parts import Ctx
from . import batches
from .circuit import _evaluate
from .dirty import _segments

if TYPE_CHECKING:
    from ..parts import PartType
    from .circuit import Circuit, _Batch

KMAX = 4  # inputs: up to 4^4 = 256 rows a table, and a row's offset fits a byte


class COSTS:
    """See batches.COSTS. Every unit is looked up at once, so running everything is
    cheap and picking out has no cost per kind; picked out, each costs more. (Chosen
    from a sweep over bogobips kinds on an i5-9600K.)"""

    FULL_FIXED, FULL_PER_PART = 20.0, 0.015
    PICK_FIXED, PICK_PER_KIND, PICK_PER_PART = 70.0, 0.0, 0.12


# ---- tables -------------------------------------------------------------------------

_tables: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # PartType -> table


def tabulate(t: PartType) -> np.ndarray | None:
    """t's outputs for every input combination: codes, (outputs, 4^k), row = in0 +
    4 * in1 + ... Or None: t can't be tabulated (not pure, too many inputs, or its
    eval fails or doesn't give an answer per row)."""
    try:
        return _tables[t]
    except KeyError:
        pass
    table = None
    k = len(t.ins)
    if t.pure and t.has("eval") and t.outs and k <= KMAX:
        rows = np.arange(4**k)
        ins = [((rows >> (2 * i)) & 3).astype(CODE) for i in range(k)]
        ctx = Ctx([None] * len(rows), 0, 0.0)  # (a pure eval needs nothing from it)
        try:
            outs = _evaluate(t, ctx, ins)
            table = np.stack([np.asarray(o, CODE) for o in outs])
            if table.shape != (len(t.outs), len(rows)) or table.max() > 3:
                table = None
        except Exception:
            table = None
    _tables[t] = table
    return table


# ---- the plan: units, made from the circuit's batches ---------------------------------


class _Plan:
    """Every unit of the board, laid out batch by batch (kinds with more inputs first,
    so input i is needed by a prefix of the units: the first ends[i]), part by part,
    output by output."""

    def __init__(self, c: Circuit, all_batches: list[_Batch]) -> None:
        self.batches = all_batches  # (what it was made from: rebuilt when that changes)
        lut: list[tuple[int, _Batch, np.ndarray]] = []
        self.other: list[_Batch] = []  # batches that still run a kind at a time ...
        self.other_numbered: list[tuple[int, _Batch]] = []  # ... with their numbers
        for b, batch in enumerate(all_batches):
            table = None if batch.type.kind in c.faults else tabulate(batch.type)
            if table is None:
                self.other.append(batch)
                self.other_numbered.append((b, batch))
            else:
                lut.append((b, batch, table))
        lut.sort(key=lambda e: -len(e[1].type.ins))  # (stable: batch order otherwise)

        tables, starts, at = [], {}, 0
        for _, batch, table in lut:
            if batch.type not in starts:
                starts[batch.type] = at
                tables.append(table.reshape(-1))  # (output j's rows at j * 4^k)
                at += table.size
        self.table = np.concatenate(tables) if tables else np.zeros(0, CODE)

        n_part_slots = c._n_part_slots
        self.unit_first = np.zeros(n_part_slots, np.intp)  # per part slot: its first unit
        self.unit_count = np.zeros(n_part_slots, np.intp)  # ... and how many (0: none)
        base, outs = [], []
        ins: list[list[np.ndarray]] = [[] for _ in range(KMAX)]
        self.spans: dict[int, tuple[int, int, int]] = {}  # batch number -> (first unit, parts, outs)
        u = 0
        for b, batch, table in lut:
            t = batch.type
            n, n_out, k = len(batch.slots), len(t.outs), len(t.ins)
            rows = 4**k
            # part-major: unit u0 + p * n_out + j is part p's output j
            base.append(np.tile(starts[t] + rows * np.arange(n_out), n))
            outs.append(np.stack(batch.outs, axis=1).reshape(-1) if n_out else np.zeros(0, np.intp))
            for i in range(k):
                ins[i].append(np.repeat(batch.ins[i], n_out))
            self.unit_first[batch.slots] = u + n_out * np.arange(n)
            self.unit_count[batch.slots] = n_out
            self.spans[b] = (u, n, n_out)
            u += n * n_out
        self.n = u
        self.base = np.concatenate(base).astype(np.intp) if base else np.zeros(0, np.intp)
        self.out = np.concatenate(outs).astype(np.intp) if outs else np.zeros(0, np.intp)
        self.ins = [np.concatenate(p).astype(np.intp) for p in ins if p]  # input i ...
        self.ends = [len(p) for p in self.ins]  # ... for units [0, ends[i])

    def rows(self, states: np.ndarray) -> np.ndarray:
        """Every unit's row in the table."""
        local = np.zeros(self.n, np.uint8)
        for i, pins in enumerate(self.ins):
            local[: self.ends[i]] |= states[pins] << (2 * i)
        return self.base + local

    def rows_of(self, states: np.ndarray, units: np.ndarray) -> np.ndarray:
        """Some units' rows."""
        local = np.zeros(len(units), np.uint8)
        for i, pins in enumerate(self.ins):
            m = units < self.ends[i]
            local[m] |= states[pins[units[m]]] << (2 * i)
        return self.base[units] + local


def _plan(c: Circuit) -> _Plan:
    all_batches = c._eval_batches()
    plan = c._lut_plan
    if plan is None or plan.batches is not all_batches:
        plan = c._lut_plan = _Plan(c, all_batches)
    return plan


# ---- the evaluator --------------------------------------------------------------------


def run_all(c: Circuit) -> None:
    plan = _plan(c)
    states = c._pins.states
    values = plan.table[plan.rows(states)]  # (all computed before anything is written)
    if c._settling:
        _settle(c, plan, values)
        return
    for batch, outs in batches._results(c, plan.other):
        for idx, v in zip(batch.outs, outs):
            states[idx] = v
    states[plan.out] = values


def run_all_tracked(c: Circuit) -> bool:
    plan = _plan(c)
    states = c._pins.states
    values = plan.table[plan.rows(states)]
    if c._settling:
        return _settle(c, plan, values)
    moved = False
    for batch, outs in batches._results(c, plan.other):
        for idx, v in zip(batch.outs, outs):
            moved = moved or not np.array_equal(states[idx], v)
            states[idx] = v
    moved = moved or not np.array_equal(states[plan.out], values)
    states[plan.out] = values
    return moved


def run_some(c: Circuit, dirty: np.ndarray, bid: np.ndarray) -> list[np.ndarray]:
    plan = _plan(c)
    states = c._pins.states
    count = plan.unit_count[dirty]
    tab = count > 0
    if not tab.any():  # (no units: often nothing at all)
        return batches.some(c, dirty, bid, plan.other_numbered) if plan.other else []
    units = _segments(plan.unit_first[dirty[tab]], count[tab])
    values = plan.table[plan.rows_of(states, units)]
    # the rest (kinds that run a kind at a time) compute and write; then the units
    moved = batches.some(c, dirty[~tab], bid[~tab], plan.other_numbered) if plan.other else []
    out = plan.out[units]
    diff = states[out] != values
    if diff.any():
        out = out[diff]
        moved.append(out)
        states[out] = values[diff]
    return moved


def _settle(c: Circuit, plan: _Plan, values: np.ndarray) -> bool:
    """run_all while parts settle: each takes its new outputs only half the time, the
    random draws going batch by batch in the circuit's batch order (as batches.py
    does), so both evaluators settle alike. Did any output change?"""
    states = c._pins.states
    others = {id(batch): outs for batch, outs in batches._results(c, plan.other)}
    moved = False
    for b, batch in enumerate(plan.batches):
        span = plan.spans.get(b)
        if span is None and id(batch) not in others:
            continue  # (faulted, or its eval just failed: no draw, as in batches.py)
        keep = (c._settle[batch.slots] <= 0) | (c.rng.random(len(batch.parts)) < 0.5)
        if span is None:
            pairs = [(idx[keep], v[keep]) for idx, v in zip(batch.outs, others[id(batch)])]
        else:
            u0, n, n_out = span
            k = np.repeat(keep, n_out)
            pairs = [(plan.out[u0 : u0 + n * n_out][k], values[u0 : u0 + n * n_out][k])]
        for idx, v in pairs:
            moved = moved or not np.array_equal(states[idx], v)
            states[idx] = v
    return moved
