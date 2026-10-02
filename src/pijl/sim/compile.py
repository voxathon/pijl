"""Compiled macros (engine option compile=mixed / zero; off: macros are flattened).

A macro instance on the board normally becomes its body's gates, flattened, each
one a tick of delay. Compiled, the instance runs as one *program* instead: the whole
body (nested macros included) as lookup-table units (see sim/lut.py) in levels, so
a pass is one batch of lookups per level, for every instance of that macro at once.
The body's hidden parts and wires are still built (the "look inside" view shows
them); the engine just doesn't run them, and the program's values are copied into
them when something looks (sync).

Loops (latches) can't be put in levels as they are, so a few of the body's nets are
cut: *state* nets. A pass reads the macro's inputs and the state nets' values from
before, evaluates every level, and commits the state nets' new values at the end.

  mixed: one pass per tick. Logic without loops takes a tick in all, however deep;
         every trip round a loop takes another. An instance whose state changed runs
         again next tick, inputs changed or not.
  zero:  passes until the state stops changing, within the tick, at most
         `cap` of them; whatever is still changing then goes X (an oscillator, a
         race). The whole macro takes one tick, loops included.

The macro's own pins: its inputs read the nets outside, its outputs drive them (an
eval does the same), so the macro takes a tick from input to output like a gate.

Not compiled (stays flattened): a macro with a part inside that can't be tabulated
(not pure, more than 4 inputs, ...), or with something inside driving one of its
input lines (inside and outside are one net only when flattened).

Nothing about compiling is saved: a program is made from the instance when it's
placed, so every macro works under every setting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..logic import CODE, X, Z
from . import lut
from .circuit import _priority

if TYPE_CHECKING:
    from .circuit import Circuit

ZERO_MODE_PASSES = 4  # zero mode: passes per tick, at most, per state net (+ a few)


# ---- programs -------------------------------------------------------------------------


class _Level:
    __slots__ = ("units", "base", "ins", "solo_nets", "solo_units", "multi")

    def __init__(self, units, base, ins, solo_nets, solo_units, multi) -> None:
        self.units = units  # unit numbers
        self.base = base  # (units, 1): where each one's rows start in the table
        self.ins = ins  # per input i: the local nets it reads (the zero net: none)
        self.solo_nets, self.solo_units = solo_nets, solo_units  # nets one unit drives
        self.multi = multi  # (nets, strong groups, weak groups) or None (see _resolve)


class Program:
    """A macro's body, compiled (see the module docstring). Local nets 0..m-1; m is
    the zero net, always Z (what unconnected inputs read). Values: V[net, column],
    a column per instance."""

    def __init__(self) -> None:
        self.m = 0
        self.table = np.zeros(0, CODE)
        self.n_units = 0
        self.levels: list[_Level] = []
        self.in_nets = np.zeros(0, np.intp)  # per macro input: the net it drives
        self.in_used = np.zeros(0, np.intp)  # the inputs that drive one (not the zero net)
        self.out_nets = np.zeros(0, np.intp)  # per macro output: the net it reads
        self.state = np.zeros(0, np.intp)  # cut nets: committed at the end of a pass
        self.state_resolve = None  # (how: see _resolve)
        self.init = np.zeros(1, CODE)  # every net's power-on value (m + 1)
        self.pin_rel = np.zeros(0, np.intp)  # body pins (relative to the instance's first) ...
        self.pin_net = np.zeros(0, np.intp)  # ... and their nets: for sync
        self.net_rel = np.zeros(0, np.intp)  # per net: one of its pins (for the global net)

    def pass_(self, V: np.ndarray, D: np.ndarray) -> np.ndarray:
        """One pass over every column of V (inputs already in; D: a row per unit plus
        one that's always Z). Returns, per column, whether a state net changed."""
        table = self.table
        for lv in self.levels:
            local = None
            for i, nets in enumerate(lv.ins):
                bits = V[nets] << (2 * i)
                local = bits if local is None else local | bits
            if local is None:  # (parts without inputs: constants)
                rows = np.broadcast_to(lv.base, (len(lv.units), V.shape[1]))
            else:
                rows = lv.base + local
            D[lv.units] = table[rows]
            if len(lv.solo_nets):
                V[lv.solo_nets] = D[lv.solo_units]
            if lv.multi is not None:
                V[lv.multi[0]] = _resolve(D, lv.multi)
        if not len(self.state):
            return np.zeros(V.shape[1], bool)
        new = _resolve(D, self.state_resolve)
        changed = (V[self.state] != new).any(axis=0)
        V[self.state] = new
        return changed


def _resolve(D: np.ndarray, how) -> np.ndarray:
    """Some nets' values from their drivers' (rows of D): OR of the strong ones (a net
    without any has D's always-Z row), and where that's Z, of the top priority weak
    ones (only nets that have some: w_at)."""
    _nets, (s_units, s_starts), (w_units, w_starts, w_at) = how
    value = np.bitwise_or.reduceat(D[s_units], s_starts, axis=0)
    if len(w_units):
        weak = np.bitwise_or.reduceat(D[w_units], w_starts, axis=0)
        sub = value[w_at]
        value[w_at] = np.where(sub == Z, weak, sub)
    return value


def build(c: Circuit, inst: int) -> Program | None:
    """The program for this macro instance's body, or None if it can't be compiled.
    Pins are kept relative to the instance's first pin, so the program serves every
    instance stamped from the same blueprint."""
    store, ws = c._pins, c._wire_slots
    tree = c._tree(inst)
    body = [s for s in tree if s != inst]
    p0 = int(c._pin0[inst])
    t_inst = c._types[c._type_id[inst]]
    n = store.n
    in_body = np.zeros(c._n_part_slots, bool)
    in_body[body] = True
    pins = np.flatnonzero(in_body[store.part[:n]] & store.alive[:n])
    wires = np.concatenate([np.asarray(c._inner_wires[s], np.intp) for s in tree if s in c._inner_wires] or [np.zeros(0, np.intp)])
    links = [c._links[s] for s in body if s in c._links]

    # local connectivity: pins and wires of the body, joined by wire ends and links
    P, W = len(pins), len(wires)
    pin_node = {int(p): i for i, p in enumerate(pins.tolist())}
    wire_node = {int(w): P + i for i, w in enumerate(wires.tolist())}
    a, b = [], []
    for w in wires.tolist():
        for e in range(2):
            if ws.end_is_wire[w, e]:
                other = wire_node.get(int(ws.end_slot[w, e]))
            else:
                other = pin_node.get(int(ws.end_slot[w, e]))
            if other is not None:
                a.append(wire_node[w])
                b.append(other)
    for pairs in links:
        for x, y in np.asarray(pairs).reshape(-1, 2).tolist():
            if x in pin_node and y in pin_node:
                a.append(pin_node[x])
                b.append(pin_node[y])
    member = np.zeros(P + W, bool)
    member[a] = member[b] = True
    graph = coo_matrix((np.ones(len(a), bool), (a, b)), shape=(P + W, P + W))
    _, label = connected_components(graph, directed=False)
    _, local = np.unique(label[member], return_inverse=True)
    net_of_node = np.full(P + W, -1, np.intp)
    net_of_node[member] = local
    m = int(local.max()) + 1 if local.size else 0
    ZN = m  # the zero net

    def net(pin: int) -> int:
        v = net_of_node[pin_node[pin]]
        return ZN if v < 0 else int(v)

    # the parts: ports, containers (nested macros), and units
    by_uid = {int(c._uid[s]): s for s in body if c._owner[s] == inst}
    prog = Program()
    tables, starts = [], {}
    at = 0
    units = []  # (part, base, input nets, driven net, weak, priority)
    for s in body:
        t = c._types[c._type_id[s]]
        if t.port or getattr(t, "body", None) is not None:
            continue
        table = lut.tabulate(t)
        if table is None:
            return None
        if t not in starts:
            starts[t] = at
            tables.append(table.reshape(-1))
            at += table.size
        k, rows = len(t.ins), 4 ** len(t.ins)
        drives = c._drive_slots(s)
        for j in range(len(t.outs)):
            col = table[j].reshape((4,) * k) if k else table[j]
            ins = []
            for i in range(k):
                axis = k - 1 - i  # (row = in0 + 4 * in1 + ...: input 0 is the last axis)
                if k and np.all(col == col.take([0], axis=axis)):
                    ins.append(ZN)  # (doesn't matter: read nothing, row stays valid)
                else:
                    ins.append(net(c._in_pin(s, i)))
            d = int(drives[j])
            units.append((s, starts[t] + j * rows, ins, net(d), bool(store.weak[d]), _priority(c._handle(s)) if store.weak[d] else 0))
    prog.table = np.concatenate(tables) if tables else np.zeros(0, CODE)
    prog.n_units = U = len(units)

    in_nets = [net(c._out_pin(by_uid[uid], 0)) for uid in t_inst.in_ids]
    out_nets = [net(c._in_pin(by_uid[uid], 0)) for uid in t_inst.out_ids]
    driven_nets = {d for _, _, _, d, _, _ in units}
    if any(n_ != ZN and n_ in driven_nets for n_ in in_nets) or len(set(x for x in in_nets if x != ZN)) < len([x for x in in_nets if x != ZN]):
        return None  # (something inside drives an input line; or two inputs are one net)
    prog.in_nets = np.array(in_nets, np.intp)
    prog.in_used = np.flatnonzero(prog.in_nets != ZN)  # (an input wired to nothing inside: skip it)
    prog.out_nets = np.array(out_nets, np.intp)

    # drivers per net: strong, and the top-priority weak ones
    strong: dict[int, list[int]] = {}
    weak: dict[int, list[tuple[int, int]]] = {}
    for u, (_, _, _, d, is_weak, prio) in enumerate(units):
        if d == ZN:
            continue
        (weak.setdefault(d, []).append((prio, u)) if is_weak else strong.setdefault(d, []).append(u))
    for d, lst in weak.items():
        top = max(p for p, _ in lst)
        weak[d] = [(p, u) for p, u in lst if p == top]
    drivers_of = {d: strong.get(d, []) + [u for _, u in weak.get(d, [])] for d in set(strong) | set(weak)}
    readers_of: dict[int, list[int]] = {}
    for u, (_, _, ins, _, _, _) in enumerate(units):
        for x in set(ins):
            if x != ZN:
                readers_of.setdefault(x, []).append(u)

    # Cut loops. First every pair of units that read each other (a latch's cross-coupled
    # gates): both their nets are state, so a chain of latches is cut at every latch,
    # not just once somewhere (a depth-first walk alone may cut each latch on the inside
    # and leave a path through all of them: thousands of levels for a register).
    state: set[int] = set()
    for u, (_, _, ins, d, _, _) in enumerate(units):
        if d == ZN:
            continue
        for v in readers_of.get(d, ()):
            if v != u and units[v][3] != ZN and units[v][3] in ins:
                state.add(d)
                state.add(units[v][3])
    # Then the rest: a depth-first walk over unit -> net -> unit; a net on a back edge
    # is state.
    color = [0] * U  # 0 new, 1 on the stack, 2 done
    for root in range(U):
        if color[root]:
            continue
        stack = [(root, iter(_next_edges(root, units, readers_of)))]
        color[root] = 1
        while stack:
            u, it = stack[-1]
            for d, v in it:
                if d in state:
                    continue
                if color[v] == 1:
                    state.add(d)  # (closes a loop)
                elif color[v] == 0:
                    color[v] = 1
                    stack.append((v, iter(_next_edges(v, units, readers_of))))
                    break
            else:
                color[u] = 2
                stack.pop()

    # levels: a unit after every unit driving a (non-state) net it reads
    level = [-1] * U
    ready = {}  # net -> the level after which it's resolved
    order = _topo(U, units, drivers_of, state, ZN)
    for u in order:
        lv = 0
        for x in units[u][2]:
            if x != ZN and x not in state and x in drivers_of:
                lv = max(lv, max(level[w] for w in drivers_of[x]) + 1)
        level[u] = lv
    for d, drv in drivers_of.items():
        if d not in state:
            ready[d] = max(level[w] for w in drv)

    base = np.array([u[1] for u in units], np.intp)
    for L in range(max(level) + 1 if units else 0):
        us = [u for u in range(U) if level[u] == L]
        k = max((len(units[u][2]) for u in us), default=0)
        ins = [np.array([units[u][2][i] if i < len(units[u][2]) else ZN for u in us], np.intp) for i in range(k)]
        nets = sorted(d for d, r in ready.items() if r == L)
        solo = [d for d in nets if len(drivers_of[d]) == 1 and d in strong]
        multi = [d for d in nets if d not in solo]
        prog.levels.append(
            _Level(
                np.array(us, np.intp),
                base[us][:, None],
                ins,
                np.array(solo, np.intp),
                np.array([strong[d][0] for d in solo], np.intp),
                _resolver(multi, strong, weak, U) if multi else None,
            )
        )
    prog.state = np.array(sorted(state), np.intp)
    prog.state_resolve = _resolver(sorted(state), strong, weak, U)

    prog.m = m
    init = np.full(m + 1, Z, CODE)
    init[list(drivers_of)] = X  # driven nets power up X, like gate outputs
    init[ZN] = Z
    prog.init = init
    rel = pins - p0
    prog.pin_rel = rel
    prog.pin_net = np.array([net(int(p)) for p in pins.tolist()], np.intp)
    first = np.full(m, -1, np.intp)
    for p, x in zip(rel.tolist(), prog.pin_net.tolist()):
        if x != ZN and first[x] < 0:
            first[x] = p
    prog.net_rel = first
    return prog


def _next_edges(u, units, readers_of):
    d = units[u][3]
    for v in readers_of.get(d, ()):
        yield d, v


def _topo(U, units, drivers_of, state, ZN) -> list[int]:
    """Units in an order where every unit comes after the drivers of the (non-state)
    nets it reads."""
    deps = [set() for _ in range(U)]
    users: list[list[int]] = [[] for _ in range(U)]
    for u in range(U):
        for x in units[u][2]:
            if x != ZN and x not in state and x in drivers_of:
                for w in drivers_of[x]:
                    if w not in deps[u]:
                        deps[u].add(w)
                        users[w].append(u)
    left = [len(d) for d in deps]
    todo = [u for u in range(U) if not left[u]]
    out = []
    while todo:
        u = todo.pop()
        out.append(u)
        for v in users[u]:
            left[v] -= 1
            if not left[v]:
                todo.append(v)
    assert len(out) == U, "cut left a loop"
    return out


def _resolver(nets, strong, weak, zero_unit: int):
    """What _resolve needs for these nets. `zero_unit`: D's always-Z row, the strong
    "driver" of a net that only has weak ones (reduceat can't do empty groups)."""
    s_units, s_starts, w_units, w_starts, w_at = [], [], [], [], []
    for i, d in enumerate(nets):
        s_starts.append(len(s_units))
        s_units += strong.get(d) or [zero_unit]
        if d in weak:
            w_at.append(i)
            w_starts.append(len(w_units))
            w_units += [u for _, u in weak[d]]
    arr = lambda x: np.array(x, np.intp)  # noqa: E731
    return arr(nets), (arr(s_units), arr(s_starts)), (arr(w_units), arr(w_starts), arr(w_at))


# ---- in the circuit -------------------------------------------------------------------
#
# Bound by Circuit._bind, from the functions below for the config's compile option:
#   choose(c)            at a net rebuild, first: which board macros run compiled
#                        (sets c._in_compiled, c._compiled_out, c._compiled_links)
#   attach(c, net_of)    at its end: the instances' groups (values kept by slot)
#   run_all(c)           every compiled instance takes its tick (see the docstring)
#   run_some(c, dirty)   just those whose inputs changed (dirty part slots), plus
#                        the ones still settling (mixed: state changed last tick)
#   sync(c)              copy the programs' values into the hidden pins and nets
# run_*: (output pins that changed, did any state change?).


class _Group:
    """The instances of one program, a column each."""

    def __init__(self, prog: Program, slots: np.ndarray, c: Circuit, net_of: np.ndarray, saved: dict) -> None:
        self.prog = prog
        self.slots = slots
        C = len(slots)
        p0 = c._pin0[slots].astype(np.intp)
        n_in, n_out = len(prog.in_nets), len(prog.out_nets)
        self.in_pins = p0[None, :] + prog.in_used[:, None]  # (just the ones used)
        self.in_nets = prog.in_nets[prog.in_used]
        self.out_pins = p0[None, :] + n_in + np.arange(n_out)[:, None]
        self.V = np.repeat(prog.init[:, None], C, axis=1)
        self.D = np.zeros((prog.n_units + 1, C), CODE)  # (+ the always-Z row)
        fresh = []
        for col, s in enumerate(slots.tolist()):
            old = saved.get(s)
            if old is not None and len(old) == len(prog.init):
                self.V[:, col] = old
            else:
                fresh.append(col)
        if fresh and c.settle_ticks and len(prog.state):  # power-on noise: latches pick a side
            self.V[np.ix_(prog.state, fresh)] = c.rng.integers(1, 3, (len(prog.state), len(fresh)), dtype=CODE)
        self.pending = np.ones(C, bool)  # (run every one once)
        self.col_of = np.full(c._n_part_slots, -1, np.intp)
        self.col_of[slots] = np.arange(C)
        self.hidden = p0[None, :] + prog.pin_rel[:, None]
        rel = prog.net_rel
        has = rel >= 0
        self.global_nets = np.full((len(rel), C), -1, np.intp)
        self.global_nets[has] = net_of[p0[None, :] + rel[has][:, None]]


_MISSING = object()


def choose(c: Circuit) -> None:
    n = c._n_part_slots
    compiled = np.zeros(n, bool)
    groups: dict = {}
    for p in c._parts:
        s = p.slot
        bp = c._bp_of.get(s)
        if bp is None:
            continue
        prog = c._programs.get(bp, _MISSING)
        if prog is _MISSING:
            prog = c._programs[bp] = build(c, s)
        if prog is not None:
            compiled[s] = True
            groups.setdefault(bp, []).append(s)
    # parts inside them: whose outermost owner is one
    top = np.arange(n)
    owner = c._owner[:n].astype(np.intp)
    while True:
        up = owner[top]
        more = up >= 0
        if not more.any():
            break
        top = np.where(more, up, top)
    inner = compiled[top] & (top != np.arange(n))
    c._in_compiled = inner
    c._compiled_groups_spec = [(c._programs[bp], np.array(slots, np.intp)) for bp, slots in groups.items()]
    out = np.zeros(c._pins.n, bool)
    for prog, slots in c._compiled_groups_spec:
        p0 = c._pin0[slots].astype(np.intp)
        n_in = len(prog.in_nets)
        for j in range(len(prog.out_nets)):
            out[p0 + n_in + j] = True
    c._compiled_out = out
    c._compiled_links = compiled


def attach(c: Circuit, net_of: np.ndarray) -> None:
    saved = {}
    for g in c._groups:
        for col, s in enumerate(g.slots.tolist()):
            saved[s] = g.V[:, col].copy()
    c._groups = [_Group(prog, slots, c, net_of, saved) for prog, slots in c._compiled_groups_spec]
    c._compiled_stale = True


def _tick_mixed(prog: Program, V: np.ndarray, D: np.ndarray) -> np.ndarray:
    return prog.pass_(V, D)


def _tick_zero(prog: Program, V: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Passes until the state stops changing. Past the cap, every state value still
    changing goes X, until nothing changes (or the cap again); one more pass then shows
    it at the outputs. Nothing is left for later: returns all False."""
    none = np.zeros(V.shape[1], bool)
    cap = ZERO_MODE_PASSES * len(prog.state) + 2
    for _ in range(cap):
        if not prog.pass_(V, D).any():
            return none
    st = prog.state
    for _ in range(cap):
        before = V[st]
        if not prog.pass_(V, D).any():
            return none
        after = V[st]
        after[after != before] = X
        V[st] = after
    prog.pass_(V, D)
    return none


def _run(c: Circuit, tick, cols_of) -> tuple[list[np.ndarray], bool]:
    states = c._pins.states
    moved: list[np.ndarray] = []
    unstable = False
    for g in c._groups:
        cols = cols_of(g)
        if cols is None:  # every column
            V, D, ins, outs = g.V, g.D, g.in_pins, g.out_pins
        elif not len(cols):
            continue
        else:
            V, D, ins, outs = g.V[:, cols], g.D[:, cols], g.in_pins[:, cols], g.out_pins[:, cols]
        prog = g.prog
        V[g.in_nets] = states[ins]
        changed = tick(prog, V, D)
        new = V[prog.out_nets]
        if cols is None:
            g.pending = changed
        else:
            g.V[:, cols] = V
            g.pending[cols] = changed
        unstable = unstable or bool(changed.any())
        diff = states[outs] != new
        if diff.any():
            idx = outs[diff]
            moved.append(idx)
            states[idx] = new[diff]
    c._compiled_stale = True
    return moved, unstable


def _dirty_cols(dirty: np.ndarray):
    def cols(g: _Group):
        mine = g.col_of[dirty[dirty < len(g.col_of)]]
        mine = mine[mine >= 0]
        if g.pending.any():
            mine = np.union1d(mine, np.flatnonzero(g.pending))
        return np.unique(mine) if len(mine) > 1 else mine

    return cols


def run_all_mixed(c: Circuit):
    return _run(c, _tick_mixed, lambda g: None)


def run_some_mixed(c: Circuit, dirty: np.ndarray):
    return _run(c, _tick_mixed, _dirty_cols(dirty))


def run_all_zero(c: Circuit):
    return _run(c, _tick_zero, lambda g: None)


def run_some_zero(c: Circuit, dirty: np.ndarray):
    return _run(c, _tick_zero, _dirty_cols(dirty))


def sync(c: Circuit) -> None:
    if not c._compiled_stale:
        return
    states = c._pins.states
    for g in c._groups:
        prog = g.prog
        if len(prog.pin_rel):
            states[g.hidden] = g.V[prog.pin_net]
        has = g.global_nets >= 0
        if has.any() and len(c.net_value):
            c.net_value[g.global_nets[has]] = g.V[: prog.m][has]
    c._compiled_stale = False


# compile=off: nothing runs compiled

def choose_off(c: Circuit) -> None:
    pass


def attach_off(c: Circuit, net_of: np.ndarray) -> None:
    pass


def run_all_off(c: Circuit):
    return _NOTHING


def run_some_off(c: Circuit, dirty: np.ndarray):
    return _NOTHING


def sync_off(c: Circuit) -> None:
    pass


_NOTHING: tuple[list, bool] = ([], False)
