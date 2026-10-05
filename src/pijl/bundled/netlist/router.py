"""A small maze router on the grid: one net at a time, each net grown as a tree.

No pyglet here. Everything is in grid cells (world / step, rounded).

A net is some pins to join. Its tree starts from its root pin, from the wires it
has already (seeds), or from a spine (a bus: one straight free-floating wire laid
first, see _lay_spine). Every pin still to join (a sink) is routed to the nearest
point of the tree so far, so a fan-out comes out as a trunk with branches, each
branch a wire of its own joined to the tree at a junction.

What a path may do:
- Never enter a part's body (its outline included), except at this net's own pins.
- Never run along another net's wire, turn on it or end on it. Crossing one at a
  right angle, straight through, is fine (wires only join where they're told to).
- Each bend costs extra, so paths come out as L's and Z's, not staircases. Passing
  right in front of another part's pin costs a little (it reads as wired to it).

Seeds and spines go down first (a spine may cross wires too), then nets are routed
shortest first, sinks nearest first. Nothing that's routed is ripped up again: a sink with no way through comes
back as unrouted.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

Cell = tuple[int, int]
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
H, V = 1, 2  # occupancy: a wire runs across the cell horizontally / vertically

STEP, BEND, CROSS, STUB = 1, 4, 2, 6  # costs
MARGIN = 12  # cells around a search's start and targets it may wander into
MAX_EXPANSIONS = 400_000
SPINE_REACH = 40  # rows (or columns) either side of the middle a spine may move to
SPINE = "spine"  # Route.seed of a sink that taps a bus's spine


@dataclass
class Pin:
    cell: Cell
    out: int  # the direction a wire leaves it in: +1 right (outputs), -1 left (inputs)


@dataclass
class Net:
    """`root`: the pin the tree starts from (None: it starts from the seeds or the
    spine, or failing those from a sink: see NetResult.rooted). `seeds`: wires the
    net has already, as (key, polyline of cells). `spine`: lay a bus spine first."""

    sinks: list[Pin]
    root: Pin | None = None
    seeds: list[tuple[Any, list[Cell]]] = field(default_factory=list)
    spine: bool = False


@dataclass
class Route:
    """How a sink was reached. `path`: cells from where it joins the net to the
    sink's pin. It joins: the root pin (`joins` and `seed` None), the wire of route
    `joins` (an index into this net's routes), or the seed wire with key `seed`
    (SPINE: the bus's spine)."""

    sink: int
    path: list[Cell] | None  # None: unroutable
    joins: int | None = None
    seed: Any = None


@dataclass
class NetResult:
    routes: list[Route]
    spine: tuple[Cell, Cell] | None = None  # the bus's spine, end to end
    rooted: int | None = None  # a sink that became the root (gets no wire of its own)


@dataclass
class Board:
    """Obstacles: `blocked` cells (part bodies and outlines, pins included), `stubs`
    (the cell in front of each pin) and the cells wires run through (`used`)."""

    blocked: set[Cell] = field(default_factory=set)
    stubs: set[Cell] = field(default_factory=set)
    used: dict[Cell, tuple[int, int]] = field(default_factory=dict)  # cell -> (owner, H|V)

    def add_rect(self, x0: int, y0: int, x1: int, y1: int) -> None:
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                self.blocked.add((x, y))

    def add_stub(self, pin: Pin) -> None:
        x, y = pin.cell
        self.stubs.add((x + pin.out, y))

    def add_wire(self, owner: int, cells: list[Cell]) -> None:
        """A polyline (corners, or every cell: either works) owned by `owner`."""
        last = len(cells) - 1
        for i, (a, b) in enumerate(zip(cells, cells[1:])):
            if a[0] != b[0] and a[1] != b[1]:
                continue  # (diagonal: not on the grid's lines, not ours to avoid)
            axis = _axis(a, b)
            for c in _between(a, b):
                self._use(c, owner, axis)
            # a corner, or an end: nothing may cross there
            if i == 0:
                self._use(a, owner, H | V)
            if i == last - 1 or (i + 2 <= last and _axis(b, cells[i + 2]) != axis):
                self._use(b, owner, H | V)
        if len(cells) == 1:
            self._use(cells[0], owner, H | V)

    def _use(self, c: Cell, owner: int, axis: int) -> None:
        had = self.used.get(c)
        if had is None:
            self.used[c] = (owner, axis)
        elif had[0] == owner:
            self.used[c] = (owner, had[1] | axis)
        else:
            self.used[c] = (-1, H | V)  # (two others: crossed already)


def _axis(a: Cell, b: Cell) -> int:
    return H if a[1] == b[1] else V


def _between(a: Cell, b: Cell) -> list[Cell]:
    (x0, y0), (x1, y1) = a, b
    if y0 == y1:
        s = 1 if x1 >= x0 else -1
        return [(x, y0) for x in range(x0, x1 + s, s)]
    s = 1 if y1 >= y0 else -1
    return [(x0, y) for y in range(y0, y1 + s, s)]


def _cells(polyline: list[Cell]) -> list[Cell]:
    """Every cell along a polyline's straight runs."""
    out: list[Cell] = []
    for a, b in zip(polyline, polyline[1:]):
        if a[0] == b[0] or a[1] == b[1]:
            out += _between(a, b)
    return out or list(polyline[:1])


def route(board: Board, nets: list[Net], first_owner: int = 1_000_000) -> list[NetResult]:
    """Routes for every net, in the order given. Each net's routes are in the order
    they were made (a route only joins routes before it). What's routed becomes an
    obstacle for the nets after it."""
    owners = [first_owner + i for i in range(len(nets))]
    results = [NetResult([]) for _ in nets]
    for net, owner in zip(nets, owners):  # (all seeds first: everyone must avoid them)
        for _, cells in net.seeds:
            board.add_wire(owner, cells)
    order = sorted(range(len(nets)), key=lambda i: _span(nets[i]))
    for i in order:
        if nets[i].spine:
            results[i].spine = _lay_spine(board, nets[i], owners[i])
    for i in order:
        _route_net(board, nets[i], owners[i], results[i])
    return results


def _span(net: Net) -> int:
    cells = [s.cell for s in net.sinks] + ([net.root.cell] if net.root else [])
    cells += [c for _, poly in net.seeds for c in poly]
    xs, ys = [c[0] for c in cells], [c[1] for c in cells]
    return max(xs) - min(xs) + max(ys) - min(ys)


def _lay_spine(board: Board, net: Net, owner: int) -> tuple[Cell, Cell] | None:
    """A straight line for a bus's pins to tap: across them (or down, if they're
    spread more up and down than across), on the row (column) nearest their middle,
    each wire it has to cross costing a little more distance. None if there's no
    such line nearby."""
    pins = net.sinks + ([net.root] if net.root else [])
    taps = [(p.cell[0] + p.out, p.cell[1]) for p in pins]  # (where each wire leaves)
    own = {p.cell for p in pins} | set(taps)
    xs, ys = [t[0] for t in taps], [t[1] for t in taps]
    across = max(xs) - min(xs) >= max(ys) - min(ys)
    for horizontal in (across, not across):
        lo, hi = (min(xs), max(xs)) if horizontal else (min(ys), max(ys))
        along = sorted(ys if horizontal else xs)
        mid = along[len(along) // 2]
        best = None
        for at in range(mid - SPINE_REACH, mid + SPINE_REACH + 1):
            line = [(v, at) if horizontal else (at, v) for v in range(lo, hi + 1)]
            crosses = _spine_crossings(board, line, own, V if horizontal else H)
            if crosses is not None:
                cost = abs(at - mid) + crosses * CROSS
                if best is None or cost < best[0]:
                    best = (cost, line)
        if best is not None:
            ends = (best[1][0], best[1][-1])
            board.add_wire(owner, list(ends))
            return ends
    return None


def _spine_crossings(board: Board, line: list[Cell], own: set[Cell], across: int) -> int | None:
    """How many wires a spine along `line` crosses (wires running `across` it, straight
    through), or None if it can't go there: a part, someone else's pin stub, a wire
    along it, a corner or an end, or any wire under the spine's own ends."""
    n = 0
    for i, c in enumerate(line):
        if c in board.blocked or (c in board.stubs and c not in own):
            return None
        u = board.used.get(c)
        if u is not None:
            if u[1] != across or i in (0, len(line) - 1):
                return None
            n += 1
    return n


def _route_net(board: Board, net: Net, owner: int, result: NetResult) -> None:
    own = {s.cell for s in net.sinks} | ({net.root.cell} if net.root else set())
    # where a sink may join the net: cell -> ("root",) / ("route", n) / ("seed", key)
    tree: dict[Cell, tuple] = {}
    # pin cells in the tree: (the direction a wire must arrive in, the pin's side)
    arrive: dict[Cell, tuple[Cell, int]] = {}
    for key, poly in net.seeds:
        for c in _cells(poly):
            tree.setdefault(c, ("seed", key))
    if result.spine is not None:
        for c in _between(*result.spine):
            tree.setdefault(c, ("seed", SPINE))
    sinks = list(range(len(net.sinks)))
    root = net.root
    if root is None and not tree:
        # nothing to start from: an output if there is one, else the first pin
        k = next((k for k in sinks if net.sinks[k].out > 0), 0)
        result.rooted = k
        sinks.remove(k)
        root = net.sinks[k]
    if root is not None:
        tree[root.cell] = ("root",)
        arrive[root.cell] = ((-root.out, 0), root.out)
    xs = [c[0] for c in tree]
    ys = [c[1] for c in tree]
    box = (min(xs), max(xs), min(ys), max(ys))
    # nearest first; with a root pin, the pins it can take a wire from go before the
    # ones of its own side (an output can't be wired to an output: those need wires
    # to join, which the others lay)
    side = root.out if root is not None else 0
    sinks.sort(key=lambda k: (net.sinks[k].out == side, _to_box(net.sinks[k].cell, box)))
    for k in sinks:
        sink = net.sinks[k]
        if len(tree) == len(arrive) and all(a[1] == sink.out for a in arrive.values()):
            # only pins of its own kind to join (an input can't feed an input): no
            # search will find a way, and a failing one is the slow kind
            result.routes.append(Route(k, None))
            continue
        for margin in (MARGIN, MARGIN * 4):  # (close by first: it's much cheaper)
            path = _search(board, sink, tree, arrive, own, owner, margin)
            if path is not None:
                break
        if path is None:
            result.routes.append(Route(k, None))
            continue
        tag = tree[path[0]]
        r = Route(k, path)
        if tag[0] == "route":
            r.joins = tag[1]
        elif tag[0] == "seed":
            r.seed = tag[1]
        result.routes.append(r)
        n = len(result.routes) - 1
        for c in path[1:-1]:  # (not the pin it ends on, not the junction it starts at)
            tree.setdefault(c, ("route", n))
        board.add_wire(owner, path)


def _to_box(c: Cell, box) -> int:
    x, y = c
    x0, x1, y0, y1 = box
    return max(x0 - x, 0, x - x1) + max(y0 - y, 0, y - y1)


def _search(board, sink: Pin, tree, arrive, own, owner, margin: int) -> list[Cell] | None:
    """A* from the sink's pin to any cell of the tree, within `margin` cells of the
    box around both. The path comes back from the tree to the sink. (The first step
    leaves the pin outward: its outline is blocked.)"""
    xs = [c[0] for c in tree]
    ys = [c[1] for c in tree]
    tbox = (min(xs), max(xs), min(ys), max(ys))
    bx0 = min(tbox[0], sink.cell[0]) - margin
    bx1 = max(tbox[1], sink.cell[0]) + margin
    by0 = min(tbox[2], sink.cell[1]) - margin
    by1 = max(tbox[3], sink.cell[1]) + margin

    blocked, stubs, used = board.blocked, board.stubs, board.used
    start_dir = (sink.out, 0)
    start = (sink.cell, start_dir)
    best = {start: 0}
    came: dict = {}
    heap = [(_to_box(sink.cell, tbox) * STEP, 0, 0, sink.cell, start_dir)]
    tick = 0
    expansions = 0
    while heap:
        _, g, _, cell, d = heapq.heappop(heap)
        if best.get((cell, d), 1 << 60) < g:
            continue
        if cell in tree and cell != sink.cell:
            if cell in arrive and (d != arrive[cell][0] or arrive[cell][1] == sink.out):
                continue  # (into a pin only from in front of it, and output to input)
            return _unwind(came, (cell, d))
        expansions += 1
        if expansions > MAX_EXPANSIONS:
            return None
        crossing = cell not in own and cell in used and used[cell][0] != owner
        for nd in DIRS:
            if nd[0] == -d[0] and nd[1] == -d[1]:
                continue  # (no U-turns)
            if crossing and nd != d:
                continue  # (straight through a wire we're crossing)
            if cell == sink.cell and nd != start_dir:
                continue
            n = (cell[0] + nd[0], cell[1] + nd[1])
            if not (bx0 <= n[0] <= bx1 and by0 <= n[1] <= by1):
                continue
            if n in blocked and n not in own:
                continue
            if n in own and n not in tree:
                continue  # (another of this net's sinks: not a way through)
            cost = STEP + (BEND if nd != d else 0)
            u = used.get(n)
            if u is not None and u[0] != owner and n not in own:
                # someone else's wire: cross it at a right angle, or not at all
                across = V if nd[1] == 0 else H  # (moving across: it must run the other way)
                if u[1] != across or n in tree:
                    continue
                cost += CROSS
            if n in stubs and n not in tree:
                cost += STUB
            ng = g + cost
            key = (n, nd)
            if ng < best.get(key, 1 << 60):
                best[key] = ng
                came[key] = (cell, d)
                tick += 1
                heapq.heappush(heap, (ng + _to_box(n, tbox) * STEP, ng, tick, n, nd))
    return None


def _unwind(came: dict, state) -> list[Cell]:
    cells = [state[0]]
    while state in came:
        state = came[state]
        cells.append(state[0])
    return cells  # (tree end first: the search ran from the sink)


def corners(path: list[Cell]) -> list[Cell]:
    """The path's inner cells where it turns (what a wire's bends are)."""
    out = []
    for a, b, c in zip(path, path[1:], path[2:]):
        if _axis(a, b) != _axis(b, c):
            out.append(b)
    return out
