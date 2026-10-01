"""Taking snapshots of the board and applying them: undo/redo, copy/paste, loading.

A Snapshot (see pijl/snapshot.py) is plain data -- part kinds, labels, positions,
props, wire endpoints and bends -- keyed by stable uids.

Undo history doesn't keep a snapshot per step: that's the whole board, hundreds of
times over. It keeps one snapshot of what's on screen (History.current) and, per
step, only what changed: uid -> (before, after). Views report which parts and wires
they changed (views.Touched), so recording an edit costs what the edit touched,
not what the board holds.

Undo doesn't rebuild the board either. `restore` diffs a target snapshot against
what's on screen -- only at the uids an undo step names, when it has them -- and
only adds/removes/moves what differs, because creating views (shapes, labels) is
the slow part. A side effect: parts that survive an undo keep their switch states.

Wire uids grow in creation order and a wire is always created after the wires
it attaches to, so iterating wires by uid always visits parents first. (Splicing
keeps that true: the surviving wire is the older one.)
"""

from __future__ import annotations

import copy
import itertools
from operator import eq, itemgetter
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np

from ..sim import Pin, Wire
from ..snapshot import EMPTY, EndRef, PartData, Snapshot, WireData
from .views import PartView, Touched, WireView, paused_gc

if TYPE_CHECKING:
    from .editor import Editor

__all__ = [
    "EMPTY",
    "STAMPS",
    "Change",
    "History",
    "Snapshot",
    "capture",
    "changes",
    "instantiate",
    "instantiate_keyed",
    "internal_wires",
    "restore",
]

# Undo step ids, shared by every History (and the library's, see Editor._undo): so
# they grow across boards and the two timelines can be told apart by which came last.
STAMPS = itertools.count(1)

# One undo step: for parts, wires and wire colors, what the uids it touches looked like
# before and after, as two dicts (uid -> value). A uid missing from one of them was
# absent then. (Two dicts instead of uid -> (before, after) pairs: a step that adds or
# removes a big batch is one dict of the values, not a pair per uid as well.)
#
# Plus `moves`: groups of uids that only moved, all by the same (dx, dy), stored as
# (uids, dx, dy) instead of their values -- dragging a whole board is one delta, not
# a copy of every part twice (see _compress). Undo takes the delta off again.
Move = tuple[np.ndarray, float, float]
Section = tuple[dict[int, object], dict[int, object], tuple[Move, ...]]
Change = tuple[Section, Section, Section]

MIN_MOVE_GROUP = 8  # fewer than this moved by one delta: kept as plain values

NO_PROPS: dict = {}  # props of a part that has none (shared: never write to it)


def capture(editor: Editor, views: Iterable[PartView] | None = None) -> Snapshot:
    """The whole board, or just `views` plus every wire fully inside that set
    (both ends on those parts, or on wires that are themselves inside). Parts come in
    board order (uid order for `views`, which may be a set: copies made from the
    snapshot are made in its order, which decides their slots and draw order)."""
    views = (
        list(editor.part_views.values())
        if views is None
        else sorted(views, key=lambda v: v.part.uid)
    )
    parts = {v.part.uid: part_data(v) for v in views}
    wires, colors = {}, {}
    for view in internal_wires(editor, views):
        wires[view.wire.uid] = wire_data(view)
        if view.color:
            colors[view.wire.uid] = view.color
    return Snapshot(parts, wires, colors)


def changes(
    editor: Editor, parts: Iterable[int], wires: Iterable[int]
) -> tuple[dict, dict, dict]:
    """What those parts and wires (uids) look like now, for History.record: their
    data, their wire colors -- or None where they're gone (or a wire has no color)."""
    c = editor.circuit
    pd, wd, cd = {}, {}, {}
    for uid in parts:
        part = c.part_by_uid.get(uid)
        view = editor.part_views.get(part) if part is not None else None
        pd[uid] = part_data(view) if view is not None else None
    for uid in wires:
        wire = c.wire_by_uid.get(uid)
        view = editor.wire_views.get(wire) if wire is not None else None
        wd[uid] = wire_data(view) if view is not None else None
        cd[uid] = view.color if view is not None and view.color else None
    return pd, wd, cd


def part_data(view: PartView) -> PartData:
    p = view.part
    return p.kind, p.label, view.x, view.y, copy.deepcopy(p.props) if p.props else NO_PROPS


def wire_data(view: WireView) -> WireData:
    w = view.wire
    return (
        _ref(w.src),
        _ref(w.dst),
        tuple(view.bends),
        None if isinstance(w.src, Pin) else view.src,
        None if isinstance(w.dst, Pin) else view.dst,
    )


def internal_wires(editor: Editor, views: Iterable[PartView]) -> list[WireView]:
    """Wire views whose every end lands on those parts or on other internal wires."""
    c = editor.circuit
    parts = {v.part for v in views}
    if 2 * len(parts) > len(editor.part_views):
        candidates = c.wires  # most of the board: just go through all of them
    else:  # only wires reachable from those parts' pins can qualify
        near = {w for part in parts for pin in part.pins for w in c.ends_on(pin)}
        candidates = sorted(near.union(c.descendants(*near)), key=lambda w: w.uid)
    inside: set[Wire] = set()
    result = []
    for w in candidates:  # creation order: parents first
        if all(e.part in parts if isinstance(e, Pin) else e in inside for e in w.ends):
            inside.add(w)
            result.append(editor.wire_views[w])
    return result


def restore(
    editor: Editor,
    target: Snapshot,
    only: tuple[Iterable[int], Iterable[int]] | None = None,
) -> None:
    """Make the board match `target`, touching only what differs. `only`: the part and
    wire uids to look at (an undo step's); everything else is known to match already."""
    with paused_gc():
        _restore(editor, target, only)


def _restore(
    editor: Editor, target: Snapshot, only: tuple[Iterable[int], Iterable[int]] | None
) -> None:
    c = editor.circuit
    if only is None:
        part_uids = c.part_by_uid.keys() | target.parts.keys()
        wire_uids = c.wire_by_uid.keys() | target.wires.keys()
    else:
        part_uids, wire_uids = set(only[0]), set(only[1])

    def view_of(uid: int) -> PartView | None:
        part = c.part_by_uid.get(uid)
        return editor.part_views[part] if part is not None else None

    # 1. parts that shouldn't exist. Their wires (and branches) go with them; any of
    #    those the target does have get rebuilt in step 4.
    doomed = [
        view
        for uid in part_uids - target.parts.keys()
        if (view := view_of(uid)) is not None
    ]
    wire_uids.update(w.uid for w in editor.remove_parts(doomed))
    # 2. wires that shouldn't exist -- or exist with different endpoints (cut-deletion
    #    splices a branch onto its trunk, re-pointing the trunk's far end). Those are
    #    rebuilt in step 4; so are branches that get removed along with them.
    for uid in sorted(wire_uids):
        wire = c.wire_by_uid.get(uid)
        if wire is None:
            continue  # gone already (or never here)
        data = target.wires.get(uid)
        if data is None or (_ref(wire.src), _ref(wire.dst)) != data[:2]:
            wire_uids.update(w.uid for w in editor.remove_wire(editor.wire_views[wire]))
    # 3. parts: add missing (all at once), update moved/relabeled/re-propped
    moved = set()
    missing = []
    edited: dict[tuple, list] = {}  # (type, setting key) -> [(part, old value)]
    for uid in part_uids & target.parts.keys():
        kind, label, x, y, props = target.parts[uid]
        view = view_of(uid)
        if view is None:
            missing.append((kind, x, y, uid, label, props))
            continue
        if (view.x, view.y) != (x, y):
            view.move_to(x, y)
            moved.add(view.part)
        if view.part.label != label:
            view.part.label = label
            view.refresh_name()
            view.name.move_to(*view.name_pos())
        part = view.part
        if part.props != props:
            was, part.props = part.props, copy.deepcopy(props)
            c.props_changed(part)
            for key in part.type.settings:
                if was.get(key) != props.get(key):
                    edited.setdefault((part.type, key), []).append((part, was.get(key)))
            if was.get("color") != props.get("color"):
                Touched.part(uid)
            else:
                Touched.parts.add(uid)  # (settings don't change colors)
    editor.add_parts(missing)
    for (_t, key), group in edited.items():  # one changed() per kind and setting
        c.settings_changed([p for p, _ in group], key, [o for _, o in group])
    # 4. wires, parents first: add missing (their views all at once), update bends / junction points
    changed: list[WireView] = []
    with editor.wire_batch():
        for uid in sorted(wire_uids & target.wires.keys()):
            src_ref, dst_ref, bends, src_pt, dst_pt = target.wires[uid]
            wire = c.wire_by_uid.get(uid)
            if wire is None:
                src = _resolve(src_ref, c.part_by_uid, c.wire_by_uid)
                dst = _resolve(dst_ref, c.part_by_uid, c.wire_by_uid)
                editor.connect(
                    src,
                    dst,
                    list(bends),
                    src_pt,
                    dst_pt,
                    uid=uid,
                    color=target.wire_colors.get(uid),
                    check=False,
                )
                continue
            view = editor.wire_views[wire]
            view.color = target.wire_colors.get(uid)  # paint() redoes the gradients
            if (
                tuple(view.bends) != bends
                or (src_pt and view.src != src_pt)
                or (dst_pt and view.dst != dst_pt)
            ):
                view.src, view.dst = src_pt or view.src, dst_pt or view.dst
                view.set_bends(list(bends))
                changed.append(view)
    # 5. re-attach ends: wires on moved parts, changed wires, and whatever hangs off them.
    #    Only those: touching every wire made undoing one moved part on a 2000-part
    #    board take ~80 ms.
    editor.refresh_wires([*editor.wires_touching(moved), *changed])


def instantiate(
    editor: Editor, clip: Snapshot, live: bool = True
) -> tuple[list[PartView], list[WireView]]:
    """Add a copy of `clip` at its original coordinates, with fresh uids (for paste).
    `live=False`: the parts are ghosts until the caller opens them (see Circuit.open_part)."""
    parts, wires = instantiate_keyed(editor, clip, live)
    return list(parts.values()), list(wires.values())


def instantiate_keyed(
    editor: Editor, clip: Snapshot, live: bool = True
) -> tuple[dict[int, PartView], dict[int, WireView]]:
    """instantiate(), with the new views keyed by the clip's uids they're copies of."""
    with paused_gc():
        return _instantiate(editor, clip, live)


def _instantiate(
    editor: Editor, clip: Snapshot, live: bool
) -> tuple[dict[int, PartView], dict[int, WireView]]:
    views = editor.add_parts(
        [
            (kind, x, y, None, label, props)
            for kind, label, x, y, props in clip.parts.values()
        ],
        live=live,
    )
    new = dict(zip(clip.parts, views))
    new_parts = {uid: view.part for uid, view in new.items()}
    new_wires: dict[int, Wire] = {}
    with editor.wire_batch():
        for uid in sorted(clip.wires):
            src_ref, dst_ref, bends, src_pt, dst_pt = clip.wires[uid]
            src = _resolve(src_ref, new_parts, new_wires)
            dst = _resolve(dst_ref, new_parts, new_wires)
            new_wires[uid] = editor.connect(
                src,
                dst,
                list(bends),
                src_pt,
                dst_pt,
                color=clip.wire_colors.get(uid),
                check=False,
            )
    if not _colored(clip):
        # The copies are wired only to each other: with no colors among them, paint()
        # would leave every one of them neutral, which is how new views start.
        Touched.paint_parts.difference_update(p.uid for p in new_parts.values())
        Touched.paint_wires.difference_update(w.uid for w in new_wires.values())
    return new, {uid: editor.wire_views[w] for uid, w in new_wires.items()}


def _colored(clip: Snapshot) -> bool:
    """Does anything in `clip` have a color of its own (see paint.py)?"""
    return bool(clip.wire_colors) or any(
        "color" in props for *_, props in clip.parts.values()
    )


class History:
    """Undo/redo as a timeline of changes (see the module docstring).

    `current` is the snapshot of what's on screen, kept up to date in place. `state`
    names the point in the timeline: every new step gets a new one, so "has anything
    changed since it was saved?" is a comparison of two numbers."""

    def __init__(self, initial: Snapshot, limit: int = 500) -> None:
        self.current = Snapshot(
            dict(initial.parts), dict(initial.wires), dict(initial.wire_colors)
        )
        self.undo_stack: list[
            tuple[Change, int, int]
        ] = []  # (change, state before, state after)
        self.redo_stack: list[tuple[Change, int, int]] = []
        self.limit = limit
        self.state = 0

    def _sections(self) -> tuple[dict, dict, dict]:
        return self.current.parts, self.current.wires, self.current.wire_colors

    def _diff(self, parts: dict, wires: dict, colors: dict) -> Change:
        """The entries that differ from `current`: (before, after) per section (no
        moves yet: see _compress)."""
        out = []
        for now, cur in zip((parts, wires, colors), self._sections()):
            before, after = {}, {}
            for uid, new in now.items():
                old = cur.get(uid)
                if old != new:
                    if old is not None:
                        before[uid] = old
                    if new is not None:
                        after[uid] = new
            out.append((before, after, ()))
        return tuple(out)

    def record(
        self, parts: dict, wires: dict, colors: dict, moves: tuple = ((), ())
    ) -> bool:
        """A new step: what these uids look like now (see changes()); None = gone.
        No-op (returns False) if that's what they looked like already.

        `moves`: (part moves, wire moves), each a list of (uids, dx, dy): uids that
        only moved, exactly (see views.Touched.moved), and aren't in parts / wires.
        They're shifted here, without their data being taken again."""
        change = self._diff(parts, wires, colors)
        if _empty(change) and not any(moves):
            return False
        _apply(self._sections(), change, 1)
        change = _compress(change)
        if any(moves):
            change = _add_moves(change, moves, self._sections())
        after = next(STAMPS)
        self.undo_stack.append((change, self.state, after))
        del self.undo_stack[: -self.limit]
        self.redo_stack.clear()
        self.state = after
        return True

    def commit(self, snap: Snapshot) -> bool:
        """Record the board becoming `snap`, compared in full (record() is the fast way)."""
        cur = self.current
        return self.record(
            {u: snap.parts.get(u) for u in cur.parts.keys() | snap.parts.keys()},
            {u: snap.wires.get(u) for u in cur.wires.keys() | snap.wires.keys()},
            {
                u: snap.wire_colors.get(u)
                for u in cur.wire_colors.keys() | snap.wire_colors.keys()
            },
        )

    def amend(self, parts: dict, wires: dict, colors: dict) -> bool:
        """Like record, but folded into the newest step (a run of small tweaks = one undo step)."""
        if not self.undo_stack:
            return self.record(parts, wires, colors)
        change = self._diff(parts, wires, colors)
        if _empty(change):
            return False
        top, before, _ = self.undo_stack[-1]
        # (its moves spelled out as values, while `current` is still where it left off)
        merged = _expand(top, self._sections())
        _apply(self._sections(), change, 1)
        for (into_b, into_a, _), (b, a, _) in zip(merged, change):
            for uid in b.keys() | a.keys():
                # what it was before the merged step: the older step's, if it had it
                had = uid in into_b or uid in into_a
                first = into_b.get(uid) if had else b.get(uid)
                new = a.get(uid)
                into_b.pop(uid, None)
                into_a.pop(uid, None)
                if first != new:
                    if first is not None:
                        into_b[uid] = first
                    if new is not None:
                        into_a[uid] = new
        self.state = next(STAMPS)
        self.undo_stack[-1] = (_compress(merged), before, self.state)
        self.redo_stack.clear()
        return True

    def undo(self) -> Change | None:
        """Step back; returns the change undone (restore() the board at its uids)."""
        if not self.undo_stack:
            return None
        entry = self.undo_stack.pop()
        _apply(self._sections(), entry[0], 0)
        self.redo_stack.append(entry)
        self.state = entry[1]
        return entry[0]

    def redo(self) -> Change | None:
        if not self.redo_stack:
            return None
        entry = self.redo_stack.pop()
        _apply(self._sections(), entry[0], 1)
        self.undo_stack.append(entry)
        self.state = entry[2]
        return entry[0]


def _apply(sections: tuple[dict, dict, dict], change: Change, side: int) -> None:
    """Set every entry of `change` to its before (side 0) or after (side 1) value."""
    sign = 1 if side else -1
    for d, section, shift in zip(sections, change, _SHIFTS):
        to, other = section[side], section[1 - side]
        for uid in other.keys() - to.keys():  # absent on that side
            d.pop(uid, None)
        d.update(to)
        for uids, dx, dy in section[2]:
            for uid in uids.tolist():
                d[uid] = shift(d[uid], sign * dx, sign * dy)


def _add_moves(change: Change, moves: tuple, sections: tuple[dict, dict, dict]) -> Change:
    """Shift `sections` (the board) by these moves, and add them to the step."""
    out = list(change)
    for i, (section_moves, d, shift) in enumerate(zip(moves, sections, _SHIFTS)):
        groups = []
        for uids, dx, dy in section_moves:
            for uid in uids:
                d[uid] = shift(d[uid], dx, dy)
            groups.append((np.array(uids, np.int64), dx, dy))
        if groups:
            before, after, old = out[i]
            out[i] = (before, after, (*old, *groups))
    return tuple(out)


def _empty(change: Change) -> bool:
    return not any(b or a or m for b, a, m in change)


def section_uids(section: Section) -> set[int]:
    uids = section[0].keys() | section[1].keys()
    for moved, _, _ in section[2]:
        uids.update(moved.tolist())
    return uids


# ---- moves -----------------------------------------------------------------------
# What moving by (dx, dy) does to a part's / wire's data, and the delta between two
# values that differ only by a move (None: they differ otherwise). A delta is only
# used if it takes before to after AND back bit for bit: float addition isn't always
# undone by subtraction, and undo must restore exactly what was there.


def _shift_part(d: tuple, dx: float, dy: float) -> tuple:
    kind, label, x, y, props = d
    return kind, label, x + dx, y + dy, props


def _shift_point(p, dx: float, dy: float):
    return None if p is None else (p[0] + dx, p[1] + dy)


def _shift_wire(d: tuple, dx: float, dy: float) -> tuple:
    src, dst, bends, src_pt, dst_pt = d
    return (
        src,
        dst,
        tuple((x + dx, y + dy) for x, y in bends),
        _shift_point(src_pt, dx, dy),
        _shift_point(dst_pt, dx, dy),
    )


def _part_delta(a: tuple, b: tuple) -> tuple[float, float] | None:
    if a[0] != b[0] or a[1] != b[1] or not (a[4] is b[4] or a[4] == b[4]):
        return None
    return b[2] - a[2], b[3] - a[3]


def _wire_delta(a: tuple, b: tuple) -> tuple[float, float] | None:
    if a[:2] != b[:2] or len(a[2]) != len(b[2]) or (a[3] is None) != (b[3] is None):
        return None
    if (a[4] is None) != (b[4] is None):
        return None
    pa = [*a[2], *(p for p in a[3:] if p is not None)]
    pb = [*b[2], *(p for p in b[3:] if p is not None)]
    if not pa:
        return None
    return pb[0][0] - pa[0][0], pb[0][1] - pa[0][1]


_SHIFTS: tuple[Callable, ...] = (_shift_part, _shift_wire, None)
_DELTAS: tuple[Callable | None, ...] = (_part_delta, _wire_delta, None)


def _compress(change: Change) -> Change:
    """Pull entries that only moved, by a delta shared by at least MIN_MOVE_GROUP of
    them, out of a step's values and into its moves."""
    out = []
    for (before, after, moves), shift, delta in zip(change, _SHIFTS, _DELTAS):
        if delta is None or len(after) < MIN_MOVE_GROUP:
            out.append((before, after, moves))
            continue
        groups = (_part_groups if delta is _part_delta else _groups)(before, after, shift, delta)
        big = [(d, uids) for d, uids in groups.items() if len(uids) >= MIN_MOVE_GROUP]
        if not big:
            out.append((before, after, moves))
            continue
        # (made anew, not copied and deleted from: dicts don't shrink, and a step
        # that kept an emptied 100k-entry table would defeat the point)
        gone = {uid for _, uids in big for uid in uids}
        if len(gone) == len(before) == len(after):  # (everything moved: a plain drag)
            before, after = {}, {}
        else:
            before = {u: v for u, v in before.items() if u not in gone}
            after = {u: v for u, v in after.items() if u not in gone}
        moves = (*moves, *((np.array(uids, np.int64), dx, dy) for (dx, dy), uids in big))
        out.append((before, after, moves))
    return tuple(out)


def _groups(before: dict, after: dict, shift, delta) -> dict[tuple, list[int]]:
    """The entries in both that differ only by an exact move, by delta."""
    groups: dict[tuple[float, float], list[int]] = {}
    for uid, new in after.items():
        old = before.get(uid)
        if old is None:
            continue
        d = delta(old, new)
        if (
            d is not None
            and _all_floats(old)
            and _all_floats(new)
            and shift(old, *d) == new
            and shift(new, -d[0], -d[1]) == old
        ):
            groups.setdefault(d, []).append(uid)
    return groups


def _all_floats(wire: tuple) -> bool:
    """Every coordinate in a wire's data is a float (see _part_groups)."""
    return all(
        type(p[0]) is float and type(p[1]) is float
        for p in (*wire[2], *(q for q in wire[3:] if q is not None))
    )


def _part_groups(before: dict, after: dict, shift, delta) -> dict[tuple, list[int]]:
    """_groups for parts, with the work done by C-level maps and arrays (a dragged
    board is all of them)."""
    uids = [u for u in after if u in before]
    n = len(uids)
    if not n:
        return {}
    olds, news = list(map(before.__getitem__, uids)), list(map(after.__getitem__, uids))
    get_x, get_y = itemgetter(2), itemgetter(3)
    x0, y0 = list(map(get_x, olds)), list(map(get_y, olds))
    x1, y1 = list(map(get_x, news)), list(map(get_y, news))
    # Floats only: undo must give back an int as an int, not as 200.0. (So the first
    # move of parts placed at whole numbers is kept as plain values; later ones aren't.)
    same = np.ones(n, bool)
    for col in (x0, x1, y0, y1):
        types = set(map(type, col))
        if float not in types:
            return {}
        if types != {float}:
            same &= np.fromiter((type(v) is float for v in col), bool, n)
    for get in (itemgetter(0), itemgetter(1), itemgetter(4)):  # kind, label, props
        same &= np.fromiter(map(eq, map(get, olds), map(get, news)), bool, n)
    a = np.column_stack((np.array(x0, np.float64), np.array(y0, np.float64)))
    b = np.column_stack((np.array(x1, np.float64), np.array(y1, np.float64)))
    d = b - a
    ok = same & ((a + d) == b).all(1) & ((b - d) == a).all(1)  # exact both ways
    if not ok.any():
        return {}
    idx = np.flatnonzero(ok)
    d, uid_arr = d[idx], np.array(uids, np.int64)[idx]
    if (d == d[0]).all():  # (the usual case: one drag, one delta)
        return {(float(d[0, 0]), float(d[0, 1])): uid_arr.tolist()}
    # several: group by delta (as complex numbers: an exact 1-D sort, unlike axis=0)
    keys, inverse = np.unique(d[:, 0] + 1j * d[:, 1], return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(len(keys) + 1))
    uid_arr = uid_arr[order]
    return {
        (float(keys[k].real), float(keys[k].imag)): uid_arr[bounds[k] : bounds[k + 1]].tolist()
        for k in range(len(keys))
    }


def _expand(change: Change, sections: tuple[dict, dict, dict]) -> Change:
    """A step's moves spelled out as before / after values again; `sections` must be
    the board right after the step (its after side)."""
    out = []
    for (before, after, moves), d, shift in zip(change, sections, _SHIFTS):
        before, after = dict(before), dict(after)
        for uids, dx, dy in moves:
            for uid in uids.tolist():
                after[uid] = now = d[uid]
                before[uid] = shift(now, -dx, -dy)
        out.append((before, after, ()))
    return tuple(out)


def change_uids(change: Change) -> tuple[set[int], set[int]]:
    """The part and wire uids an undo step touches (wire colors count as wires)."""
    parts, wires, colors = change
    return section_uids(parts), section_uids(wires) | section_uids(colors)


def _ref(end) -> EndRef:
    if isinstance(end, Pin):
        return "p", end.part.uid, end.is_input, end.index
    return "w", end.uid


def _resolve(ref: EndRef, parts: dict, wires: dict[int, Wire]):
    """`parts`: uid -> Part."""
    if ref[0] == "p":
        _, part_uid, is_input, index = ref
        part = parts[part_uid]
        return (part.inputs if is_input else part.outputs)[index]
    return wires[ref[1]]
