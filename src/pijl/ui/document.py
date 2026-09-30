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
from typing import TYPE_CHECKING, Iterable

from ..sim import Pin, Wire
from ..snapshot import EMPTY, EndRef, PartData, Snapshot, WireData
from .views import PartView, Touched, WireView, paused_gc

if TYPE_CHECKING:
    from .editor import Editor

__all__ = ["EMPTY", "Change", "History", "Snapshot", "capture", "changes", "instantiate", "internal_wires",
           "restore"]

# One undo step: for parts, wires and wire colors, uid -> (before, after). None = absent.
Change = tuple[dict[int, tuple], dict[int, tuple], dict[int, tuple]]


def capture(editor: Editor, views: Iterable[PartView] | None = None) -> Snapshot:
    """The whole board, or just `views` plus every wire fully inside that set
    (both ends on those parts, or on wires that are themselves inside)."""
    views = list(editor.part_views.values() if views is None else views)
    parts = {v.part.uid: part_data(v) for v in views}
    wires, colors = {}, {}
    for view in internal_wires(editor, views):
        wires[view.wire.uid] = wire_data(view)
        if view.color:
            colors[view.wire.uid] = view.color
    return Snapshot(parts, wires, colors)


def changes(editor: Editor, parts: Iterable[int], wires: Iterable[int]) -> tuple[dict, dict, dict]:
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
    return p.kind, p.label, view.x, view.y, copy.deepcopy(p.props) if p.props else {}


def wire_data(view: WireView) -> WireData:
    w = view.wire
    return (_ref(w.src), _ref(w.dst), tuple(view.bends),
            None if isinstance(w.src, Pin) else view.src,
            None if isinstance(w.dst, Pin) else view.dst)


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


def restore(editor: Editor, target: Snapshot, only: tuple[Iterable[int], Iterable[int]] | None = None) -> None:
    """Make the board match `target`, touching only what differs. `only`: the part and
    wire uids to look at (an undo step's); everything else is known to match already."""
    with paused_gc():
        _restore(editor, target, only)


def _restore(editor: Editor, target: Snapshot, only: tuple[Iterable[int], Iterable[int]] | None) -> None:
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
    doomed = [view for uid in part_uids - target.parts.keys() if (view := view_of(uid)) is not None]
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
                editor.connect(src, dst, list(bends), src_pt, dst_pt, uid=uid,
                               color=target.wire_colors.get(uid), check=False)
                continue
            view = editor.wire_views[wire]
            view.color = target.wire_colors.get(uid)  # paint() redoes the gradients
            if tuple(view.bends) != bends or (src_pt and view.src != src_pt) or (dst_pt and view.dst != dst_pt):
                view.src, view.dst = src_pt or view.src, dst_pt or view.dst
                view.set_bends(list(bends))
                changed.append(view)
    # 5. re-attach ends: wires on moved parts, changed wires, and whatever hangs off them.
    #    Only those: touching every wire made undoing one moved part on a 2000-part
    #    board take ~80 ms.
    editor.refresh_wires([*editor.wires_touching(moved), *changed])


def instantiate(editor: Editor, clip: Snapshot, live: bool = True) -> tuple[list[PartView], list[WireView]]:
    """Add a copy of `clip` at its original coordinates, with fresh uids (for paste).
    `live=False`: the parts are ghosts until the caller opens them (see Circuit.open_part)."""
    with paused_gc():
        return _instantiate(editor, clip, live)


def _instantiate(editor: Editor, clip: Snapshot, live: bool) -> tuple[list[PartView], list[WireView]]:
    views = editor.add_parts([(kind, x, y, None, label, props)
                              for kind, label, x, y, props in clip.parts.values()], live=live)
    new = dict(zip(clip.parts, views))
    new_parts = {uid: view.part for uid, view in new.items()}
    new_wires: dict[int, Wire] = {}
    with editor.wire_batch():
        for uid in sorted(clip.wires):
            src_ref, dst_ref, bends, src_pt, dst_pt = clip.wires[uid]
            src = _resolve(src_ref, new_parts, new_wires)
            dst = _resolve(dst_ref, new_parts, new_wires)
            new_wires[uid] = editor.connect(src, dst, list(bends), src_pt, dst_pt,
                                            color=clip.wire_colors.get(uid), check=False)
    return list(new.values()), [editor.wire_views[w] for w in new_wires.values()]


class History:
    """Undo/redo as a timeline of changes (see the module docstring).

    `current` is the snapshot of what's on screen, kept up to date in place. `state`
    names the point in the timeline: every new step gets a new one, so "has anything
    changed since it was saved?" is a comparison of two numbers."""

    def __init__(self, initial: Snapshot, limit: int = 500) -> None:
        self.current = Snapshot(dict(initial.parts), dict(initial.wires), dict(initial.wire_colors))
        self.undo_stack: list[tuple[Change, int, int]] = []  # (change, state before, state after)
        self.redo_stack: list[tuple[Change, int, int]] = []
        self.limit = limit
        self._ids = itertools.count(1)
        self.state = 0

    def _sections(self) -> tuple[dict, dict, dict]:
        return self.current.parts, self.current.wires, self.current.wire_colors

    def _diff(self, parts: dict, wires: dict, colors: dict) -> Change:
        """uid -> (before, after) for the entries that differ from `current`."""
        return tuple({uid: (cur.get(uid), new) for uid, new in now.items() if cur.get(uid) != new}
                     for now, cur in zip((parts, wires, colors), self._sections()))

    def record(self, parts: dict, wires: dict, colors: dict) -> bool:
        """A new step: what these uids look like now (see changes()); None = gone.
        No-op (returns False) if that's what they looked like already."""
        change = self._diff(parts, wires, colors)
        if not any(change):
            return False
        _apply(self._sections(), change, 1)
        after = next(self._ids)
        self.undo_stack.append((change, self.state, after))
        del self.undo_stack[:-self.limit]
        self.redo_stack.clear()
        self.state = after
        return True

    def commit(self, snap: Snapshot) -> bool:
        """Record the board becoming `snap`, compared in full (record() is the fast way)."""
        cur = self.current
        return self.record({u: snap.parts.get(u) for u in cur.parts.keys() | snap.parts.keys()},
                           {u: snap.wires.get(u) for u in cur.wires.keys() | snap.wires.keys()},
                           {u: snap.wire_colors.get(u) for u in cur.wire_colors.keys() | snap.wire_colors.keys()})

    def amend(self, parts: dict, wires: dict, colors: dict) -> bool:
        """Like record, but folded into the newest step (a run of small tweaks = one undo step)."""
        if not self.undo_stack:
            return self.record(parts, wires, colors)
        change = self._diff(parts, wires, colors)
        if not any(change):
            return False
        _apply(self._sections(), change, 1)
        top, before, _ = self.undo_stack[-1]
        merged = tuple(dict(section) for section in top)
        for into, section in zip(merged, change):
            for uid, (old, new) in section.items():
                first = into[uid][0] if uid in into else old
                if first == new:
                    into.pop(uid, None)
                else:
                    into[uid] = (first, new)
        self.state = next(self._ids)
        self.undo_stack[-1] = (merged, before, self.state)
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
    for d, section in zip(sections, change):
        for uid, pair in section.items():
            if pair[side] is None:
                d.pop(uid, None)
            else:
                d[uid] = pair[side]


def change_uids(change: Change) -> tuple[set[int], set[int]]:
    """The part and wire uids an undo step touches (wire colors count as wires)."""
    parts, wires, colors = change
    return set(parts), set(wires) | set(colors)


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
