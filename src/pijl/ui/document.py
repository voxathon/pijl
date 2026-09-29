"""Snapshots of the board: the basis for undo/redo and copy/paste (and later, save files).

A Snapshot is plain data -- part kinds, labels, positions, wire endpoints and
bends -- keyed by stable uids (parts and wires both have one, since a wire can
be attached to another wire). Simulation state (which switches are on) is
deliberately NOT part of it: toggling is using the circuit, not editing it.

Undo doesn't rebuild the board from scratch. `restore` diffs the target
snapshot against what's on screen and only adds/removes/moves what changed,
because creating pyglet shapes is the slow part (~0.8 ms per part). A side
effect: parts that survive an undo keep their switch states.

Wire uids grow in creation order and a wire is always created after the wires
it attaches to, so iterating wires by uid always visits parents first. (Splicing
keeps that true: the surviving wire is the older one.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from ..sim import Pin, Wire
from .views import PartView, Point, WireView

if TYPE_CHECKING:
    from .editor import Editor

PartData = tuple[str, str, float, float]  # kind, label, x, y
# ("p", part uid, is_input, pin index) or ("w", wire uid)
EndRef = tuple
# src ref, dst ref, bends (src to dst), src junction point, dst junction point
# (junction points are None for pin ends: those follow from the part's position)
WireData = tuple[EndRef, EndRef, tuple[Point, ...], Point | None, Point | None]


@dataclass(frozen=True)
class Snapshot:
    parts: dict[int, PartData]
    wires: dict[int, WireData]


def capture(editor: Editor, views: Iterable[PartView] | None = None) -> Snapshot:
    """The whole board, or just `views` plus every wire fully inside that set
    (both ends on those parts, or on wires that are themselves inside)."""
    views = list(editor.part_views.values() if views is None else views)
    uids = {v.part.uid for v in views}
    parts = {v.part.uid: (v.part.kind, v.part.label, v.x, v.y) for v in views}
    inside = internal_wires(editor, uids)
    wires = {}
    for view in inside:
        w = view.wire
        wires[w.uid] = (_ref(w.src), _ref(w.dst), tuple(view.bends),
                        None if isinstance(w.src, Pin) else view.src,
                        None if isinstance(w.dst, Pin) else view.dst)
    return Snapshot(parts, wires)


def internal_wires(editor: Editor, part_uids: set[int]) -> list[WireView]:
    """Wire views whose every end lands on those parts or on other internal wires."""
    inside: set[Wire] = set()
    result = []
    for w in editor.circuit.wires:  # creation order: parents first
        if all(e.part.uid in part_uids if isinstance(e, Pin) else e in inside for e in w.ends):
            inside.add(w)
            result.append(editor.wire_views[w])
    return result


def restore(editor: Editor, target: Snapshot) -> None:
    """Make the board match `target`, touching only what differs."""
    by_uid = {v.part.uid: v for v in editor.part_views.values()}
    # 1. parts that shouldn't exist (their wires and branches go with them)
    for uid in by_uid.keys() - target.parts.keys():
        editor.remove_part(by_uid.pop(uid))
    # 2. wires that shouldn't exist -- or exist with different endpoints (cut-deletion
    #    splices a branch onto its trunk, re-pointing the trunk's far end). Those are
    #    rebuilt in step 4; branches that get removed along with them are too.
    def stale(view: WireView) -> bool:
        data = target.wires.get(view.wire.uid)
        return data is None or (_ref(view.wire.src), _ref(view.wire.dst)) != data[:2]

    for view in [v for v in editor.wire_views.values() if stale(v)]:
        if view.wire in editor.wire_views:  # may be gone already, as a branch of an earlier one
            editor.remove_wire(view)
    # 3. parts: add missing, update moved/relabeled
    moved = set()
    for uid, (kind, label, x, y) in target.parts.items():
        view = by_uid.get(uid)
        if view is None:
            view = by_uid[uid] = editor.add_part(kind, x, y, uid=uid)
        elif (view.x, view.y) != (x, y):
            view.move_to(x, y)
            moved.add(view.part)
        if view.part.label != label:
            view.part.label = label
            view.refresh_name()
            view.name.move_to(*view.name_pos())
    # 4. wires, parents first: add missing, update bends / junction points
    wire_by_uid = {v.wire.uid: v.wire for v in editor.wire_views.values()}
    changed: list[WireView] = []
    for uid in sorted(target.wires):
        src_ref, dst_ref, bends, src_pt, dst_pt = target.wires[uid]
        wire = wire_by_uid.get(uid)
        if wire is None:
            src = _resolve(src_ref, by_uid, wire_by_uid)
            dst = _resolve(dst_ref, by_uid, wire_by_uid)
            wire = editor.connect(src, dst, list(bends), src_pt, dst_pt, uid=uid)
            wire_by_uid[uid] = wire
            continue
        view = editor.wire_views[wire]
        if tuple(view.bends) != bends or (src_pt and view.src != src_pt) or (dst_pt and view.dst != dst_pt):
            view.src, view.dst = src_pt or view.src, dst_pt or view.dst
            view.set_bends(list(bends))
            changed.append(view)
    # 5. re-attach ends: wires on moved parts, changed wires, and whatever hangs off them.
    #    Only those: touching every wire made undoing one moved part on a 2000-part
    #    board take ~80 ms.
    editor.refresh_wires([*editor.wires_touching(moved), *changed])


def instantiate(editor: Editor, clip: Snapshot) -> tuple[list[PartView], list[WireView]]:
    """Add a copy of `clip` at its original coordinates, with fresh uids (for paste)."""
    new: dict[int, PartView] = {}
    for uid, (kind, label, x, y) in clip.parts.items():
        view = new[uid] = editor.add_part(kind, x, y)
        if label:
            view.part.label = label
            view.refresh_name()
            view.name.move_to(*view.name_pos())
    new_wires: dict[int, Wire] = {}
    for uid in sorted(clip.wires):
        src_ref, dst_ref, bends, src_pt, dst_pt = clip.wires[uid]
        src = _resolve(src_ref, new, new_wires)
        dst = _resolve(dst_ref, new, new_wires)
        new_wires[uid] = editor.connect(src, dst, list(bends), src_pt, dst_pt)
    return list(new.values()), [editor.wire_views[w] for w in new_wires.values()]


class History:
    """Undo/redo as a timeline of snapshots. `current` is always what's on screen."""

    def __init__(self, initial: Snapshot, limit: int = 500) -> None:
        self.current = initial
        self.undo_stack: list[Snapshot] = []
        self.redo_stack: list[Snapshot] = []
        self.limit = limit

    def commit(self, snap: Snapshot) -> bool:
        """Record a new state; no-op (returns False) if nothing changed."""
        if snap == self.current:
            return False
        self.undo_stack.append(self.current)
        del self.undo_stack[:-self.limit]
        self.redo_stack.clear()
        self.current = snap
        return True

    def undo(self) -> Snapshot | None:
        if not self.undo_stack:
            return None
        self.redo_stack.append(self.current)
        self.current = self.undo_stack.pop()
        return self.current

    def redo(self) -> Snapshot | None:
        if not self.redo_stack:
            return None
        self.undo_stack.append(self.current)
        self.current = self.redo_stack.pop()
        return self.current


def _ref(end) -> EndRef:
    if isinstance(end, Pin):
        return "p", end.part.uid, end.is_input, end.index
    return "w", end.uid


def _resolve(ref: EndRef, parts: dict[int, PartView], wires: dict[int, Wire]):
    if ref[0] == "p":
        _, part_uid, is_input, index = ref
        part = parts[part_uid].part
        return (part.inputs if is_input else part.outputs)[index]
    return wires[ref[1]]
