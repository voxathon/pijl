"""Snapshots of the board: the basis for undo/redo and copy/paste (and later, save files).

A Snapshot is plain data -- chip kinds, labels, positions, wire endpoints and
bends -- keyed by the chips' stable uids. Simulation state (which switches are
on) is deliberately NOT part of it: toggling is using the circuit, not editing it.

Undo doesn't rebuild the board from scratch. `restore` diffs the target
snapshot against what's on screen and only adds/removes/moves what changed,
because creating pyglet shapes is the slow part (~0.8 ms per chip). A side
effect: chips that survive an undo keep their switch states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from .views import ChipView, Point, WireView

if TYPE_CHECKING:
    from .editor import Editor

ChipData = tuple[str, str, float, float]  # kind, label, x, y
WireKey = tuple[int, int, int, int]       # src uid, output index, dst uid, input index


@dataclass(frozen=True)
class Snapshot:
    chips: dict[int, ChipData]
    wires: dict[WireKey, tuple[Point, ...]]  # -> bends, src to dst


def capture(editor: Editor, views: Iterable[ChipView] | None = None) -> Snapshot:
    """The whole board, or just `views` plus every wire running between two of them."""
    views = list(editor.chip_views.values() if views is None else views)
    uids = {v.chip.uid for v in views}
    chips = {v.chip.uid: (v.chip.kind, v.chip.label, v.x, v.y) for v in views}
    wires = {_key(w.wire): tuple(w.bends) for w in editor.wire_views.values()
             if w.wire.src.chip.uid in uids and w.wire.dst.chip.uid in uids}
    return Snapshot(chips, wires)


def restore(editor: Editor, target: Snapshot) -> None:
    """Make the board match `target`, touching only what differs."""
    by_uid = {v.chip.uid: v for v in editor.chip_views.values()}
    # 1. chips that shouldn't exist (their wires go with them)
    for uid in by_uid.keys() - target.chips.keys():
        editor.remove_chip(by_uid.pop(uid))
    # 2. wires that shouldn't exist
    for view in [v for v in editor.wire_views.values() if _key(v.wire) not in target.wires]:
        editor.remove_wire(view)
    # 3. chips: add missing, update moved/relabeled
    moved = set()
    for uid, (kind, label, x, y) in target.chips.items():
        view = by_uid.get(uid)
        if view is None:
            view = by_uid[uid] = editor.add_chip(kind, x, y, uid=uid)
        elif (view.x, view.y) != (x, y):
            view.move_to(x, y)
            moved.add(view.chip)
        if view.chip.label != label:
            view.chip.label = label
            view.refresh_name()
            view.name.move_to(*view.name_pos())
    # 4. wires: add missing (created at the right pins already), update bends,
    #    and re-attach the ends of wires on moved chips. Only those: touching every
    #    wire made undoing one moved chip on a 2000-chip board take ~80 ms.
    existing = {_key(v.wire): v for v in editor.wire_views.values()}
    for (su, si, du, di), bends in target.wires.items():
        view = existing.get((su, si, du, di))
        if view is None:
            editor.connect(by_uid[su].chip.outputs[si], by_uid[du].chip.inputs[di], list(bends))
        elif tuple(view.bends) != bends:
            view.set_bends(list(bends))
    if moved:
        editor.refresh_wires_touching(moved)


def instantiate(editor: Editor, clip: Snapshot) -> tuple[list[ChipView], list[WireView]]:
    """Add a copy of `clip` at its original coordinates, with fresh uids (for paste)."""
    new: dict[int, ChipView] = {}
    for uid, (kind, label, x, y) in clip.chips.items():
        view = new[uid] = editor.add_chip(kind, x, y)
        if label:
            view.chip.label = label
            view.refresh_name()
            view.name.move_to(*view.name_pos())
    wires = []
    for (su, si, du, di), bends in clip.wires.items():
        wire = editor.connect(new[su].chip.outputs[si], new[du].chip.inputs[di], list(bends))
        wires.append(editor.wire_views[wire])
    return list(new.values()), wires


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


def _key(wire) -> WireKey:
    return wire.src.chip.uid, wire.src.index, wire.dst.chip.uid, wire.dst.index
