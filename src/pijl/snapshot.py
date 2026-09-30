"""Snapshots of a board: plain data, no pyglet. The basis for undo/redo, copy/paste
and save files (see ui/document.py for taking and applying them, storage.py for
reading and writing them).

Everything is keyed by stable uids (parts and wires both have one, since a wire
can be attached to another wire). Simulation state (which switches are on) is
deliberately NOT part of it: toggling is using the circuit, not editing it.

Wire uids grow in creation order and a wire is always created after the wires
it attaches to, so iterating wires by uid always visits parents first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MACRO = "macro:"  # kind prefix of a placed macro ("macro:half adder"); part type kinds can't contain ':'

Point = tuple[float, float]
PartData = tuple[str, str, float, float, dict[str, Any]]  # kind, label, x, y, props
# ("p", part uid, is_input, pin index) or ("w", wire uid)
EndRef = tuple
# src ref, dst ref, bends (src to dst), src junction point, dst junction point
# (junction points are None for pin ends: those follow from the part's position)
WireData = tuple[EndRef, EndRef, tuple[Point, ...], Point | None, Point | None]


@dataclass(frozen=True)
class Snapshot:
    parts: dict[int, PartData]
    wires: dict[int, WireData]
    wire_colors: dict[int, str] = field(
        default_factory=dict
    )  # wire uid -> color name; absent = default


EMPTY = Snapshot({}, {})
