"""Snapshots of a board: plain data, no pyglet. The basis for undo/redo, copy/paste
and save files (see ui/document.py for taking and applying them, storage.py for
reading and writing them).

Everything is keyed by stable uids (parts and wires both have one, since a wire
can be attached to another wire). Simulation state (which switches are on) is
deliberately NOT part of it: toggling is using the circuit, not editing it.

Wire uids grow in creation order and a wire is always created after the wires
it attaches to, so iterating wires by uid always visits parents first.

Wire layers (2.5D), like a chip's metal layers: the parts are the silicon, and the
wiring stacks up above them, layer 0 (the lowest, the only one pins connect to) up
to LAYER_COUNT - 1. Every wire runs on one layer. Layers only say where a wire is
drawn and what it can be wired to from where, never what it connects: that's its
ends, as ever. A *via* is a wire of zero length with both ends free (see is_via);
others attach to it like to any wire, so going through one is a junction. Its layer
is its floor: it reaches that layer and every layer above it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MACRO = "macro:"  # kind prefix of a placed macro ("macro:half adder"); part type kinds can't contain ':'

Point = tuple[float, float]
PartData = tuple[str, str, float, float, dict[str, Any]]  # kind, label, x, y, props
# ("p", part uid, is_input, pin index) or ("w", wire uid). A wire end naming the
# wire's own uid is a free end (attached to nothing: see sim.Wire).
EndRef = tuple
# src ref, dst ref, bends (src to dst), src junction point, dst junction point
# (junction points are None for pin ends: those follow from the part's position;
# a free end's point is where it is)
WireData = tuple[EndRef, EndRef, tuple[Point, ...], Point | None, Point | None]


LAYER_COUNT = 16


def is_via(data: WireData, uid: int) -> bool:
    """Is this wire (its data and uid) a via: no length, attached to nothing?"""
    src, dst, bends, src_pt, dst_pt = data
    return src == dst == ("w", uid) and not bends and src_pt == dst_pt


@dataclass(frozen=True)
class Snapshot:
    parts: dict[int, PartData]
    wires: dict[int, WireData]
    wire_colors: dict[int, str] = field(
        default_factory=dict
    )  # wire uid -> color name; absent = default
    wire_layers: dict[int, int] = field(
        default_factory=dict
    )  # wire uid -> its layer (a via's: its floor); absent = 0, the lowest


EMPTY = Snapshot({}, {})
