"""Looking inside a placed macro (right-click -> View): read-only, live.

Nothing gets simulated for it. A macro instance's body already runs in the board's
circuit as hidden parts and wires (see Circuit._expand), with that instance's
inputs; viewing it only draws the body's layout (MacroType.body: positions, labels,
bends, colors) with each shape bound to the instance's hidden pins and wires, so
ViewSync shows their states like any other. Views inside views work the same way.

A scene is what the editor draws and hit-tests: a canvas with its tables, spatial
indexes, views and selection (SCENE). Going in swaps the board's scene out (kept
whole: building views is the slow part, so coming back is instant) for a fresh one
made of the body; coming out frees that one and swaps the board's back. The editor
keeps all edits away while inside (see Editor._inside_press / _inside_key).

PinProbe: the hovered part's pin levels (0 / 1 / X / Z), written inside its body
next to each pin. On the board too, not only inside.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pyglet

from ..sim import Part, Pin
from . import theme as T
from .canvas import Canvas
from .paint import paint
from .sdf_text import SDFLabel, SDFText
from .selection import Selection
from .spatial import SpatialIndex
from .sync import ViewSync
from .views import Layers, PartTable, PartView, Touched, WireTable, WireView, paused_gc

if TYPE_CHECKING:
    from .editor import Editor

# What makes a scene: the Editor attributes swapped in and out together
SCENE = (
    "world",
    "layers",
    "text",
    "part_index",
    "wire_index",
    "wire_table",
    "part_table",
    "part_views",
    "wire_views",
    "selection",
    "view_sync",
)


@dataclass
class Level:
    """One step in: the instance looked into, and what was on screen before."""

    inst: Part
    outer: dict[str, Any]  # the scene it replaced (SCENE attributes)
    camera: dict[str, Any] = field(default_factory=dict)  # the camera there


def take_scene(editor: Editor) -> dict[str, Any]:
    return {name: getattr(editor, name) for name in SCENE}


def put_scene(editor: Editor, scene: dict[str, Any]) -> None:
    for name, value in scene.items():
        setattr(editor, name, value)


@contextmanager
def untouched():
    """Making and freeing views reports them to Touched (for the undo history). The
    views inside a macro aren't the board's: keep what was there, drop the rest."""
    saved = (
        set(Touched.parts),
        set(Touched.wires),
        set(Touched.paint_parts),
        set(Touched.paint_wires),
        list(Touched.moved),
    )
    try:
        yield
    finally:
        Touched.parts, Touched.wires, Touched.paint_parts, Touched.paint_wires = saved[:4]
        Touched.moved = saved[4]


def build_scene(inst: Part, pin_labels: bool) -> dict[str, Any]:
    """A fresh scene showing `inst`'s body, bound to its hidden parts and wires."""
    world = Canvas(pyglet.graphics.Batch())
    layers = Layers()
    text = SDFText(world, layers.text_order)
    part_index, wire_index = SpatialIndex(), SpatialIndex()
    part_table = PartTable(world, layers, text, part_index)
    wire_table = WireTable(world, layers, wire_index)
    body = inst.type.body
    inner = inst.inner
    with untouched(), paused_gc():
        placed = [
            (inner[uid], x, y)
            for uid, (_kind, _label, x, y, _props) in body.parts.items()
            if uid in inner
        ]
        views = PartView.many(placed, part_table, pin_labels=pin_labels)
        part_views = {p: v for (p, _, _), v in zip(placed, views)}
        by_uid = {w.uid: w for w in inst.inner_wires}
        specs = []
        for uid in sorted(body.wires):  # parents first
            wire = by_uid.get(uid)
            if wire is None:
                continue
            _src, _dst, bends, src_pt, dst_pt = body.wires[uid]
            src = part_views[wire.src.part].pin_pos(wire.src) if isinstance(wire.src, Pin) else src_pt
            dst = part_views[wire.dst.part].pin_pos(wire.dst) if isinstance(wire.dst, Pin) else dst_pt
            specs.append((wire, src, list(bends), dst, body.wire_colors.get(uid)))
        wire_views = dict(zip((s[0] for s in specs), WireView.many(specs, wire_table)))
        paint(_Painted([s[0] for s in specs], part_views, wire_views))
    return {
        "world": world,
        "layers": layers,
        "text": text,
        "part_index": part_index,
        "wire_index": wire_index,
        "wire_table": wire_table,
        "part_table": part_table,
        "part_views": part_views,
        "wire_views": wire_views,
        "selection": Selection(),
        "view_sync": ViewSync(),
    }


def free_scene(scene: dict[str, Any]) -> None:
    """Done with a scene from build_scene: its GL buffers go (the rest is garbage)."""
    scene["world"].delete()


class _Painted:
    """What paint() asks an editor for, for a body's views: its wires (parents first)
    and the wires on each pin or wire."""

    def __init__(self, wires: list, part_views: dict, wire_views: dict) -> None:
        self.part_views, self.wire_views = part_views, wire_views
        self.circuit = self
        self.wires = wires
        self._at: dict[Any, list] = {}
        for w in wires:
            for end in w.ends:
                self._at.setdefault(end, []).append(w)

    def ends_on(self, end) -> list:
        return self._at.get(end, [])


class PinProbe:
    """The pin levels of one part view (the hovered one), as text inside its body: a
    character per pin, in the level's color, kept current every frame (update)."""

    GAP = 3  # world units between a pin dot's edge and its level

    def __init__(self) -> None:
        self.view: PartView | None = None
        self.labels: list[SDFLabel | None] = []
        self.codes: list[int] = []
        self.slots = np.empty(0, np.intp)
        self.at: tuple = ()  # where the view was when the labels were placed

    def clear(self) -> None:
        for label in self.labels:
            if label is not None:
                label.delete()
        self.view, self.labels, self.codes, self.at = None, [], [], ()

    def update(self, view: PartView | None) -> None:
        """Show `view`'s pin levels (None: nobody's). Lifted views (being dragged) show none."""
        if view is not None and view.lifted:
            view = None
        if view is not self.view:
            self.clear()
            self.view = view
            if view is None:
                return
            pins = view.part.pins
            self.slots = np.fromiter((p.slot for p in pins), np.intp, len(pins))
            self.labels = [None] * len(pins)
            self.codes = [-1] * len(pins)
        if view is None:
            return
        codes = view.part.circuit.pin_codes(self.slots).tolist()
        at = (view.x, view.y)
        moved = at != self.at
        self.at = at
        for i, (pin, code) in enumerate(zip(view.part.pins, codes)):
            if code == self.codes[i] and not moved:
                continue
            if self.labels[i] is not None:
                self.labels[i].delete()
            px, py = view.pin_pos(pin)
            off = T.PIN_RADIUS + self.GAP
            self.labels[i] = SDFLabel(
                view.text,
                "Z01X"[code],
                px + off if pin.is_input else px - off,
                py,
                T.LEVEL_SIZE,
                T.LEVEL_TEXT[code],
                "left" if pin.is_input else "right",
            )
            self.codes[i] = code
