"""The editor window: input handling (a small state machine), sim ticking, drawing.

Design rule for carpal-tunnel friendliness: nothing *requires* holding a
mouse button. Placing parts and drawing wires are "click to start, click to
finish". The press that starts an action never finishes it on release, so
holding and dragging simply carries the thing along until the next click.

Controls
  part picker (left panel; its "«" button collapses it, see picker.py)
    click a part           pick it up; it follows the cursor (dragging it out of the panel does too)
      click                place it (shift+click: place and keep another)
      click another part   swap to that part instead
      right-click / Esc    cancel (so does a click anywhere on the panel; clicking the carried
                           part puts it back, so a double-click leaves you ready to drag rows)
    click a collection     expand / collapse it (a quick second click is ignored: mouse bounce)
    drag a part            move it into another collection (drop on its header or between its
                           parts), or below everything to take it out of collections; the other
                           rows slide apart to show where it lands
    drag a collection      reorder the collections
    "+" button             new collection; type its name, Enter commits, Esc keeps the default
    right-click            menus: rename / delete / collapse-or-expand-all collections, take a part
                           out of its collection or into a new one
    scroll                 scroll the list (smoothly; dragging near its edges scrolls too)
  click a pin              start a wire; it follows the cursor
  press+drag on a wire     start a branch from that spot, like from a pin (a plain click
                           selects the wire; Alt+click or right-click -> Branch also work)
    click empty space      add a bend point
    click a pin or wire    connect (green preview = valid target); ending on a wire
                           makes a junction. Wires joined this way form one net; if its
                           drivers disagree the net shows amber (conflict, see sim/circuit.py)
    right-click/Backspace  remove last bend point (or cancel if none)
    Esc / click start pin  cancel
  click a part / wire      select it (a plain click on an IN switch toggles it instead)
  shift+click              add to / remove from the selection (works on switches too)
  drag on empty space      box-select anything the box touches (shift: add to selection)
  click empty space / Esc  clear the selection;  Ctrl+A select all;  Del/Backspace delete it
  drag a part              move it; dragging a selected part moves the whole selection
  right-click part/wire    select just it + context menu (click outside or Esc closes)
    Delete (wires)         cuts the wire like Digital Logic Sim: from the nearest junction
                           before the spot you clicked, onward. See cut_wire.
    Label...               type in place; Enter commits, Esc reverts, clicking elsewhere commits
    Recolor                wires and IN/OUT parts: 8 colors, or Default (inherit). Colors blend
                           along wires, see paint.py. New branches take their wire's color
    (settings)             whatever settings a part's type offers (PartType.settings, e.g. a
                           pull's Priority) and its actions. On a part inside a selection of
                           parts all of one kind, the menu edits all of them ("mixed" where
                           their values differ); Ctrl+right-click: just the part clicked.
                           Numbers open a popover: drag the slider (the parts follow live;
                           each release is one undo step), or type + Enter. Esc takes back
                           a drag in progress; a click outside closes it. See popover.py.
    Edit (wires)           hold+drag square handles to move bends, "+" handles or the wire
                           itself to add one; right-click a square to remove it. Round handles
                           are junctions (its own ends and branches off it): drag to slide
                           them along their wire (Alt: grab a junction hidden under a bend).
                           Enter or a click elsewhere finishes, Esc reverts. See wire_edit.py.
  Ctrl+C / Ctrl+X         copy / cut the selected parts (+ wires running between them)
  Ctrl+V                   paste: the copy follows the cursor like a new part; click to place
                           (shift+click: place and keep another copy), Esc/right-click cancels
  Ctrl+D                   duplicate into a block: each press doubles it, right then down
                           (1, 2x1, 2x2, 4x2...). While it's still selected, Ctrl+scroll spaces
                           it out along the last doubling, Ctrl+Shift+scroll the other way.
                           See duplicate.py.
  Ctrl+S                   save the board as a macro (the first save asks for a name)
  Ctrl+Shift+S             save under another name
  Ctrl+O                   open a macro: type to filter, arrows + Enter (or click); saved macros
                           are also in the picker's MACROS section: right-click -> Open
  Ctrl+N                   new, empty board
  Tab                      pin names on placed parts: hidden -> on hover -> always -> hidden
                           (opening, new and closing the window ask first if there are unsaved changes)
  cogwheel (bottom right)  Controls (the short version of this list, see controls.py: keep them in
                           step), Open macro..., Projects: switch to another project or make a new one
  Ctrl+Z / Ctrl+Y          undo / redo (also Ctrl+Shift+Z). During an action, Ctrl+Z cancels it.
                           Every finished edit is recorded automatically; see document.py.
  right-drag empty space   pan (middle-drag pans in any mode)
  hold Ctrl                snap parts and wire bends to the grid
  hold Ctrl+Shift          snap to the finer subgrid instead
  scroll                   zoom
  Home                     reset the camera
  M                        minimap (top right): press (any button) or drag in it to go there;
                           works mid-action too, like panning
  G                        lens: magnifies the board around the cursor. Hold G + scroll, or
                           scroll on the lens or the minimap, to change how much (see minimap.py)
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto

import pyglet
from pyglet import shapes
from pyglet.math import Mat4
from pyglet.window import key, mouse

from ..macros import Catalog
from ..parts import Choice, Number, Toggle
from ..parts import load as load_parts
from ..project import (
    Project,
    last_project,
    project_names,
    remember_project,
    write_atomic,
)
from ..storage import NAME_MAX, FormatError, MacroStore, check_name
from ..sim import Part, Circuit, Pin, Wire
from ..snapshot import (
    MACRO,
)  # library entries (and part kinds) of saved macros: "macro:<name>"
from . import theme as T
from .camera import MIN_LEVEL, Camera
from .canvas import Canvas
from .controls import ControlsSheet
from .document import (
    EMPTY,
    Change,
    History,
    Snapshot,
    capture,
    change_uids,
    changes,
    instantiate,
    internal_wires,
    restore,
)
from .duplicate import DOWN, RIGHT, Cell, Tiling
from .grid import Grid
from .library import Library
from .line_edit import LineEdit
from .menu import RAINBOW, ContextMenu, MenuItem
from .minimap import Panels
from .paint import paint, part_color
from .picker import PartPicker, Row
from .popover import NumberPopover
from .prompt import Prompt
from .sdf_text import SDFText
from .selection import Selection
from .spatial import SpatialIndex, ordered
from .status_bar import BAR_H, StatusBar
from .views import (
    PartView,
    Layers,
    Point,
    Polyline,
    Touched,
    WireView,
    arc_length_at,
    points_before,
    delete_views,
    lift,
    paused_gc,
    project_onto,
    put_down,
    theme_color,
)
from .wire_edit import WireEditSession

SIM_STEPS_PER_FRAME = 1
HOME = (400, 300)
LABEL_MAX = 32
PIN_LABELS_HIDDEN, PIN_LABELS_HOVER, PIN_LABELS_ALWAYS = 0, 1, 2
PIN_LABEL_MODES = ("hidden", "on hover", "always shown")
SETTLE_TICKS = 64  # settling noise for new parts, see sim/circuit.py
MACROS = "MACROS"  # the picker collection new macros land in
NOTICE_SECONDS = 4.0
DOUBLE_CLICK = 0.4  # s: a second click on the same picker row within this is treated as mouse bounce
FIT_MARGIN, FIT_MARGIN_SHARE = (
    60,
    0.05,
)  # room around the board when fitting it: world units, or of its size
STATS_EVERY = 0.25  # s: how often the status bar's numbers are refreshed (averaged over that time)


class Mode(Enum):
    IDLE = auto()
    PRESSING_PART = (
        auto()
    )  # mouse down on a part, not moved yet: could become a click or a drag
    DRAGGING_PART = auto()  # moving the selection (or one part) with the mouse held
    PRESSING_WIRE = (
        auto()
    )  # mouse down on a wire: a click selects it, a drag starts a branch
    BOX_SELECTING = auto()  # dragging a selection rectangle over empty space
    PLACING_PART = auto()  # a new part follows the cursor until a click places it
    WIRING = (
        auto()
    )  # a wire follows the cursor from its start pin until a click on a pin
    MENU = auto()  # context menu open; the next click picks an item or closes it
    EDITING_LABEL = auto()  # typing a part's label in place
    EDITING_WIRE = auto()  # moving / adding / removing one wire's bend points
    PICKER_PRESS = (
        auto()
    )  # mouse down on a picker row: a click picks / toggles it, a drag moves it
    PICKER_DRAG = auto()  # carrying a picker row to another spot in the list
    RENAMING = auto()  # typing a collection's name in the picker
    PROMPT = (
        auto()
    )  # a Prompt box is up (save as / open / unsaved changes); see _open_prompt
    POPOVER = auto()  # editing a Number setting in its popover; see _open_popover


# Modes you can steer with the minimap in (the rest hold a button down, or have the keyboard)
STEERABLE = frozenset({Mode.IDLE, Mode.PLACING_PART, Mode.WIRING, Mode.EDITING_WIRE})


def _make_config() -> pyglet.gl.Config | None:
    """Ask for 4x multisampling (smooth shape edges); fall back to the default if unsupported."""
    screen = pyglet.display.get_display().get_default_screen()
    try:
        return screen.get_best_config(
            pyglet.gl.Config(double_buffer=True, sample_buffers=1, samples=4)
        )
    except pyglet.window.NoSuchConfigException:
        return None


class Editor(pyglet.window.Window):
    def __init__(self) -> None:
        self.history: History | None = (
            None  # set up by _start_document; checked by dispatch_event
        )
        super().__init__(
            1280, 720, caption="pijl", resizable=True, vsync=True, config=_make_config()
        )
        # the open document: a macro, or an untitled board (set early: the picker asks about it)
        self.doc: str | None = None  # its name; None = untitled
        self.saved_state = 0  # history.state of what's on disk (dirty = it differs)
        self.load_problems: list[str] = []
        try:
            project = Project.open(
                last_project()
            )  # created on first run (see project.py)
        except (OSError, ValueError) as e:
            self.load_problems.append(
                f"can't open the last project ({e}); opened the default one"
            )
            project = Project.open()
        self._bind_project(project)
        self.pin_label_mode = PIN_LABELS_HOVER  # Tab cycles it
        self.hover_view: PartView | None = None  # the part whose pin names hover shows
        self._wire_batch: dict | None = (
            None  # wires waiting for their views (see wire_batch)
        )
        self.camera = Camera()
        self.grid = Grid()
        self.keys = key.KeyStateHandler()  # live "is this key down?" lookups
        self.push_handlers(self.keys)

        self.world = Canvas(pyglet.graphics.Batch())  # parts and wires; see canvas.py
        self.layers = Layers()
        self.text = SDFText(self.world, self.layers.text_order)
        self.hud = pyglet.graphics.Batch()
        self.library = self._load_library()
        self.picker = PartPicker(
            self.library,
            self.hud,
            self.height,
            self._pixel_ratio(),
            swatch=self._swatch,
            name_of=lambda entry: entry.removeprefix(MACRO),
            disabled=self._unplaceable,
        )
        # Screen-space things that belong to the board, drawn over the HUD: the selection
        # box (the lens magnifies it with the board) and the context menu (the lens shows
        # it at its own size, where it was opened)
        self.overlay = pyglet.graphics.Batch()
        self.menu = ContextMenu(pyglet.graphics.Batch())
        # Problems (part scripts, files) in red; notices ("saved adder") in grey, for a few seconds
        self.status = pyglet.text.Label(
            "",
            font_name="Consolas",
            font_size=10,
            color=T.MENU_DANGER,
            x=self.picker.width + 8,
            y=self.height - 8,
            anchor_y="top",
            batch=self.hud,
        )
        self.bar = StatusBar(self.hud, self.width)
        self.panels = Panels(
            self.overlay, self.menu
        )  # minimap (M) and lens (G), top right
        self.lens_key: list | None = (
            None  # while G is down: [lens was open before, scrolled since]
        )
        # runtime numbers for the bar, summed over STATS_EVERY seconds
        self.stats = {"frames": 0, "time": 0.0, "sim": 0.0, "steps": 0, "draw": 0.0}
        for msg in self.load_problems:
            self._report(msg)

        self.part_views: dict[Part, PartView] = {}
        self.wire_views: dict[Wire, WireView] = {}
        # Where every view is, for hit testing without looking at all of them (views keep it current)
        self.part_index = SpatialIndex()
        self.wire_index = SpatialIndex()

        # interaction state
        self.mode = Mode.IDLE
        self.panning = False  # orthogonal to mode: you can pan while carrying things
        self.map_button: int | None = (
            None  # same for steering with the minimap: the button held in it
        )
        self.mouse = (0, 0)  # last known cursor position, screen space
        self.mouse_in = False  # is it over the window at all?
        self.active: PartView | None = None  # part being pressed / dragged
        self.grab = (0.0, 0.0)  # part origin minus cursor, world units
        self.press_at = (0, 0)  # screen pos of the press on a part / wire / picker row
        self.picker_row: Row | None = None  # picker row pressed / being dragged
        self.picker_bounce = (
            False  # that press is a double-click's 2nd half: its click does nothing
        )
        self.last_picker_click: tuple = (
            None,
            0.0,
        )  # (row key, time) of the last click on a picker row
        self.pressed_wire: WireView | None = None
        self.press_world: Point = (0.0, 0.0)
        self.wire_start: Pin | Wire | None = None  # where the wire being drawn starts
        self.wire_start_pos: Point = (0.0, 0.0)
        self.wire_bends: list[Point] = []
        self.preview: Polyline | None = None
        # label editing
        self.edit_view: PartView | None = None
        self.edit: LineEdit | None = None
        self.caret: shapes.Rectangle | None = None
        self.wire_edit: WireEditSession | None = None
        # placing (new part or paste): ghosts that follow the cursor until a click
        self.placing_views: list[PartView] = []
        self.placing_wires: list[WireView] = []
        self.place_again = None  # shift+click: start another of the same
        self.placing_kind: str | None = (
            None  # the picker part on the cursor (None for a paste)
        )
        self.clipboard: Snapshot | None = None
        self.tiling: Tiling | None = (
            None  # the Ctrl+D block being grown (see duplicate.py)
        )
        # selection
        self.selection = Selection()
        # Moving a group (dragging, or carrying new parts / a paste): see _begin_move
        self.drag_group: list[
            tuple[PartView, float, float]
        ] = []  # parts moving + their start origins
        self.drag_wires: list[
            tuple[WireView, list[Point], Point, Point]
        ] = []  # wires inside the group + start shape
        self.drag_origin = (0.0, 0.0)  # start origin of the grabbed part
        self.drag_delta = (0.0, 0.0)  # how far the group has moved so far
        self.stretched: list[
            tuple[WireView, Point, Point, tuple[bool, bool]]
        ] = []  # see _begin_move
        self.stretched_tail: list[WireView] = []
        self.box_start: Point = (0.0, 0.0)  # world point where the box drag began
        self.box_base: tuple[set, set] = (
            set(),
            set(),
        )  # selection to add to (shift) or empty
        self.box_shapes: tuple[shapes.Rectangle, shapes.Box] | None = None
        self.hud_box_group = pyglet.graphics.Group(order=8)  # below menus
        # prompt box (Mode.PROMPT)
        self.prompt: Prompt | None = None
        self.prompt_enter = (
            None  # what Enter (or clicking a list item) does: fn(prompt)
        )
        self.prompt_key = None  # other keys: fn(symbol) -> handled
        # Number setting popover (Mode.POPOVER)
        self.popover: NumberPopover | None = None
        self.pop_edit: _NumberEdit | None = None
        self._caption_for: tuple = ()

        self._start_document()
        pyglet.clock.schedule_interval(self.update, 1 / 60)

    # ======================================================================
    # model + view bookkeeping
    # ======================================================================

    def add_part(
        self, kind: str, x: float, y: float, uid: int | None = None, live: bool = True
    ) -> PartView:
        """`live=False`: a ghost, for carrying on the cursor (see _carry / _commit_placing)."""
        return self.add_parts([(kind, x, y, uid, "", None)], live)[0]

    def add_parts(self, specs: list[tuple], live: bool = True) -> list[PartView]:
        """add_part for each (kind, x, y, uid, label, props) -- label "" and props None
        leave the new part's own -- with the views made all at once (undo, loading, paste)."""
        parts = []
        with paused_gc():
            try:
                for kind, _x, _y, uid, label, props in specs:
                    part = self.circuit.add_part(
                        kind, uid, live
                    )  # (KeyError: no such kind)
                    if label:
                        part.label = label
                    if props is not None:
                        part.props = copy.deepcopy(props) if props else {}
                    parts.append(part)
            finally:  # (what did get added gets its view, even if a later kind was missing)
                views = PartView.many(
                    [(p, s[1], s[2]) for p, s in zip(parts, specs)],
                    self.world,
                    self.layers,
                    self.text,
                    pin_labels=self.pin_label_mode == PIN_LABELS_ALWAYS,
                    index=self.part_index,
                )
                for part, view in zip(parts, views):
                    self.part_views[part] = view
        return views

    def remove_part(self, view: PartView) -> list[Wire]:
        """Returns the wires that went with it."""
        return self.remove_parts([view])

    def remove_parts(self, views: list[PartView]) -> list[Wire]:
        """Remove many parts at once (a big selection: one by one was seconds).
        Returns the wires that went with them."""
        if self.hover_view in views:
            self.hover_view = None
        removed = self.circuit.remove_parts([v.part for v in views])
        for v in views:
            del self.part_views[v.part]
        self._drop_views(views, removed)
        return removed

    def _drop_wire_view(self, wire: Wire) -> None:
        if self._wire_batch is not None and wire in self._wire_batch:
            del self._wire_batch[
                wire
            ]  # (connected and gone again in the same batch: no view yet)
            return
        self._drop_views([], [wire])

    @contextmanager
    def wire_batch(self):
        """Wires connected inside get their views when it ends, all at once (undo,
        loading, paste). Until then wire_views doesn't have them."""
        if self._wire_batch is not None:  # (nested: the outer one makes them)
            yield
            return
        self._wire_batch = {}
        try:
            with paused_gc():
                yield
        finally:
            pending, self._wire_batch = self._wire_batch, None
            with paused_gc():
                views = WireView.many(
                    list(pending.values()),
                    self.world,
                    self.layers,
                    index=self.wire_index,
                )
            for wire, view in zip(pending, views):
                self.wire_views[wire] = view

    def _drop_views(self, parts: list[PartView], wires: list[Wire]) -> None:
        """Forget and delete these part views and the views of these (removed) wires."""
        wire_views = [self.wire_views.pop(w) for w in wires]
        self.selection.parts.difference_update(
            parts
        )  # (their highlights go with the views)
        self.selection.wires.difference_update(wire_views)
        delete_views(parts, wire_views)

    def connect(
        self,
        a: Pin | Wire,
        b: Pin | Wire,
        bends: list[Point] = (),
        a_pos: Point | None = None,
        b_pos: Point | None = None,
        uid: int | None = None,
        color: str | None = None,
        check: bool = True,
    ) -> Wire | None:
        """Connect two endpoints (pins or wires); `bends` are ordered from a to b.
        `a_pos` / `b_pos` say where on a wire endpoint the junction sits.
        `check=False`: rebuilding wiring that existed before (see Circuit.connect)."""
        wire, replaced = self.circuit.connect(a, b, uid, check)
        for old in replaced:
            self._drop_wire_view(old)
        if wire is not None:
            if (
                wire.src is not a
            ):  # the circuit put the output side first; flip our layout too
                bends, a_pos, b_pos = list(reversed(bends)), b_pos, a_pos
            src = self.pin_pos(wire.src) if isinstance(wire.src, Pin) else a_pos
            dst = self.pin_pos(wire.dst) if isinstance(wire.dst, Pin) else b_pos
            if self._wire_batch is not None:
                self._wire_batch[wire] = (wire, src, list(bends), dst, color)
            else:
                self.wire_views[wire] = WireView(
                    wire,
                    src,
                    list(bends),
                    dst,
                    self.world,
                    self.layers,
                    color,
                    index=self.wire_index,
                )
        return wire

    def remove_wire(self, view: WireView) -> list[Wire]:
        """Returns the wires removed: this one and its branches."""
        return self.remove_wires([view])

    def remove_wires(self, views: list[WireView]) -> list[Wire]:
        removed = self.circuit.remove_wires([v.wire for v in views])
        self._drop_views([], removed)
        return removed

    def cut_wire(self, view: WireView, at: Point) -> None:
        """Delete a wire the way Digital Logic Sim does, from the spot `at` onward.

        Walk back from `at` toward the wire's source to the nearest junction J
        (a point where another wire attaches). Everything from J onward goes:
        the rest of this wire and whatever hangs off that part. What's left
        before J doesn't dangle -- it's spliced together with the wire attached
        at J into one wire: source -> ... -> J -> that wire's far end.
        No junction before `at` means the whole wire (and its branches) goes.
        """
        w, pts = view.wire, view.points
        eps = 1e-6
        s_cut = arc_length_at(pts, at)
        attached = [
            (arc_length_at(pts, self._attach_point(x, w)), x)
            for x in self.circuit.attachments(w)
        ]
        before = [s for s, _ in attached if s <= s_cut + eps]
        if not before:
            self.remove_wire(view)
            return
        s_j = max(before)
        after = [x for s, x in attached if s > s_j + eps]
        doomed = set(after)
        for x in after:
            doomed.update(self.circuit.descendants(x))
        # The wire at J to splice on. Its far end must survive the cut, of course.
        at_j = [x for s, x in attached if abs(s - s_j) <= eps]
        splice = next((x for x in at_j if self._far_end(x, w) not in doomed), None)
        if splice is None:
            self.remove_wire(view)
            return

        for x in after:
            if x in self.wire_views:
                self.remove_wire(self.wire_views[x])
        # geometry of the merged wire: our points up to J, then the spliced wire's from J outward
        j = self._attach_point(splice, w)
        sv = self.wire_views[splice]
        tail, far_pos = (
            (sv.bends, sv.dst)
            if splice.src is w
            else (list(reversed(sv.bends)), sv.src)
        )
        bends = [*points_before(pts, s_j)[1:], j, *tail]
        src_pos, color = view.src, view.color

        self.circuit.merge(
            w, splice
        )  # w keeps its identity (uid); splice's branches move to w
        for x in self.circuit.attachments(w):  # their ends now name w
            Touched.wire(x.uid)
        for gone in (w, splice):
            self._drop_wire_view(gone)
        self.wire_views[w] = WireView(
            w,
            src_pos,
            bends,
            self.end_pos(w.dst, far_pos),
            self.world,
            self.layers,
            color,
            index=self.wire_index,
        )
        self.refresh_wires([self.wire_views[w]])

    def _attach_point(self, x: Wire, parent: Wire) -> Point:
        """Where wire `x` touches `parent`."""
        xv = self.wire_views[x]
        return xv.src if x.src is parent else xv.dst

    @staticmethod
    def _far_end(x: Wire, parent: Wire):
        return x.dst if x.src is parent else x.src

    def delete_selection(self) -> None:
        self.remove_wires(list(self.selection.wires))
        self.remove_parts(list(self.selection.parts))
        self.selection.clear()

    def pin_pos(self, pin: Pin) -> Point:
        return self.part_views[pin.part].pin_pos(pin)

    def wires_touching(self, parts: set[Part]) -> list[WireView]:
        wires = {
            w for part in parts for pin in part.pins for w in self.circuit.ends_on(pin)
        }
        return [self.wire_views[w] for w in wires]

    def refresh_wires_touching(self, parts: set[Part]) -> None:
        self.refresh_wires(self.wires_touching(parts))

    def refresh_wires(self, views) -> None:
        """Re-attach the ends of `views` -- and of every wire hanging off them -- to
        their pins / parent wires. Pin ends snap to the pin; junction ends slide to
        the nearest point on their parent wire. Parents go first (creation order)."""
        todo = {v.wire for v in views}
        if not todo:
            return
        for wire in sorted(
            todo.union(self.circuit.descendants(*todo)), key=lambda w: w.uid
        ):  # parents first
            view = self.wire_views[wire]
            view.set_ends(
                self.end_pos(wire.src, view.src), self.end_pos(wire.dst, view.dst)
            )

    def end_pos(self, end: Pin | Wire, near: Point) -> Point:
        if isinstance(end, Pin):
            return self.pin_pos(end)
        return project_onto(self.wire_views[end].points, near)

    def _build_demo(self) -> None:
        # Everything on grid points, so the wires come out straight.
        if "NAND" not in self.parts:
            return  # the project's part scripts no longer have one
        a = self.add_part("IN", 200, 360)  # output pin at (240, 380)
        b = self.add_part("IN", 200, 220)  # output pin at (240, 240)
        g = self.add_part(
            "NAND", 380, 280
        )  # inputs at y=320 / y=300, output at (460, 310)
        out = self.add_part("OUT", 580, 290)  # input pin at (580, 310)
        self.connect(a.part.outputs[0], g.part.inputs[0], [(320, 380), (320, 320)])
        self.connect(b.part.outputs[0], g.part.inputs[1], [(320, 240), (320, 300)])
        self.connect(g.part.outputs[0], out.part.inputs[0])

    # ======================================================================
    # hit testing (world coordinates)
    # ======================================================================

    @property
    def slop(self) -> float:
        """HIT_SLOP_PX converted to world units, so clicking feels the same at any zoom."""
        return T.HIT_SLOP_PX / self.camera.zoom

    @property
    def pins_clickable(self) -> bool:
        """Zoomed far out, pins are specks and their (screen-sized) reach would cover
        the whole part: clicks go to parts instead, and wires can't be started or ended."""
        return T.PIN_RADIUS * self.camera.zoom >= T.PIN_HIT_MIN_PX

    def pin_at(self, wx: float, wy: float) -> Pin | None:
        if not self.pins_clickable:
            return None
        for view in ordered(self.part_index.near(wx, wy, self.slop), newest_first=True):
            if self.mode is Mode.PLACING_PART and view in self.placing_views:
                continue  # parts on the cursor aren't targets
            if pin := view.pin_at(wx, wy, self.slop):
                return pin
        return None

    def part_at(self, wx: float, wy: float) -> PartView | None:
        hit = next(
            (
                v
                for v in ordered(self.part_index.near(wx, wy, 0), newest_first=True)
                if v.contains(wx, wy)
            ),
            None,
        )
        if hit is None and not self.pins_clickable:
            # Zoomed far out parts are a few px big: the nearest within reach will do
            # (only here, where pins don't claim the space around parts).
            near = (
                (v.distance_to(wx, wy), -v.seq, v)
                for v in self.part_index.near(wx, wy, self.slop)
            )
            d, _, hit = min(near, default=(math.inf, 0, None))
            hit = hit if d <= self.slop else None
        return hit

    def wire_at(self, wx: float, wy: float) -> WireView | None:
        """The nearest wire within reach (the older one on a tie, e.g. right on a junction)."""
        limit = T.WIRE_THICKNESS / 2 + self.slop
        near = (
            (v.distance_to(wx, wy), v.seq, v)
            for v in self.wire_index.near(wx, wy, limit)
        )
        d, _, best = min(near, default=(math.inf, 0, None))
        return best if d <= limit else None

    def wire_target(self, wx: float, wy: float) -> tuple[Pin | Wire, Point] | None:
        """What a wire being drawn would connect to here: a pin (preferred) or a
        point on another wire."""
        if pin := self.pin_at(wx, wy):
            return pin, self.pin_pos(pin)
        if view := self.wire_at(wx, wy):
            return view.wire, project_onto(view.points, self.snapped(wx, wy))
        return None

    def can_wire_to(self, end: Pin | Wire | None) -> bool:
        return (
            end is not None
            and self.wire_start is not None
            and self.circuit.can_connect(self.wire_start, end)
        )

    # ======================================================================
    # input: the state machine
    # ======================================================================

    def on_mouse_press(self, x, y, button, modifiers):
        self.mouse = (x, y)
        wx, wy = self.camera.screen_to_world(x, y)
        in_picker = self.picker.hit(x, y)  # None unless the cursor is over the picker
        tool = (
            in_picker.part
            if isinstance(in_picker, Row) and in_picker.what == "part"
            else None
        )

        if self.mode is Mode.POPOVER:
            self._popover_press(x, y, button)
            return

        if self.mode is Mode.PROMPT:
            item = self.prompt.item_at(x, y)
            if button == mouse.LEFT and item is not None:
                self.prompt.selected = item
                self.prompt_enter(self.prompt)
            elif not self.prompt.contains(x, y):
                self._close_prompt()  # a click outside is "never mind"
            return

        if self.mode is Mode.MENU:
            item = self.menu.item_at(x, y)
            if button == mouse.LEFT and item is not None and item.submenu:
                self.menu.hover(x, y)  # opens on hover already; a click just makes sure
                return
            inside = self.menu.contains(x, y)
            self._close_menu()
            if button == mouse.LEFT and item is not None:
                # Done: the cursor goes back to where you right-clicked (what the menu was
                # about, and where the lens still looks), so e.g. Branch starts from there
                self._warp(*self.menu.anchor)
                item.action()  # after closing: may start another mode (e.g. label editing)
                return
            if button == mouse.MIDDLE:
                self.panning = True
            if button != mouse.RIGHT or inside:
                return  # a click outside the menu only closes it
            # right-click elsewhere: fall through and open a menu there instead

        if self.mode is Mode.EDITING_LABEL:
            self._finish_edit(commit=True)  # clicking anywhere else commits
            return
        if self.mode is Mode.RENAMING:
            self._finish_rename(commit=True)  # same for collection names
            return

        if self.panels.minimap.contains(x, y):
            # Steer from the map (with any button, in any mode that isn't holding a button already).
            if self.map_button is None and not self.panning and self.mode in STEERABLE:
                self.map_button = button
                self.panels.minimap.held = True
                self._look_at(*self.panels.minimap.to_world(x, y))
            return
        if button == mouse.MIDDLE:
            self.panning = True
            return
        if self.bar.contains(x, y):
            if (
                button == mouse.LEFT
                and self.bar.cog_hit(x, y)
                and self.mode is Mode.IDLE
            ):
                self._cog_menu()
            return  # the bar isn't board: no placing, bends or selecting under it
        if self.panels.contains(x, y):
            return  # nor is the lens

        if self.mode is Mode.EDITING_WIRE and in_picker:
            self._finish_wire_edit(commit=True)  # like any click away from the wire
            return
        if self.mode is Mode.EDITING_WIRE:
            view = self.wire_edit.view
            self._wire_edit_press(x, y, wx, wy, button)
            self.refresh_wires([view])  # branches slide along the edited wire
            return

        if self.mode is Mode.PLACING_PART:
            if button == mouse.LEFT and in_picker:
                # Back onto the panel: never mind. A press there then acts like any picker press
                # (so rows can be dragged right away; clicking another part swaps to it), except
                # that pressing the part being carried just puts it back -- which is also what
                # the second half of a double-click does.
                held = self.placing_kind
                self._cancel()
                self._picker_press(
                    in_picker, x, y, put_back=tool is not None and tool == held
                )
            elif button == mouse.LEFT:
                # Shift = "keep another", but Ctrl+Shift is subgrid snapping, not a request for more.
                again = bool(modifiers & key.MOD_SHIFT) and not modifiers & key.MOD_CTRL
                self._commit_placing(again=again)
            elif button == mouse.RIGHT:
                self._cancel()
            return

        if self.mode is Mode.WIRING:
            if button == mouse.LEFT and tool:
                self._cancel()
                self._start_placing(tool)
            elif button == mouse.LEFT and in_picker:
                pass  # no bend points under the panel
            elif button == mouse.LEFT:
                target = self.wire_target(wx, wy)
                end, end_pos = target if target else (None, None)
                if end is self.wire_start:
                    self._cancel()
                elif self.can_wire_to(end):
                    start_pos = (
                        None
                        if isinstance(self.wire_start, Pin)
                        else self.wire_start_pos
                    )
                    # a branch takes the color of the wire it comes off (or joins)
                    color = next(
                        (
                            self.wire_views[e].color
                            for e in (self.wire_start, end)
                            if isinstance(e, Wire) and self.wire_views[e].color
                        ),
                        None,
                    )
                    self.connect(
                        self.wire_start,
                        end,
                        self.wire_bends,
                        start_pos,
                        None if isinstance(end, Pin) else end_pos,
                        color=color,
                    )
                    self._cancel()  # clears the preview; the wire now exists
                elif end is None:
                    self.wire_bends.append(self.snapped(wx, wy))
                    self._update_preview()
                # clicking an invalid target (in->in pin, same part, ...) does nothing
            elif button == mouse.RIGHT:
                self._pop_bend_or_cancel()
            return

        if self.mode in (
            Mode.PRESSING_PART,
            Mode.DRAGGING_PART,
            Mode.BOX_SELECTING,
            Mode.PRESSING_WIRE,
            Mode.PICKER_PRESS,
            Mode.PICKER_DRAG,
        ):
            return  # another button while the left one is held down: ignore (middle already panned)

        # IDLE
        if in_picker:
            if button == mouse.LEFT:
                self._picker_press(in_picker, x, y)
            elif button == mouse.RIGHT:
                self._picker_menu(in_picker, x, y)
            return
        if button == mouse.LEFT:
            if pin := self.pin_at(wx, wy):
                self._start_wiring(pin, self.pin_pos(pin))
            elif modifiers & key.MOD_ALT and (view := self.wire_at(wx, wy)):
                self._start_wiring(
                    view.wire, project_onto(view.points, self.snapped(wx, wy))
                )
            elif view := self.part_at(wx, wy):
                if modifiers & key.MOD_SHIFT and not modifiers & key.MOD_CTRL:
                    self.selection.toggle(
                        view
                    )  # never toggles a switch; that's how you select one
                    return  # (Ctrl+Shift is subgrid snapping: press + drag as usual)
                self.active = view
                self.grab = (view.x - wx, view.y - wy)
                self.press_at = (x, y)
                self.mode = Mode.PRESSING_PART
            elif wire := self.wire_at(wx, wy):
                if modifiers & key.MOD_SHIFT:
                    self.selection.toggle(wire)
                else:
                    self.pressed_wire, self.press_world, self.press_at = (
                        wire,
                        (wx, wy),
                        (x, y),
                    )
                    self.mode = Mode.PRESSING_WIRE
            else:
                shift = modifiers & key.MOD_SHIFT
                self.box_base = (
                    (set(self.selection.parts), set(self.selection.wires))
                    if shift
                    else (set(), set())
                )
                if not shift:
                    self.selection.clear()  # a plain click on empty space ends here: cleared
                self.box_start = (wx, wy)
                self.press_at = (x, y)
                self.mode = Mode.BOX_SELECTING

        elif button == mouse.RIGHT:
            # Right-clicking narrows the selection to what the menu will act on, so the
            # highlight shows exactly that: the clicked item -- or, on a part inside a
            # selection of parts all of one kind, those parts (Ctrl: just the clicked one).
            if view := self.part_at(wx, wy):
                sel = self.selection.parts
                group = (
                    sorted(sel, key=lambda v: v.part.uid)
                    if view in sel
                    and not modifiers & key.MOD_CTRL
                    and len({v.part.type for v in sel}) == 1
                    else [view]
                )
                self.selection.set(parts=group)
                items = (
                    [MenuItem("Label...", lambda: self._start_edit(view))]
                    if len(group) == 1
                    else []
                )
                if view.look.lit:  # switches and LEDs are color sources (paint.py)
                    items.append(
                        MenuItem(
                            "Recolor",
                            submenu=self._recolor_items(
                                _common(part_color(v.part) for v in group),
                                lambda c: [self._set_part_color(v, c) for v in group],
                            ),
                        )
                    )
                items += self._settings_items(group)
                items.append(
                    MenuItem("Delete", lambda: self.remove_parts(group), danger=True)
                )
                self._open_menu(x, y, items)
            elif wire := self.wire_at(wx, wy):
                self.selection.set(wires=[wire])
                at = project_onto(wire.points, (wx, wy))
                self._open_menu(
                    x,
                    y,
                    [
                        MenuItem("Edit", lambda: self._start_wire_edit(wire)),
                        MenuItem("Branch", lambda: self._start_wiring(wire.wire, at)),
                        MenuItem(
                            "Recolor",
                            submenu=self._recolor_items(
                                wire.color, lambda c: setattr(wire, "color", c)
                            ),
                        ),
                        MenuItem(
                            "Delete", lambda: self.cut_wire(wire, at), danger=True
                        ),
                    ],
                )
            else:
                self.panning = True

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.mouse = (x, y)
        if self.mode is Mode.POPOVER:
            self._popover_drag(x)
            return
        if self.map_button is not None:
            self._look_at(*self.panels.minimap.to_world(x, y))
            return
        if self.panning:
            self.camera.pan(dx, dy)
        elif self.mode is Mode.PRESSING_PART:
            px, py = self.press_at
            if abs(x - px) + abs(y - py) >= T.DRAG_THRESHOLD_PX:
                self._begin_group_drag(self.active)
        elif self.mode is Mode.PRESSING_WIRE:
            px, py = self.press_at
            if abs(x - px) + abs(y - py) >= T.DRAG_THRESHOLD_PX:
                # Branch from where the press happened, not where the drag got to.
                view = self.pressed_wire
                self.pressed_wire = None
                self._start_wiring(
                    view.wire,
                    project_onto(view.points, self.snapped(*self.press_world)),
                )
        elif self.mode is Mode.PICKER_PRESS:
            px, py = self.press_at
            if abs(x - px) + abs(y - py) >= T.DRAG_THRESHOLD_PX:
                self.picker.begin_drag(self.picker_row, self.press_at)
                self.mode = Mode.PICKER_DRAG
        if self.mode is Mode.PICKER_DRAG:
            row = self.picker_row
            if row.what == "part" and not self.picker.contains(x, y):
                # Carried out of the panel: it becomes the part itself, on the cursor.
                self.picker.cancel_drag()
                self.mode, self.picker_row = Mode.IDLE, None
                self._start_placing(row.part)
            else:
                self.picker.drag_to(x, y)
        self._follow_cursor()

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse, self.mouse_in = (x, y), True
        self.bar.set_hover(self.mode is Mode.IDLE and self.bar.cog_hit(x, y))
        if self.mode is Mode.PROMPT:
            self.prompt.hover(x, y)
            return
        hover_ok = self.mode in (Mode.IDLE, Mode.PLACING_PART, Mode.WIRING)
        self.picker.set_hover(self.picker.hit(x, y) if hover_ok else None)
        if self.mode is Mode.MENU:
            self.menu.hover(x, y)
        elif self.mode is Mode.EDITING_WIRE and self.wire_edit.set_hover(
            self._wire_edit_target(x, y)
        ):
            hand = self.get_system_mouse_cursor(self.CURSOR_HAND)
            self.set_mouse_cursor(hand if self.wire_edit.hover else None)
        self._follow_cursor()
        self._update_pin_labels()

    def on_mouse_enter(self, x, y):
        self.mouse, self.mouse_in = (x, y), True

    def on_mouse_leave(self, x, y):
        self.mouse_in = False

    def on_mouse_release(self, x, y, button, modifiers):
        if button == self.map_button:
            self.map_button, self.panels.minimap.held = None, False
            return
        if self.mode is Mode.POPOVER:
            if button == mouse.LEFT:
                self._popover_release()
            return
        # Releases never *finish* a click-to-place or click-to-wire action.
        # That is what makes press-and-hold on a picker part or pin harmless.
        if button in (mouse.MIDDLE, mouse.RIGHT) and self.panning:
            self.panning = False
        elif button == mouse.LEFT and self.mode is Mode.PRESSING_PART:
            # A click without movement: clickable parts (switches) get the click,
            # everything else gets selected.
            if not self.circuit.click(self.active.part):
                self.selection.set(parts=[self.active])
            self.mode, self.active = Mode.IDLE, None
        elif button == mouse.LEFT and self.mode is Mode.BOX_SELECTING:
            self._end_box()
        elif button == mouse.LEFT and self.mode is Mode.PRESSING_WIRE:
            self.selection.set(wires=[self.pressed_wire])  # a click, not a drag: select
            self.pressed_wire, self.mode = None, Mode.IDLE
        elif button == mouse.LEFT and self.mode is Mode.DRAGGING_PART:
            self._end_move()
            self.mode, self.active = Mode.IDLE, None
        elif button == mouse.LEFT and self.mode is Mode.EDITING_WIRE:
            self.wire_edit.end_drag()
        elif button == mouse.LEFT and self.mode is Mode.PICKER_PRESS:
            row, self.picker_row, self.mode = self.picker_row, None, Mode.IDLE
            self.last_picker_click = (row.key, time.monotonic())
            if self.picker_bounce:
                pass  # 2nd half of a double-click, or putting the carried part back
            elif row.what == "part":
                self._start_placing(row.part)
            else:
                row.collection.open = not row.collection.open
                self.picker.refresh()
        elif button == mouse.LEFT and self.mode is Mode.PICKER_DRAG:
            self.picker.end_drag()
            self.mode, self.picker_row = Mode.IDLE, None

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.mouse = (x, y)
        if self.mode is Mode.POPOVER:
            return  # (it's pinned to where it opened: don't let the board move away under it)
        if self.mode is Mode.PROMPT:
            if scroll_y:
                self.prompt.move(-1 if scroll_y > 0 else 1)
            return
        if self.mode is Mode.MENU and self.picker.contains(x, y):
            self._close_menu()  # (the picker would scroll away under a menu on it)
        if self.picker.contains(x, y):
            self.picker.scroll_by(
                scroll_y
            )  # eases there; a drag follows along (see PartPicker.update)
            if self.mode is not Mode.PICKER_DRAG:
                self.picker.set_hover(self.picker.hit(x, y))
            return
        if (
            scroll_y
            and self.snapping
            and self.mode is Mode.IDLE
            and self._tiling_active()
        ):
            self._space_tiling(scroll_y)
            return
        if self.panels.scrolls_lens(x, y) or (
            self.lens_key is not None and self.keys[key.G] and self.panels.lens.open
        ):
            self.panels.lens.adjust(
                scroll_y
            )  # the lens's magnification; the view stays put
            if self.lens_key is not None:
                self.lens_key[1] = True
            return
        if (
            self.mode is Mode.MENU
        ):  # zoom around what the menu belongs to, so it stays on it
            self.camera.scroll(*self.menu.anchor, scroll_y)
            return
        if (at := self.panels.anchor(x, y, self._visible_world())) is not None:
            # On the map: zoom around the spot pointed at -- after jumping there if it's
            # outside the view (zooming around a point off screen would slide the view away).
            p, jump = at
            if jump:
                self._look_at(*p)
            x, y = self.camera.world_to_screen(*p)
        elif self.panels.contains(x, y):
            x, y = self._board_center()
        self.camera.scroll(x, y, scroll_y)
        self._follow_cursor()

    def on_key_press(self, symbol, modifiers):
        # Deliberately NOT calling super(): pyglet's default closes the window on Esc.
        if self.mode is Mode.POPOVER:
            # Typing goes through on_text / on_text_motion. No editor shortcuts while it's up.
            if symbol in (key.ENTER, key.NUM_ENTER):
                if self.pop_edit.drag_old is None and self._popover_enter():
                    self._close_popover()
            elif symbol == key.ESCAPE:
                self._close_popover()
            return
        if self.mode is Mode.PROMPT:
            # Typing goes through on_text / on_text_motion. No editor shortcuts while it's up.
            if symbol in (key.ENTER, key.NUM_ENTER):
                self.prompt_enter(self.prompt)
            elif symbol == key.ESCAPE:
                self._close_prompt()
            elif symbol in (key.UP, key.DOWN):
                self.prompt.move(-1 if symbol == key.UP else 1)
            elif self.prompt_key is not None and not modifiers & (
                key.MOD_CTRL | key.MOD_ALT
            ):
                self.prompt_key(symbol)
            return
        if self.mode is Mode.EDITING_LABEL:
            # Text goes through on_text / on_text_motion; only Enter/Esc matter here.
            # Everything else (Home, Backspace, Ctrl...) must not trigger editor shortcuts.
            if symbol in (key.ENTER, key.NUM_ENTER):
                self._finish_edit(commit=True)
            elif symbol == key.ESCAPE:
                self._finish_edit(commit=False)
            return
        if self.mode is Mode.RENAMING:
            if symbol in (key.ENTER, key.NUM_ENTER):
                self._finish_rename(commit=True)
            elif symbol == key.ESCAPE:
                self._finish_rename(commit=False)
            return
        if symbol in (key.LCTRL, key.RCTRL, key.LSHIFT, key.RSHIFT):
            self._follow_cursor()  # (sub)snap whatever is on the cursor right away
        elif self.mode is Mode.EDITING_WIRE and symbol in (key.ENTER, key.NUM_ENTER):
            self._finish_wire_edit(commit=True)
        elif (
            self.mode is Mode.EDITING_WIRE
            and symbol in (key.DELETE, key.BACKSPACE)
            and self.wire_edit.hover
            and self.wire_edit.hover[0] == "bend"
        ):
            self.wire_edit.remove(self.wire_edit.hover[1])  # delete the hovered bend
            self.refresh_wires([self.wire_edit.view])
        elif symbol == key.ESCAPE:
            if self.mode is Mode.IDLE:
                self.selection.clear()
            else:
                self._cancel()
        elif symbol in (key.DELETE, key.BACKSPACE) and self.mode is Mode.IDLE:
            self.delete_selection()
        elif symbol == key.A and modifiers & key.MOD_CTRL and self.mode is Mode.IDLE:
            self.selection.set(self.part_views.values(), self.wire_views.values())
        elif symbol == key.S and modifiers & key.MOD_CTRL and self.mode is Mode.IDLE:
            if modifiers & key.MOD_SHIFT:
                self._save_as()
            else:
                self._save()
        elif symbol == key.O and modifiers & key.MOD_CTRL and self.mode is Mode.IDLE:
            self._open_dialog()
        elif symbol == key.N and modifiers & key.MOD_CTRL and self.mode is Mode.IDLE:
            self._unsaved_then(self._new)
        elif modifiers & key.MOD_CTRL and symbol in (key.Z, key.Y):
            if self.mode is not Mode.IDLE:
                self._cancel()  # mid-action: undo means "never mind", not "and the step before"
            elif symbol == key.Y or modifiers & key.MOD_SHIFT:
                self._apply(self.history.redo())
            else:
                self._apply(self.history.undo())
        elif (
            modifiers & key.MOD_CTRL
            and symbol in (key.C, key.X)
            and self.mode is Mode.IDLE
        ):
            if self.selection.parts:
                self.clipboard = capture(self, self.selection.parts)
                if symbol == key.X:
                    self.delete_selection()
        elif (
            modifiers & key.MOD_CTRL
            and symbol == key.V
            and self.mode is Mode.IDLE
            and self.clipboard
        ):
            self._start_paste()
        elif modifiers & key.MOD_CTRL and symbol == key.D and self.mode is Mode.IDLE:
            self._duplicate()
        elif symbol == key.BACKSPACE and self.mode is Mode.WIRING:
            self._pop_bend_or_cancel()
        elif symbol == key.TAB:
            self.pin_label_mode = (self.pin_label_mode + 1) % len(PIN_LABEL_MODES)
            self._update_pin_labels()
            self._notice(f"pin names: {PIN_LABEL_MODES[self.pin_label_mode]}")
        elif symbol == key.M and not modifiers & (key.MOD_CTRL | key.MOD_ALT):
            self.panels.minimap.open = not self.panels.minimap.open
        elif symbol == key.G and not modifiers & (key.MOD_CTRL | key.MOD_ALT):
            # Opens right away (so G+scroll has something to adjust); closes on release,
            # unless it was just opened or you scrolled while holding G.
            self.lens_key = [self.panels.lens.open, False]
            self.panels.lens.open = True
        elif symbol == key.HOME:
            self.camera.set_level(0, 0, 0)
            self.camera.center_on(*HOME, self.width, self.height)
            self._follow_cursor()

    def on_key_release(self, symbol, modifiers):
        if symbol == key.G and self.lens_key is not None:
            was_open, scrolled = self.lens_key
            self.lens_key = None
            if was_open and not scrolled:
                self.panels.lens.open = False
        if symbol in (key.LCTRL, key.RCTRL, key.LSHIFT, key.RSHIFT):
            self._follow_cursor()  # un-snap / back to the normal grid

    def on_text(self, text):
        if self.mode is Mode.POPOVER:
            self.popover.type_text(text)
        elif self.mode is Mode.PROMPT:
            self.prompt.type_text(text)
        elif self.mode is Mode.EDITING_LABEL:
            self.edit.insert(text)
            self._update_edit()
        elif self.mode is Mode.RENAMING:
            self.picker.rename_text(text)

    def on_text_motion(self, motion):
        if self.mode is Mode.POPOVER:
            self.popover.motion(motion)
        elif self.mode is Mode.PROMPT:
            self.prompt.motion(motion)
        elif self.mode is Mode.EDITING_LABEL:
            self.edit.motion(motion)
            self._update_edit()
        elif self.mode is Mode.RENAMING:
            self.picker.rename_motion(motion)

    # ---- helpers -----------------------------------------------------------

    def _open_menu(self, x: float, y: float, items: list[MenuItem]) -> None:
        self.menu.open(x, y, items, self.width, self.height)
        self.menu.hover(x, y)
        self.mode = Mode.MENU

    @staticmethod
    def _recolor_items(current: str | None, set_color) -> list[MenuItem]:
        """The palette, then Default (inherit a color: paint.py), for a wire or an IN/OUT part."""
        return [
            *(
                MenuItem(
                    name.capitalize(),
                    lambda c=name: set_color(c),
                    swatch=on,
                    checked=current == name,
                )
                for name, (_, on) in T.WIRE_COLORS.items()
            ),
            MenuItem(
                "Default",
                lambda: set_color(None),
                swatch=RAINBOW,
                checked=current is None,
            ),
        ]

    def _settings_items(self, views: list[PartView]) -> list[MenuItem]:
        """Menu rows for the settings and actions of the parts' type (PartType.settings,
        .actions): all of `views` are of that one kind, and every row acts on all of them.
        Where their values differ, a row says "mixed" and nothing is checked."""
        t = views[0].part.type
        items = []
        for k, s in t.settings.items():
            value = _common(v.part.props.get(k) for v in views)
            shown = MIXED if value is MIXED else s.show(value)
            title = f"{s.title(k)}: {shown}"
            if isinstance(s, Choice):
                items.append(
                    MenuItem(
                        title,
                        submenu=[
                            MenuItem(
                                s.show(c),
                                lambda k=k, c=c: self._set_setting(views, k, c),
                                checked=value is not MIXED
                                and value == c
                                and type(value) is type(c),
                            )
                            for c in s.values
                        ],
                    )
                )
            elif isinstance(s, Toggle):  # (a mixed one turns on)
                items.append(
                    MenuItem(
                        title,
                        lambda k=k, on=value is not True: self._set_setting(
                            views, k, on
                        ),
                        checked=value is True,
                    )
                )
            elif isinstance(s, Number):
                items.append(
                    MenuItem(
                        title + "...",
                        lambda k=k, s=s, value=value: self._open_popover(
                            views, k, s, value
                        ),
                    )
                )
            else:  # Text: typed into a prompt
                items.append(
                    MenuItem(
                        title + "...",
                        lambda k=k, s=s, value=value: self._setting_prompt(
                            views, k, s, value
                        ),
                    )
                )
        for name, a in t.actions.items():
            items.append(
                MenuItem(
                    a.title(name),
                    lambda name=name: self._run_action(views, name),
                    danger=a.danger,
                )
            )
        return items

    def _set_setting(self, views: list[PartView], key: str, value) -> bool:
        """One finished edit of a setting on all of `views` (see Circuit.set_setting).
        False (and the reason reported) if the value doesn't parse."""
        try:
            self.circuit.set_setting([v.part for v in views], key, value)
        except ValueError as e:
            self._report(str(e))
            return False
        Touched.parts.update(
            v.part.uid for v in views
        )  # (settings don't change colors: no repaint)
        return True

    def _setting_prompt(self, views: list[PartView], key: str, s, value) -> None:
        """Type a Text setting's new value (mixed values start out empty)."""
        hint = (
            f"{s.hint}   Enter: set   Esc: cancel"
            if s.hint
            else "Enter: set   Esc: cancel"
        )
        p = Prompt(
            self.hud,
            self.width,
            self.height,
            s.title(key) + _count(views),
            text="" if value is MIXED else value,
            max_len=s.max_len,
            hint=hint,
        )

        def enter(p: Prompt) -> None:
            if self._set_setting(views, key, p.text):
                self._close_prompt()

        self._open_prompt(p, enter)

    # ---- the Number popover: a slider and a typed field ---------------------------
    # Dragging the slider changes the props live (applied once a frame, in update());
    # the part's changed() hook runs on release, unless the setting is live. Every
    # release and every Enter is one undo step. Esc takes back a drag in progress and
    # closes; a click outside closes (taking a typed, valid value first).

    def _open_popover(self, views: list[PartView], key: str, s: Number, value) -> None:
        self._cancel()
        hint = s.hint or f"{s.show(s.min)} to {s.show(s.max)}"
        self.popover = NumberPopover(
            self.hud,
            self.width,
            self.height,
            self.mouse,
            s.title(key) + _count(views),
            s,
            None if value is MIXED else value,
            hint,
        )
        self.pop_edit = _NumberEdit([v.part for v in views], key, s)
        self.mode = Mode.POPOVER

    def _popover_press(self, x: float, y: float, button: int) -> None:
        pop, e = self.popover, self.pop_edit
        if not pop.contains(x, y):
            self._close_popover(
                take_typed=True
            )  # a click outside: done (the click goes no further)
        elif button == mouse.LEFT and pop.on_slider(x, y):
            e.drag_old = [p.props.get(e.key) for p in e.parts]
            e.pending = pop.value_at(x)

    def _popover_drag(self, x: float) -> None:
        if self.pop_edit.drag_old is not None:
            self.pop_edit.pending = self.popover.value_at(
                x
            )  # applied in update(): once a frame

    def _popover_apply(self) -> None:
        """Write the slider's latest value (at most once a frame)."""
        e = self.pop_edit
        if e is not None and e.pending is not None:
            value, e.pending = e.pending, None
            self.circuit.set_setting(
                e.parts, e.key, value, notify=False
            )  # (a live setting notifies)
            self.popover.set_value(value)

    def _popover_release(self) -> None:
        e = self.pop_edit
        if e.drag_old is None:
            return
        self._popover_apply()
        if not e.setting.live:  # (a live one heard about every frame already)
            self.circuit.settings_changed(e.parts, e.key, e.drag_old)
        e.drag_old = None
        self._record_setting(e.parts)

    def _popover_enter(self) -> bool:
        """Set the typed value. False (the reason in the hint line) if it isn't one."""
        e, pop = self.pop_edit, self.popover
        try:
            value = e.setting.parse_text(pop.text)
            self.circuit.set_setting(e.parts, e.key, value)
        except ValueError as err:
            pop.set_hint(str(err), danger=True)
            return False
        pop.set_value(value)
        self._record_setting(e.parts)
        return True

    def _record_setting(self, parts: list[Part]) -> None:
        """One undo step for a settings edit, now (the popover isn't idle, so the
        after-event recording in dispatch_event doesn't do it)."""
        Touched.parts.update(
            p.uid for p in parts
        )  # (no repaint: settings don't change colors)
        self._record()

    def _close_popover(self, take_typed: bool = False) -> None:
        e, pop = self.pop_edit, self.popover
        if e.drag_old is not None:  # Esc mid-drag: take it back
            e.pending = None
            self.circuit.put_setting(e.parts, e.key, e.drag_old, notify=e.setting.live)
        elif take_typed and pop.text.strip() and pop.text != pop.shown_text:
            if not self._popover_enter():
                return  # not a number: stay open, the hint says why
        pop.delete()
        self.popover = self.pop_edit = None
        self.mode = Mode.IDLE

    def _run_action(self, views: list[PartView], name: str) -> None:
        """Run a part action; whatever props it changed become an undo step."""
        before = [copy.deepcopy(v.part.props) for v in views]
        self.circuit.run_action([v.part for v in views], name)
        Touched.parts.update(
            v.part.uid for v, was in zip(views, before) if v.part.props != was
        )

    @staticmethod
    def _set_part_color(view: PartView, color: str | None) -> None:
        if color is None:
            view.part.props.pop("color", None)
        else:
            view.part.props["color"] = color
        Touched.part(view.part.uid)

    def _close_menu(self) -> None:
        self.menu.close()
        self.mode = Mode.IDLE

    def _start_wire_edit(self, view: WireView) -> None:
        self.selection.discard(view)  # one glow at a time
        self.mode = Mode.EDITING_WIRE
        parents = {
            end: self.wire_views[e]
            for end, e in zip(("src", "dst"), view.wire.ends)
            if isinstance(e, Wire)
        }
        branches = [
            (self.wire_views[w], end)
            for w in self.circuit.attachments(view.wire)
            for end, e in zip(("src", "dst"), w.ends)
            if e is view.wire
        ]
        self.wire_edit = WireEditSession(
            view, self.camera, self.world, self.layers, parents, branches
        )
        self.wire_edit.set_hover(self._wire_edit_target(*self.mouse))

    def _wire_edit_target(self, x: float, y: float):
        return self.wire_edit.target_at(
            x, y, prefer_junction=self.keys[key.LALT] or self.keys[key.RALT]
        )

    def _wire_edit_press(
        self, x: float, y: float, wx: float, wy: float, button: int
    ) -> None:
        s = self.wire_edit
        target = self._wire_edit_target(x, y)
        if button == mouse.LEFT:
            if target and target[0] in ("bend", "junction"):
                s.begin_drag(target, wx, wy)
            elif target:  # "+" handle: new bend at the midpoint
                s.begin_drag(
                    ("bend", s.insert(target[1], s.add_handle_pos(target[1]))), wx, wy
                )
            elif hit := s.segment_at(wx, wy):  # on the wire itself: split right there
                s.begin_drag(("bend", s.insert(*hit)), wx, wy)
            else:
                self._finish_wire_edit(commit=True)  # click elsewhere finishes
        elif button == mouse.RIGHT:
            if target and target[0] == "bend":
                s.remove(target[1])
            elif target and target[0] == "junction":
                pass  # nothing to remove; don't end the edit over a near miss
            else:
                self._finish_wire_edit(commit=True)

    def _finish_wire_edit(self, commit: bool) -> None:
        if not commit:
            self.wire_edit.revert()
            self.refresh_wires([self.wire_edit.view])
        self.wire_edit.close()
        self.wire_edit = None
        self.set_mouse_cursor(None)
        self.mode = Mode.IDLE

    def _start_edit(self, view: PartView) -> None:
        self.mode = Mode.EDITING_LABEL
        self.edit_view = view
        self.edit = LineEdit(view.part.label, LABEL_MAX)
        self.caret = shapes.Rectangle(
            0, 0, 1, 1, color=T.CARET, batch=self.world.batch, group=self.layers.overlay
        )
        pyglet.clock.schedule_interval(self._blink_caret, 0.5)
        self._update_edit()

    def _update_edit(self) -> None:
        """Show the in-progress text and put the caret where the next character goes."""
        name = self.edit_view.name
        name.set_text(self.edit.text)
        name.move_to(*self.edit_view.name_pos())
        h = name.cap_height * 1.6
        self.caret.position = (name.caret_x(self.edit.caret) - 0.6, name.y - h / 2)
        self.caret.width, self.caret.height = 1.2, h
        self.caret.visible = True  # restart the blink so the caret shows while typing

    def _blink_caret(self, dt: float) -> None:
        if self.caret is not None:
            self.caret.visible = not self.caret.visible

    def _finish_edit(self, commit: bool) -> None:
        view = self.edit_view
        if commit:
            view.part.label = self.edit.text.strip()
        view.refresh_name()  # shows the committed label, or reverts to the old one
        view.name.move_to(*view.name_pos())
        pyglet.clock.unschedule(self._blink_caret)
        self.caret.delete()
        self.caret = None
        self.edit_view, self.edit = None, None
        self.mode = Mode.IDLE

    def _picker_press(self, hit, x: float, y: float, put_back: bool = False) -> None:
        if hit == "toggle":
            self.picker.toggle()
        elif hit == "new":
            self._start_rename(self.library.new_collection(), fresh=True)
        elif isinstance(hit, Row) and hit.what in ("part", "section"):
            last_key, last_time = self.last_picker_click
            bounce = last_key == hit.key and time.monotonic() - last_time < DOUBLE_CLICK
            self.picker_row, self.press_at, self.picker_bounce = (
                hit,
                (x, y),
                put_back or bounce,
            )
            self.mode = (
                Mode.PICKER_PRESS
            )  # a click or a drag; the release / movement decides

    def _picker_menu(self, hit, x: float, y: float) -> None:
        lib = self.library

        def new_collection(index=None):
            self._start_rename(lib.new_collection(index), fresh=True)

        def move(part, dest):
            lib.move_part(part, dest)
            self.picker.refresh()

        def into_new_collection(part, index):
            c = lib.new_collection(index)
            lib.move_part(part, c)
            self._start_rename(c, fresh=True)

        def delete(c):
            lib.delete_collection(c)
            self.picker.refresh()

        if isinstance(hit, Row) and hit.what in ("section", "empty"):
            c = hit.collection
            any_open = any(x.open for x in lib.collections)
            items = [
                MenuItem("Rename...", lambda: self._start_rename(c)),
                MenuItem(
                    "New collection",
                    lambda: new_collection(lib.collections.index(c) + 1),
                ),
                MenuItem(
                    "Collapse all" if any_open else "Expand all",
                    lambda: self.picker.set_all_open(not any_open),
                ),
            ]
            if not c.builtin:
                items.append(MenuItem("Delete", lambda: delete(c), danger=True))
        elif isinstance(hit, Row) and hit.what == "part":
            p, at = (
                hit.part,
                (lib.collections.index(hit.collection) + 1 if hit.collection else None),
            )
            items = [
                MenuItem("Move to new collection", lambda: into_new_collection(p, at))
            ]
            if p.startswith(MACRO):
                items.insert(
                    0,
                    MenuItem("Open", lambda: self._request_open(p.removeprefix(MACRO))),
                )
            if hit.collection is not None:
                items.append(MenuItem("Remove from collection", lambda: move(p, None)))
        else:
            items = [MenuItem("New collection", new_collection)]
        if items:
            self._open_menu(x, y, items)

    def _start_rename(self, c, fresh: bool = False) -> None:
        self.picker.start_rename(c, fresh)
        self.mode = Mode.RENAMING

    def _finish_rename(self, commit: bool) -> None:
        self.picker.finish_rename(commit)
        self.mode = Mode.IDLE

    def _warp(self, sx: float, sy: float) -> None:
        """Move the cursor to screen point (sx, sy), in window px (the OS wants physical px)."""
        r = self._pixel_ratio()
        self.set_mouse_position(round(sx * r), round(sy * r))
        self.mouse = (sx, sy)  # (before the OS's motion event arrives)

    def _pixel_ratio(self) -> float:
        fb_w, _ = self.get_framebuffer_size()
        return fb_w / self.width if self.width else 1.0

    def _start_placing(self, kind: str) -> None:
        if self._unplaceable(kind):
            self._notice(
                f"{kind.removeprefix(MACRO)} can't go in here: it contains {self.doc}"
                if kind != MACRO + self.doc
                else "a macro can't contain itself"
            )
            return
        try:
            self.catalog.get(kind)
        except KeyError as e:
            self._report(
                f"can't place {kind.removeprefix(MACRO)}: {e.args[0] if e.args else e}"
            )
            return
        self._carry(
            [self.add_part(kind, 0, 0, live=False)],
            [],
            again=lambda: self._start_placing(kind),
        )
        self.placing_kind = kind

    def _start_paste(self) -> None:
        views, wires = instantiate(self, self.clipboard, live=False)
        paint(
            self, {v.part.uid for v in views}, {w.wire.uid for w in wires}
        )  # ghosts show their colors too
        self._carry(views, wires, again=self._start_paste)
        self.placing_kind = None

    def _selection_signature(self):
        """What the selection is and where it sits: a Ctrl+D block keeps growing only
        while this is unchanged since its last step."""
        return (
            frozenset((v, v.x, v.y) for v in self.selection.parts),
            frozenset(self.selection.wires),
        )

    def _tiling_active(self) -> bool:
        return (
            self.tiling is not None
            and self.tiling.signature == self._selection_signature()
        )

    def _duplicate(self) -> None:
        if not self.selection.parts:
            return
        if not self._tiling_active():
            unit = list(self.selection.parts)
            wires = internal_wires(self, unit)
            self.tiling = Tiling(capture(self, unit), unit, wires)
        t = self.tiling
        t.grow(t.next_axis(), lambda: Cell.of(*instantiate(self, t.unit)))
        t.adjusting = (
            False  # this press is its own undo step (committed by dispatch_event)
        )
        self._place_tiling()

    def _space_tiling(self, scroll_y: float) -> None:
        t = self.tiling
        if t.last is None:
            return
        shift = self.keys[key.LSHIFT] or self.keys[key.RSHIFT]
        axis = t.last if not shift else (DOWN if t.last == RIGHT else RIGHT)
        if not t.adjust(axis, 1 if scroll_y > 0 else -1):
            return
        self._place_tiling()
        if t.adjusting:
            self._record(amend=True)  # a run of Ctrl+scroll notches undoes as one step
        else:
            t.adjusting = self._record()

    def _place_tiling(self) -> None:
        t = self.tiling
        wires = t.layout()
        self.refresh_wires(
            [*self.wires_touching({v.part for v in t.all_parts()}), *wires]
        )
        self.selection.set(t.all_parts(), t.all_wires())
        t.signature = self._selection_signature()

    def _carry(self, views: list[PartView], wires: list[WireView], again) -> None:
        """Attach new (ghost) parts + wires to the cursor, centered on it, until a click."""
        self.selection.clear()
        for v in views:
            v.set_ghost(True)
        for w in wires:
            w.set_ghost(True)
        x0 = min(v.x for v in views)
        y0 = min(v.y for v in views)
        x1 = max(v.x + v.w for v in views)
        y1 = max(v.y + v.h for v in views)
        # The first part is the anchor: Ctrl snaps *its* origin, so a pasted layout
        # that was on the grid lands on the grid again.
        anchor = views[0]
        self.grab = (anchor.x - (x0 + x1) / 2, anchor.y - (y0 + y1) / 2)
        self.drag_origin = (anchor.x, anchor.y)
        self._begin_move(views, wires)
        self.placing_views, self.placing_wires, self.place_again = views, wires, again
        self.mode = Mode.PLACING_PART
        self._follow_cursor()

    def _commit_placing(self, again: bool) -> None:
        self._end_move()
        views, wires, place_again = (
            self.placing_views,
            self.placing_wires,
            self.place_again,
        )
        for v in views:
            v.set_ghost(False)
            self.circuit.open_part(v.part)
        for w in wires:
            w.set_ghost(False)
        self.placing_views, self.placing_wires, self.place_again = [], [], None
        self.placing_kind = None
        self.mode = Mode.IDLE
        if again:
            place_again()
        elif wires or len(views) > 1:
            self.selection.set(
                views, wires
            )  # a paste stays selected, ready to move/delete

    def _apply(self, change: Change | None) -> None:
        """Show the board after an undo/redo step (history.current), which changed `change`."""
        if change is None:
            return
        self.selection.clear()
        restore(self, self.history.current, change_uids(change))
        # parts at the ends of wires that came or went: their pins' colors may change
        for uid in _pin_part_uids(pair for pair in change[1].values()):
            Touched.part(uid)

    def _begin_group_drag(self, grabbed: PartView) -> None:
        """Start moving the selection, or just `grabbed` if it isn't part of it."""
        if grabbed not in self.selection:
            self.selection.set(parts=[grabbed])
        group = self.selection.parts
        self.drag_origin = (grabbed.x, grabbed.y)
        # Wires with BOTH ends in the group move rigidly with it, bends included.
        # Wires with one end outside keep their bends; only that end follows.
        self._begin_move(list(group), internal_wires(self, group))
        self.mode = Mode.DRAGGING_PART

    def _begin_move(self, views: list[PartView], wires: list[WireView]) -> None:
        """Start moving `views` and the `wires` running inside the group. They're *lifted*
        (see canvas.py): until _end_move they're drawn shifted by the canvas's offset and
        keep their old coordinates, so a mouse move costs one offset change -- plus
        re-shaping the few wires stretched between the group and the rest of the board."""
        self.drag_group = [(v, v.x, v.y) for v in views]
        self.drag_wires = [(w, list(w.bends), w.src, w.dst) for w in wires]
        self.drag_delta = (0.0, 0.0)
        lift(views, wires, True)
        c, parts, inside = (
            self.circuit,
            {v.part for v in views},
            {w.wire for w in wires},
        )
        # Stretched: an end on the group's pins or on one of its wires, the other end outside.
        # The ends on the group move along with it (junction ends too: their wire moves rigidly).
        stretched = {w for part in parts for pin in part.pins for w in c.ends_on(pin)}
        stretched.update(w for i in inside for w in c.ends_on(i))
        stretched -= inside
        self.stretched = [
            (
                self.wire_views[w],
                self.wire_views[w].src,
                self.wire_views[w].dst,
                tuple(
                    e.part in parts if isinstance(e, Pin) else e in inside
                    for e in w.ends
                ),
            )
            for w in sorted(stretched, key=lambda w: w.uid)
        ]
        # ... and whatever hangs off those, re-attached as they change shape
        self.stretched_tail = [
            self.wire_views[w]
            for w in c.descendants(*stretched)
            if w not in stretched and w not in inside
        ]

    def _move_group(self) -> None:
        wx, wy = self.camera.screen_to_world(*self.mouse)
        # Snap the grabbed part's origin; everything else moves by the same delta, so
        # the group keeps its shape (and snapped layouts stay snapped).
        ox, oy = self.snapped(wx + self.grab[0], wy + self.grab[1])
        dx, dy = ox - self.drag_origin[0], oy - self.drag_origin[1]
        if (dx, dy) == self.drag_delta:
            return
        self.drag_delta = self.world.offset = (dx, dy)
        for view, (sx, sy), (tx, ty), (src_moves, dst_moves) in self.stretched:
            view.set_ends(
                (sx + dx, sy + dy) if src_moves else (sx, sy),
                (tx + dx, ty + dy) if dst_moves else (tx, ty),
            )
        for (
            view
        ) in self.stretched_tail:  # parents first; their pins are all outside the group
            w = view.wire
            view.set_ends(self.end_pos(w.src, view.src), self.end_pos(w.dst, view.dst))

    def _end_move(self) -> None:
        """Put the lifted group down where it was dragged to, for real."""
        dx, dy = self.drag_delta
        self.world.offset = (0.0, 0.0)
        parts, wires = (
            [v for v, *_ in self.drag_group],
            [v for v, *_ in self.drag_wires],
        )
        put_down(
            parts, wires, dx, dy
        )  # (lifted, they kept their coordinates: from where they were picked up)
        # exact final attachment, as if it had been moved step by step
        live = [v for v, *_ in self.stretched if v.wire in self.wire_views]
        self.refresh_wires([*live, *(v for v, *_ in self.drag_wires)])
        self.drag_group, self.drag_wires, self.stretched, self.stretched_tail = (
            [],
            [],
            [],
            [],
        )
        self.drag_delta = (0.0, 0.0)

    def _update_box(self) -> None:
        sx, sy = self.mouse
        px, py = self.press_at
        if (
            self.box_shapes is None
            and abs(sx - px) + abs(sy - py) < T.DRAG_THRESHOLD_PX
        ):
            return  # not a drag yet; a plain click just leaves the selection cleared
        # Anchor in world space so panning/zooming mid-drag keeps the start corner in place.
        ax, ay = self.camera.world_to_screen(*self.box_start)
        x0, x1 = sorted((ax, sx))
        y0, y1 = sorted((ay, sy))
        if self.box_shapes is None:
            self.box_shapes = (
                shapes.Rectangle(
                    0,
                    0,
                    1,
                    1,
                    color=T.SELECT_BOX_FILL,
                    batch=self.overlay,
                    group=self.hud_box_group,
                ),
                shapes.Box(
                    0,
                    0,
                    1,
                    1,
                    thickness=1,
                    color=T.SELECT,
                    batch=self.overlay,
                    group=self.hud_box_group,
                ),
            )
        for shape in self.box_shapes:
            shape.position = (x0, y0)
            shape.width, shape.height = max(x1 - x0, 1), max(y1 - y0, 1)
        (wx0, wy0), (wx1, wy1) = (
            self.camera.screen_to_world(x0, y0),
            self.camera.screen_to_world(x1, y1),
        )
        base_parts, base_wires = self.box_base
        self.selection.set(
            base_parts
            | {
                v
                for v in self.part_index.query(wx0, wy0, wx1, wy1)
                if v.intersects(wx0, wy0, wx1, wy1)
            },
            base_wires
            | {
                v
                for v in self.wire_index.query(wx0, wy0, wx1, wy1)
                if v.inside(wx0, wy0, wx1, wy1)
            },
        )

    def _end_box(self) -> None:
        if self.box_shapes is not None:
            for shape in self.box_shapes:
                shape.delete()
            self.box_shapes = None
        self.mode = Mode.IDLE

    @property
    def snapping(self) -> bool:
        return self.keys[key.LCTRL] or self.keys[key.RCTRL]

    @property
    def grid_divisions(self) -> int:
        """1 = normal grid; Ctrl+Shift switches to the subgrid."""
        shift = self.keys[key.LSHIFT] or self.keys[key.RSHIFT]
        return T.SUBGRID_DIVISIONS if self.snapping and shift else 1

    def snapped(self, wx: float, wy: float) -> Point:
        """Round to the nearest grid (or, with Ctrl+Shift, subgrid) point while Ctrl is held."""
        if not self.snapping:
            return wx, wy
        step = T.GRID / self.grid_divisions
        return round(wx / step) * step, round(wy / step) * step

    def _follow_cursor(self) -> None:
        """Keep whatever is attached to the cursor under the cursor (also after pan/zoom)."""
        if self.mode in (Mode.DRAGGING_PART, Mode.PLACING_PART):
            self._move_group()
        elif self.mode is Mode.BOX_SELECTING:
            self._update_box()
        elif self.mode is Mode.WIRING:
            self._update_preview()
        elif self.mode is Mode.EDITING_WIRE:
            if self.wire_edit.dragging is not None:
                self.wire_edit.drag_to(
                    *self.camera.screen_to_world(*self.mouse), self.snapped
                )
                self.refresh_wires([self.wire_edit.view])
            else:
                self.wire_edit.refresh()  # zoom changes handle sizes

    def _start_wiring(self, start: Pin | Wire, start_pos: Point) -> None:
        self.selection.clear()
        self.mode = Mode.WIRING
        self.wire_start, self.wire_start_pos = start, start_pos
        self.wire_bends = []
        self.preview = Polyline([], T.WIRE_PREVIEW, self.world, self.layers.overlay)
        self._update_preview()

    def _update_preview(self) -> None:
        wx, wy = self.camera.screen_to_world(*self.mouse)
        target = self.wire_target(wx, wy)
        valid = target is not None and self.can_wire_to(target[0])
        end = target[1] if valid else self.snapped(wx, wy)  # valid targets always win
        self.preview.set_points([self.wire_start_pos, *self.wire_bends, end])
        self.preview.color = T.WIRE_PREVIEW_SNAP if valid else T.WIRE_PREVIEW

    def _pop_bend_or_cancel(self) -> None:
        if self.wire_bends:
            self.wire_bends.pop()
            self._update_preview()
        else:
            self._cancel()

    def _cancel(self) -> None:
        """Abort whatever is in progress and return to IDLE."""
        if self.mode is Mode.MENU:
            self._close_menu()
        elif self.mode is Mode.EDITING_LABEL:
            self._finish_edit(commit=False)
        elif self.mode is Mode.EDITING_WIRE:
            self._finish_wire_edit(commit=False)
        elif self.mode is Mode.BOX_SELECTING:
            self._end_box()
        elif self.mode is Mode.PICKER_DRAG:
            self.picker.cancel_drag()
        elif self.mode is Mode.RENAMING:
            self._finish_rename(commit=False)
        elif self.mode is Mode.PROMPT:
            self._close_prompt()
        elif self.mode is Mode.POPOVER:
            self._close_popover()
        self.pressed_wire = None
        self.picker_row = None
        if self.mode in (Mode.DRAGGING_PART, Mode.PLACING_PART):
            self._end_move()  # a drag stays where it got to (a paste is removed next)
        if self.mode is Mode.PLACING_PART:
            for view in self.placing_views:
                self.remove_part(view)  # takes the ghost wires with it
            self.placing_views, self.placing_wires, self.place_again = [], [], None
            self.placing_kind = None
        if self.preview is not None:
            self.preview.delete()
            self.preview = None
        self.mode = Mode.IDLE
        self.active = None
        self.wire_start = None
        self.wire_bends = []

    # ======================================================================
    # history: every finished edit is recorded, automatically
    # ======================================================================

    _EDIT_EVENTS = frozenset({"on_mouse_press", "on_mouse_release", "on_key_press"})

    def dispatch_event(self, event_type, *args):
        """After any click/release/key that leaves the editor idle, snapshot the board
        and record it if it changed. So every action -- including ones added later --
        is undoable without writing inverse operations, and no-ops (a click on
        empty space, a switch toggle) don't clutter the history.

        Only what views say they changed gets looked at (see _record)."""
        result = super().dispatch_event(event_type, *args)
        if (
            event_type in self._EDIT_EVENTS
            and self.history is not None
            and self.mode is Mode.IDLE
            and (Touched.parts or Touched.wires)
        ):
            self._record()
        if event_type in self._EDIT_EVENTS and self.history is not None:
            self._save_library()  # picker rearrangements are saved as they happen (no-op if unchanged)
        return result

    # ======================================================================
    # tick + draw
    # ======================================================================

    def update(self, dt: float) -> None:
        self.picker.update(dt)
        if self.prompt is not None:
            self.prompt.tick(dt)
        if self.popover is not None:
            self._popover_apply()  # the slider's latest value, once a frame
            self.popover.tick(dt)
        self._update_caption()
        self.status.x = self.picker.width + 8  # follows the panel sliding in / out
        self.bar.place(self.picker.width, self.width)
        self._update_panels(dt)
        t0 = time.perf_counter()
        self.circuit.frame()
        for _ in range(SIM_STEPS_PER_FRAME):
            self.circuit.step()
        self._tally(dt, time.perf_counter() - t0)
        while self.circuit.errors:
            self._report(self.circuit.errors.pop(0))
        # Only what the sim says changed: touching every view each frame costs ~13 ms per 10k parts.
        everything, parts, wires = self.circuit.take_changes()
        if everything:
            for view in self.part_views.values():
                view.sync()
            for wire, view in self.wire_views.items():
                view.sync(self.circuit.wire_state(wire))
            return
        for part in parts:
            if view := self.part_views.get(
                part
            ):  # (hidden parts inside macros have none)
                view.sync()
        for wire in wires:
            if view := self.wire_views.get(wire):
                view.sync(self.circuit.wire_state(wire))

    def on_resize(self, width, height):
        super().on_resize(width, height)  # keeps the projection matrix in sync
        self.status.y = height - 8
        self.bar.place(self.picker.width, width)
        if self.prompt is not None:
            self.prompt.layout(width, height)
        if self.popover is not None:
            self.popover.layout(width, height)
        self.picker.resize(height, self._pixel_ratio())

    def on_draw(self):
        t0 = time.perf_counter()
        self.clear()
        self.grid.draw(
            self,
            self.camera,
            emphasized=self.snapping,  # also paints the background
            divisions=self.grid_divisions,
        )
        self.view = self.camera.matrix()
        self.world.draw()
        self.view = Mat4()  # identity: HUD is in screen pixels
        self.panels.draw(self, self.world, self.grid)
        self.hud.draw()
        self.overlay.draw()
        self.menu.batch.draw()
        self.stats["draw"] += (
            time.perf_counter() - t0
        )  # CPU side: issuing the draws, not the GPU's work

    def _board_center(self) -> tuple[float, float]:
        """The middle of the visible board (right of the picker, above the bar), screen px."""
        return (self.picker.width + self.width) / 2, (BAR_H + self.height) / 2

    def _visible_world(self) -> tuple[float, float, float, float]:
        """The world rect you can see: right of the picker, above the bar."""
        (x0, y0), (x1, y1) = (
            self.camera.screen_to_world(self.picker.width, BAR_H),
            self.camera.screen_to_world(self.width, self.height),
        )
        return x0, y0, x1, y1

    def _look_at(self, wx: float, wy: float) -> None:
        """Move the camera so (wx, wy) is in the middle of the visible board."""
        sx, sy = self._board_center()
        z = self.camera.zoom
        self.camera.x, self.camera.y = wx - sx / z, wy - sy / z
        self._follow_cursor()

    def _board_bounds(self) -> tuple[float, float, float, float] | None:
        """The box around every part and wire, or None for an empty board."""
        boxes = [b for b in (self.part_index.bounds(), self.wire_index.bounds()) if b]
        if not boxes:
            return None
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )

    def _fit(self, board) -> tuple[tuple[float, float, float, float], int]:
        """`board` with a margin around it, and the level at which that fits the visible board."""
        x0, y0, x1, y1 = board
        m = max(FIT_MARGIN, FIT_MARGIN_SHARE * max(x1 - x0, y1 - y0))
        x0, y0, x1, y1 = x0 - m, y0 - m, x1 + m, y1 + m
        level = Camera.fit_level(
            x1 - x0, y1 - y0, self.width - self.picker.width, self.height - BAR_H
        )
        return (x0, y0, x1, y1), level

    def _update_zoom_floor(self, board) -> None:
        """Zooming out goes as far as it takes to see the whole board (at least to MIN_LEVEL)."""
        self.camera.min_level = (
            MIN_LEVEL if board is None else min(MIN_LEVEL, self._fit(board)[1])
        )

    def _update_panels(self, dt: float) -> None:
        board = self._board_bounds()
        self._update_zoom_floor(board)
        on_board = not self.picker.contains(*self.mouse) and not self.bar.contains(
            *self.mouse
        )
        self.panels.update(
            dt,
            self.camera,
            board,
            self._visible_world(),
            self.mouse if self.mouse_in else None,
            on_board,
            self.width,
            self.height,
        )

    def _tally(self, dt: float, sim: float) -> None:
        """Add up one frame's numbers; every STATS_EVERY seconds, show their averages in the bar."""
        st = self.stats
        st["frames"] += 1
        st["time"] += dt
        st["sim"] += sim
        if st["time"] < STATS_EVERY:
            return
        n, c = st["frames"], self.circuit
        hidden = len(c.hidden_parts)
        self.bar.set_stats(
            "   ".join(
                (
                    f"{n / st['time']:.0f} fps",
                    f"sim {1000 * st['sim'] / n:.2f} ms",
                    f"draw {1000 * st['draw'] / n:.1f} ms",
                    f"tick {c.tick}",
                    f"{len(c.parts)} parts"
                    + (f" (+{hidden} in macros)" if hidden else ""),
                    f"{len(c.wires)} wires",
                    f"{len(c.net_value)} nets",
                    f"zoom {100 * self.camera.zoom:.0f}%",
                )
            )
        )
        self.stats = dict.fromkeys(st, 0)

    def on_close(self):
        # Not calling super() (it closes right away): unsaved changes get asked about first.
        if self.mode is Mode.PROMPT:
            return  # already asking something; answer that first
        self._cancel()
        self._unsaved_then(self._quit)

    def _quit(self) -> None:
        self._save_library()
        self.circuit.close_all()  # every opened part gets its close()
        self.close()

    def _report(self, msg: str) -> None:
        """Show a problem at the top of the window (the newest one) and on stderr."""
        print(msg, file=sys.stderr)
        pyglet.clock.unschedule(self._clear_status)
        self.status.text, self.status.color = msg, T.MENU_DANGER

    def _notice(self, msg: str) -> None:
        """Show something worth knowing (not a problem) for a few seconds."""
        pyglet.clock.unschedule(self._clear_status)
        self.status.text, self.status.color = msg, T.HELP_TEXT
        pyglet.clock.schedule_once(self._clear_status, NOTICE_SECONDS)

    def _clear_status(self, dt: float = 0.0) -> None:
        self.status.text = ""

    # ======================================================================
    # documents: every board is a macro (see storage.py)
    # ======================================================================

    @property
    def dirty(self) -> bool:
        """Unsaved changes: the board isn't at the point in its history that was last saved or opened."""
        return self.history.state != self.saved_state

    def _start_document(self) -> None:
        """Reopen what was open last time. With nothing to reopen: an untitled board,
        with the demo on it if there are no macros at all yet (a first run)."""
        self.history = History(EMPTY)
        name = self.project.last_open()
        name = name and self.store.find(name)
        if name:
            self._load(name)
            if self.doc is not None:
                return
        if not self.store.names():
            self._build_demo()
        self._reset_history(None)
        self.camera.center_on(*HOME, self.width, self.height)

    def _record(self, amend: bool = False) -> bool:
        """Write what views reported changed (views.Touched) into the undo history, as a
        new step (or folded into the last one), and rework the colors around it (paint.py).
        Returns whether there was a change to record."""
        with paused_gc():  # (a big edit is many entries: see paused_gc)
            return self._record_touched(amend)

    def _record_touched(self, amend: bool) -> bool:
        parts, wires, paint_parts, paint_wires = Touched.take()
        now = changes(self, parts, wires)
        # parts at the ends of a changed wire, before and after: their pins' colors may change
        # (before: from the history, so ask before recording updates it)
        before = self.history.current.wires
        ends = _pin_part_uids((before.get(uid), now[1][uid]) for uid in paint_wires)
        recorded = (self.history.amend if amend else self.history.record)(*now)
        if paint_parts or paint_wires:
            paint(self, paint_parts | ends, paint_wires)
        return recorded

    def _reset_history(self, doc: str | None) -> None:
        """A fresh undo timeline for what's on the board now, which counts as saved."""
        self.history = History(capture(self))
        Touched.take()  # all of it is in there now
        paint(self)
        self.saved_state = self.history.state
        self.doc = doc
        self.picker.refresh()  # what's greyed out depends on what's open

    def _clear_board(self) -> None:
        self._cancel()
        self.selection.clear()
        self.tiling = None
        restore(self, EMPTY)  # removes (and closes) everything

    def _load(self, name: str) -> None:
        try:
            loaded = self.store.load(name, self.catalog)
        except (OSError, FormatError) as e:
            self._report(f"can't open {name}: {e}")
            return
        self._clear_board()
        restore(self, loaded.snapshot)
        # Undo starts at what was actually built, which is also what counts as saved:
        # if the file needed repairs, the board shows them and saving writes them.
        self._reset_history(name)
        self.project.remember_open(name)
        self._fit_camera()
        if loaded.warnings:
            for w in loaded.warnings:
                print(f"{name}: {w}", file=sys.stderr)
            more = (
                f" (+{len(loaded.warnings) - 1} more, see the console)"
                if len(loaded.warnings) > 1
                else ""
            )
            self._report(f"{name}: {loaded.warnings[0]}{more}")
        else:
            self._notice(f"opened {name}")

    def _new(self) -> None:
        self._clear_board()
        self._reset_history(None)
        self.project.remember_open(None)
        self.camera.set_level(0, 0, 0)
        self.camera.center_on(*HOME, self.width, self.height)

    def _save(self, then=None) -> None:
        """Save under the current name (asking for one if untitled), then call `then`."""
        if self.doc is None:
            self._save_as(then)
        elif self._write(self.doc) and then is not None:
            then()

    def _save_as(self, then=None) -> None:
        p = Prompt(
            self.hud,
            self.width,
            self.height,
            "Save macro as",
            text=self.doc or "",
            max_len=NAME_MAX,
            hint="Enter: save   Esc: cancel",
        )
        confirmed = [None]  # the existing name the user already agreed to overwrite

        def enter(p: Prompt) -> None:
            try:
                name = check_name(p.text)
            except ValueError as e:
                p.set_hint(str(e), danger=True)
                return
            existing = self.store.find(name)
            mine = (
                self.doc is not None
                and existing is not None
                and existing.casefold() == self.doc.casefold()
            )
            if existing is not None and not mine and confirmed[0] != name:
                confirmed[0] = name
                p.set_hint(
                    f"{existing} already exists. Enter again to overwrite it.",
                    danger=True,
                )
                return
            self._close_prompt()
            if self._write(name) and then is not None:
                then()

        self._open_prompt(p, enter)

    def _write(self, name: str) -> bool:
        snap = self.history.current
        name = check_name(name)
        for kind in {d[0] for d in snap.parts.values() if d[0].startswith(MACRO)}:
            inner = kind.removeprefix(MACRO)
            if inner.casefold() == name.casefold() or self.catalog.book.contains(
                inner, name
            ):
                self._report(
                    f"can't save as {name}: the board has {inner} on it, which would then contain itself"
                )
                return False
        try:
            self.store.save(name, snap, self.catalog)
        except (OSError, ValueError, KeyError) as e:
            self._report(f"couldn't save {name}: {e}")
            return False
        self.catalog.book.forget()  # definitions changed; boards opened later see the new version
        self.doc, self.saved_state = check_name(name), self.history.state
        self.project.remember_open(self.doc)
        self._sync_library()
        self._notice(f"saved {self.doc}")
        return True

    def _open_dialog(self) -> None:
        names = self.store.names()
        p = Prompt(
            self.hud,
            self.width,
            self.height,
            "Open macro",
            text="",
            max_len=NAME_MAX,
            items=names,
            hint="Enter: open   Up/Down: choose   Esc: cancel",
            empty="no match"
            if names
            else "no saved macros yet (Ctrl+S saves this board)",
        )

        def enter(p: Prompt) -> None:
            if p.choice is not None:
                self._close_prompt()
                self._request_open(p.choice)

        self._open_prompt(p, enter)

    def _request_open(self, name: str) -> None:
        if (
            self.doc is not None
            and name.casefold() == self.doc.casefold()
            and not self.dirty
        ):
            self._notice(f"{name} is already open")
            return
        self._unsaved_then(lambda: self._load(name))

    def _unsaved_then(self, then) -> None:
        """Run `then` -- right away, or once unsaved changes are saved or discarded."""
        if not self.dirty:
            then()
            return

        def discard(symbol: int) -> None:
            if symbol == key.D:
                self._close_prompt()
                then()

        def save(p: Prompt) -> None:
            self._close_prompt()
            self._save(then)

        p = Prompt(
            self.hud,
            self.width,
            self.height,
            f"Unsaved changes to {self.doc or 'the untitled board'}",
            hint="Enter: save them   D: discard them   Esc: cancel",
        )
        self._open_prompt(p, save, discard)

    def _open_prompt(self, prompt: Prompt, enter, on_key=None) -> None:
        self._cancel()  # also closes a prompt that's already up
        self.prompt, self.prompt_enter, self.prompt_key = prompt, enter, on_key
        self.picker.set_hover(None)
        self.mode = Mode.PROMPT

    def _close_prompt(self) -> None:
        if self.prompt is not None:
            self.prompt.delete()
        self.prompt = self.prompt_enter = self.prompt_key = None
        self.mode = Mode.IDLE

    def _update_caption(self) -> None:
        state = (self.history.state, self.saved_state, self.doc, self.project)
        if state == self._caption_for:
            return  # nothing changed since last frame
        self._caption_for = state
        name = f"{self.doc or 'untitled'}{' *' if self.dirty else ''}"
        self.set_caption(f"pijl - {name}")
        self.bar.set_doc(f"{self.project.name} / {name}")

    def _fit_camera(self) -> None:
        """Show everything on the board, centered in the space right of the picker.
        Zooms out if it doesn't fit, never in past 1:1."""
        board = self._board_bounds()
        self._update_zoom_floor(board)
        if board is None:
            self.camera.set_level(0, 0, 0)
            self.camera.center_on(*HOME, self.width, self.height)
            return
        (x0, y0, x1, y1), level = self._fit(board)
        self.camera.level = max(self.camera.min_level, min(0, level))
        sx, sy = self._board_center()
        self.camera.x = (x0 + x1) / 2 - sx / self.camera.zoom
        self.camera.y = (y0 + y1) / 2 - sy / self.camera.zoom

    # ---- projects and the cogwheel menu ---------------------------------------------

    def _bind_project(self, project: Project) -> None:
        """Point everything at `project`: its part scripts, its macros, a fresh circuit.
        (Its picker library is loaded separately, see _load_library.)"""
        self.project = project
        try:
            remember_project(project.name)
        except OSError as e:
            self.load_problems.append(f"couldn't save settings.json: {e}")
        self.parts = load_parts(
            project.parts_dir
        )  # the project's part scripts (see pijl.parts)
        self.load_problems += [f"part script {msg}" for msg in self.parts.errors]
        self.store = MacroStore(project.macros_dir)
        # Part scripts plus macros. A macro's definition is read from its file when first needed.
        self.catalog = Catalog(
            self.parts, lambda name: self.store.load(name, self.catalog).snapshot
        )
        self.circuit = Circuit(self.catalog, settle_ticks=SETTLE_TICKS)

    def _cog_menu(self) -> None:
        current = self.project.name
        projects = [
            MenuItem(
                name, lambda n=name: self._request_project(n), checked=name == current
            )
            for name in project_names()
        ]
        projects.append(MenuItem("New project...", self._new_project_dialog))
        self._open_menu(
            *self.bar.cog_anchor,
            [
                MenuItem("Controls", self._show_controls),
                MenuItem("Open macro...", self._open_dialog),
                MenuItem("Projects", submenu=projects),
            ],
        )

    def _show_controls(self) -> None:
        sheet = ControlsSheet(self.hud, self.width, self.height, self._pixel_ratio())

        def page(symbol: int) -> None:
            if symbol in (key.PAGEUP, key.PAGEDOWN):
                sheet.move(-8 if symbol == key.PAGEUP else 8)

        self._open_prompt(sheet, lambda _: self._close_prompt(), page)

    def _new_project_dialog(self) -> None:
        p = Prompt(
            self.hud,
            self.width,
            self.height,
            "New project",
            text="",
            max_len=NAME_MAX,
            hint="Enter: create it   Esc: cancel",
        )

        def enter(p: Prompt) -> None:
            try:
                name = check_name(p.text)
            except ValueError as e:
                p.set_hint(str(e), danger=True)
                return
            existing = next(
                (n for n in project_names() if n.casefold() == name.casefold()), None
            )
            if existing is not None:
                p.set_hint(f"{existing} already exists", danger=True)
                return
            self._close_prompt()
            self._request_project(name)

        self._open_prompt(p, enter)

    def _request_project(self, name: str) -> None:
        if name == self.project.name:
            self._notice(f"{name} is already open")
            return
        self._unsaved_then(lambda: self._switch_project(name))

    def _switch_project(self, name: str) -> None:
        """Close this project and open (or create) another, reopening what was open in it last."""
        try:
            project = Project.open(name)
        except (OSError, ValueError) as e:
            self._report(f"can't open project {name}: {e}")
            return
        self._save_library()
        self._clear_board()
        self.circuit.close_all()
        # the clipboard's parts may not exist over there
        self.clipboard, self.tiling = None, None
        self.load_problems = []
        self._bind_project(project)
        self.library = self._load_library()
        self.picker.set_library(self.library)
        self._clear_status()
        for msg in self.load_problems:
            self._report(msg)
        self._start_document()

    # ---- the picker's library ----------------------------------------------------

    def _library_entries(self) -> list[tuple[str, str]]:
        return [(t.kind, t.category) for t in self.parts] + [
            (MACRO + name, MACROS) for name in self.store.names()
        ]

    def _load_library(self) -> Library:
        entries = self._library_entries()
        lib = Library(entries)
        try:
            lib = Library.from_dict(
                json.loads(self.project.library_file.read_text(encoding="utf-8")),
                entries,
            )
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as e:
            # (too early to _report on start: the status line doesn't exist yet; the caller shows it)
            self.load_problems.append(
                f"library.json unreadable ({e}); using the default layout"
            )
        self._library_saved = lib.to_dict()
        return lib

    def _sync_library(self) -> None:
        """After macros were added / renamed: bring the picker up to date."""
        self.library.sync(self._library_entries())
        self.picker.refresh()
        self._save_library()

    def _save_library(self) -> None:
        data = self.library.to_dict()
        if data == self._library_saved:
            return
        try:
            write_atomic(self.project.library_file, json.dumps(data, indent=2) + "\n")
            self._library_saved = data
        except OSError as e:
            self._report(f"couldn't save library.json: {e}")

    def _update_pin_labels(self) -> None:
        """Pin name tags per the Tab mode. On hover: the part under the cursor (its body
        or one of its pins) shows them; ghosts on the cursor don't count."""
        mode = self.pin_label_mode
        hovered = None
        if (
            mode == PIN_LABELS_HOVER
            and self.mode is not Mode.PROMPT
            and not self.picker.contains(*self.mouse)
        ):
            wx, wy = self.camera.screen_to_world(*self.mouse)
            pin = self.pin_at(wx, wy)
            hovered = (
                self.part_views[pin.part] if pin is not None else self.part_at(wx, wy)
            )
            if hovered in self.placing_views:
                hovered = None
        if mode == PIN_LABELS_HOVER and hovered is self.hover_view:
            return  # nothing changed (the common case: called on every mouse move)
        self.hover_view = hovered
        for view in self.part_views.values():
            view.set_pin_labels(mode == PIN_LABELS_ALWAYS or view is hovered)

    def _unplaceable(self, entry: str) -> bool:
        """Macros that can't go on the open board: itself, and anything containing it."""
        if not entry.startswith(MACRO) or self.doc is None:
            return False
        name = entry.removeprefix(MACRO)
        return name.casefold() == self.doc.casefold() or self.catalog.book.contains(
            name, self.doc
        )

    def _swatch(self, entry: str) -> tuple:
        if entry.startswith(MACRO):
            return T.MACRO_SWATCH
        return theme_color(self.parts.get(entry).look.swatch)


MIXED = "mixed"  # what a menu row shows when the edited parts' values differ


@dataclass
class _NumberEdit:
    """The popover's edit: which parts and setting, and the slider drag in progress."""

    parts: list[Part]
    key: str
    setting: Number
    drag_old: list | None = (
        None  # each part's value when the drag started; None: not dragging
    )
    pending: object = None  # the slider's latest value, not written yet (see update)


def _count(views) -> str:
    return f" ({len(views)} parts)" if len(views) > 1 else ""


def _common(values):
    """The one value all of them have, or MIXED."""
    values = iter(values)
    first = next(values)
    return (
        first if all(v == first and type(v) is type(first) for v in values) else MIXED
    )


def _pin_part_uids(wire_data) -> set[int]:
    """The uids of parts that wires end on, from (before, after) pairs of wire data."""
    return {
        ref[1]
        for pair in wire_data
        for data in pair
        if data
        for ref in data[:2]
        if ref[0] == "p"
    }


def run() -> None:
    Editor()
    pyglet.app.run()
