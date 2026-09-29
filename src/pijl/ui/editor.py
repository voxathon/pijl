"""The editor window: input handling (a small state machine), sim ticking, drawing.

Design rule for carpal-tunnel friendliness: nothing *requires* holding a
mouse button. Placing parts and drawing wires are "click to start, click to
finish". The press that starts an action never finishes it on release, so
holding and dragging simply carries the thing along until the next click.

Controls
  part picker (left panel; Tab or its "«" button collapses it, see picker.py)
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
    Edit (wires)           hold+drag square handles to move bends, "+" handles or the wire
                           itself to add one; right-click a square to remove it. Enter or a
                           click elsewhere finishes, Esc reverts. See wire_edit.py.
  Ctrl+C / Ctrl+X         copy / cut the selected parts (+ wires running between them)
  Ctrl+V                   paste: the copy follows the cursor like a new part; click to place
                           (shift+click: place and keep another copy), Esc/right-click cancels
  Ctrl+S                   save the board as a macro (the first save asks for a name)
  Ctrl+Shift+S             save under another name
  Ctrl+O                   open a macro: type to filter, arrows + Enter (or click); saved macros
                           are also in the picker's MACROS section: right-click -> Open
  Ctrl+N                   new, empty board
                           (opening, new and closing the window ask first if there are unsaved changes)
  Ctrl+Z / Ctrl+Y          undo / redo (also Ctrl+Shift+Z). During an action, Ctrl+Z cancels it.
                           Every finished edit is recorded automatically; see document.py.
  right-drag empty space   pan (middle-drag pans in any mode)
  hold Ctrl                snap parts and wire bends to the grid
  hold Ctrl+Shift          snap to the finer subgrid instead
  scroll                   zoom
  Home                     reset the camera
"""

from __future__ import annotations

import json
import math
import sys
import time
from enum import Enum, auto

import pyglet
from pyglet import shapes
from pyglet.math import Mat4
from pyglet.window import key, mouse

from ..parts import load as load_parts
from ..project import Project, write_atomic
from ..storage import NAME_MAX, FormatError, MacroStore, check_name
from ..sim import Part, Circuit, Pin, Wire
from . import theme as T
from .camera import MIN_LEVEL, STEPS_PER_OCTAVE, Camera
from .document import EMPTY, History, Snapshot, capture, instantiate, internal_wires, restore
from .grid import Grid
from .library import Library
from .line_edit import LineEdit
from .menu import ContextMenu, MenuItem
from .picker import PartPicker, Row
from .prompt import Prompt
from .sdf_text import SDFText
from .selection import Selection
from .views import (PartView, Layers, Point, Polyline, WireView, arc_length_at, points_before, project_onto,
                    theme_color)
from .wire_edit import WireEditSession

SIM_STEPS_PER_FRAME = 1
HOME = (400, 300)
LABEL_MAX = 32
MACRO = "macro:"  # library entries for saved macros; part kinds can't contain ':'
MACROS = "MACROS"  # the picker collection new macros land in
NOTICE_SECONDS = 4.0
DOUBLE_CLICK = 0.4  # s: a second click on the same picker row within this is treated as mouse bounce


class Mode(Enum):
    IDLE = auto()
    PRESSING_PART = auto()  # mouse down on a part, not moved yet: could become a click or a drag
    DRAGGING_PART = auto()  # moving the selection (or one part) with the mouse held
    PRESSING_WIRE = auto()  # mouse down on a wire: a click selects it, a drag starts a branch
    BOX_SELECTING = auto()  # dragging a selection rectangle over empty space
    PLACING_PART = auto()   # a new part follows the cursor until a click places it
    WIRING = auto()         # a wire follows the cursor from its start pin until a click on a pin
    MENU = auto()           # context menu open; the next click picks an item or closes it
    EDITING_LABEL = auto()  # typing a part's label in place
    EDITING_WIRE = auto()   # moving / adding / removing one wire's bend points
    PICKER_PRESS = auto()   # mouse down on a picker row: a click picks / toggles it, a drag moves it
    PICKER_DRAG = auto()    # carrying a picker row to another spot in the list
    RENAMING = auto()       # typing a collection's name in the picker
    PROMPT = auto()         # a Prompt box is up (save as / open / unsaved changes); see _open_prompt


def _make_config() -> pyglet.gl.Config | None:
    """Ask for 4x multisampling (smooth shape edges); fall back to the default if unsupported."""
    screen = pyglet.display.get_display().get_default_screen()
    try:
        return screen.get_best_config(pyglet.gl.Config(double_buffer=True, sample_buffers=1, samples=4))
    except pyglet.window.NoSuchConfigException:
        return None


class Editor(pyglet.window.Window):
    def __init__(self) -> None:
        self.history: History | None = None  # set up by _start_document; checked by dispatch_event
        super().__init__(1280, 720, caption="pijl", resizable=True, vsync=True, config=_make_config())
        self.project = Project.open()  # where saves go; created on first run (see project.py)
        self.parts = load_parts(self.project.parts_dir)  # the project's part scripts (see pijl.parts)
        self.store = MacroStore(self.project.macros_dir)
        self.circuit = Circuit(self.parts)
        self.camera = Camera()
        self.grid = Grid()
        self.keys = key.KeyStateHandler()  # live "is this key down?" lookups
        self.push_handlers(self.keys)

        self.world = pyglet.graphics.Batch()
        self.layers = Layers()
        self.text = SDFText(self.world, self.layers.text_order)
        self.hud = pyglet.graphics.Batch()
        self.startup_problems: list[str] = [f"part script {msg}" for msg in self.parts.errors]
        self.library = self._load_library()
        self.picker = PartPicker(self.library, self.hud, self.height, self._pixel_ratio(),
                                 swatch=self._swatch, name_of=lambda entry: entry.removeprefix(MACRO))
        self.menu = ContextMenu(self.hud)
        self.help = pyglet.text.Label(
            "click pin: wire | click: select | drag empty: box select | Del: delete | Ctrl+C/X/V | "
            "Ctrl+Z/Y | Ctrl+S/O/N: save/open/new | right-click: menu | middle-drag: pan | scroll: zoom | "
            "hold Ctrl: snap",
            font_name="Consolas", font_size=10, color=T.HELP_TEXT,
            x=self.picker.width + 8, y=self.height - 8, anchor_y="top", batch=self.hud)
        # Problems (part scripts, files) in red; notices ("saved adder") in grey, for a few seconds
        self.status = pyglet.text.Label(
            "", font_name="Consolas", font_size=10, color=T.MENU_DANGER,
            x=self.picker.width + 8, y=self.height - 24, anchor_y="top", batch=self.hud)
        for msg in self.startup_problems:
            self._report(msg)

        self.part_views: dict[Part, PartView] = {}
        self.wire_views: dict[Wire, WireView] = {}

        # interaction state
        self.mode = Mode.IDLE
        self.panning = False                 # orthogonal to mode: you can pan while carrying things
        self.mouse = (0, 0)                  # last known cursor position, screen space
        self.active: PartView | None = None  # part being pressed / dragged
        self.grab = (0.0, 0.0)               # part origin minus cursor, world units
        self.press_at = (0, 0)               # screen pos of the press on a part / wire / picker row
        self.picker_row: Row | None = None   # picker row pressed / being dragged
        self.picker_bounce = False           # that press is a double-click's 2nd half: its click does nothing
        self.last_picker_click: tuple = (None, 0.0)  # (row key, time) of the last click on a picker row
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
        self.place_again = None               # shift+click: start another of the same
        self.placing_kind: str | None = None  # the picker part on the cursor (None for a paste)
        self.clipboard: Snapshot | None = None
        # selection
        self.selection = Selection()
        self.drag_group: list[tuple[PartView, float, float]] = []  # parts moving + their start origins
        self.drag_wires: list[tuple[WireView, list[Point], Point, Point]] = []  # wires inside the group + start shape
        self.drag_origin = (0.0, 0.0)          # start origin of the grabbed part
        self.box_start: Point = (0.0, 0.0)     # world point where the box drag began
        self.box_base: tuple[set, set] = (set(), set())  # selection to add to (shift) or empty
        self.box_shapes: tuple[shapes.Rectangle, shapes.Box] | None = None
        self.hud_box_group = pyglet.graphics.Group(order=8)  # above the picker, below menus
        # prompt box (Mode.PROMPT)
        self.prompt: Prompt | None = None
        self.prompt_enter = None  # what Enter (or clicking a list item) does: fn(prompt)
        self.prompt_key = None    # other keys: fn(symbol) -> handled
        # the open document: a macro, or an untitled board
        self.doc: str | None = None          # its name; None = untitled
        self.saved: Snapshot = EMPTY         # what's on disk (dirty = history.current differs)
        self._caption_for: tuple = ()

        self._start_document()
        pyglet.clock.schedule_interval(self.update, 1 / 60)

    # ======================================================================
    # model + view bookkeeping
    # ======================================================================

    def add_part(self, kind: str, x: float, y: float, uid: int | None = None, live: bool = True) -> PartView:
        """`live=False`: a ghost, for carrying on the cursor (see _carry / _commit_placing)."""
        part = self.circuit.add_part(kind, uid, live)
        view = PartView(part, x, y, self.world, self.layers, self.text)
        self.part_views[part] = view
        return view

    def remove_part(self, view: PartView) -> None:
        for wire in self.circuit.remove_part(view.part):
            self._drop_wire_view(wire)
        self.selection.discard(view)
        del self.part_views[view.part]
        view.delete()

    def _drop_wire_view(self, wire: Wire) -> None:
        view = self.wire_views.pop(wire)
        self.selection.discard(view)
        view.delete()

    def connect(self, a: Pin | Wire, b: Pin | Wire, bends: list[Point] = (),
                a_pos: Point | None = None, b_pos: Point | None = None, uid: int | None = None) -> Wire | None:
        """Connect two endpoints (pins or wires); `bends` are ordered from a to b.
        `a_pos` / `b_pos` say where on a wire endpoint the junction sits."""
        wire, replaced = self.circuit.connect(a, b, uid)
        for old in replaced:
            self._drop_wire_view(old)
        if wire is not None:
            if wire.src is not a:  # the circuit put the output side first; flip our layout too
                bends, a_pos, b_pos = list(reversed(bends)), b_pos, a_pos
            src = self.pin_pos(wire.src) if isinstance(wire.src, Pin) else a_pos
            dst = self.pin_pos(wire.dst) if isinstance(wire.dst, Pin) else b_pos
            self.wire_views[wire] = WireView(wire, src, list(bends), dst, self.world, self.layers)
        return wire

    def remove_wire(self, view: WireView) -> None:
        for wire in self.circuit.remove_wire(view.wire):  # plus its branches
            self._drop_wire_view(wire)

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
        attached = [(arc_length_at(pts, self._attach_point(x, w)), x) for x in self.circuit.attachments(w)]
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
        tail, far_pos = (sv.bends, sv.dst) if splice.src is w else (list(reversed(sv.bends)), sv.src)
        bends = [*points_before(pts, s_j)[1:], j, *tail]
        src_pos = view.src

        self.circuit.merge(w, splice)  # w keeps its identity (uid); splice's branches move to w
        for gone in (w, splice):
            self._drop_wire_view(gone)
        self.wire_views[w] = WireView(w, src_pos, bends, self.end_pos(w.dst, far_pos), self.world, self.layers)
        self.refresh_wires([self.wire_views[w]])

    def _attach_point(self, x: Wire, parent: Wire) -> Point:
        """Where wire `x` touches `parent`."""
        xv = self.wire_views[x]
        return xv.src if x.src is parent else xv.dst

    @staticmethod
    def _far_end(x: Wire, parent: Wire):
        return x.dst if x.src is parent else x.src

    def delete_selection(self) -> None:
        for view in list(self.selection.wires):
            if view.wire in self.wire_views:  # may already be gone with a deleted part
                self.remove_wire(view)
        for view in list(self.selection.parts):
            self.remove_part(view)
        self.selection.clear()

    def pin_pos(self, pin: Pin) -> Point:
        return self.part_views[pin.part].pin_pos(pin)

    def wires_touching(self, parts: set[Part]) -> list[WireView]:
        return [v for w, v in self.wire_views.items()
                if any(isinstance(e, Pin) and e.part in parts for e in w.ends)]

    def refresh_wires_touching(self, parts: set[Part]) -> None:
        self.refresh_wires(self.wires_touching(parts))

    def refresh_wires(self, views) -> None:
        """Re-attach the ends of `views` -- and of every wire hanging off them -- to
        their pins / parent wires. Pin ends snap to the pin; junction ends slide to
        the nearest point on their parent wire. Parents go first (creation order)."""
        todo = {v.wire for v in views}
        if not todo:
            return
        for wire in self.circuit.wires:  # creation order: parents before children
            if wire in todo or any(e in todo for e in wire.ends):
                todo.add(wire)
                view = self.wire_views[wire]
                view.set_ends(self.end_pos(wire.src, view.src), self.end_pos(wire.dst, view.dst))

    def end_pos(self, end: Pin | Wire, near: Point) -> Point:
        if isinstance(end, Pin):
            return self.pin_pos(end)
        return project_onto(self.wire_views[end].points, near)

    def _build_demo(self) -> None:
        # Everything on grid points, so the wires come out straight.
        if "NAND" not in self.parts:
            return  # the project's part scripts no longer have one
        a = self.add_part("IN", 200, 360)     # output pin at (240, 380)
        b = self.add_part("IN", 200, 220)     # output pin at (240, 240)
        g = self.add_part("NAND", 380, 280)   # inputs at y=320 / y=300, output at (460, 310)
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

    def pin_at(self, wx: float, wy: float) -> Pin | None:
        for view in reversed(self.part_views.values()):
            if self.mode is Mode.PLACING_PART and view in self.placing_views:
                continue  # parts on the cursor aren't targets
            if pin := view.pin_at(wx, wy, self.slop):
                return pin
        return None

    def part_at(self, wx: float, wy: float) -> PartView | None:
        return next((v for v in reversed(self.part_views.values()) if v.contains(wx, wy)), None)

    def wire_at(self, wx: float, wy: float) -> WireView | None:
        limit = T.WIRE_THICKNESS / 2 + self.slop
        best = min(self.wire_views.values(), key=lambda v: v.distance_to(wx, wy), default=None)
        return best if best is not None and best.distance_to(wx, wy) <= limit else None

    def wire_target(self, wx: float, wy: float) -> tuple[Pin | Wire, Point] | None:
        """What a wire being drawn would connect to here: a pin (preferred) or a
        point on another wire."""
        if pin := self.pin_at(wx, wy):
            return pin, self.pin_pos(pin)
        if view := self.wire_at(wx, wy):
            return view.wire, project_onto(view.points, self.snapped(wx, wy))
        return None

    def can_wire_to(self, end: Pin | Wire | None) -> bool:
        return end is not None and self.wire_start is not None and self.circuit.can_connect(self.wire_start, end)

    # ======================================================================
    # input: the state machine
    # ======================================================================

    def on_mouse_press(self, x, y, button, modifiers):
        self.mouse = (x, y)
        wx, wy = self.camera.screen_to_world(x, y)
        in_picker = self.picker.hit(x, y)  # None unless the cursor is over the picker
        tool = in_picker.part if isinstance(in_picker, Row) and in_picker.what == "part" else None

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
            inside = self.menu.contains(x, y)
            self._close_menu()
            if button == mouse.LEFT and item is not None:
                self.menu.activate_last(item)  # may start another mode (e.g. label editing)
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

        if button == mouse.MIDDLE:
            self.panning = True
            return

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
                self._picker_press(in_picker, x, y, put_back=tool is not None and tool == held)
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
                pass                            # no bend points under the panel
            elif button == mouse.LEFT:
                target = self.wire_target(wx, wy)
                end, end_pos = target if target else (None, None)
                if end is self.wire_start:
                    self._cancel()
                elif self.can_wire_to(end):
                    start_pos = None if isinstance(self.wire_start, Pin) else self.wire_start_pos
                    self.connect(self.wire_start, end, self.wire_bends, start_pos,
                                 None if isinstance(end, Pin) else end_pos)
                    self._cancel()              # clears the preview; the wire now exists
                elif end is None:
                    self.wire_bends.append(self.snapped(wx, wy))
                    self._update_preview()
                # clicking an invalid target (in->in pin, same part, ...) does nothing
            elif button == mouse.RIGHT:
                self._pop_bend_or_cancel()
            return

        if self.mode in (Mode.PRESSING_PART, Mode.DRAGGING_PART, Mode.BOX_SELECTING, Mode.PRESSING_WIRE,
                         Mode.PICKER_PRESS, Mode.PICKER_DRAG):
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
                self._start_wiring(view.wire, project_onto(view.points, self.snapped(wx, wy)))
            elif view := self.part_at(wx, wy):
                if modifiers & key.MOD_SHIFT and not modifiers & key.MOD_CTRL:
                    self.selection.toggle(view)  # never toggles a switch; that's how you select one
                    return  # (Ctrl+Shift is subgrid snapping: press + drag as usual)
                self.active = view
                self.grab = (view.x - wx, view.y - wy)
                self.press_at = (x, y)
                self.mode = Mode.PRESSING_PART
            elif wire := self.wire_at(wx, wy):
                if modifiers & key.MOD_SHIFT:
                    self.selection.toggle(wire)
                else:
                    self.pressed_wire, self.press_world, self.press_at = wire, (wx, wy), (x, y)
                    self.mode = Mode.PRESSING_WIRE
            else:
                shift = modifiers & key.MOD_SHIFT
                self.box_base = (set(self.selection.parts), set(self.selection.wires)) if shift else (set(), set())
                if not shift:
                    self.selection.clear()  # a plain click on empty space ends here: cleared
                self.box_start = (wx, wy)
                self.press_at = (x, y)
                self.mode = Mode.BOX_SELECTING

        elif button == mouse.RIGHT:
            # Right-clicking narrows the selection to the clicked item, so the
            # highlight shows exactly what the menu will act on.
            if view := self.part_at(wx, wy):
                self.selection.set(parts=[view])
                self._open_menu(x, y, [MenuItem("Label...", lambda: self._start_edit(view)),
                                       MenuItem("Delete", lambda: self.remove_part(view), danger=True)])
            elif wire := self.wire_at(wx, wy):
                self.selection.set(wires=[wire])
                at = project_onto(wire.points, (wx, wy))
                self._open_menu(x, y, [MenuItem("Edit", lambda: self._start_wire_edit(wire)),
                                       MenuItem("Branch", lambda: self._start_wiring(wire.wire, at)),
                                       MenuItem("Delete", lambda: self.cut_wire(wire, at), danger=True)])
            else:
                self.panning = True

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.mouse = (x, y)
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
                self._start_wiring(view.wire, project_onto(view.points, self.snapped(*self.press_world)))
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
        self.mouse = (x, y)
        if self.mode is Mode.PROMPT:
            self.prompt.hover(x, y)
            return
        hover_ok = self.mode in (Mode.IDLE, Mode.PLACING_PART, Mode.WIRING)
        self.picker.set_hover(self.picker.hit(x, y) if hover_ok else None)
        if self.mode is Mode.MENU:
            self.menu.hover(x, y)
        elif self.mode is Mode.EDITING_WIRE and self.wire_edit.set_hover(self.wire_edit.target_at(x, y)):
            hand = self.get_system_mouse_cursor(self.CURSOR_HAND)
            self.set_mouse_cursor(hand if self.wire_edit.hover else None)
        self._follow_cursor()

    def on_mouse_release(self, x, y, button, modifiers):
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
        if self.mode is Mode.PROMPT:
            if scroll_y:
                self.prompt.move(-1 if scroll_y > 0 else 1)
            return
        if self.mode is Mode.MENU:
            self._close_menu()  # the menu belongs to what's under it; don't let the world slide away
        if self.picker.contains(x, y):
            self.picker.scroll_by(scroll_y)  # eases there; a drag follows along (see PartPicker.update)
            if self.mode is not Mode.PICKER_DRAG:
                self.picker.set_hover(self.picker.hit(x, y))
            return
        self.camera.scroll(x, y, scroll_y)
        self._follow_cursor()

    def on_key_press(self, symbol, modifiers):
        # Deliberately NOT calling super(): pyglet's default closes the window on Esc.
        if self.mode is Mode.PROMPT:
            # Typing goes through on_text / on_text_motion. No editor shortcuts while it's up.
            if symbol in (key.ENTER, key.NUM_ENTER):
                self.prompt_enter(self.prompt)
            elif symbol == key.ESCAPE:
                self._close_prompt()
            elif symbol in (key.UP, key.DOWN):
                self.prompt.move(-1 if symbol == key.UP else 1)
            elif self.prompt_key is not None and not modifiers & (key.MOD_CTRL | key.MOD_ALT):
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
        elif (self.mode is Mode.EDITING_WIRE and symbol in (key.DELETE, key.BACKSPACE)
              and self.wire_edit.hover and self.wire_edit.hover[0] == "bend"):
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
        elif modifiers & key.MOD_CTRL and symbol in (key.C, key.X) and self.mode is Mode.IDLE:
            if self.selection.parts:
                self.clipboard = capture(self, self.selection.parts)
                if symbol == key.X:
                    self.delete_selection()
        elif modifiers & key.MOD_CTRL and symbol == key.V and self.mode is Mode.IDLE and self.clipboard:
            self._start_paste()
        elif symbol == key.BACKSPACE and self.mode is Mode.WIRING:
            self._pop_bend_or_cancel()
        elif symbol == key.TAB and self.mode not in (Mode.PICKER_PRESS, Mode.PICKER_DRAG):
            self.picker.toggle()
        elif symbol == key.HOME:
            self.camera.set_level(0, 0, 0)
            self.camera.center_on(*HOME, self.width, self.height)
            self._follow_cursor()

    def on_key_release(self, symbol, modifiers):
        if symbol in (key.LCTRL, key.RCTRL, key.LSHIFT, key.RSHIFT):
            self._follow_cursor()  # un-snap / back to the normal grid

    def on_text(self, text):
        if self.mode is Mode.PROMPT:
            self.prompt.type_text(text)
        elif self.mode is Mode.EDITING_LABEL:
            self.edit.insert(text)
            self._update_edit()
        elif self.mode is Mode.RENAMING:
            self.picker.rename_text(text)

    def on_text_motion(self, motion):
        if self.mode is Mode.PROMPT:
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

    def _close_menu(self) -> None:
        self.menu.close()
        self.mode = Mode.IDLE

    def _start_wire_edit(self, view: WireView) -> None:
        self.selection.discard(view)  # one glow at a time
        self.mode = Mode.EDITING_WIRE
        self.wire_edit = WireEditSession(view, self.camera, self.world, self.layers)
        self.wire_edit.set_hover(self.wire_edit.target_at(*self.mouse))

    def _wire_edit_press(self, x: float, y: float, wx: float, wy: float, button: int) -> None:
        s = self.wire_edit
        target = s.target_at(x, y)
        if button == mouse.LEFT:
            if target and target[0] == "bend":
                s.begin_drag(target[1], wx, wy)
            elif target:  # "+" handle: new bend at the midpoint
                s.begin_drag(s.insert(target[1], s.add_handle_pos(target[1])), wx, wy)
            elif hit := s.segment_at(wx, wy):  # on the wire itself: split right there
                s.begin_drag(s.insert(*hit), wx, wy)
            else:
                self._finish_wire_edit(commit=True)  # click elsewhere finishes
        elif button == mouse.RIGHT:
            if target and target[0] == "bend":
                s.remove(target[1])
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
        self.caret = shapes.Rectangle(0, 0, 1, 1, color=T.CARET, batch=self.world, group=self.layers.overlay)
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
            self.picker_row, self.press_at, self.picker_bounce = hit, (x, y), put_back or bounce
            self.mode = Mode.PICKER_PRESS  # a click or a drag; the release / movement decides

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
            items = [MenuItem("Rename...", lambda: self._start_rename(c)),
                     MenuItem("New collection", lambda: new_collection(lib.collections.index(c) + 1)),
                     MenuItem("Collapse all" if any_open else "Expand all",
                              lambda: self.picker.set_all_open(not any_open))]
            if not c.builtin:
                items.append(MenuItem("Delete", lambda: delete(c), danger=True))
        elif isinstance(hit, Row) and hit.what == "part":
            p, at = hit.part, (lib.collections.index(hit.collection) + 1 if hit.collection else None)
            items = [MenuItem("Move to new collection", lambda: into_new_collection(p, at))]
            if p.startswith(MACRO):
                items.insert(0, MenuItem("Open", lambda: self._request_open(p.removeprefix(MACRO))))
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

    def _pixel_ratio(self) -> float:
        fb_w, _ = self.get_framebuffer_size()
        return fb_w / self.width if self.width else 1.0

    def _start_placing(self, kind: str) -> None:
        if kind.startswith(MACRO):
            self._notice("placing macros comes later; right-click it -> Open to edit it")
            return
        self._carry([self.add_part(kind, 0, 0, live=False)], [], again=lambda: self._start_placing(kind))
        self.placing_kind = kind

    def _start_paste(self) -> None:
        views, wires = instantiate(self, self.clipboard, live=False)
        self._carry(views, wires, again=self._start_paste)
        self.placing_kind = None

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
        self.drag_group = [(v, v.x, v.y) for v in views]
        self.drag_wires = [(w, list(w.bends), w.src, w.dst) for w in wires]
        self.drag_origin = (anchor.x, anchor.y)
        self.placing_views, self.placing_wires, self.place_again = views, wires, again
        self.mode = Mode.PLACING_PART
        self._follow_cursor()

    def _commit_placing(self, again: bool) -> None:
        views, wires, place_again = self.placing_views, self.placing_wires, self.place_again
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
            self.selection.set(views, wires)  # a paste stays selected, ready to move/delete

    def _apply(self, snap: Snapshot | None) -> None:
        """Show an undo/redo state."""
        if snap is not None:
            self.selection.clear()
            restore(self, snap)

    def _begin_group_drag(self, grabbed: PartView) -> None:
        """Start moving the selection, or just `grabbed` if it isn't part of it."""
        if grabbed not in self.selection:
            self.selection.set(parts=[grabbed])
        group = self.selection.parts
        parts = {v.part for v in group}
        self.drag_group = [(v, v.x, v.y) for v in group]
        # Wires with BOTH ends in the group move rigidly with it, bends included.
        # Wires with one end outside keep their bends; only that end follows.
        self.drag_wires = [(v, list(v.bends), v.src, v.dst)
                           for v in internal_wires(self, {c.uid for c in parts})]
        self.drag_origin = (grabbed.x, grabbed.y)
        self.mode = Mode.DRAGGING_PART

    def _move_group(self) -> None:
        wx, wy = self.camera.screen_to_world(*self.mouse)
        # Snap the grabbed part's origin; everything else moves by the same delta, so
        # the group keeps its shape (and snapped layouts stay snapped).
        ox, oy = self.snapped(wx + self.grab[0], wy + self.grab[1])
        dx, dy = ox - self.drag_origin[0], oy - self.drag_origin[1]
        for view, sx, sy in self.drag_group:
            view.move_to(sx + dx, sy + dy)
        for view, bends, (sx, sy), (tx, ty) in self.drag_wires:
            view.src, view.dst = (sx + dx, sy + dy), (tx + dx, ty + dy)  # junction ends ride along
            view.set_bends([(bx + dx, by + dy) for bx, by in bends])
        self.refresh_wires([*self.wires_touching({v.part for v, _, _ in self.drag_group}),
                            *(v for v, *_ in self.drag_wires)])

    def _update_box(self) -> None:
        sx, sy = self.mouse
        px, py = self.press_at
        if self.box_shapes is None and abs(sx - px) + abs(sy - py) < T.DRAG_THRESHOLD_PX:
            return  # not a drag yet; a plain click just leaves the selection cleared
        # Anchor in world space so panning/zooming mid-drag keeps the start corner in place.
        ax, ay = self.camera.world_to_screen(*self.box_start)
        x0, x1 = sorted((ax, sx))
        y0, y1 = sorted((ay, sy))
        if self.box_shapes is None:
            self.box_shapes = (shapes.Rectangle(0, 0, 1, 1, color=T.SELECT_BOX_FILL,
                                                batch=self.hud, group=self.hud_box_group),
                               shapes.Box(0, 0, 1, 1, thickness=1, color=T.SELECT,
                                          batch=self.hud, group=self.hud_box_group))
        for shape in self.box_shapes:
            shape.position = (x0, y0)
            shape.width, shape.height = max(x1 - x0, 1), max(y1 - y0, 1)
        (wx0, wy0), (wx1, wy1) = self.camera.screen_to_world(x0, y0), self.camera.screen_to_world(x1, y1)
        base_parts, base_wires = self.box_base
        self.selection.set(
            base_parts | {v for v in self.part_views.values() if v.intersects(wx0, wy0, wx1, wy1)},
            base_wires | {v for v in self.wire_views.values() if v.inside(wx0, wy0, wx1, wy1)})

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
                self.wire_edit.drag_to(*self.camera.screen_to_world(*self.mouse), self.snapped)
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
        self.pressed_wire = None
        self.picker_row = None
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
        empty space, a switch toggle) don't clutter the history."""
        result = super().dispatch_event(event_type, *args)
        if event_type in self._EDIT_EVENTS and self.history is not None and self.mode is Mode.IDLE:
            self.history.commit(capture(self))
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
        self._update_caption()
        self.help.x = self.status.x = self.picker.width + 8  # follows the panel sliding in / out
        self.circuit.frame()
        for _ in range(SIM_STEPS_PER_FRAME):
            self.circuit.step()
        while self.circuit.errors:
            self._report(self.circuit.errors.pop(0))
        for view in self.part_views.values():
            view.sync()
        for wire, view in self.wire_views.items():
            view.sync(self.circuit.wire_state(wire))

    def on_resize(self, width, height):
        super().on_resize(width, height)  # keeps the projection matrix in sync
        self.help.y = height - 8
        self.status.y = height - 24
        if self.prompt is not None:
            self.prompt.layout(width, height)
        self.picker.resize(height, self._pixel_ratio())

    def on_draw(self):
        self.clear()
        self.grid.draw(self, self.camera, emphasized=self.snapping,  # also paints the background
                       divisions=self.grid_divisions)
        self.view = self.camera.matrix()
        self.world.draw()
        self.view = Mat4()  # identity: HUD is in screen pixels
        self.hud.draw()

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
        """Unsaved changes: the board differs from what was last saved or opened."""
        return self.history.current != self.saved

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

    def _reset_history(self, doc: str | None) -> None:
        """A fresh undo timeline for what's on the board now, which counts as saved."""
        self.history = History(capture(self))
        self.saved = self.history.current
        self.doc = doc

    def _clear_board(self) -> None:
        self._cancel()
        self.selection.clear()
        restore(self, EMPTY)  # removes (and closes) everything

    def _load(self, name: str) -> None:
        try:
            loaded = self.store.load(name, self.parts)
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
            more = f" (+{len(loaded.warnings) - 1} more, see the console)" if len(loaded.warnings) > 1 else ""
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
        p = Prompt(self.hud, self.width, self.height, "Save macro as", text=self.doc or "", max_len=NAME_MAX,
                   hint="Enter: save   Esc: cancel")
        confirmed = [None]  # the existing name the user already agreed to overwrite

        def enter(p: Prompt) -> None:
            try:
                name = check_name(p.text)
            except ValueError as e:
                p.set_hint(str(e), danger=True)
                return
            existing = self.store.find(name)
            mine = self.doc is not None and existing is not None and existing.casefold() == self.doc.casefold()
            if existing is not None and not mine and confirmed[0] != name:
                confirmed[0] = name
                p.set_hint(f"{existing} already exists. Enter again to overwrite it.", danger=True)
                return
            self._close_prompt()
            if self._write(name) and then is not None:
                then()

        self._open_prompt(p, enter)

    def _write(self, name: str) -> bool:
        snap = self.history.current
        try:
            self.store.save(name, snap)
        except (OSError, ValueError) as e:
            self._report(f"couldn't save {name}: {e}")
            return False
        self.doc, self.saved = check_name(name), snap
        self.project.remember_open(self.doc)
        self._sync_library()
        self._notice(f"saved {self.doc}")
        return True

    def _open_dialog(self) -> None:
        names = self.store.names()
        p = Prompt(self.hud, self.width, self.height, "Open macro", text="", max_len=NAME_MAX, items=names,
                   hint="Enter: open   Up/Down: choose   Esc: cancel",
                   empty="no match" if names else "no saved macros yet (Ctrl+S saves this board)")

        def enter(p: Prompt) -> None:
            if p.choice is not None:
                self._close_prompt()
                self._request_open(p.choice)

        self._open_prompt(p, enter)

    def _request_open(self, name: str) -> None:
        if self.doc is not None and name.casefold() == self.doc.casefold() and not self.dirty:
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

        p = Prompt(self.hud, self.width, self.height, f"Unsaved changes to {self.doc or 'the untitled board'}",
                   hint="Enter: save them   D: discard them   Esc: cancel")
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
        state = (self.history.current, self.saved, self.doc)
        if len(self._caption_for) == 3 and all(a is b for a, b in zip(state, self._caption_for)):
            return  # nothing changed since last frame (compared by identity: cheap)
        self._caption_for = state
        self.set_caption(f"pijl - {self.doc or 'untitled'}{' *' if self.dirty else ''}")

    def _fit_camera(self) -> None:
        """Show everything on the board, centered in the space right of the picker.
        Zooms out if it doesn't fit, never in past 1:1."""
        views = list(self.part_views.values())
        if not views:
            self.camera.set_level(0, 0, 0)
            self.camera.center_on(*HOME, self.width, self.height)
            return
        margin = 60
        x0 = min(v.x for v in views) - margin
        y0 = min(v.y for v in views) - margin
        x1 = max(v.x + v.w for v in views) + margin
        y1 = max(v.y + v.h for v in views) + margin
        avail = max(1, self.width - self.picker.width)
        fit = min(avail / (x1 - x0), self.height / (y1 - y0))
        self.camera.level = max(MIN_LEVEL, min(0, math.floor(STEPS_PER_OCTAVE * math.log2(fit))))
        self.camera.center_on((x0 + x1) / 2, (y0 + y1) / 2, self.width, self.height)
        self.camera.x -= self.picker.width / 2 / self.camera.zoom  # center in the free space, not the window

    # ---- the picker's library ----------------------------------------------------

    def _library_entries(self) -> list[tuple[str, str]]:
        return ([(t.kind, t.category) for t in self.parts]
                + [(MACRO + name, MACROS) for name in self.store.names()])

    def _load_library(self) -> Library:
        entries = self._library_entries()
        lib = Library(entries)
        try:
            lib = Library.from_dict(json.loads(self.project.library_file.read_text(encoding="utf-8")), entries)
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as e:
            # (too early to _report: the status line doesn't exist yet; __init__ shows it)
            self.startup_problems.append(f"library.json unreadable ({e}); using the default layout")
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

    def _swatch(self, entry: str) -> tuple:
        if entry.startswith(MACRO):
            return T.MACRO_SWATCH
        return theme_color(self.parts.get(entry).look.swatch)


def run() -> None:
    Editor()
    pyglet.app.run()
