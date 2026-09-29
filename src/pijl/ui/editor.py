"""The editor window: input handling (a small state machine), sim ticking, drawing.

Design rule for carpal-tunnel friendliness: nothing *requires* holding a
mouse button. Placing chips and drawing wires are "click to start, click to
finish". The press that starts an action never finishes it on release, so
holding and dragging simply carries the thing along until the next click.

Controls
  toolbar click            pick up a chip; it follows the cursor
    click                  place it (shift+click: place and keep another)
    toolbar click          swap to that chip instead
    right-click / Esc      cancel
  click a pin              start a wire; it follows the cursor
  Alt+click a wire         start a branch from that spot (also: right-click wire -> Branch)
    click empty space      add a bend point
    click a pin or wire    connect (green preview = valid target); ending on a wire
                           makes a junction. Wires joined this way form one net; if its
                           drivers disagree the net shows amber (conflict, see sim/circuit.py)
    right-click/Backspace  remove last bend point (or cancel if none)
    Esc / click start pin  cancel
  click a chip / wire      select it (a plain click on an IN switch toggles it instead)
  shift+click              add to / remove from the selection (works on switches too)
  drag on empty space      box-select anything the box touches (shift: add to selection)
  click empty space / Esc  clear the selection;  Ctrl+A select all;  Del/Backspace delete it
  drag a chip              move it; dragging a selected chip moves the whole selection
  right-click chip/wire    select just it + context menu (click outside or Esc closes)
    Label...               type in place; Enter commits, Esc reverts, clicking elsewhere commits
    Edit (wires)           hold+drag square handles to move bends, "+" handles or the wire
                           itself to add one; right-click a square to remove it. Enter or a
                           click elsewhere finishes, Esc reverts. See wire_edit.py.
  Ctrl+C / Ctrl+X         copy / cut the selected chips (+ wires running between them)
  Ctrl+V                   paste: the copy follows the cursor like a new part; click to place
                           (shift+click: place and keep another copy), Esc/right-click cancels
  Ctrl+Z / Ctrl+Y          undo / redo (also Ctrl+Shift+Z). During an action, Ctrl+Z cancels it.
                           Every finished edit is recorded automatically; see document.py.
  right-drag empty space   pan (middle-drag pans in any mode)
  hold Ctrl                snap chips and wire bends to the grid
  hold Ctrl+Shift          snap to the finer subgrid instead
  scroll                   zoom
  Home                     reset the camera
"""

from __future__ import annotations

from enum import Enum, auto

import pyglet
from pyglet import shapes
from pyglet.math import Mat4
from pyglet.window import key, mouse

from ..sim import Chip, Circuit, Pin, Wire
from . import theme as T
from .camera import Camera
from .document import History, Snapshot, capture, instantiate, internal_wires, restore
from .grid import Grid
from .menu import ContextMenu, MenuItem
from .sdf_text import SDFText
from .selection import Selection
from .toolbar import Toolbar
from .views import ChipView, Layers, Point, Polyline, WireView, project_onto
from .wire_edit import WireEditSession

SIM_STEPS_PER_FRAME = 1
PALETTE = ["IN", "OUT", "NAND", "AND", "OR", "NOT"]
HOME = (400, 300)
LABEL_MAX = 32


class Mode(Enum):
    IDLE = auto()
    PRESSING_CHIP = auto()  # mouse down on a chip, not moved yet: could become a click or a drag
    DRAGGING_CHIP = auto()  # moving the selection (or one chip) with the mouse held
    BOX_SELECTING = auto()  # dragging a selection rectangle over empty space
    PLACING_CHIP = auto()   # a new chip follows the cursor until a click places it
    WIRING = auto()         # a wire follows the cursor from its start pin until a click on a pin
    MENU = auto()           # context menu open; the next click picks an item or closes it
    EDITING_LABEL = auto()  # typing a chip's label in place
    EDITING_WIRE = auto()   # moving / adding / removing one wire's bend points


def _make_config() -> pyglet.gl.Config | None:
    """Ask for 4x multisampling (smooth shape edges); fall back to the default if unsupported."""
    screen = pyglet.display.get_display().get_default_screen()
    try:
        return screen.get_best_config(pyglet.gl.Config(double_buffer=True, sample_buffers=1, samples=4))
    except pyglet.window.NoSuchConfigException:
        return None


class Editor(pyglet.window.Window):
    def __init__(self) -> None:
        self.history: History | None = None  # set up after the demo; checked by dispatch_event
        super().__init__(1280, 720, caption="pijl", resizable=True, vsync=True, config=_make_config())
        self.circuit = Circuit()
        self.camera = Camera()
        self.grid = Grid()
        self.keys = key.KeyStateHandler()  # live "is this key down?" lookups
        self.push_handlers(self.keys)

        self.world = pyglet.graphics.Batch()
        self.layers = Layers()
        self.text = SDFText(self.world, self.layers.text_order)
        self.hud = pyglet.graphics.Batch()
        self.toolbar = Toolbar(PALETTE, self.hud)
        self.menu = ContextMenu(self.hud)
        self.help = pyglet.text.Label(
            "click pin: wire | click: select | drag empty: box select | Del: delete | Ctrl+C/X/V | "
            "Ctrl+Z/Y | right-click: menu | middle-drag: pan | scroll: zoom | hold Ctrl: snap",
            font_name="Consolas", font_size=10, color=T.HELP_TEXT,
            x=8, y=self.height - 8, anchor_y="top", batch=self.hud)

        self.chip_views: dict[Chip, ChipView] = {}
        self.wire_views: dict[Wire, WireView] = {}

        # interaction state
        self.mode = Mode.IDLE
        self.panning = False                 # orthogonal to mode: you can pan while carrying things
        self.mouse = (0, 0)                  # last known cursor position, screen space
        self.active: ChipView | None = None  # chip being pressed / dragged
        self.grab = (0.0, 0.0)               # chip origin minus cursor, world units
        self.press_at = (0, 0)               # screen pos of the press on a chip
        self.wire_start: Pin | Wire | None = None  # where the wire being drawn starts
        self.wire_start_pos: Point = (0.0, 0.0)
        self.wire_bends: list[Point] = []
        self.preview: Polyline | None = None
        # label editing
        self.edit_view: ChipView | None = None
        self.edit_text = ""
        self.edit_caret = 0                  # insertion index into edit_text
        self.caret: shapes.Rectangle | None = None
        self.wire_edit: WireEditSession | None = None
        # placing (new part or paste): ghosts that follow the cursor until a click
        self.placing_views: list[ChipView] = []
        self.placing_wires: list[WireView] = []
        self.place_again = None               # shift+click: start another of the same
        self.clipboard: Snapshot | None = None
        # selection
        self.selection = Selection()
        self.drag_group: list[tuple[ChipView, float, float]] = []  # chips moving + their start origins
        self.drag_wires: list[tuple[WireView, list[Point], Point, Point]] = []  # wires inside the group + start shape
        self.drag_origin = (0.0, 0.0)          # start origin of the grabbed chip
        self.box_start: Point = (0.0, 0.0)     # world point where the box drag began
        self.box_base: tuple[set, set] = (set(), set())  # selection to add to (shift) or empty
        self.box_shapes: tuple[shapes.Rectangle, shapes.Box] | None = None
        self.hud_box_group = pyglet.graphics.Group(order=8)  # above toolbar, below menus

        self._build_demo()
        self.history = History(capture(self))
        self.camera.center_on(*HOME, self.width, self.height)
        pyglet.clock.schedule_interval(self.update, 1 / 60)

    # ======================================================================
    # model + view bookkeeping
    # ======================================================================

    def add_chip(self, kind: str, x: float, y: float, uid: int | None = None) -> ChipView:
        chip = self.circuit.add_chip(kind, uid)
        view = ChipView(chip, x, y, self.world, self.layers, self.text)
        self.chip_views[chip] = view
        return view

    def remove_chip(self, view: ChipView) -> None:
        for wire in self.circuit.remove_chip(view.chip):
            self._drop_wire_view(wire)
        self.selection.discard(view)
        del self.chip_views[view.chip]
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

    def delete_selection(self) -> None:
        for view in list(self.selection.wires):
            if view.wire in self.wire_views:  # may already be gone with a deleted chip
                self.remove_wire(view)
        for view in list(self.selection.chips):
            self.remove_chip(view)
        self.selection.clear()

    def pin_pos(self, pin: Pin) -> Point:
        return self.chip_views[pin.chip].pin_pos(pin)

    def wires_touching(self, chips: set[Chip]) -> list[WireView]:
        return [v for w, v in self.wire_views.items()
                if any(isinstance(e, Pin) and e.chip in chips for e in w.ends)]

    def refresh_wires_touching(self, chips: set[Chip]) -> None:
        self.refresh_wires(self.wires_touching(chips))

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
        a = self.add_chip("IN", 200, 360)     # output pin at (240, 380)
        b = self.add_chip("IN", 200, 220)     # output pin at (240, 240)
        g = self.add_chip("NAND", 380, 280)   # inputs at y=320 / y=300, output at (460, 310)
        out = self.add_chip("OUT", 580, 290)  # input pin at (580, 310)
        self.connect(a.chip.outputs[0], g.chip.inputs[0], [(320, 380), (320, 320)])
        self.connect(b.chip.outputs[0], g.chip.inputs[1], [(320, 240), (320, 300)])
        self.connect(g.chip.outputs[0], out.chip.inputs[0])

    # ======================================================================
    # hit testing (world coordinates)
    # ======================================================================

    @property
    def slop(self) -> float:
        """HIT_SLOP_PX converted to world units, so clicking feels the same at any zoom."""
        return T.HIT_SLOP_PX / self.camera.zoom

    def pin_at(self, wx: float, wy: float) -> Pin | None:
        for view in reversed(self.chip_views.values()):
            if self.mode is Mode.PLACING_CHIP and view in self.placing_views:
                continue  # chips on the cursor aren't targets
            if pin := view.pin_at(wx, wy, self.slop):
                return pin
        return None

    def chip_at(self, wx: float, wy: float) -> ChipView | None:
        return next((v for v in reversed(self.chip_views.values()) if v.contains(wx, wy)), None)

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
        tool = self.toolbar.button_at(x, y)

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

        if button == mouse.MIDDLE:
            self.panning = True
            return

        if self.mode is Mode.EDITING_WIRE:
            view = self.wire_edit.view
            self._wire_edit_press(x, y, wx, wy, button)
            self.refresh_wires([view])  # branches slide along the edited wire
            return

        if self.mode is Mode.PLACING_CHIP:
            if button == mouse.LEFT and tool:
                self._cancel()
                self._start_placing(tool)       # swap to a different part
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
                # clicking an invalid target (in->in pin, same chip, ...) does nothing
            elif button == mouse.RIGHT:
                self._pop_bend_or_cancel()
            return

        if self.mode in (Mode.PRESSING_CHIP, Mode.DRAGGING_CHIP, Mode.BOX_SELECTING):
            return  # another button while the left one is held down: ignore (middle already panned)

        # IDLE
        if button == mouse.LEFT:
            if tool:
                self._start_placing(tool)
            elif pin := self.pin_at(wx, wy):
                self._start_wiring(pin, self.pin_pos(pin))
            elif modifiers & key.MOD_ALT and (view := self.wire_at(wx, wy)):
                self._start_wiring(view.wire, project_onto(view.points, self.snapped(wx, wy)))
            elif view := self.chip_at(wx, wy):
                if modifiers & key.MOD_SHIFT and not modifiers & key.MOD_CTRL:
                    self.selection.toggle(view)  # never toggles a switch; that's how you select one
                    return  # (Ctrl+Shift is subgrid snapping: press + drag as usual)
                self.active = view
                self.grab = (view.x - wx, view.y - wy)
                self.press_at = (x, y)
                self.mode = Mode.PRESSING_CHIP
            elif wire := self.wire_at(wx, wy):
                if modifiers & key.MOD_SHIFT:
                    self.selection.toggle(wire)
                else:
                    self.selection.set(wires=[wire])
            else:
                shift = modifiers & key.MOD_SHIFT
                self.box_base = (set(self.selection.chips), set(self.selection.wires)) if shift else (set(), set())
                if not shift:
                    self.selection.clear()  # a plain click on empty space ends here: cleared
                self.box_start = (wx, wy)
                self.press_at = (x, y)
                self.mode = Mode.BOX_SELECTING

        elif button == mouse.RIGHT:
            # Right-clicking narrows the selection to the clicked item, so the
            # highlight shows exactly what the menu will act on.
            if view := self.chip_at(wx, wy):
                self.selection.set(chips=[view])
                self._open_menu(x, y, [MenuItem("Label...", lambda: self._start_edit(view)),
                                       MenuItem("Delete", lambda: self.remove_chip(view), danger=True)])
            elif wire := self.wire_at(wx, wy):
                self.selection.set(wires=[wire])
                at = project_onto(wire.points, (wx, wy))
                self._open_menu(x, y, [MenuItem("Edit", lambda: self._start_wire_edit(wire)),
                                       MenuItem("Branch", lambda: self._start_wiring(wire.wire, at)),
                                       MenuItem("Delete", lambda: self.remove_wire(wire), danger=True)])
            else:
                self.panning = True

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.mouse = (x, y)
        if self.panning:
            self.camera.pan(dx, dy)
        elif self.mode is Mode.PRESSING_CHIP:
            px, py = self.press_at
            if abs(x - px) + abs(y - py) >= T.DRAG_THRESHOLD_PX:
                self._begin_group_drag(self.active)
        self._follow_cursor()

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse = (x, y)
        self.toolbar.set_hover(self.toolbar.button_at(x, y))
        if self.mode is Mode.MENU:
            self.menu.hover(x, y)
        elif self.mode is Mode.EDITING_WIRE and self.wire_edit.set_hover(self.wire_edit.target_at(x, y)):
            hand = self.get_system_mouse_cursor(self.CURSOR_HAND)
            self.set_mouse_cursor(hand if self.wire_edit.hover else None)
        self._follow_cursor()

    def on_mouse_release(self, x, y, button, modifiers):
        # Releases never *finish* a click-to-place or click-to-wire action.
        # That is what makes press-and-hold on a toolbar button or pin harmless.
        if button in (mouse.MIDDLE, mouse.RIGHT) and self.panning:
            self.panning = False
        elif button == mouse.LEFT and self.mode is Mode.PRESSING_CHIP:
            # A click without movement: switches toggle, everything else gets selected.
            if self.active.chip.kind == "IN":
                self.active.chip.toggle()
            else:
                self.selection.set(chips=[self.active])
            self.mode, self.active = Mode.IDLE, None
        elif button == mouse.LEFT and self.mode is Mode.BOX_SELECTING:
            self._end_box()
        elif button == mouse.LEFT and self.mode is Mode.DRAGGING_CHIP:
            self.mode, self.active = Mode.IDLE, None
        elif button == mouse.LEFT and self.mode is Mode.EDITING_WIRE:
            self.wire_edit.end_drag()

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.mouse = (x, y)
        if self.mode is Mode.MENU:
            self._close_menu()  # the menu belongs to what's under it; don't let the world slide away
        self.camera.scroll(x, y, scroll_y)
        self._follow_cursor()

    def on_key_press(self, symbol, modifiers):
        # Deliberately NOT calling super(): pyglet's default closes the window on Esc.
        if self.mode is Mode.EDITING_LABEL:
            # Text goes through on_text / on_text_motion; only Enter/Esc matter here.
            # Everything else (Home, Backspace, Ctrl...) must not trigger editor shortcuts.
            if symbol in (key.ENTER, key.NUM_ENTER):
                self._finish_edit(commit=True)
            elif symbol == key.ESCAPE:
                self._finish_edit(commit=False)
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
            self.selection.set(self.chip_views.values(), self.wire_views.values())
        elif modifiers & key.MOD_CTRL and symbol in (key.Z, key.Y):
            if self.mode is not Mode.IDLE:
                self._cancel()  # mid-action: undo means "never mind", not "and the step before"
            elif symbol == key.Y or modifiers & key.MOD_SHIFT:
                self._apply(self.history.redo())
            else:
                self._apply(self.history.undo())
        elif modifiers & key.MOD_CTRL and symbol in (key.C, key.X) and self.mode is Mode.IDLE:
            if self.selection.chips:
                self.clipboard = capture(self, self.selection.chips)
                if symbol == key.X:
                    self.delete_selection()
        elif modifiers & key.MOD_CTRL and symbol == key.V and self.mode is Mode.IDLE and self.clipboard:
            self._start_paste()
        elif symbol == key.BACKSPACE and self.mode is Mode.WIRING:
            self._pop_bend_or_cancel()
        elif symbol == key.HOME:
            self.camera.set_level(0, 0, 0)
            self.camera.center_on(*HOME, self.width, self.height)
            self._follow_cursor()

    def on_key_release(self, symbol, modifiers):
        if symbol in (key.LCTRL, key.RCTRL, key.LSHIFT, key.RSHIFT):
            self._follow_cursor()  # un-snap / back to the normal grid

    def on_text(self, text):
        if self.mode is not Mode.EDITING_LABEL:
            return
        text = "".join(c for c in text if c.isprintable())  # drops Enter's carriage return
        text = text[:max(LABEL_MAX - len(self.edit_text), 0)]
        if text:
            t, i = self.edit_text, self.edit_caret
            self.edit_text, self.edit_caret = t[:i] + text + t[i:], i + len(text)
            self._update_edit()

    def on_text_motion(self, motion):
        if self.mode is not Mode.EDITING_LABEL:
            return
        t, i = self.edit_text, self.edit_caret
        if motion == key.MOTION_BACKSPACE and i > 0:
            self.edit_text, self.edit_caret = t[:i - 1] + t[i:], i - 1
        elif motion == key.MOTION_DELETE:
            self.edit_text = t[:i] + t[i + 1:]
        elif motion == key.MOTION_LEFT:
            self.edit_caret = max(0, i - 1)
        elif motion == key.MOTION_RIGHT:
            self.edit_caret = min(len(t), i + 1)
        elif motion in (key.MOTION_BEGINNING_OF_LINE, key.MOTION_BEGINNING_OF_FILE):
            self.edit_caret = 0
        elif motion in (key.MOTION_END_OF_LINE, key.MOTION_END_OF_FILE):
            self.edit_caret = len(t)
        self._update_edit()

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

    def _start_edit(self, view: ChipView) -> None:
        self.mode = Mode.EDITING_LABEL
        self.edit_view = view
        self.edit_text = view.chip.label
        self.edit_caret = len(self.edit_text)
        self.caret = shapes.Rectangle(0, 0, 1, 1, color=T.CARET, batch=self.world, group=self.layers.overlay)
        pyglet.clock.schedule_interval(self._blink_caret, 0.5)
        self._update_edit()

    def _update_edit(self) -> None:
        """Show the in-progress text and put the caret where the next character goes."""
        name = self.edit_view.name
        name.set_text(self.edit_text)
        name.move_to(*self.edit_view.name_pos())
        h = name.cap_height * 1.6
        self.caret.position = (name.caret_x(self.edit_caret) - 0.6, name.y - h / 2)
        self.caret.width, self.caret.height = 1.2, h
        self.caret.visible = True  # restart the blink so the caret shows while typing

    def _blink_caret(self, dt: float) -> None:
        if self.caret is not None:
            self.caret.visible = not self.caret.visible

    def _finish_edit(self, commit: bool) -> None:
        view = self.edit_view
        if commit:
            view.chip.label = self.edit_text.strip()
        view.refresh_name()  # shows the committed label, or reverts to the old one
        view.name.move_to(*view.name_pos())
        pyglet.clock.unschedule(self._blink_caret)
        self.caret.delete()
        self.caret = None
        self.edit_view = None
        self.mode = Mode.IDLE

    def _start_placing(self, kind: str) -> None:
        self._carry([self.add_chip(kind, 0, 0)], [], again=lambda: self._start_placing(kind))

    def _start_paste(self) -> None:
        views, wires = instantiate(self, self.clipboard)
        self._carry(views, wires, again=self._start_paste)

    def _carry(self, views: list[ChipView], wires: list[WireView], again) -> None:
        """Attach new (ghost) chips + wires to the cursor, centered on it, until a click."""
        self.selection.clear()
        for v in views:
            v.set_ghost(True)
        for w in wires:
            w.set_ghost(True)
        x0 = min(v.x for v in views)
        y0 = min(v.y for v in views)
        x1 = max(v.x + v.w for v in views)
        y1 = max(v.y + v.h for v in views)
        # The first chip is the anchor: Ctrl snaps *its* origin, so a pasted layout
        # that was on the grid lands on the grid again.
        anchor = views[0]
        self.grab = (anchor.x - (x0 + x1) / 2, anchor.y - (y0 + y1) / 2)
        self.drag_group = [(v, v.x, v.y) for v in views]
        self.drag_wires = [(w, list(w.bends), w.src, w.dst) for w in wires]
        self.drag_origin = (anchor.x, anchor.y)
        self.placing_views, self.placing_wires, self.place_again = views, wires, again
        self.mode = Mode.PLACING_CHIP
        self._follow_cursor()

    def _commit_placing(self, again: bool) -> None:
        views, wires, place_again = self.placing_views, self.placing_wires, self.place_again
        for v in views:
            v.set_ghost(False)
        for w in wires:
            w.set_ghost(False)
        self.placing_views, self.placing_wires, self.place_again = [], [], None
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

    def _begin_group_drag(self, grabbed: ChipView) -> None:
        """Start moving the selection, or just `grabbed` if it isn't part of it."""
        if grabbed not in self.selection:
            self.selection.set(chips=[grabbed])
        group = self.selection.chips
        chips = {v.chip for v in group}
        self.drag_group = [(v, v.x, v.y) for v in group]
        # Wires with BOTH ends in the group move rigidly with it, bends included.
        # Wires with one end outside keep their bends; only that end follows.
        self.drag_wires = [(v, list(v.bends), v.src, v.dst)
                           for v in internal_wires(self, {c.uid for c in chips})]
        self.drag_origin = (grabbed.x, grabbed.y)
        self.mode = Mode.DRAGGING_CHIP

    def _move_group(self) -> None:
        wx, wy = self.camera.screen_to_world(*self.mouse)
        # Snap the grabbed chip's origin; everything else moves by the same delta, so
        # the group keeps its shape (and snapped layouts stay snapped).
        ox, oy = self.snapped(wx + self.grab[0], wy + self.grab[1])
        dx, dy = ox - self.drag_origin[0], oy - self.drag_origin[1]
        for view, sx, sy in self.drag_group:
            view.move_to(sx + dx, sy + dy)
        for view, bends, (sx, sy), (tx, ty) in self.drag_wires:
            view.src, view.dst = (sx + dx, sy + dy), (tx + dx, ty + dy)  # junction ends ride along
            view.set_bends([(bx + dx, by + dy) for bx, by in bends])
        self.refresh_wires([*self.wires_touching({v.chip for v, _, _ in self.drag_group}),
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
        base_chips, base_wires = self.box_base
        self.selection.set(
            base_chips | {v for v in self.chip_views.values() if v.intersects(wx0, wy0, wx1, wy1)},
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
        if self.mode in (Mode.DRAGGING_CHIP, Mode.PLACING_CHIP):
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
        if self.mode is Mode.PLACING_CHIP:
            for view in self.placing_views:
                self.remove_chip(view)  # takes the ghost wires with it
            self.placing_views, self.placing_wires, self.place_again = [], [], None
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
        return result

    # ======================================================================
    # tick + draw
    # ======================================================================

    def update(self, dt: float) -> None:
        for _ in range(SIM_STEPS_PER_FRAME):
            self.circuit.step()
        for view in self.chip_views.values():
            view.sync()
        for wire, view in self.wire_views.items():
            view.sync(self.circuit.wire_state(wire))

    def on_resize(self, width, height):
        super().on_resize(width, height)  # keeps the projection matrix in sync
        self.help.y = height - 8

    def on_draw(self):
        self.clear()
        self.grid.draw(self, self.camera, emphasized=self.snapping,  # also paints the background
                       divisions=self.grid_divisions)
        self.view = self.camera.matrix()
        self.world.draw()
        self.view = Mat4()  # identity: HUD is in screen pixels
        self.hud.draw()


def run() -> None:
    Editor()
    pyglet.app.run()
