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
    click empty space      add a bend point
    click a pin            connect (green preview = valid target)
    right-click/Backspace  remove last bend point (or cancel if none)
    Esc / click start pin  cancel
  drag a chip              move it (hold-to-drag; a plain click toggles an IN switch)
  right-click chip/wire    delete it
  right-drag empty space   pan (middle-drag pans in any mode)
  hold Ctrl                snap chips and wire bends to the grid
  scroll                   zoom
  Home                     reset the camera
"""

from __future__ import annotations

from enum import Enum, auto

import pyglet
from pyglet.math import Mat4
from pyglet.window import key, mouse

from ..sim import Chip, Circuit, Pin, Wire
from . import theme as T
from .camera import Camera
from .grid import Grid
from .sdf_text import SDFText
from .toolbar import Toolbar
from .views import ChipView, Layers, Point, Polyline, WireView

SIM_STEPS_PER_FRAME = 1
PALETTE = ["IN", "OUT", "NAND", "AND", "OR", "NOT"]
HOME = (400, 300)


class Mode(Enum):
    IDLE = auto()
    PRESSING_CHIP = auto()  # mouse down on a chip, not moved yet: could become a click or a drag
    DRAGGING_CHIP = auto()
    PLACING_CHIP = auto()   # a new chip follows the cursor until a click places it
    WIRING = auto()         # a wire follows the cursor from its start pin until a click on a pin


def _make_config() -> pyglet.gl.Config | None:
    """Ask for 4x multisampling (smooth shape edges); fall back to the default if unsupported."""
    screen = pyglet.display.get_display().get_default_screen()
    try:
        return screen.get_best_config(pyglet.gl.Config(double_buffer=True, sample_buffers=1, samples=4))
    except pyglet.window.NoSuchConfigException:
        return None


class Editor(pyglet.window.Window):
    def __init__(self) -> None:
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
        self.help = pyglet.text.Label(
            "click part: pick up/place | click pin: wire (click empty: bend) | drag: move | "
            "click IN: toggle | right-click: delete/cancel | middle/right-drag: pan | scroll: zoom | hold Ctrl: snap",
            font_name="Consolas", font_size=10, color=T.HELP_TEXT,
            x=8, y=self.height - 8, anchor_y="top", batch=self.hud)

        self.chip_views: dict[Chip, ChipView] = {}
        self.wire_views: dict[Wire, WireView] = {}

        # interaction state
        self.mode = Mode.IDLE
        self.panning = False                 # orthogonal to mode: you can pan while carrying things
        self.mouse = (0, 0)                  # last known cursor position, screen space
        self.active: ChipView | None = None  # chip being pressed / dragged / placed
        self.grab = (0.0, 0.0)               # chip origin minus cursor, world units
        self.press_at = (0, 0)               # screen pos of the press on a chip
        self.wire_start: Pin | None = None
        self.wire_bends: list[Point] = []
        self.preview: Polyline | None = None

        self._build_demo()
        self.camera.center_on(*HOME, self.width, self.height)
        pyglet.clock.schedule_interval(self.update, 1 / 60)

    # ======================================================================
    # model + view bookkeeping
    # ======================================================================

    def add_chip(self, kind: str, x: float, y: float) -> ChipView:
        chip = self.circuit.add_chip(kind)
        view = ChipView(chip, x, y, self.world, self.layers, self.text)
        self.chip_views[chip] = view
        return view

    def remove_chip(self, view: ChipView) -> None:
        for wire in self.circuit.remove_chip(view.chip):
            self.wire_views.pop(wire).delete()
        del self.chip_views[view.chip]
        view.delete()

    def connect(self, a: Pin, b: Pin, bends: list[Point] = ()) -> Wire | None:
        """Connect pins; `bends` are ordered from a to b."""
        wire, replaced = self.circuit.connect(a, b)
        if replaced is not None:
            self.wire_views.pop(replaced).delete()
        if wire is not None:
            ordered = list(bends) if wire.src is a else list(reversed(bends))
            self.wire_views[wire] = WireView(wire, self.pin_pos(wire.src), ordered, self.pin_pos(wire.dst),
                                             self.world, self.layers)
        return wire

    def remove_wire(self, view: WireView) -> None:
        self.circuit.remove_wire(view.wire)
        del self.wire_views[view.wire]
        view.delete()

    def pin_pos(self, pin: Pin) -> Point:
        return self.chip_views[pin.chip].pin_pos(pin)

    def refresh_wires_of(self, chip: Chip) -> None:
        for wire, view in self.wire_views.items():
            if wire.src.chip is chip or wire.dst.chip is chip:
                view.set_ends(self.pin_pos(wire.src), self.pin_pos(wire.dst))

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
            if view is self.active and self.mode is Mode.PLACING_CHIP:
                continue  # the chip on the cursor isn't a target
            if pin := view.pin_at(wx, wy, self.slop):
                return pin
        return None

    def chip_at(self, wx: float, wy: float) -> ChipView | None:
        return next((v for v in reversed(self.chip_views.values()) if v.contains(wx, wy)), None)

    def wire_at(self, wx: float, wy: float) -> WireView | None:
        limit = T.WIRE_THICKNESS / 2 + self.slop
        best = min(self.wire_views.values(), key=lambda v: v.distance_to(wx, wy), default=None)
        return best if best is not None and best.distance_to(wx, wy) <= limit else None

    def can_wire_to(self, pin: Pin | None) -> bool:
        s = self.wire_start
        return pin is not None and s is not None and pin.is_input != s.is_input and pin.chip is not s.chip

    # ======================================================================
    # input: the state machine
    # ======================================================================

    def on_mouse_press(self, x, y, button, modifiers):
        self.mouse = (x, y)
        wx, wy = self.camera.screen_to_world(x, y)
        tool = self.toolbar.button_at(x, y)

        if button == mouse.MIDDLE:
            self.panning = True
            return

        if self.mode is Mode.PLACING_CHIP:
            if button == mouse.LEFT and tool:
                self._cancel()
                self._start_placing(tool)       # swap to a different part
            elif button == mouse.LEFT:
                kind = self.active.chip.kind
                self.active.set_ghost(False)    # commit
                self.active = None
                self.mode = Mode.IDLE
                if modifiers & key.MOD_SHIFT:
                    self._start_placing(kind)   # keep placing more of the same
            elif button == mouse.RIGHT:
                self._cancel()
            return

        if self.mode is Mode.WIRING:
            if button == mouse.LEFT and tool:
                self._cancel()
                self._start_placing(tool)
            elif button == mouse.LEFT:
                pin = self.pin_at(wx, wy)
                if pin is self.wire_start:
                    self._cancel()
                elif self.can_wire_to(pin):
                    self.connect(self.wire_start, pin, self.wire_bends)
                    self._cancel()              # clears the preview; the wire now exists
                elif pin is None:
                    self.wire_bends.append(self.snapped(wx, wy))
                    self._update_preview()
                # clicking an invalid pin (in->in, same chip) does nothing
            elif button == mouse.RIGHT:
                self._pop_bend_or_cancel()
            return

        # IDLE
        if button == mouse.LEFT:
            if tool:
                self._start_placing(tool)
            elif pin := self.pin_at(wx, wy):
                self.mode = Mode.WIRING
                self.wire_start = pin
                self.wire_bends = []
                self.preview = Polyline([], T.WIRE_PREVIEW, self.world, self.layers.overlay)
                self._update_preview()
            elif view := self.chip_at(wx, wy):
                self.active = view
                self.grab = (view.x - wx, view.y - wy)
                self.press_at = (x, y)
                self.mode = Mode.PRESSING_CHIP

        elif button == mouse.RIGHT:
            if view := self.chip_at(wx, wy):
                self.remove_chip(view)
            elif wire := self.wire_at(wx, wy):
                self.remove_wire(wire)
            else:
                self.panning = True

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.mouse = (x, y)
        if self.panning:
            self.camera.pan(dx, dy)
        elif self.mode is Mode.PRESSING_CHIP:
            px, py = self.press_at
            if abs(x - px) + abs(y - py) >= T.DRAG_THRESHOLD_PX:
                self.mode = Mode.DRAGGING_CHIP
        self._follow_cursor()

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse = (x, y)
        self.toolbar.set_hover(self.toolbar.button_at(x, y))
        self._follow_cursor()

    def on_mouse_release(self, x, y, button, modifiers):
        # Releases never *finish* a click-to-place or click-to-wire action.
        # That is what makes press-and-hold on a toolbar button or pin harmless.
        if button in (mouse.MIDDLE, mouse.RIGHT) and self.panning:
            self.panning = False
        elif button == mouse.LEFT and self.mode is Mode.PRESSING_CHIP:
            self.active.chip.toggle()  # click without movement (no-op for non-switches)
            self.mode, self.active = Mode.IDLE, None
        elif button == mouse.LEFT and self.mode is Mode.DRAGGING_CHIP:
            self.mode, self.active = Mode.IDLE, None

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.mouse = (x, y)
        self.camera.scroll(x, y, scroll_y)
        self._follow_cursor()

    def on_key_press(self, symbol, modifiers):
        # Deliberately NOT calling super(): pyglet's default closes the window on Esc.
        if symbol in (key.LCTRL, key.RCTRL):
            self._follow_cursor()  # snap whatever is on the cursor right away
        elif symbol == key.ESCAPE:
            self._cancel()
        elif symbol == key.BACKSPACE and self.mode is Mode.WIRING:
            self._pop_bend_or_cancel()
        elif symbol == key.HOME:
            self.camera.set_level(0, 0, 0)
            self.camera.center_on(*HOME, self.width, self.height)
            self._follow_cursor()

    # ---- helpers -----------------------------------------------------------

    def _start_placing(self, kind: str) -> None:
        view = self.add_chip(kind, 0, 0)
        view.set_ghost(True)
        self.active = view
        self.grab = (-view.w / 2, -view.h / 2)  # carry it by its center
        self.mode = Mode.PLACING_CHIP
        self._follow_cursor()

    def on_key_release(self, symbol, modifiers):
        if symbol in (key.LCTRL, key.RCTRL):
            self._follow_cursor()  # un-snap

    @property
    def snapping(self) -> bool:
        return self.keys[key.LCTRL] or self.keys[key.RCTRL]

    def snapped(self, wx: float, wy: float) -> Point:
        """Round to the nearest grid point while Ctrl is held."""
        if not self.snapping:
            return wx, wy
        return round(wx / T.GRID) * T.GRID, round(wy / T.GRID) * T.GRID

    def _follow_cursor(self) -> None:
        """Keep whatever is attached to the cursor under the cursor (also after pan/zoom)."""
        if self.mode in (Mode.DRAGGING_CHIP, Mode.PLACING_CHIP):
            wx, wy = self.camera.screen_to_world(*self.mouse)
            # Snap the chip's origin: chip sizes are grid multiples, so its pins land on grid points.
            self.active.move_to(*self.snapped(wx + self.grab[0], wy + self.grab[1]))
            self.refresh_wires_of(self.active.chip)
        elif self.mode is Mode.WIRING:
            self._update_preview()

    def _update_preview(self) -> None:
        wx, wy = self.camera.screen_to_world(*self.mouse)
        target = self.pin_at(wx, wy)
        valid = self.can_wire_to(target)
        end = self.pin_pos(target) if valid else self.snapped(wx, wy)  # valid pins always win
        self.preview.set_points([self.pin_pos(self.wire_start), *self.wire_bends, end])
        self.preview.color = T.WIRE_PREVIEW_SNAP if valid else T.WIRE_PREVIEW

    def _pop_bend_or_cancel(self) -> None:
        if self.wire_bends:
            self.wire_bends.pop()
            self._update_preview()
        else:
            self._cancel()

    def _cancel(self) -> None:
        """Abort whatever is in progress and return to IDLE."""
        if self.mode is Mode.PLACING_CHIP and self.active is not None:
            self.remove_chip(self.active)
        if self.preview is not None:
            self.preview.delete()
            self.preview = None
        self.mode = Mode.IDLE
        self.active = None
        self.wire_start = None
        self.wire_bends = []

    # ======================================================================
    # tick + draw
    # ======================================================================

    def update(self, dt: float) -> None:
        for _ in range(SIM_STEPS_PER_FRAME):
            self.circuit.step()
        for view in self.chip_views.values():
            view.sync()
        for view in self.wire_views.values():
            view.sync()

    def on_resize(self, width, height):
        super().on_resize(width, height)  # keeps the projection matrix in sync
        self.help.y = height - 8

    def on_draw(self):
        self.clear()
        self.grid.draw(self, self.camera, emphasized=self.snapping)  # also paints the background
        self.view = self.camera.matrix()
        self.world.draw()
        self.view = Mat4()  # identity: HUD is in screen pixels
        self.hud.draw()


def run() -> None:
    Editor()
    pyglet.app.run()
