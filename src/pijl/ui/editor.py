"""The editor window: input handling (a small state machine), sim ticking, drawing.

Controls
  left-drag chip        move it
  left-click switch     toggle an IN switch
  left-drag from pin    draw a wire, release on another pin to connect
  right-click           delete the chip or wire under the cursor
  right/middle-drag     pan
  scroll                zoom
  toolbar button        spawn a chip and start dragging it
  Esc                   cancel the current action
  Home                  reset the camera
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
from .toolbar import Toolbar
from .views import ChipView, Layers, WireView

SIM_STEPS_PER_FRAME = 1
PALETTE = ["IN", "OUT", "NAND", "AND", "OR", "NOT"]


class Mode(Enum):
    IDLE = auto()
    PRESSING_CHIP = auto()  # mouse down on a chip, not moved yet: could become a click or a drag
    DRAGGING_CHIP = auto()
    DRAWING_WIRE = auto()
    PANNING = auto()


class Editor(pyglet.window.Window):
    def __init__(self) -> None:
        super().__init__(1280, 720, caption="pijl", resizable=True, vsync=True)
        self.circuit = Circuit()
        self.camera = Camera()

        self.world = pyglet.graphics.Batch()
        self.layers = Layers()
        self.hud = pyglet.graphics.Batch()
        self.toolbar = Toolbar(PALETTE, self.hud)
        self.help = pyglet.text.Label(
            "drag: move | click IN: toggle | pin->pin: wire | right-click: delete | "
            "right/middle-drag: pan | scroll: zoom | Home: reset view",
            font_name="Consolas", font_size=10, color=T.HELP_TEXT,
            x=8, y=self.height - 8, anchor_y="top", batch=self.hud)

        self.chip_views: dict[Chip, ChipView] = {}
        self.wire_views: dict[Wire, WireView] = {}

        # interaction state
        self.mode = Mode.IDLE
        self.active: ChipView | None = None       # chip being pressed/dragged
        self.grab = (0.0, 0.0)                    # chip origin minus mouse, in world units
        self.press_at = (0, 0)                    # screen pos of the mouse press
        self.wire_start: Pin | None = None
        self.preview: shapes.Line | None = None

        self._build_demo()
        self.camera.center_on(400, 300, self.width, self.height)
        pyglet.clock.schedule_interval(self.update, 1 / 60)

    # ======================================================================
    # model + view bookkeeping
    # ======================================================================

    def add_chip(self, kind: str, x: float, y: float) -> ChipView:
        chip = self.circuit.add_chip(kind)
        view = ChipView(chip, x, y, self.world, self.layers)
        self.chip_views[chip] = view
        return view

    def remove_chip(self, view: ChipView) -> None:
        for wire in self.circuit.remove_chip(view.chip):
            self.wire_views.pop(wire).delete()
        del self.chip_views[view.chip]
        view.delete()

    def connect(self, a: Pin, b: Pin) -> None:
        wire, replaced = self.circuit.connect(a, b)
        if replaced is not None:
            self.wire_views.pop(replaced).delete()
        if wire is not None:
            self.wire_views[wire] = WireView(wire, self.pin_pos(wire.src), self.pin_pos(wire.dst),
                                             self.world, self.layers)

    def remove_wire(self, view: WireView) -> None:
        self.circuit.remove_wire(view.wire)
        del self.wire_views[view.wire]
        view.delete()

    def pin_pos(self, pin: Pin) -> tuple[float, float]:
        return self.chip_views[pin.chip].pin_pos(pin)

    def refresh_wires_of(self, chip: Chip) -> None:
        for wire, view in self.wire_views.items():
            if wire.src.chip is chip or wire.dst.chip is chip:
                view.set_ends(self.pin_pos(wire.src), self.pin_pos(wire.dst))

    def _build_demo(self) -> None:
        a = self.add_chip("IN", 200, 360)
        b = self.add_chip("IN", 200, 240)
        g = self.add_chip("NAND", 380, 290)
        out = self.add_chip("OUT", 580, 305)
        self.connect(a.chip.outputs[0], g.chip.inputs[0])
        self.connect(b.chip.outputs[0], g.chip.inputs[1])
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
            if pin := view.pin_at(wx, wy, self.slop):
                return pin
        return None

    def chip_at(self, wx: float, wy: float) -> ChipView | None:
        return next((v for v in reversed(self.chip_views.values()) if v.contains(wx, wy)), None)

    def wire_at(self, wx: float, wy: float) -> WireView | None:
        limit = T.WIRE_THICKNESS / 2 + self.slop
        best = min(self.wire_views.values(), key=lambda v: v.distance_to(wx, wy), default=None)
        return best if best is not None and best.distance_to(wx, wy) <= limit else None

    # ======================================================================
    # input: the state machine
    # ======================================================================

    def on_mouse_press(self, x, y, button, modifiers):
        wx, wy = self.camera.screen_to_world(x, y)

        if button == mouse.LEFT:
            if kind := self.toolbar.button_at(x, y):
                view = self.add_chip(kind, 0, 0)
                self.grab = (-view.w / 2, -view.h / 2)
                self.active = view
                self._drag_active_to(wx, wy)
                self.mode = Mode.DRAGGING_CHIP
            elif pin := self.pin_at(wx, wy):
                self.wire_start = pin
                px, py = self.pin_pos(pin)
                self.preview = shapes.Line(px, py, wx, wy, thickness=T.WIRE_THICKNESS,
                                           color=T.WIRE_PREVIEW, batch=self.world, group=self.layers.overlay)
                self.mode = Mode.DRAWING_WIRE
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
                self.mode = Mode.PANNING

        elif button == mouse.MIDDLE:
            self.mode = Mode.PANNING

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        wx, wy = self.camera.screen_to_world(x, y)

        if self.mode is Mode.PRESSING_CHIP:
            px, py = self.press_at
            if abs(x - px) + abs(y - py) >= T.DRAG_THRESHOLD_PX:
                self.mode = Mode.DRAGGING_CHIP

        if self.mode is Mode.DRAGGING_CHIP:
            self._drag_active_to(wx, wy)
        elif self.mode is Mode.DRAWING_WIRE:
            self._update_preview(wx, wy)
        elif self.mode is Mode.PANNING:
            self.camera.pan(dx, dy)

    def on_mouse_release(self, x, y, button, modifiers):
        wx, wy = self.camera.screen_to_world(x, y)

        if self.mode is Mode.PRESSING_CHIP and self.active is not None:
            self.active.chip.toggle()  # a click without movement (no-op for non-switches)
        elif self.mode is Mode.DRAWING_WIRE:
            if (target := self.pin_at(wx, wy)) and self.wire_start is not None:
                self.connect(self.wire_start, target)
        self._reset_mode()

    def on_mouse_motion(self, x, y, dx, dy):
        self.toolbar.set_hover(self.toolbar.button_at(x, y))

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.camera.zoom_at(x, y, 1.1 ** scroll_y)

    def on_key_press(self, symbol, modifiers):
        # Deliberately NOT calling super(): pyglet's default closes the window on Esc.
        if symbol == key.ESCAPE:
            self._reset_mode()
        elif symbol == key.HOME:
            self.camera.zoom = 1.0
            self.camera.center_on(400, 300, self.width, self.height)

    def _drag_active_to(self, wx: float, wy: float) -> None:
        assert self.active is not None
        self.active.move_to(wx + self.grab[0], wy + self.grab[1])
        self.refresh_wires_of(self.active.chip)

    def _update_preview(self, wx: float, wy: float) -> None:
        assert self.preview is not None and self.wire_start is not None
        target = self.pin_at(wx, wy)
        valid = (target is not None and target.is_input != self.wire_start.is_input
                 and target.chip is not self.wire_start.chip)
        if valid:
            wx, wy = self.pin_pos(target)  # snap to the pin
        self.preview.x2, self.preview.y2 = wx, wy
        self.preview.color = T.WIRE_PREVIEW_SNAP if valid else T.WIRE_PREVIEW

    def _reset_mode(self) -> None:
        if self.preview is not None:
            self.preview.delete()
            self.preview = None
        self.mode = Mode.IDLE
        self.active = None
        self.wire_start = None

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
        pyglet.gl.glClearColor(*(c / 255 for c in T.BACKGROUND))
        self.clear()
        self.view = self.camera.matrix()
        self.world.draw()
        self.view = Mat4()  # identity: HUD is in screen pixels
        self.hud.draw()


def run() -> None:
    Editor()
    pyglet.app.run()
