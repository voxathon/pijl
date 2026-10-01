"""A viewport: a window onto the board -- its camera, where the mouse is in it, and
which of its screen px actually show board (the main window's picker and status bar
cover some).

There is one editor (one document, one interaction state machine: there's only one
mouse) and any number of viewports. A viewport is just the window: it hands every input
event to the editor saying where it came from (the editor's `vp` while it handles it),
so screen points always mean that viewport's screen. Drawing, resizing and closing go to
the editor too; the board itself is drawn in every viewport from the same GPU buffers
(see canvas.py: only the vertex arrays are per window).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pyglet
from pyglet import shapes

from . import glshare
from . import theme as T
from .camera import Camera
from .grid import Grid

glshare.install()  # any pyglet batch draws in any viewport (menus, prompts, handles)

if TYPE_CHECKING:
    from .editor import Editor

Rect = tuple[float, float, float, float]  # x0, y0, x1, y1

# What a viewport hands to the editor (as the editor's methods of the same name)
INPUT_EVENTS = (
    "on_mouse_press",
    "on_mouse_release",
    "on_mouse_drag",
    "on_mouse_motion",
    "on_mouse_scroll",
    "on_mouse_enter",
    "on_mouse_leave",
    "on_key_press",
    "on_key_release",
    "on_text",
    "on_text_motion",
    "on_deactivate",
)


class Viewport(pyglet.window.Window):
    def __init__(
        self,
        editor: Editor,
        insets: Callable[[], tuple[float, float]] = lambda: (0.0, 0.0),
        **window: object,
    ) -> None:
        """`insets()`: the screen px along the left and bottom edges that aren't board.
        `window`: pyglet.window.Window's arguments."""
        self.editor: Editor | None = None  # (events during construction are dropped)
        super().__init__(**window)
        self.camera = Camera()
        self.insets = insets
        self.mouse = (0.0, 0.0)  # last known cursor position, its screen px
        self.mouse_in = False  # is it over the window at all?
        self.grid = Grid()  # (its quad's vertex array is per window too)
        self.hud = pyglet.graphics.Batch()  # screen-space things of its own
        self.overlay = pyglet.graphics.Batch()  # the selection box, when it's in this one
        self.editor = editor

    # ---- events: all to the editor -----------------------------------------------------

    def on_resize(self, width, height):
        super().on_resize(width, height)  # keeps the projection matrix in sync
        if self.editor is not None:
            self.editor.resized(self, width, height)

    def on_draw(self):
        if self.editor is not None:
            self.editor.draw(self)

    def on_close(self):
        # Not calling super() (it closes right away): the editor decides (unsaved changes).
        if self.editor is not None:
            self.editor.close_viewport(self)

    # ---- geometry ----------------------------------------------------------------------

    def pixel_ratio(self) -> float:
        """Framebuffer px per window px (HiDPI)."""
        fb_w, _ = self.get_framebuffer_size()
        return fb_w / self.width if self.width else 1.0

    def warp(self, sx: float, sy: float) -> None:
        """Move the cursor to screen point (sx, sy), in window px (the OS wants physical px)."""
        r = self.pixel_ratio()
        self.set_mouse_position(round(sx * r), round(sy * r))
        self.mouse = (sx, sy)  # (before the OS's motion event arrives)

    def board_rect(self) -> Rect:
        """The screen rect that shows board."""
        left, bottom = self.insets()
        return left, bottom, self.width, self.height

    def board_size(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.board_rect()
        return x1 - x0, y1 - y0

    def board_center(self) -> tuple[float, float]:
        """The middle of the board part, screen px."""
        x0, y0, x1, y1 = self.board_rect()
        return (x0 + x1) / 2, (y0 + y1) / 2

    def visible_world(self) -> Rect:
        """The world rect you can see in it."""
        x0, y0, x1, y1 = self.board_rect()
        (ax, ay), (bx, by) = (
            self.camera.screen_to_world(x0, y0),
            self.camera.screen_to_world(x1, y1),
        )
        return ax, ay, bx, by

    def center_on(self, wx: float, wy: float) -> None:
        """Move the camera so (wx, wy) is in the middle of the board part."""
        sx, sy = self.board_center()
        z = self.camera.zoom
        self.camera.x, self.camera.y = wx - sx / z, wy - sy / z


def _forward(event_type: str):
    def handler(self: Viewport, *args) -> None:
        if self.editor is not None:
            self.editor.handle(self, event_type, *args)

    handler.__name__ = event_type
    return handler


# (as methods: pyglet calls a window's own handlers last, after pushed ones like the
# editor's KeyStateHandler -- and replacing on_key_press drops pyglet's Esc-closes-it)
for _event in INPUT_EVENTS:
    setattr(Viewport, _event, _forward(_event))


class ExtraViewport(Viewport):
    """A viewport window besides the main one: just the board, and its zoom in a corner."""

    def __init__(self, editor: Editor, **window: object) -> None:
        super().__init__(editor, **window)
        self._shade_batch = pyglet.graphics.Batch()
        self.shade = shapes.Rectangle(
            0, 0, 1, 1, color=T.PROMPT_SHADE, batch=self._shade_batch
        )
        self.zoom_label = pyglet.text.Label(
            "",
            font_name="Consolas",
            font_size=9 * T.UI_SCALE,
            color=T.HELP_TEXT,
            anchor_x="right",
            anchor_y="bottom",
            batch=self.hud,
        )

    def show_zoom(self) -> None:
        text = f"{100 * self.camera.zoom:.0f}%"
        if self.zoom_label.text != text:
            self.zoom_label.text = text
        pad = 6 * T.UI_SCALE
        self.zoom_label.position = (self.width - pad, pad, 0)

    def draw_shade(self) -> None:
        """Dim it all (a prompt is up in the main window)."""
        self.shade.width, self.shade.height = self.width, self.height
        self._shade_batch.draw()
