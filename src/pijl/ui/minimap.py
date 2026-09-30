"""Panels in the top-right corner that show the board a second time, each toggled by
its own key and open at any zoom:

  Minimap (M)  the whole board, with what you're looking at outlined and a crosshair
               where the cursor is. Click or drag in it to go there.
  Lens (G)     a magnified view around the cursor (crosshair in the middle), with a
               frame on the board showing what it covers. Hold G and scroll to change
               the magnification.

Both show the grid too (grid.py's shader, drawn for the panel's view).

The lens sits under the minimap when both are open and moves up when the minimap
closes. Panels slide in from the right edge and back out.

No second copy of the world exists: the canvas's instance buffers are simply drawn
a second time through another view matrix, clipped to the panel (glScissor). So the
panels are live -- signals light up in them too -- and cost one draw call per buffer.
The shapes anti-alias per pixel (fwidth), so they stay clean at any scale; text
fades out by itself once it's too small to read.
"""

from __future__ import annotations

import pyglet
from pyglet import gl, shapes
from pyglet.math import Mat4

from . import theme as T
from .camera import MAX_LEVEL, STEPS_PER_OCTAVE, Camera
from .canvas import Canvas
from .grid import Grid

S = T.UI_SCALE
W, H = round(T.MINIMAP_SIZE[0] * S), round(T.MINIMAP_SIZE[1] * S)
MARGIN = round(T.MINIMAP_MARGIN * S)  # from the window's edges, and between the panels
PAD = 0.06        # the minimap leaves this much room around what it shows (of its size)
SLIDE_S = 0.18    # seconds to slide in or out
EASE_S = 0.12     # the minimap's region eases toward its target with this time constant
MIN_VIEW_PX = 4   # the view outline never gets smaller than this

Rect = tuple[float, float, float, float]  # x0, y0, x1, y1


def _ease(t: float) -> float:
    return 1 - (1 - t) ** 3


class Panel:
    """What both panels share: a box that slides in and out, with the world drawn in
    it at `scale` (screen px per world unit) around `center` (a world point)."""

    def __init__(self, cross_color) -> None:
        self.front = pyglet.graphics.Batch()  # over the world
        self.line = max(1, round(S))
        # a crosshair spanning the whole panel: horizontal, vertical
        self.cross = [shapes.Rectangle(0, 0, 1, 1, color=cross_color, batch=self.front) for _ in range(2)]
        self.border = shapes.Box(0, 0, W, H, thickness=self.line, color=T.PICKER_BORDER, batch=self.front)
        self.open = False
        self.shown = 0.0          # 0 = tucked away past the right edge, 1 = in place
        self.x = self.y = 0.0     # bottom-left, screen px (after sliding)
        self.scale = 1.0
        self.center = (0.0, 0.0)

    @property
    def ready(self) -> bool:
        """Has something to show (else it stays tucked away, open or not)."""
        return True

    def slide(self, dt: float, win_w: int, top: float) -> None:
        """Ease toward open / closed; `top`: where the panel's top edge goes."""
        step = dt / SLIDE_S
        on = self.open and self.ready
        self.shown = min(1.0, self.shown + step) if on else max(0.0, self.shown - step)
        self.x = win_w - MARGIN - W + (1 - _ease(self.shown)) * (W + MARGIN + 2)
        self.y = top - H
        self.border.position = (self.x, self.y)

    def place_cross(self, at: tuple[float, float] | None) -> None:
        """Put the crosshair through screen point `at` (hidden if None or off the panel)."""
        h, v = self.cross
        if at is None:
            h.visible = v.visible = False
            return
        sx, sy = round(at[0]), round(at[1])
        h.visible = self.y <= sy < self.y + H
        v.visible = self.x <= sx < self.x + W
        h.position, h.width, h.height = (self.x, sy), W, self.line
        v.position, v.width, v.height = (sx, self.y), self.line, H

    def contains(self, sx: float, sy: float) -> bool:
        return self.shown > 0 and self.x <= sx <= self.x + W and self.y <= sy <= self.y + H

    def to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        cx, cy = self.center
        return self.x + W / 2 + (wx - cx) * self.scale, self.y + H / 2 + (wy - cy) * self.scale

    def to_world(self, sx: float, sy: float) -> tuple[float, float]:
        cx, cy = self.center
        return cx + (sx - self.x - W / 2) / self.scale, cy + (sy - self.y - H / 2) / self.scale

    def draw(self, window: pyglet.window.Window, world: Canvas, grid: Grid, snap: bool = False) -> None:
        """Call with the window's view at identity (screen px); leaves it that way.
        `snap`: whole-pixel translation (text doesn't shimmer as the view follows the cursor)."""
        if not self.shown or not self.ready:
            return
        s = self.scale
        cx, cy = self.center
        tx, ty = self.x + W / 2 - cx * s, self.y + H / 2 - cy * s
        if snap:
            tx, ty = round(tx), round(ty)
        ratio = window.get_framebuffer_size()[0] / window.width
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(round(self.x * ratio), round(self.y * ratio), round(W * ratio), round(H * ratio))
        grid.draw_at(window, (tx, ty), s, T.MINIMAP_BG, T.GRID_COLORS)  # (also paints the background)
        window.view = Mat4(s, 0, 0, 0,
                           0, s, 0, 0,
                           0, 0, 1, 0,
                           tx, ty, 0, 1)
        world.draw_instances()
        window.view = Mat4()
        gl.glDisable(gl.GL_SCISSOR_TEST)
        self.front.draw()


class Minimap(Panel):
    """The whole board: its bounds together with what's on screen (so the outline is
    always inside), eased toward as they change so panning doesn't make it jump, and
    held still while you drag in it."""

    def __init__(self) -> None:
        super().__init__(T.MINIMAP_CROSS)
        self.view_fill = shapes.Rectangle(0, 0, 1, 1, color=T.MINIMAP_VIEW_FILL, batch=self.front)
        self.view_box = shapes.Box(0, 0, 1, 1, thickness=self.line, color=T.MINIMAP_VIEW, batch=self.front)
        self.held = False                # being dragged in: don't move what it maps
        self.region: Rect | None = None  # world rect mapped onto the panel (eased)

    @property
    def ready(self) -> bool:
        return self.region is not None

    def update(self, dt: float, camera: Camera, board: Rect | None, visible: Rect,
               cursor: tuple[float, float] | None, win_w: int, top: float) -> None:
        """`board`: bounds of everything on it (None if empty). `visible`: the world
        rect you can actually see (right of the picker, above the status bar).
        `cursor`: the mouse, screen px, if it's over the board."""
        if board is None and not self.held:
            self.region = None  # nothing to map
        elif self.open and not self.held:
            target = _fit(_pad(_union(board, visible)), W / H)
            if self.region is None or not self.shown:
                self.region = target  # coming in: fitted right away, not easing in from before
            else:
                k = min(1.0, dt / EASE_S)
                self.region = tuple(a + (b - a) * k for a, b in zip(self.region, target))
        self.slide(dt, win_w, top)
        if self.region is None:
            return
        x0, y0, x1, y1 = self.region
        self.scale = W / (x1 - x0)
        self.center = ((x0 + x1) / 2, (y0 + y1) / 2)
        self._place_view(visible)
        # the crosshair: on the cursor itself while it's over the map (where a click goes),
        # else where the cursor is on the board
        if cursor is not None and not self.contains(*cursor):
            cursor = self.to_screen(*camera.screen_to_world(*cursor))
        self.place_cross(cursor)

    def _place_view(self, visible: Rect) -> None:
        (ax, ay), (bx, by) = self.to_screen(*visible[:2]), self.to_screen(*visible[2:])
        # keep it at least a few px big (around its center) and inside the panel
        cx, cy = (ax + bx) / 2, (ay + by) / 2
        hw, hh = max(bx - ax, MIN_VIEW_PX) / 2, max(by - ay, MIN_VIEW_PX) / 2
        ax, bx = max(self.x, cx - hw), min(self.x + W, cx + hw)
        ay, by = max(self.y, cy - hh), min(self.y + H, cy + hh)
        w, h = max(1.0, bx - ax), max(1.0, by - ay)
        for s in (self.view_fill, self.view_box):
            s.position = (ax, ay)
            s.width, s.height = w, h


class Lens(Panel):
    """The board around the cursor, `steps` zoom levels closer than the camera."""

    def __init__(self) -> None:
        super().__init__(T.LENS_CROSS)
        self.steps = T.LENS_STEPS
        self._accum = 0.0
        self.focus: tuple[float, float] | None = None  # screen point it magnifies
        # a frame on the board around the cursor, and the magnification in a corner
        self.frame = shapes.Box(0, 0, 1, 1, thickness=self.line, color=T.LENS_FRAME, batch=self.front)
        self.tag = shapes.Rectangle(0, 0, 1, 1, color=T.PIN_TAG_BG, batch=self.front,
                                    group=pyglet.graphics.Group(order=0))
        self.label = pyglet.text.Label("", font_name="Consolas", font_size=9 * S, color=T.HELP_TEXT,
                                       anchor_x="right", anchor_y="bottom", batch=self.front,
                                       group=pyglet.graphics.Group(order=1))

    @property
    def ready(self) -> bool:
        return self.focus is not None

    def adjust(self, notches: float) -> None:
        """More (or less) magnification: scroll notches, one zoom level each."""
        self._accum += notches  # (trackpads send fractions: they add up to whole levels)
        whole = int(self._accum)
        self._accum -= whole
        lo, hi = T.LENS_STEPS_RANGE
        self.steps = max(lo, min(hi, self.steps + whole))

    def update(self, dt: float, camera: Camera, cursor: tuple[float, float] | None,
               win_w: int, top: float) -> None:
        """`cursor`: the mouse, screen px, if it's over the board (else it keeps the last spot)."""
        if cursor is not None:
            self.focus = cursor
        self.slide(dt, win_w, top)
        if self.focus is None:
            return
        level = min(MAX_LEVEL, camera.level + self.steps)
        self.scale = 2 ** (level / STEPS_PER_OCTAVE)
        self.center = camera.screen_to_world(*self.focus)
        m = self.scale / camera.zoom
        self.label.text = f"×{m:.1f}" if m < 10 else f"×{m:.0f}"
        pad = 3 * S
        self.label.position = (self.x + W - 5 * S, self.y + 4 * S, 0)
        self.tag.position = (self.label.x - self.label.content_width - pad, self.label.y - pad / 2)
        self.tag.width, self.tag.height = self.label.content_width + 2 * pad, self.label.content_height + pad
        # the frame: what the lens covers, drawn on the board around the cursor
        fx, fy = self.focus
        fw, fh = W * camera.zoom / self.scale, H * camera.zoom / self.scale
        self.frame.position = (round(fx - fw / 2), round(fy - fh / 2))
        self.frame.width, self.frame.height = max(2, round(fw)), max(2, round(fh))
        self.frame.opacity = round(T.LENS_FRAME[3] * self.shown)
        self.place_cross((self.x + W / 2, self.y + H / 2))  # the middle: where the cursor is

    def draw(self, window: pyglet.window.Window, world: Canvas, grid: Grid, snap: bool = True) -> None:
        super().draw(window, world, grid, snap)


class Panels:
    """Both panels, stacked: the lens under the minimap, moving up when it closes."""

    def __init__(self) -> None:
        self.minimap = Minimap()
        self.lens = Lens()

    def update(self, dt: float, camera: Camera, board: Rect | None, visible: Rect,
               cursor: tuple[float, float] | None, win_w: int, win_h: int) -> None:
        top = win_h - MARGIN
        self.minimap.update(dt, camera, board, visible, cursor, win_w, top)
        below = top - _ease(self.minimap.shown) * (H + MARGIN)
        self.lens.update(dt, camera, cursor, win_w, below)

    def contains(self, sx: float, sy: float) -> bool:
        return self.minimap.contains(sx, sy) or self.lens.contains(sx, sy)

    def draw(self, window: pyglet.window.Window, world: Canvas, grid: Grid) -> None:
        self.minimap.draw(window, world, grid)
        self.lens.draw(window, world, grid)


def _union(a: Rect, b: Rect) -> Rect:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _pad(r: Rect) -> Rect:
    x0, y0, x1, y1 = r
    p = PAD * max(x1 - x0, y1 - y0, 1.0)
    return x0 - p, y0 - p, x1 + p, y1 + p


def _fit(r: Rect, aspect: float) -> Rect:
    """Grow the shorter side (around the center) so r has this width / height."""
    x0, y0, x1, y1 = r
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:
        grow = (h * aspect - w) / 2
        return x0 - grow, y0, x1 + grow, y1
    grow = (w / aspect - h) / 2
    return x0, y0 - grow, x1, y1 + grow
