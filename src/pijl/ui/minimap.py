"""Panels in the top-right corner that show the board a second time, each toggled by
its own key and open at any zoom:

  Minimap (M)  the whole board, with what you're looking at outlined (blue), what the
               lens shows outlined (yellow), and a crosshair where the cursor points.
               Press (any button) or drag in it to go there.
  Lens (G)     a magnified view around the cursor (crosshair in the middle), with a
               frame on the board showing what it covers. It also shows the
               screen-space things that belong to the board -- the selection box, the
               context menu -- magnified the same.

Scrolling (the editor decides, with scrolls_lens / anchor):
  on the minimap, lens open     the lens's magnification
  on the minimap, lens closed   zooms the view: around the cursor if it's inside the
                                blue outline, else the view jumps there first
  on the lens                   the lens's magnification
  on the board, G held          the lens's magnification
  on the board                  zooms the view

The minimap doesn't rescale while the cursor is on it (what you point at stays put
while you steer and zoom), unless the view no longer fits in it.

Both follow the *pointer*: the world point the cursor refers to, whichever view it's
in. On the board that's what's under it; over the minimap, the spot it maps to (so
the lens magnifies what you point at on the map); over the lens, the spot under it
in the lens (the lens itself holds still).

Over a context menu, the lens eases out to fit the menu (a menu is readable at 1x
already) and the menu holds still in it while the crosshair follows the cursor.
Along a side where the menu can't fit (it's never shown below 1x), the crosshair
moves across the lens as the cursor moves across the menu instead, panning it edge
to edge. Off the menu, the lens eases back to its own magnification, crosshair in
the middle on the cursor's true spot.

Both show the grid too (grid.py's shader, drawn for the panel's view). The lens sits
under the minimap when both are open and moves up when the minimap closes. Panels
slide in from the right edge and back out.

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
EASE_S = 0.12     # the minimap's region (and the lens's scale) ease with this time constant
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
        self.draw_over(window, tx, ty)
        window.view = Mat4()
        gl.glDisable(gl.GL_SCISSOR_TEST)
        self.front.draw()

    def draw_over(self, window: pyglet.window.Window, tx: float, ty: float) -> None:
        """Anything else to show over the world, still clipped (the world's view is set)."""


class Minimap(Panel):
    """The whole board: its bounds together with what's on screen (so the outline is
    always inside), eased toward as they change so panning doesn't make it jump, and
    held still while you drag in it."""

    def __init__(self) -> None:
        super().__init__(T.MINIMAP_CROSS)
        self.view_fill = shapes.Rectangle(0, 0, 1, 1, color=T.MINIMAP_VIEW_FILL, batch=self.front)
        self.view_box = shapes.Box(0, 0, 1, 1, thickness=self.line, color=T.MINIMAP_VIEW, batch=self.front)
        self.lens_box = shapes.Box(0, 0, 1, 1, thickness=self.line, color=T.LENS_FRAME, batch=self.front)
        self.lens_box.visible = False
        self.held = False                # being dragged in: don't move what it maps
        self.hovered = False             # the cursor is on it: nor then
        self.region: Rect | None = None  # world rect mapped onto the panel (eased)

    @property
    def ready(self) -> bool:
        return self.region is not None

    def update(self, dt: float, board: Rect | None, visible: Rect,
               pointer: tuple[float, float] | None, win_w: int, top: float) -> None:
        """`board`: bounds of everything on it (None if empty). `visible`: the world
        rect you can actually see (right of the picker, above the status bar).
        `pointer`: the world point the cursor refers to (see the module docstring)."""
        if board is None and not self.held:
            self.region = None  # nothing to map
        elif self.open and (not (self.held or self.hovered)
                            or self.region is None or not _inside(visible, self.region)):
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
        self.place_cross(None if pointer is None else self.to_screen(*pointer))

    def _place_view(self, visible: Rect) -> None:
        self._outline(visible, (self.view_fill, self.view_box))

    def place_lens(self, covers: Rect | None) -> None:
        """Outline what the lens shows (world rect), or nothing."""
        self.lens_box.visible = covers is not None and self.ready and self._outline(covers, (self.lens_box,))

    def _outline(self, r: Rect, boxes) -> bool:
        """Fit these shapes to world rect r: at least a few px big (around its center) and
        cut to the panel. False if it's entirely off the panel."""
        (ax, ay), (bx, by) = self.to_screen(*r[:2]), self.to_screen(*r[2:])
        cx, cy = (ax + bx) / 2, (ay + by) / 2
        hw, hh = max(bx - ax, MIN_VIEW_PX) / 2, max(by - ay, MIN_VIEW_PX) / 2
        ax, bx = max(self.x, cx - hw), min(self.x + W, cx + hw)
        ay, by = max(self.y, cy - hh), min(self.y + H, cy + hh)
        w, h = max(1.0, bx - ax), max(1.0, by - ay)
        for s in boxes:
            s.position = (ax, ay)
            s.width, s.height = w, h
        return bx > ax and by > ay


class Lens(Panel):
    """The board around the pointer, `steps` zoom levels closer than the camera."""

    def __init__(self, frame_batch: pyglet.graphics.Batch) -> None:
        super().__init__(T.LENS_CROSS)
        self.steps = T.LENS_STEPS
        self._accum = 0.0
        self.pointer: tuple[float, float] | None = None  # world point under the lens's crosshair
        self.at = (0.5, 0.5)  # where the crosshair is in the lens (fractions of its size)
        self.menu: tuple | None = None  # while on a menu: its (x, y, w, h) and the cursor, screen px
        self.overlay: pyglet.graphics.Batch | None = None  # screen-space things to magnify too
        self.camera: Camera | None = None  # (for mapping them)
        # a frame on the board showing what the lens covers (drawn under the panels),
        # and the magnification in a corner
        self.frame = shapes.Box(0, 0, 1, 1, thickness=self.line, color=T.LENS_FRAME, batch=frame_batch)
        self.tag = shapes.Rectangle(0, 0, 1, 1, color=T.PIN_TAG_BG, batch=self.front,
                                    group=pyglet.graphics.Group(order=0))
        self.label = pyglet.text.Label("", font_name="Consolas", font_size=9 * S, color=T.HELP_TEXT,
                                       anchor_x="right", anchor_y="bottom", batch=self.front,
                                       group=pyglet.graphics.Group(order=1))

    @property
    def ready(self) -> bool:
        return self.pointer is not None

    def adjust(self, notches: float) -> None:
        """More (or less) magnification: scroll notches, one zoom level each."""
        self._accum += notches  # (trackpads send fractions: they add up to whole levels)
        whole = int(self._accum)
        self._accum -= whole
        lo, hi = T.LENS_STEPS_RANGE
        self.steps = max(lo, min(hi, self.steps + whole))

    def update(self, dt: float, camera: Camera, win_w: int, top: float) -> None:
        """(Panels sets `pointer`, `at` and `fit` first.)"""
        self.camera = camera
        was_shown = self.shown
        self.slide(dt, win_w, top)
        self.frame.visible = bool(self.shown and self.pointer is not None)
        if self.pointer is None:
            return
        target = 2 ** (min(MAX_LEVEL, camera.level + self.steps) / STEPS_PER_OCTAVE)
        if self.menu is not None:  # out to fit the menu, but not below 1x (then it pans instead)
            _, _, mw, mh, _, _ = self.menu
            target = min(target, max(camera.zoom, camera.zoom * min(W / mw, H / mh)))
        if not was_shown:
            self.scale = target  # coming in: no easing from wherever it was
        else:  # ease, evenly in log space (zooming in and out feel the same)
            k = min(1.0, dt / EASE_S)
            self.scale *= (target / self.scale) ** k
        if self.menu is not None:
            mx, my, mw, mh, cx, cy = self.menu
            k = self.scale / camera.zoom
            self.at = (_menu_at(mx, mw, cx, k, W), _menu_at(my, mh, cy, k, H))
        u, v = self.at
        px, py = self.pointer
        self.center = (px - (u - 0.5) * W / self.scale, py - (v - 0.5) * H / self.scale)
        m = self.scale / camera.zoom
        self.label.text = f"\u00d7{m:.1f}" if m < 10 else f"\u00d7{m:.0f}"
        pad = 3 * S
        self.label.position = (self.x + W - 5 * S, self.y + 4 * S, 0)
        self.tag.position = (self.label.x - self.label.content_width - pad, self.label.y - pad / 2)
        self.tag.width, self.tag.height = self.label.content_width + 2 * pad, self.label.content_height + pad
        # the frame: what the lens covers, on the board
        cx, cy = self.center
        hw, hh = W / 2 / self.scale, H / 2 / self.scale
        (ax, ay), (bx, by) = camera.world_to_screen(cx - hw, cy - hh), camera.world_to_screen(cx + hw, cy + hh)
        self.frame.position = (round(ax), round(ay))
        self.frame.width, self.frame.height = max(2, round(bx - ax)), max(2, round(by - ay))
        self.frame.opacity = round(T.LENS_FRAME[3] * self.shown)
        self.place_cross((self.x + u * W, self.y + v * H))

    def covers(self) -> Rect | None:
        """The world rect it shows, if shown."""
        if not self.shown or self.pointer is None:
            return None
        (cx, cy), hw, hh = self.center, W / 2 / self.scale, H / 2 / self.scale
        return cx - hw, cy - hh, cx + hw, cy + hh

    def draw(self, window: pyglet.window.Window, world: Canvas, grid: Grid, snap: bool = True) -> None:
        super().draw(window, world, grid, snap)

    def draw_over(self, window: pyglet.window.Window, tx: float, ty: float) -> None:
        """The overlay (selection box, context menu), magnified like the world: the main
        view puts world point p at p * zoom + T (screen px); this puts it at p * s + t.
        So a screen point q goes to (q - T) / zoom * s + t."""
        if self.overlay is None or self.camera is None:
            return
        k = self.scale / self.camera.zoom
        mx, my = self.camera.translation()
        window.view = Mat4(k, 0, 0, 0,
                           0, k, 0, 0,
                           0, 0, 1, 0,
                           tx - mx * k, ty - my * k, 0, 1)
        self.overlay.draw()


class Panels:
    """Both panels, stacked: the lens under the minimap, moving up when it closes.
    Works out the pointer for both (see the module docstring)."""

    def __init__(self, overlay: pyglet.graphics.Batch | None = None) -> None:
        self.frames = pyglet.graphics.Batch()  # on the board, under the panels
        self.minimap = Minimap()
        self.lens = Lens(self.frames)
        self.lens.overlay = overlay
        self.pointer: tuple[float, float] | None = None  # (for the minimap's crosshair)

    def update(self, dt: float, camera: Camera, board: Rect | None, visible: Rect,
               cursor: tuple[float, float] | None, on_board: bool, menu: Rect | None,
               win_w: int, win_h: int) -> None:
        """`cursor`: the mouse (screen px) if it's in the window. `on_board`: it's over the
        board (not the picker or the status bar). `menu`: the context menu panel under
        it, (x, y, w, h), if any."""
        lens = self.lens
        lens.menu = None
        self.minimap.hovered = cursor is not None and self.minimap.contains(*cursor)
        if cursor is None:
            pass  # out of the window: everything stays
        elif menu is not None:
            x, y, w, h = menu
            self.pointer = lens.pointer = _screen_to_world(camera, cursor)
            lens.menu = (x, y, w, h, *cursor)  # (the lens works out `at`: it depends on its scale)
        elif lens.contains(*cursor) and lens.ready:
            self.pointer = lens.to_world(*cursor)  # the lens holds still; the map shows where you point
        elif self.minimap.contains(*cursor):
            self.pointer = lens.pointer = self.minimap.to_world(*cursor)
            lens.at = (0.5, 0.5)
        elif on_board:
            self.pointer = lens.pointer = _screen_to_world(camera, cursor)
            lens.at = (0.5, 0.5)
        top = win_h - MARGIN
        self.minimap.update(dt, board, visible, self.pointer, win_w, top)
        below = top - _ease(self.minimap.shown) * (H + MARGIN)
        lens.update(dt, camera, win_w, below)
        self.minimap.place_lens(lens.covers())

    def contains(self, sx: float, sy: float) -> bool:
        return self.minimap.contains(sx, sy) or self.lens.contains(sx, sy)

    def scrolls_lens(self, sx: float, sy: float) -> bool:
        """Does scrolling here change the lens's magnification (rather than zoom the view)?
        On the lens, or on the minimap while the lens is open."""
        lens = self.lens
        return lens.open and lens.ready and (lens.contains(sx, sy) or self.minimap.contains(sx, sy))

    def anchor(self, sx: float, sy: float, visible: Rect) -> tuple[tuple[float, float], bool] | None:
        """Scrolling on the minimap at (sx, sy) zooms the view around the spot pointed at:
        (that world point, whether the view has to jump there first -- it's outside the
        view). None when not on the minimap."""
        if not self.minimap.contains(sx, sy):
            return None
        px, py = self.minimap.to_world(sx, sy)
        inside = visible[0] <= px <= visible[2] and visible[1] <= py <= visible[3]
        return (px, py), not inside

    def draw(self, window: pyglet.window.Window, world: Canvas, grid: Grid) -> None:
        self.frames.draw()
        self.minimap.draw(window, world, grid)
        self.lens.draw(window, world, grid)


def _menu_at(start: float, length: float, cursor: float, k: float, size: float) -> float:
    """Along one side: where the lens's crosshair goes (fraction of the lens) for a cursor
    on a menu, at magnification k. If the menu fits, it sits centered and holds still;
    if not, cursor at the menu's edge = crosshair at the lens's edge, panning in between."""
    if k * length <= size:
        return 0.5 + (cursor - (start + length / 2)) * k / size
    return min(1.0, max(0.0, (cursor - start) / length))


def _screen_to_world(camera: Camera, p: tuple[float, float]) -> tuple[float, float]:
    """Like camera.screen_to_world, but with the whole-pixel translation the world is
    actually drawn with, so things line up exactly when magnified."""
    tx, ty = camera.translation()
    return (p[0] - tx) / camera.zoom, (p[1] - ty) / camera.zoom


def _inside(a: Rect, b: Rect) -> bool:
    return a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]


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
