"""The miniview: a panel in the top-right corner that shows the board around the
cursor a second time, at its own zoom -- closer than the view (a magnifying glass) or
further out (a map); it's the same thing either way. Its zoom is its own: zooming the
view doesn't change it (the tag in its corner shows it, like the status bar does the
view's). It goes out as far as the view can -- or further, if that's what it takes
for the board to fit in it twice over (so all of it shows wherever on it you point).

It looks at the *pointer*: the world point under the cursor on the board, with the
crosshair through it. On the board, a yellow frame shows what it covers; in it, a blue
outline shows what the view does. Over the miniview itself, it holds still (so you can
point at things in it); pressing or dragging in it (any button) moves its crosshair
there (kept on the panel's edge if you drag off it) -- the view doesn't move.

Parked (the editor drives this; see its docstring for the keys): it stops following
the pointer and stays on its own world point, moved only by nudge() (arrows, G+mouse)
-- within the board and the view -- until unparked. The tag says so, and a hint in the
top-left corner shows the keys: G lights up while held, the arrows while steering.

It also shows the screen-space things that belong to the board: the selection box,
magnified the same, and the context menu, at its own size (it's already readable) with
the spot it was opened at where the board puts it. While a context menu is open it
holds still (centered on where the menu opened) and the crosshair follows the cursor
across it -- over the menu or its submenus, or over the board. Only when that would
take the crosshair past an edge does it move: at once just enough to keep it on the
edge, then easing until the crosshair is back in the middle, where it holds still
again. Zooming (the view or it) keeps the menu where it is in it. When the menu
closes, it follows the cursor again -- picking an item puts the cursor back where the
menu was opened (the editor does that), which is what it was looking at anyway.

It shows the grid too (grid.py's shader, drawn for its view), and slides in from the
right edge and back out.

No second copy of the world exists: the canvas's instance buffers are simply drawn a
second time through another view matrix, clipped to the panel (glScissor). So it's
live -- signals light up in it too -- and costs one draw call per buffer. The shapes
anti-alias per pixel (fwidth), so they stay clean at any scale; text fades out by
itself once it's too small to read.
"""

from __future__ import annotations

import pyglet
from pyglet import gl, shapes
from pyglet.math import Mat4

from . import theme as T
from .camera import MAX_LEVEL, MIN_LEVEL, STEPS_PER_OCTAVE, Camera
from .canvas import Canvas
from .grid import Grid
from .menu import ContextMenu

S = T.UI_SCALE
W, H = round(T.MINI_SIZE[0] * S), round(T.MINI_SIZE[1] * S)
MARGIN = round(T.MINI_MARGIN * S)  # from the window's edges
PAD = 0.06  # parked, it goes this far past the board and the view (of their size)
SLIDE_S = 0.18  # seconds to slide in or out
EASE_S = 0.12  # its scale eases with this time constant
MIN_VIEW_PX = 4  # the view outline never gets smaller than this
HINT_AT = 38  # the key hint: its middle is this far in from the top-left corner (before UI_SCALE)

Rect = tuple[float, float, float, float]  # x0, y0, x1, y1


def _ease(t: float) -> float:
    return 1 - (1 - t) ** 3


class Miniview:
    def __init__(
        self,
        overlay: pyglet.graphics.Batch | None = None,
        context: ContextMenu | None = None,
    ) -> None:
        self.front = pyglet.graphics.Batch()  # over the world
        self.frames = pyglet.graphics.Batch()  # on the board, under the panel
        self.line = max(1, round(S))
        # a crosshair spanning the whole panel: horizontal, vertical
        self.cross = [
            shapes.Rectangle(0, 0, 1, 1, color=T.MINI_CROSS, batch=self.front)
            for _ in range(2)
        ]
        self.border = shapes.Box(
            0, 0, W, H, thickness=self.line, color=T.PICKER_BORDER, batch=self.front
        )
        # what the view shows, outlined in it (drawn clipped to it: past its edges is
        # simply off it)
        self.inside = pyglet.graphics.Batch()
        self.view_box = shapes.Box(
            0, 0, 1, 1, thickness=self.line, color=T.MINI_VIEW, batch=self.inside
        )
        # what it shows, framed on the board
        self.frame = shapes.Box(
            0, 0, 1, 1, thickness=self.line, color=T.MINI_FRAME, batch=self.frames
        )
        # and its zoom, in a corner
        self.tag = shapes.Rectangle(
            0,
            0,
            1,
            1,
            color=T.PIN_TAG_BG,
            batch=self.front,
            group=pyglet.graphics.Group(order=0),
        )
        self.label = pyglet.text.Label(
            "",
            font_name="Consolas",
            font_size=9 * S,
            color=T.HELP_TEXT,
            anchor_x="right",
            anchor_y="bottom",
            batch=self.front,
            group=pyglet.graphics.Group(order=1),
        )
        self.hint = _Hint(self.front)

        self.open = False
        self.shown = 0.0  # 0 = tucked away past the right edge, 1 = in place
        self.x = self.y = 0.0  # bottom-left, screen px (after sliding)
        self.scale = 1.0  # screen px per world unit
        self.center = (0.0, 0.0)  # world point in its middle
        self.level = T.MINI_LEVEL  # its zoom level (like Camera.level)
        self._accum = 0.0
        self.floor = MIN_LEVEL  # as far out as it goes (see the module docstring)
        self.reach: Rect | None = (
            None  # where parked it may go: the board and the view, padded
        )
        self.pointer: tuple[float, float] | None = None  # world point it looks at
        self.at = (
            0.5,
            0.5,
        )  # where that is in it (fractions of its size; menus move it)
        self.hovered = False  # the cursor is on it: it holds still
        self.grabbed = False  # a button pressed in it is held: it holds still too
        self.parked = (
            False  # on its own spot, not following the pointer (see the docstring)
        )
        # While a menu is open it holds still: (the cursor, screen px, if it's on the
        # menu or the board; else None,)
        self.menu: tuple | None = None
        # While held: where the menu's anchor is from its middle, px (so it stays put in
        # it as zoom changes)
        self.hold: tuple[float, float] | None = None
        self.recentering = False  # held, but the cursor got past an edge: easing to it
        self.overlay = overlay  # screen-space things to magnify too
        self.context = context  # and the one to show at its own size
        self.camera: Camera | None = None  # (for mapping them)

    @property
    def ready(self) -> bool:
        """Has something to show (else it stays tucked away, open or not)."""
        return self.pointer is not None

    # ---- zoom ----------------------------------------------------------------------

    def adjust(self, notches: float) -> None:
        """Closer (or further out): scroll notches, one zoom level each."""
        self._accum += (
            notches  # (trackpads send fractions: they add up to whole levels)
        )
        whole = int(self._accum)
        self._accum -= whole
        self.level = max(self.floor, min(MAX_LEVEL, self.level + whole))

    # ---- parking -------------------------------------------------------------------

    def park(self, at: tuple[float, float]) -> None:
        """Stay on world point `at` (or where it is, if already parked)."""
        if not self.parked:
            self.parked, self.pointer, self.at = True, at, (0.5, 0.5)

    def unpark(self) -> None:
        self.parked = False

    def look_at(self, at: tuple[float, float], level: int | None = None) -> None:
        """Park on world point `at` (moving there if already parked), at `level` if given."""
        self.parked, self.pointer, self.at = True, at, (0.5, 0.5)
        if level is not None:
            self.level = max(self.floor, min(MAX_LEVEL, level))

    def fit(self, r: Rect) -> None:
        """Park on world rect r, zoomed to fit it."""
        x0, y0, x1, y1 = r
        self.look_at(
            ((x0 + x1) / 2, (y0 + y1) / 2), Camera.fit_level(x1 - x0, y1 - y0, W, H)
        )

    def point(self, sx: float, sy: float) -> None:
        """Put the crosshair at screen point (sx, sy), clamped to it; it holds still."""
        sx = min(max(sx, self.x), self.x + W - 1)
        sy = min(max(sy, self.y), self.y + H - 1)
        self.pointer = self.to_world(sx, sy)
        self.at = ((sx - self.x) / W, (sy - self.y) / H)

    def nudge(self, dx: float, dy: float) -> None:
        """Move what it looks at by (dx, dy) of its px (parked), within reach."""
        px, py = self.pointer
        px, py = px + dx / self.scale, py + dy / self.scale
        if self.reach is not None:
            x0, y0, x1, y1 = self.reach
            px, py = min(max(px, x0), x1), min(max(py, y0), y1)
        self.pointer = (px, py)

    # ---- geometry ------------------------------------------------------------------

    def contains(self, sx: float, sy: float) -> bool:
        return (
            self.shown > 0 and self.x <= sx <= self.x + W and self.y <= sy <= self.y + H
        )

    def scrolls(self, sx: float, sy: float) -> bool:
        """Does scrolling here change its zoom (rather than the view's)?"""
        return self.open and self.ready and self.contains(sx, sy)

    def to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        cx, cy = self.center
        return (
            self.x + W / 2 + (wx - cx) * self.scale,
            self.y + H / 2 + (wy - cy) * self.scale,
        )

    def to_world(self, sx: float, sy: float) -> tuple[float, float]:
        cx, cy = self.center
        return (
            cx + (sx - self.x - W / 2) / self.scale,
            cy + (sy - self.y - H / 2) / self.scale,
        )

    def covers(self) -> Rect | None:
        """The world rect it shows, if shown."""
        if not self.shown or self.pointer is None:
            return None
        (cx, cy), hw, hh = self.center, W / 2 / self.scale, H / 2 / self.scale
        return cx - hw, cy - hh, cx + hw, cy + hh

    # ---- per frame -----------------------------------------------------------------

    def update(
        self,
        dt: float,
        camera: Camera,
        board: Rect | None,
        visible: Rect,
        cursor: tuple[float, float] | None,
        on_board: bool,
        win_w: int,
        win_h: int,
    ) -> None:
        """`board`: bounds of everything on it (None if empty). `visible`: the world rect
        the view shows (right of the picker, above the status bar). `cursor`: the mouse
        (screen px) if it's in the window. `on_board`: it's over the board (not the
        picker or the status bar)."""
        self.camera = camera
        self._follow(camera, cursor, on_board)
        self.reach = _pad(visible if board is None else _union(board, visible))
        self.floor = MIN_LEVEL
        if board is not None:  # (the board twice over: see the module docstring)
            w, h = board[2] - board[0], board[3] - board[1]
            self.floor = min(MIN_LEVEL, Camera.fit_level(2 * w, 2 * h, W, H))
        self.level = max(self.floor, self.level)

        was_shown = self.shown
        step = dt / SLIDE_S
        on = self.open and self.ready
        self.shown = min(1.0, self.shown + step) if on else max(0.0, self.shown - step)
        self.x = win_w - MARGIN - W + (1 - _ease(self.shown)) * (W + MARGIN + 2)
        self.y = win_h - MARGIN - H
        self.border.position = (self.x, self.y)
        self.frame.visible = bool(self.shown and self.ready)
        if not self.ready:
            return

        target = 2 ** (self.level / STEPS_PER_OCTAVE)
        if not was_shown:
            self.scale = target  # coming in: no easing from wherever it was
        else:  # ease, evenly in log space (zooming in and out feel the same)
            k = min(1.0, dt / EASE_S)
            self.scale *= (target / self.scale) ** k
        cross = self._place(dt, camera, was_shown)

        self.label.text = f"{100 * self.scale:.0f}%" + (
            " parked" if self.parked else ""
        )
        pad = 3 * S
        self.label.position = (self.x + W - 5 * S, self.y + 4 * S, 0)
        self.tag.position = (
            self.label.x - self.label.content_width - pad,
            self.label.y - pad / 2,
        )
        self.tag.width, self.tag.height = (
            self.label.content_width + 2 * pad,
            self.label.content_height + pad,
        )
        # what the view shows, in it; what it shows, on the board
        (ax, ay), (bx, by) = self.to_screen(*visible[:2]), self.to_screen(*visible[2:])
        mx, my = (ax + bx) / 2, (ay + by) / 2
        hw, hh = max(bx - ax, MIN_VIEW_PX) / 2, max(by - ay, MIN_VIEW_PX) / 2
        self.view_box.position = (round(mx - hw), round(my - hh))
        self.view_box.width, self.view_box.height = round(2 * hw), round(2 * hh)
        cx, cy = self.center
        hw, hh = W / 2 / self.scale, H / 2 / self.scale
        (ax, ay), (bx, by) = (
            camera.world_to_screen(cx - hw, cy - hh),
            camera.world_to_screen(cx + hw, cy + hh),
        )
        self.frame.position = (round(ax), round(ay))
        self.frame.width, self.frame.height = (
            max(2, round(bx - ax)),
            max(2, round(by - ay)),
        )
        self.frame.opacity = round(T.MINI_FRAME[3] * self.shown)
        self._place_cross(cross)
        self.hint.update(dt, self.x + HINT_AT * S, self.y + H - HINT_AT * S)

    def _follow(
        self, camera: Camera, cursor: tuple[float, float] | None, on_board: bool
    ) -> None:
        """Work out the pointer (see the module docstring)."""
        self.menu = None
        self.hovered = self.grabbed or cursor is not None and self.contains(*cursor)
        context = self.context
        if context is not None and context.visible:  # it holds still (see _push)
            if cursor is not None and (
                context.contains(*cursor) or on_board and not self.hovered
            ):
                self.menu = (cursor,)
            else:
                self.menu = (None,)
        elif cursor is None or self.hovered or self.parked or not on_board:
            pass  # stays
        else:
            self.pointer, self.at = _screen_to_world(camera, cursor), (0.5, 0.5)

    def _place(
        self, dt: float, camera: Camera, was_shown: float
    ) -> tuple[float, float]:
        """Put its middle where it looks; returns where the crosshair goes (screen px)."""
        if self.menu is not None and self.context is not None:
            ax, ay = _screen_to_world(camera, self.context.anchor)
            if self.hold is None:  # just opened: hold where the anchor is now
                self.hold = (
                    (0.0, 0.0)
                    if not was_shown
                    else (
                        (ax - self.center[0]) * self.scale,
                        (ay - self.center[1]) * self.scale,
                    )
                )
            if self.menu[0] is not None:
                self._push(dt, self.menu[0])
            self.center = (
                ax - self.hold[0] / self.scale,
                ay - self.hold[1] / self.scale,
            )
            u, v = self.at
            return self.x + u * W, self.y + v * H
        if self.hold is not None:  # just closed (the crosshair `at` was the menu's)
            self.hold, self.recentering, self.at = None, False, (0.5, 0.5)
        u, v = self.at
        px, py = self.pointer
        self.center = (
            px - (u - 0.5) * W / self.scale,
            py - (v - 0.5) * H / self.scale,
        )
        return self.to_screen(px, py)

    def _push(self, dt: float, cursor: tuple[float, float]) -> None:
        """Holding still for a menu: the crosshair goes where the cursor shows up in it.
        Once that gets past an edge, it moves: right away just enough to keep it on
        the edge, and eased the rest of the way until it's back in the middle; then it
        holds still again.
        The menu is drawn at 1x with its anchor a where the board puts that point (panel
        point L(A)), so cursor c shows up at L(A) + (c - a). Off the menu too: mapping the
        board around it at k instead would put the crosshair (c - a) * (k - 1) away the
        moment the cursor leaves the menu. (Closing the menu eases it back to the true spot.)
        """
        cx, cy = cursor
        ax, ay = self.context.anchor
        sx = self.x + W / 2 + self.hold[0] + cx - ax
        sy = self.y + H / 2 + self.hold[1] + cy - ay
        # px to move by (the crosshair moves the other way)
        dx = min(0.0, sx - self.x) + max(0.0, sx - (self.x + W))
        dy = min(0.0, sy - self.y) + max(0.0, sy - (self.y + H))
        if dx or dy:
            self.recentering = True
        if self.recentering:
            k = min(1.0, dt / EASE_S)
            ex, ey = (
                sx - dx - (self.x + W / 2),
                sy - dy - (self.y + H / 2),
            )  # (after the push)
            dx, dy = dx + ex * k, dy + ey * k
            if max(abs(ex), abs(ey)) * (1 - k) < 0.5:
                self.recentering = False
        self.hold = (self.hold[0] - dx, self.hold[1] - dy)
        self.at = ((sx - dx - self.x) / W, (sy - dy - self.y) / H)

    def _place_cross(self, at: tuple[float, float]) -> None:
        """Put the crosshair through screen point `at` (hidden where it's off the panel)."""
        h, v = self.cross
        sx, sy = round(at[0]), round(at[1])
        h.visible = self.y <= sy < self.y + H
        v.visible = self.x <= sx < self.x + W
        h.position, h.width, h.height = (self.x, sy), W, self.line
        v.position, v.width, v.height = (sx, self.y), self.line, H

    # ---- drawing -------------------------------------------------------------------

    def draw(self, window: pyglet.window.Window, world: Canvas, grid: Grid) -> None:
        """Call with the window's view at identity (screen px); leaves it that way. The
        translation is in whole px (text doesn't shimmer as it follows the cursor)."""
        self.frames.draw()
        if not self.shown or not self.ready:
            return
        s = self.scale
        cx, cy = self.center
        tx, ty = round(self.x + W / 2 - cx * s), round(self.y + H / 2 - cy * s)
        ratio = window.get_framebuffer_size()[0] / window.width
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(
            round(self.x * ratio),
            round(self.y * ratio),
            round(W * ratio),
            round(H * ratio),
        )
        grid.draw_at(
            window, (tx, ty), s, T.MINI_BG, T.GRID_COLORS
        )  # (also paints the background)
        window.view = Mat4(s, 0, 0, 0, 0, s, 0, 0, 0, 0, 1, 0, tx, ty, 0, 1)
        world.draw_instances()
        self._draw_over(window, tx, ty)
        window.view = Mat4()
        self.inside.draw()
        gl.glDisable(gl.GL_SCISSOR_TEST)
        self.front.draw()

    def _draw_over(self, window: pyglet.window.Window, tx: float, ty: float) -> None:
        """The overlay (selection box), scaled like the world: the main view puts world
        point p at p * zoom + T (screen px); this puts it at p * s + t. So a screen point
        q goes to (q - T) * k + t, k = s / zoom.
        The context menu goes at 1x instead, its anchor a where that puts it: q goes to
        (a - T) * k + t + (q - a)."""
        if self.camera is None:
            return
        k = self.scale / self.camera.zoom
        mx, my = self.camera.translation()
        if self.overlay is not None:
            window.view = Mat4(
                k, 0, 0, 0, 0, k, 0, 0, 0, 0, 1, 0, tx - mx * k, ty - my * k, 0, 1
            )
            self.overlay.draw()
        if self.context is not None and self.context.visible:
            ax, ay = self.context.anchor
            window.view = Mat4(
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                round(tx + (ax - mx) * k - ax),
                round(ty + (ay - my) * k - ay),
                0,
                1,
            )  # whole px: crisp text
            self.context.batch.draw()


class _Hint:
    """Its keys, top left: a G key cap with an arrow on each side. Each lights up
    (brighter, with a soft glow behind) while its key is held -- the arrows also for G+mouse,
    in the directions you move -- and fades back when let go."""

    CAP, GAP, ARROW = (
        17,
        25,
        7,
    )  # cap size, cap middle to arrow tip, arrow size (px, before UI_SCALE)
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))  # left, right, up, down

    def __init__(self, batch: pyglet.graphics.Batch) -> None:
        glows = pyglet.graphics.Group(order=2)
        marks = pyglet.graphics.Group(order=3)
        text = pyglet.graphics.Group(order=4)
        c, a, g = self.CAP * S, self.ARROW * S, self.GAP * S
        self.cap = shapes.BorderedRectangle(
            -c / 2, -c / 2, c, c, border=max(1, round(S)), batch=batch, group=marks
        )
        self.letter = pyglet.text.Label(
            "G",
            font_name="Consolas",
            font_size=8 * S,
            weight="bold",
            anchor_x="center",
            anchor_y="center",
            batch=batch,
            group=text,
        )
        # additive, so it brightens what's behind rather than painting a disc over it
        glow = _glow_image()
        self.glows = []
        for r in [c * 1.3] + [a * 2.0] * 4:
            sprite = pyglet.sprite.Sprite(
                glow, blend_dest=gl.GL_ONE, batch=batch, group=glows
            )
            sprite.scale = 2 * r / glow.width
            self.glows.append(sprite)
        # arrows pointing out, tip `g` from the middle (a triangle's position is its tip)
        self.arrows, self.tips = [], []
        for dx, dy in self.DIRS:
            tip, base = (dx * g, dy * g), (dx * (g - a), dy * (g - a))
            side = (-dy * a * 0.9, dx * a * 0.9)
            self.tips.append(tip)
            self.arrows.append(
                shapes.Triangle(
                    *tip,
                    base[0] + side[0],
                    base[1] + side[1],
                    base[0] - side[0],
                    base[1] - side[1],
                    batch=batch,
                    group=marks,
                )
            )
        self.centers = [(0.0, 0.0)] + [
            (dx * (g - a / 2), dy * (g - a / 2)) for dx, dy in self.DIRS
        ]
        self.want = [0.0] * 5  # G, left, right, up, down: how lit each should be (0..1)
        self.lit = [0.0] * 5  # ... and is (easing there)

    def light(self, g: bool, arrows: tuple[float, float, float, float]) -> None:
        """What to light: G (held), and each arrow (left, right, up, down) by 0..1."""
        self.want = [1.0 if g else 0.0, *arrows]

    def update(self, dt: float, x: float, y: float) -> None:
        up, down = min(1.0, dt / 0.04), min(1.0, dt / T.MINI_HINT_FADE_S)
        self.lit = [
            l + (w - l) * (up if w > l else down) for l, w in zip(self.lit, self.want)
        ]
        x, y = round(x), round(y)
        for glow, (cx, cy), l in zip(self.glows, self.centers, self.lit):
            glow.position = (x + cx, y + cy, 0)
            glow.color = (*T.MINI_HINT_GLOW[:3], round(T.MINI_HINT_GLOW[3] * l))
        for arrow, (tx, ty), l in zip(self.arrows, self.tips, self.lit[1:]):
            arrow.position = (x + tx, y + ty)
            arrow.color = _mix(T.MINI_HINT, T.MINI_HINT_LIT, l)
        g = self.lit[0]
        self.cap.position = (x - self.cap.width / 2, y - self.cap.height / 2)
        self.cap.border_color = _mix(T.MINI_HINT, T.MINI_HINT_LIT, g)
        self.cap.color = (*T.MINI_BG, 170)
        self.letter.position = (x, y + 1, 0)
        self.letter.color = _mix(T.MINI_HINT, T.MINI_HINT_LIT, g)


_glow = None


def _glow_image() -> pyglet.image.AbstractImage:
    """A white disc whose alpha falls off smoothly from the middle to nothing at the
    rim, centre-anchored; tinted and scaled per use."""
    global _glow
    if _glow is None:
        n = 64
        data = bytearray()
        for j in range(n):
            for i in range(n):
                d = min(
                    1.0,
                    ((i + 0.5 - n / 2) ** 2 + (j + 0.5 - n / 2) ** 2) ** 0.5 / (n / 2),
                )
                data += bytes((255, 255, 255, round(255 * (1 - d) ** 2.2)))
        _glow = pyglet.image.ImageData(n, n, "RGBA", bytes(data)).get_texture()
        _glow.anchor_x = _glow.anchor_y = n // 2
    return _glow


def _mix(a, b, t: float) -> tuple[int, int, int, int]:
    return tuple(round(p + (q - p) * t) for p, q in zip(a, b))


def _screen_to_world(camera: Camera, p: tuple[float, float]) -> tuple[float, float]:
    """Like camera.screen_to_world, but with the whole-pixel translation the world is
    actually drawn with, so things line up exactly when magnified."""
    tx, ty = camera.translation()
    return (p[0] - tx) / camera.zoom, (p[1] - ty) / camera.zoom


def _union(a: Rect, b: Rect) -> Rect:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _pad(r: Rect) -> Rect:
    x0, y0, x1, y1 = r
    p = PAD * max(x1 - x0, y1 - y0, 1.0)
    return x0 - p, y0 - p, x1 + p, y1 + p
