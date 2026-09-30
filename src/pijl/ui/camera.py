"""World <-> screen coordinate conversion.

Everything in the circuit lives in *world* coordinates. The camera decides
which part of the world is visible and how big it is:

    screen = (world - camera_pos) * zoom
    world  = screen / zoom + camera_pos

On the GPU side this is a single view matrix, so panning/zooming costs
nothing per object.

Zoom moves in discrete steps (2^(1/8) per scroll notch), so zoom levels are
reproducible and a trackpad's tiny fractional scrolls add up to whole notches.

How far out you can zoom depends on the board: `min_level` is kept (by the editor)
at whatever fits all of it, but never closer than MIN_LEVEL.
"""

import math

from pyglet.math import Mat4

STEPS_PER_OCTAVE = 8           # scroll notches to double the zoom
MIN_LEVEL, MAX_LEVEL = -26, 24  # ~0.1x .. 8x; a big board lowers the minimum (min_level)


class Camera:
    def __init__(self) -> None:
        self.x = 0.0  # world coordinate shown at the bottom-left of the window
        self.y = 0.0
        self.level = 0
        self.min_level = MIN_LEVEL  # how far out zooming may go (see fit_level)
        self._scroll_accum = 0.0  # trackpads send fractional scroll amounts

    @property
    def zoom(self) -> float:
        return 2 ** (self.level / STEPS_PER_OCTAVE)

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return sx / self.zoom + self.x, sy / self.zoom + self.y

    def world_to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        return (wx - self.x) * self.zoom, (wy - self.y) * self.zoom

    def pan(self, screen_dx: float, screen_dy: float) -> None:
        """Drag the world by a screen-space mouse delta."""
        self.x -= screen_dx / self.zoom
        self.y -= screen_dy / self.zoom

    def scroll(self, sx: float, sy: float, amount: float) -> bool:
        """Zoom by scroll-wheel notches around the cursor. Returns True if zoom changed."""
        self._scroll_accum += amount
        steps = int(self._scroll_accum)  # truncates toward zero, keeps the fractional rest
        self._scroll_accum -= steps
        if not steps:
            return False
        return self.set_level(self.level + steps, sx, sy)

    def set_level(self, level: int, sx: float, sy: float) -> bool:
        """Change zoom while keeping the world point under (sx, sy) fixed."""
        # (never pushed in: if the board shrank below where you are, you just can't go further out)
        level = min(MAX_LEVEL, max(min(self.min_level, self.level), level))
        if level == self.level:
            return False
        wx, wy = self.screen_to_world(sx, sy)
        self.level = level
        self.x = wx - sx / self.zoom
        self.y = wy - sy / self.zoom
        return True

    @staticmethod
    def fit_level(w: float, h: float, avail_w: float, avail_h: float) -> int:
        """The closest level at which a w x h world rect fits in avail_w x avail_h px."""
        fit = min(max(1.0, avail_w) / max(w, 1e-9), max(1.0, avail_h) / max(h, 1e-9))
        return math.floor(STEPS_PER_OCTAVE * math.log2(fit))

    def center_on(self, wx: float, wy: float, width: int, height: int) -> None:
        self.x = wx - width / 2 / self.zoom
        self.y = wy - height / 2 / self.zoom

    def translation(self) -> tuple[int, int]:
        """Screen-space translation, in whole pixels: keeps text and 1px details
        from shimmering while panning. Shared with the grid shader so they agree."""
        return round(-self.x * self.zoom), round(-self.y * self.zoom)

    def matrix(self) -> Mat4:
        z = self.zoom
        tx, ty = self.translation()
        # Column-major: scale on the diagonal, translation in the last column.
        return Mat4(z, 0, 0, 0,
                    0, z, 0, 0,
                    0, 0, 1, 0,
                    tx, ty, 0, 1)
