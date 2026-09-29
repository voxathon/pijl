"""World <-> screen coordinate conversion.

Everything in the circuit lives in *world* coordinates. The camera decides
which part of the world is visible and how big it is:

    screen = (world - camera_pos) * zoom
    world  = screen / zoom + camera_pos

On the GPU side this is a single view matrix, so panning/zooming costs
nothing per object.
"""

from pyglet.math import Mat4

MIN_ZOOM, MAX_ZOOM = 0.1, 8.0


class Camera:
    def __init__(self) -> None:
        self.x = 0.0  # world coordinate shown at the bottom-left of the window
        self.y = 0.0
        self.zoom = 1.0

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return sx / self.zoom + self.x, sy / self.zoom + self.y

    def world_to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        return (wx - self.x) * self.zoom, (wy - self.y) * self.zoom

    def pan(self, screen_dx: float, screen_dy: float) -> None:
        """Drag the world by a screen-space mouse delta."""
        self.x -= screen_dx / self.zoom
        self.y -= screen_dy / self.zoom

    def zoom_at(self, sx: float, sy: float, factor: float) -> None:
        """Zoom while keeping the world point under the cursor fixed."""
        wx, wy = self.screen_to_world(sx, sy)
        self.zoom = min(MAX_ZOOM, max(MIN_ZOOM, self.zoom * factor))
        self.x = wx - sx / self.zoom
        self.y = wy - sy / self.zoom

    def center_on(self, wx: float, wy: float, width: int, height: int) -> None:
        self.x = wx - width / 2 / self.zoom
        self.y = wy - height / 2 / self.zoom

    def matrix(self) -> Mat4:
        z = self.zoom
        # Column-major: scale on the diagonal, translation in the last column.
        return Mat4(z, 0, 0, 0,
                    0, z, 0, 0,
                    0, 0, 1, 0,
                    -self.x * z, -self.y * z, 0, 1)
