"""Infinite background grid, drawn by one full-screen shader.

No line objects exist: for every screen pixel the fragment shader converts
back to world space and asks "how far am I from the nearest grid line, in
screen pixels?". That makes the grid infinite, perfectly aligned with the
camera (it uses the exact same rounded translation), anti-aliased, and free
regardless of zoom. Line levels fade in and out with zoom so it never gets dense.
"""

from __future__ import annotations

import pyglet
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram

from . import theme as T
from .camera import Camera

_VERTEX = """#version 150 core
in vec2 position;
void main() { gl_Position = vec4(position, 0.0, 1.0); }
"""

_FRAGMENT = """#version 150 core
out vec4 final_color;
uniform vec2 translate;     // camera translation in screen px (same as the view matrix)
uniform float zoom;
uniform float px_ratio;     // framebuffer px per window px (HiDPI)
uniform float spacing;      // world units between minor lines
uniform float major_every;
uniform vec3 background;
uniform vec3 minor_color;
uniform vec3 major_color;

// Coverage of a 1px line through every multiple of `step`, at this pixel.
float lines(vec2 screen, float step_px) {
    vec2 d = abs(fract(screen / step_px + 0.5) - 0.5) * step_px;  // px to nearest line, per axis
    return clamp(1.0 - min(d.x, d.y), 0.0, 1.0);
}

void main() {
    // -0.5: pixel centers sit at .5, so this lands lines on whole pixels (crisp at zoom 1)
    vec2 screen = gl_FragCoord.xy / px_ratio - 0.5 - translate;
    // Level 0 = snap grid (minor). Levels 1+ = major lines, each major_every x coarser
    // than the last, so zooming far out shows sparse coarse lines instead of mush.
    // Each level fades in only while its lines are comfortably spaced on screen.
    float step_px = spacing * zoom;
    float minor = lines(screen, step_px) * smoothstep(8.0, 20.0, step_px);
    float major = 0.0;
    for (int level = 1; level <= 5; level++) {
        step_px *= major_every;
        major = max(major, lines(screen, step_px) * smoothstep(10.0, 28.0, step_px));
    }
    vec3 c = mix(background, minor_color, minor);
    c = mix(c, major_color, major);
    final_color = vec4(c, 1.0);
}
"""


class Grid:
    def __init__(self) -> None:
        self.program = ShaderProgram(Shader(_VERTEX, "vertex"), Shader(_FRAGMENT, "fragment"))
        self.quad = self.program.vertex_list(
            6, gl.GL_TRIANGLES,
            position=("f", (-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1)))

    def draw(self, window: pyglet.window.Window, camera: Camera, emphasized: bool,
             divisions: int = 1) -> None:
        """`divisions` > 1 shows the finer subgrid (Ctrl+Shift); major lines stay put."""
        self.draw_at(window, camera.translation(), camera.zoom, T.BACKGROUND,
                     T.GRID_SNAPPING if emphasized else T.GRID_COLORS, divisions)

    def draw_at(self, window: pyglet.window.Window, translate: tuple[float, float], zoom: float,
                background, colors, divisions: int = 1) -> None:
        """The grid for any view: `translate` is where the world's origin is on screen (px).
        Fills the whole window; clip it with glScissor (the minimap panels do)."""
        p = self.program
        p.use()
        p["translate"] = translate
        p["zoom"] = zoom
        p["px_ratio"] = window.get_framebuffer_size()[0] / window.width
        p["spacing"] = T.GRID / divisions
        p["major_every"] = float(T.GRID_MAJOR_EVERY * divisions)
        p["background"] = _rgb(background)
        minor, major = colors
        p["minor_color"] = _rgb(minor)
        p["major_color"] = _rgb(major)
        self.quad.draw(gl.GL_TRIANGLES)
        p.stop()


def _rgb(color) -> tuple[float, float, float]:
    return color[0] / 255, color[1] / 255, color[2] / 255
