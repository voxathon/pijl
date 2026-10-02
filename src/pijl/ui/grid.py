"""Infinite background grid, drawn by one full-screen shader.

No line objects exist: for every screen pixel the fragment shader converts
back to world space and asks "how far am I from the nearest grid line, in
screen pixels?". That makes the grid infinite, perfectly aligned with the
camera (it uses the exact same rounded translation), anti-aliased, and free
regardless of zoom. Line levels fade in and out with zoom so it never gets dense.

Snapping follows the levels: `snap_step` is the finest level that's at least
GRID_SNAP_MIN_PX apart on screen, so zoomed out, Ctrl snaps to the coarser lines
you see (Ctrl+Shift: one level finer). While snapping, the shader draws exactly
that level as the minor lines.
"""

from __future__ import annotations

import pyglet
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram

from . import theme as T
from .camera import Camera

_VERTEX = """#version 150 core
in vec2 position;
out vec2 ndc;
void main() { ndc = position; gl_Position = vec4(position, 0.0, 1.0); }
"""

_FRAGMENT = """#version 150 core
in vec2 ndc;  // not gl_FragCoord: its y origin came out flipped on Linux + NVIDIA
out vec4 final_color;
uniform vec2 translate;     // camera translation in screen px (same as the view matrix)
uniform float zoom;
uniform vec2 size;          // window size in screen px
uniform float spacing;      // world units between level-0 lines (GRID)
uniform float major_every;
uniform float snap_step;    // > 0 while snapping: the level things snap to (see snap_step())
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
    vec2 screen = (ndc * 0.5 + 0.5) * size - 0.5 - translate;
    // Level 0 = the pin grid (minor). Levels 1+ = major lines, each major_every x coarser
    // than the last, so zooming far out shows sparse coarse lines instead of mush.
    // Each major level fades in only while its lines are comfortably spaced on screen.
    float minor;
    if (snap_step > 0.0) {
        minor = lines(screen, snap_step * zoom);  // exactly what Ctrl snaps to, full strength
    } else {
        float px = spacing * zoom;
        minor = lines(screen, px) * smoothstep(8.0, 20.0, px);
    }
    float major = 0.0;
    float step = spacing;
    for (int level = 1; level <= 8; level++) {
        step *= major_every;
        if (step <= snap_step * 1.001) continue;  // that's the minor level now (or finer)
        major = max(major, lines(screen, step * zoom) * smoothstep(10.0, 28.0, step * zoom));
    }
    vec3 c = mix(background, minor_color, minor);
    c = mix(c, major_color, major);
    final_color = vec4(c, 1.0);
}
"""


def snap_step(zoom: float, divisions: int = 1) -> float:
    """World units between the points Ctrl snaps to at this zoom: GRID if that's at
    least GRID_SNAP_MIN_PX on screen, else the first major level that is. Ctrl+Shift
    (`divisions` > 1) is one level finer: the subgrid at GRID, else the level below.
    Without the subgrid it's always a multiple of GRID, so pins land on grid points."""
    step = float(T.GRID)
    for _ in range(8):  # (as many levels as the shader draws)
        if step * zoom >= T.GRID_SNAP_MIN_PX:
            break
        step *= T.GRID_MAJOR_EVERY
    if divisions > 1:
        return step / (divisions if step == T.GRID else T.GRID_MAJOR_EVERY)
    return step


class Grid:
    def __init__(self) -> None:
        self.program = ShaderProgram(
            Shader(_VERTEX, "vertex"), Shader(_FRAGMENT, "fragment")
        )
        self.quad = self.program.vertex_list(
            6,
            gl.GL_TRIANGLES,
            position=("f", (-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1)),
        )

    def draw(
        self,
        window: pyglet.window.Window,
        camera: Camera,
        snapping: float = 0.0,
        inside: bool = False,
    ) -> None:
        """`snapping`: the snap step (see snap_step) while Ctrl is held, else 0; it's
        drawn as the minor lines, major lines stay put.
        `inside`: viewing a macro's insides (see inside.py), in their own colors."""
        p = self.program
        p.use()
        p["translate"] = camera.translation()
        p["zoom"] = camera.zoom
        p["size"] = (float(window.width), float(window.height))
        p["spacing"] = float(T.GRID)
        p["major_every"] = float(T.GRID_MAJOR_EVERY)
        p["snap_step"] = float(snapping)
        p["background"] = _rgb(T.INSIDE_BACKGROUND if inside else T.BACKGROUND)
        minor, major = (
            T.GRID_SNAPPING if snapping else T.GRID_INSIDE if inside else T.GRID_COLORS
        )
        p["minor_color"] = _rgb(minor)
        p["major_color"] = _rgb(major)
        self.quad.draw(gl.GL_TRIANGLES)
        p.stop()


def _rgb(color) -> tuple[float, float, float]:
    return color[0] / 255, color[1] / 255, color[2] / 255
