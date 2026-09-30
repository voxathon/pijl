"""Round shapes drawn as one quad each, made round by the fragment shader.

  Dot      pins and junctions: a filled circle.
  Segment  one straight piece of a wire, optionally with round caps at its ends.
           Caps go where a wire bends, so corners come out round without a
           separate joint shape (see views.Polyline).

pyglet's Circle is a fan of triangles: to stay smooth at 8x zoom a pin needed
48 segments, i.e. 144 vertices -- most of the vertices of a whole part, and
every color change rewrote all of them from Python. Here every shape is 4
vertices whatever the zoom: the shader measures each pixel's distance to the
shape and anti-aliases the edge to one screen pixel (like sdf_text.py).

A wire's segments all use one program, so they share one group per layer and
draw in buffer order: each wire as a unit, the way pyglet shapes did. (Round
joints in a program of their own would draw above or below *every* wire line.)
"""

from __future__ import annotations

import math

import pyglet
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram

Point = tuple[float, float]

# Quads reach this far past the shape's edge (in its radius), so the outer half of
# the one-pixel anti-aliasing ramp has somewhere to go. Only while a pixel is
# under PAD - 1 radii: zoomed far out, edges just get a little harder.
PAD = 1.5

_UNIFORMS = "uniform WindowBlock { mat4 projection; mat4 view; } window;"

# The edge ramp: 1 inside, 0 outside, one screen pixel wide, centered on d = 1.
_COVERAGE = """
float coverage(float d) {
    float w = fwidth(d);
    return 1.0 - smoothstep(1.0 - 0.5 * w, 1.0 + 0.5 * w, d);
}
"""

_DOT_VERTEX = f"""#version 150 core
in vec2 position;
in vec2 offset;   // quad corner: -PAD..PAD on both axes; the rim is at length 1
in vec4 colors;
out vec2 local;
out vec4 color;
{_UNIFORMS}
void main() {{
    gl_Position = window.projection * window.view * vec4(position, 0.0, 1.0);
    local = offset;
    color = colors;
}}
"""

_DOT_FRAGMENT = f"""#version 150 core
in vec2 local;
in vec4 color;
out vec4 final_color;
{_COVERAGE}
void main() {{
    float a = coverage(length(local));
    if (a <= 0.0) discard;
    final_color = vec4(color.rgb, color.a * a);
}}
"""

_SEGMENT_VERTEX = f"""#version 150 core
in vec2 position;
in vec2 local;     // world units: along the segment from its start, and across it
in vec2 extent;    // (length, radius)
in vec4 color_a;   // at the start ...
in vec4 color_b;   // ... and at the end; blended along the length
out vec2 uv;
flat out vec2 ext;
flat out vec4 ca;
flat out vec4 cb;
{_UNIFORMS}
void main() {{
    gl_Position = window.projection * window.view * vec4(position, 0.0, 1.0);
    uv = local;
    ext = extent;
    ca = color_a;
    cb = color_b;
}}
"""

# A capsule: distance to the segment's core line, in radii. Past an end the quad only
# reaches out if that end has a cap, so an uncapped end stays square.
_SEGMENT_FRAGMENT = f"""#version 150 core
in vec2 uv;
flat in vec2 ext;
flat in vec4 ca;
flat in vec4 cb;
out vec4 final_color;
{_COVERAGE}
void main() {{
    float len = ext.x;
    vec2 p = vec2(uv.x - clamp(uv.x, 0.0, len), uv.y);
    float a = coverage(length(p) / ext.y);
    if (a <= 0.0) discard;
    vec4 c = mix(ca, cb, len > 0.0 ? clamp(uv.x / len, 0.0, 1.0) : 0.0);
    final_color = vec4(c.rgb, c.a * a);
}}
"""

_programs: dict[str, ShaderProgram] = {}
_groups: dict[tuple[str, pyglet.graphics.Group | None], _SDFGroup] = {}


def _get_program(name: str) -> ShaderProgram:
    """Built lazily: needs a GL context, i.e. a window must exist."""
    program = _programs.get(name)
    if program is None:
        vertex, fragment = {"dot": (_DOT_VERTEX, _DOT_FRAGMENT),
                            "segment": (_SEGMENT_VERTEX, _SEGMENT_FRAGMENT)}[name]
        program = _programs[name] = ShaderProgram(Shader(vertex, "vertex"), Shader(fragment, "fragment"))
    return program


class _SDFGroup(pyglet.graphics.Group):
    def __init__(self, program: ShaderProgram, parent: pyglet.graphics.Group | None) -> None:
        super().__init__(parent=parent)
        self.program = program

    def set_state(self) -> None:
        self.program.use()
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def unset_state(self) -> None:
        gl.glDisable(gl.GL_BLEND)
        self.program.stop()

    def __eq__(self, other) -> bool:
        return (self.__class__ is other.__class__ and self.program is other.program
                and self.parent == other.parent)

    def __hash__(self) -> int:
        return hash((id(self.program), self.parent))


def _vertex_list(name: str, batch: pyglet.graphics.Batch, parent: pyglet.graphics.Group | None, **data):
    """One quad; all shapes of a kind in a layer share one group, i.e. one draw call."""
    group = _groups.get((name, parent))
    if group is None:
        group = _groups[name, parent] = _SDFGroup(_get_program(name), parent)
    return group.program.vertex_list_indexed(4, gl.GL_TRIANGLES, [0, 1, 2, 0, 2, 3], batch, group, **data)


def _rgba(color: tuple, alpha: int) -> tuple[int, int, int, int]:
    """3-component colors keep the current opacity, like pyglet shapes."""
    return (*color[:3], color[3] if len(color) > 3 else alpha)


_DOT_OFFSETS = (-PAD, -PAD, PAD, -PAD, PAD, PAD, -PAD, PAD)


class Dot:
    """A filled circle. Same interface as the parts of shapes.Circle we use:
    position, radius, color, opacity, delete()."""

    def __init__(self, x: float, y: float, radius: float, color: tuple, batch: pyglet.graphics.Batch,
                 group: pyglet.graphics.Group | None = None) -> None:
        self._x, self._y, self._radius = x, y, radius
        self._rgba = _rgba(color, 255)
        self._vlist = _vertex_list("dot", batch, group, position=("f", self._corners()),
                                   offset=("f", _DOT_OFFSETS), colors=("Bn", self._rgba * 4))

    def _corners(self) -> tuple[float, ...]:
        x, y, r = self._x, self._y, self._radius * PAD
        return x - r, y - r, x + r, y - r, x + r, y + r, x - r, y + r

    @property
    def position(self) -> Point:
        return self._x, self._y

    @position.setter
    def position(self, xy: Point) -> None:
        if xy == (self._x, self._y):
            return
        self._x, self._y = xy
        self._vlist.position[:] = self._corners()

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        self._radius = value
        self._vlist.position[:] = self._corners()

    @property
    def color(self) -> tuple:
        return self._rgba

    @color.setter
    def color(self, value: tuple) -> None:
        rgba = _rgba(value, self._rgba[3])
        if rgba != self._rgba:
            self._rgba = rgba
            self._vlist.colors[:] = rgba * 4

    @property
    def opacity(self) -> int:
        return self._rgba[3]

    @opacity.setter
    def opacity(self, value: int) -> None:
        self.color = (*self._rgba[:3], value)

    def delete(self) -> None:
        if self._vlist is not None:
            self._vlist.delete()
            self._vlist = None


class Segment:
    """A straight thick line from a to b, blending from one color to another,
    with an optional round cap on either end."""

    def __init__(self, thickness: float, color: tuple, batch: pyglet.graphics.Batch,
                 group: pyglet.graphics.Group | None = None) -> None:
        self._r = thickness / 2
        self._a = self._b = (0.0, 0.0)
        self._caps = (False, False)
        self._ca = self._cb = _rgba(color, 255)
        self._vlist = _vertex_list("segment", batch, group, position=("f", (0.0,) * 8), local=("f", (0.0,) * 8),
                                   extent=("f", (0.0,) * 8), color_a=("Bn", self._ca * 4),
                                   color_b=("Bn", self._cb * 4))

    def place(self, a: Point, b: Point, cap_a: bool = False, cap_b: bool = False) -> None:
        if (a, b, (cap_a, cap_b)) == (self._a, self._b, self._caps):
            return
        self._a, self._b, self._caps = a, b, (cap_a, cap_b)
        (x0, y0), (x1, y1) = a, b
        length = math.hypot(x1 - x0, y1 - y0)
        dx, dy = ((x1 - x0) / length, (y1 - y0) / length) if length else (1.0, 0.0)
        r = self._r
        w = r * PAD                          # half width, anti-aliasing room included
        e0, e1 = (w if cap_a else 0.0), (w if cap_b else 0.0)
        nx, ny = -dy * w, dx * w             # across, to the left
        sx, sy = x0 - dx * e0, y0 - dy * e0  # the quad's ends, past a cap
        tx, ty = x1 + dx * e1, y1 + dy * e1
        self._vlist.position[:] = (sx - nx, sy - ny, tx - nx, ty - ny, tx + nx, ty + ny, sx + nx, sy + ny)
        u0, u1 = -e0, length + e1
        self._vlist.local[:] = (u0, -w, u1, -w, u1, w, u0, w)
        self._vlist.extent[:] = (length, r) * 4

    def set_colors(self, start: tuple, end: tuple) -> None:
        """Colors at a and b; 3-component colors keep the current opacity."""
        ca, cb = _rgba(start, self._ca[3]), _rgba(end, self._cb[3])
        if ca != self._ca:
            self._ca = ca
            self._vlist.color_a[:] = ca * 4
        if cb != self._cb:
            self._cb = cb
            self._vlist.color_b[:] = cb * 4

    @property
    def opacity(self) -> int:
        return self._ca[3]

    @opacity.setter
    def opacity(self, value: int) -> None:
        self.set_colors((*self._ca[:3], value), (*self._cb[:3], value))

    def delete(self) -> None:
        if self._vlist is not None:
            self._vlist.delete()
            self._vlist = None
