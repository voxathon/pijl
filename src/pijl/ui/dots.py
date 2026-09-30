"""Round dots (pins, junctions, wire joints) as one quad each, made round by the shader.

pyglet's Circle is a fan of triangles: to stay smooth at 8x zoom a pin needed
48 segments, i.e. 144 vertices -- most of the vertices of a whole part, and
every color change rewrote all of them from Python. Here a dot is 4 vertices
whatever the zoom: the fragment shader measures each pixel's distance from the
center and anti-aliases the edge to one screen pixel (like sdf_text.py).
"""

from __future__ import annotations

import pyglet
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram

Point = tuple[float, float]

_VERTEX = """#version 150 core
in vec2 position;
in vec2 offset;   // quad corner: -1..1 on both axes
in vec4 colors;
out vec2 local;
out vec4 color;
uniform WindowBlock { mat4 projection; mat4 view; } window;
void main() {
    gl_Position = window.projection * window.view * vec4(position, 0.0, 1.0);
    local = offset;
    color = colors;
}
"""

_FRAGMENT = """#version 150 core
in vec2 local;
in vec4 color;
out vec4 final_color;
void main() {
    float d = length(local);   // 1 = on the rim
    float w = fwidth(d);       // how much d changes over one screen pixel
    float a = 1.0 - smoothstep(1.0 - w, 1.0, d);
    if (a <= 0.0) discard;
    final_color = vec4(color.rgb, color.a * a);
}
"""

_OFFSETS = (-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0)

_program: ShaderProgram | None = None
_groups: dict[pyglet.graphics.Group | None, _DotGroup] = {}


def _get_program() -> ShaderProgram:
    """Built lazily: needs a GL context, i.e. a window must exist."""
    global _program
    if _program is None:
        _program = ShaderProgram(Shader(_VERTEX, "vertex"), Shader(_FRAGMENT, "fragment"))
    return _program


class _DotGroup(pyglet.graphics.Group):
    def __init__(self, program: ShaderProgram, parent: pyglet.graphics.Group | None) -> None:
        super().__init__(order=1, parent=parent)  # over the layer's other shapes (wire joints over segments)
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


def _group_for(parent: pyglet.graphics.Group | None) -> _DotGroup:
    """One shared group per layer, so all dots in it are a single draw call."""
    group = _groups.get(parent)
    if group is None:
        group = _groups[parent] = _DotGroup(_get_program(), parent)
    return group


class Dot:
    """A filled circle. Same interface as the parts of shapes.Circle we use:
    position, color (3 components keep the opacity), opacity, delete()."""

    def __init__(self, x: float, y: float, radius: float, color: tuple, batch: pyglet.graphics.Batch,
                 group: pyglet.graphics.Group | None = None) -> None:
        self._x, self._y, self._radius = x, y, radius
        self._rgba = (*color[:3], color[3] if len(color) > 3 else 255)
        self._vlist = _get_program().vertex_list_indexed(
            4, gl.GL_TRIANGLES, [0, 1, 2, 0, 2, 3], batch, _group_for(group),
            position=("f", self._corners()),
            offset=("f", _OFFSETS),
            colors=("Bn", self._rgba * 4))

    def _corners(self) -> tuple[float, ...]:
        x, y, r = self._x, self._y, self._radius
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
        rgba = (*value[:3], value[3] if len(value) > 3 else self._rgba[3])
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
