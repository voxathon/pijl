"""World shapes as canvas instances (see canvas.py), shaped by their fragment shaders.

  Rect     part bodies (fill + border), selection outlines, pin tag backings.
  Dot      pins: a filled circle.
  Segment  one straight piece of a wire, optionally with round caps at its ends.
           Caps go where a wire bends, so corners come out round without a
           separate joint shape (see views.Polyline).
  WireDot  junctions: a zero-length Segment, so it draws with its wire (a wire
           crossing over the junction covers the dot too).

Every shape is one quad whatever the zoom; round edges are measured per pixel
and anti-aliased to one screen pixel (like sdf_text.py). pyglet's Circle needed
144 vertices for a pin to stay smooth at 8x zoom.

Colors: each shape has an off and an on color and a `state` flag choosing
between them (so a pin lighting up is a one-byte write), and an `opacity`
(multiplied in; ghosts). `color = c` sets both to c.
"""

from __future__ import annotations

import numpy as np

from .canvas import UNIFORMS, Canvas, Kind

Point = tuple[float, float]

# Round shapes' quads reach this far past the edge (in radii), so the outer half of
# the one-pixel anti-aliasing ramp has somewhere to go. Only while a pixel is
# under PAD - 1 radii: zoomed far out, edges just get a little harder.
PAD = 1.5

# The edge ramp: 1 inside, 0 outside, one screen pixel wide, centered on d = 1.
_COVERAGE = """
float coverage(float d) {
    float w = fwidth(d);
    return 1.0 - smoothstep(1.0 - 0.5 * w, 1.0 + 0.5 * w, d);
}
"""

# flags: x = state (on), y = opacity, z / w = per-kind extras. u8, normalized.
_CORNER = "vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));  // triangle strip: 0,0 1,0 0,1 1,1"


def _rgba(color) -> tuple[int, int, int, int]:
    return (*color[:3], color[3] if len(color) > 3 else 255)


# ---- Rect ---------------------------------------------------------------------------

RECT = Kind("rect", 0, f"""#version 150 core
in vec4 rect;     // x, y, width, height
in float border;  // inside the rectangle
in vec4 fill; in vec4 fill_on; in vec4 edge; in vec4 edge_on;
in vec4 flags;
in float lift;
out vec2 local;
flat out vec2 size;
flat out float bw;
flat out vec4 cf;
flat out vec4 ce;
{UNIFORMS}
void main() {{
    {_CORNER}
    local = corner * rect.zw;
    size = rect.zw;
    bw = border;
    bool on = flags.x > 0.5;
    cf = on ? fill_on : fill;
    ce = on ? edge_on : edge;
    cf.a *= flags.y;
    ce.a *= flags.y;
    gl_Position = window.projection * window.view * vec4(rect.xy + lift * lift_offset + local, 0.0, 1.0);
}}
""", """#version 150 core
in vec2 local;
flat in vec2 size;
flat in float bw;
flat in vec4 cf;
flat in vec4 ce;
out vec4 final_color;
void main() {
    float d = min(min(local.x, size.x - local.x), min(local.y, size.y - local.y));  // to the outside
    float w = fwidth(d);
    float t = bw > 0.0 ? 1.0 - smoothstep(bw - 0.5 * w, bw + 0.5 * w, d) : 0.0;
    vec4 c = mix(cf, ce, t);
    if (c.a <= 0.0) discard;
    final_color = c;
}
""", np.dtype([("rect", "f4", 4), ("border", "f4"), ("fill", "u1", 4), ("fill_on", "u1", 4),
               ("edge", "u1", 4), ("edge_on", "u1", 4), ("flags", "u1", 4), ("lift", "f4")]))

# ---- Dot ----------------------------------------------------------------------------

DOT = Kind("dot", 1, f"""#version 150 core
in vec2 center;
in float radius;
in vec4 color; in vec4 color_on;
in vec4 flags;
in float lift;
out vec2 local;   // the rim is at length 1
flat out vec4 c;
{UNIFORMS}
void main() {{
    {_CORNER}
    local = (corner * 2.0 - 1.0) * {PAD};
    c = flags.x > 0.5 ? color_on : color;
    c.a *= flags.y;
    gl_Position = window.projection * window.view * vec4(center + lift * lift_offset + local * radius, 0.0, 1.0);
}}
""", f"""#version 150 core
in vec2 local;
flat in vec4 c;
out vec4 final_color;
{_COVERAGE}
void main() {{
    float a = coverage(length(local));
    if (a <= 0.0) discard;
    final_color = vec4(c.rgb, c.a * a);
}}
""", np.dtype([("center", "f4", 2), ("radius", "f4"), ("color", "u1", 4), ("color_on", "u1", 4),
               ("flags", "u1", 4), ("lift", "f4")]))

# ---- Segment ------------------------------------------------------------------------

SEGMENT = Kind("segment", 2, f"""#version 150 core
in vec2 a;
in vec2 b;
in float radius;
in vec4 ca; in vec4 ca_on;  // color at a (off / on) ...
in vec4 cb; in vec4 cb_on;  // ... and at b; blended along the length
in vec4 flags;              // z / w: round cap at a / b
in float lift;
out vec2 uv;                // world units: along the segment from a, and across it
flat out vec2 ext;          // (length, radius)
flat out vec4 c0;
flat out vec4 c1;
{UNIFORMS}
void main() {{
    {_CORNER}
    vec2 d = b - a;
    float len = length(d);
    vec2 dir = len > 0.0 ? d / len : vec2(1.0, 0.0);
    vec2 n = vec2(-dir.y, dir.x);
    float w = radius * {PAD};  // half width, anti-aliasing room included
    // Past an end the quad only reaches out if that end has a cap, so an uncapped end stays square.
    float u = mix(flags.z > 0.5 ? -w : 0.0, len + (flags.w > 0.5 ? w : 0.0), corner.x);
    float v = mix(-w, w, corner.y);
    uv = vec2(u, v);
    ext = vec2(len, radius);
    bool on = flags.x > 0.5;
    c0 = on ? ca_on : ca;
    c1 = on ? cb_on : cb;
    c0.a *= flags.y;
    c1.a *= flags.y;
    gl_Position = window.projection * window.view * vec4(a + lift * lift_offset + dir * u + n * v, 0.0, 1.0);
}}
""", f"""#version 150 core
in vec2 uv;
flat in vec2 ext;
flat in vec4 c0;
flat in vec4 c1;
out vec4 final_color;
{_COVERAGE}
void main() {{
    float len = ext.x;
    vec2 p = vec2(uv.x - clamp(uv.x, 0.0, len), uv.y);  // to the nearest point of the core line
    float a = coverage(length(p) / ext.y);
    if (a <= 0.0) discard;
    vec4 c = mix(c0, c1, len > 0.0 ? clamp(uv.x / len, 0.0, 1.0) : 0.0);
    final_color = vec4(c.rgb, c.a * a);
}}
""", np.dtype([("a", "f4", 2), ("b", "f4", 2), ("radius", "f4"), ("ca", "u1", 4), ("ca_on", "u1", 4),
               ("cb", "u1", 4), ("cb_on", "u1", 4), ("flags", "u1", 4), ("lift", "f4")]))


class _Shape:
    """One instance: a slot in its kind's buffer for that layer."""

    kind: Kind

    def __init__(self, canvas: Canvas, layer) -> None:
        self.buf = canvas.buffer(self.kind, layer)
        self.slot: int | None = self.buf.alloc()
        self._set("flags", (0, 255, 0, 0))

    def _set(self, field: str, value) -> None:
        self.buf.f[field][self.slot] = value
        self.buf.mark(self.slot)

    def _get(self, field: str):
        return self.buf.f[field][self.slot]

    @property
    def state(self) -> bool:
        return bool(self._get("flags")[0])

    @state.setter
    def state(self, on: bool) -> None:
        flags = self.buf.f["flags"]
        v = 255 if on else 0
        if flags[self.slot, 0] != v:
            flags[self.slot, 0] = v
            self.buf.mark(self.slot)

    @property
    def lifted(self) -> bool:
        return bool(self._get("lift"))

    @lifted.setter
    def lifted(self, on: bool) -> None:
        """Drawn shifted by the canvas's offset (see canvas.py)."""
        self._set("lift", 1.0 if on else 0.0)

    @property
    def opacity(self) -> int:
        return int(self._get("flags")[1])

    @opacity.setter
    def opacity(self, value: int) -> None:
        self.buf.f["flags"][self.slot, 1] = value
        self.buf.mark(self.slot)

    def delete(self) -> None:
        if self.slot is not None:
            self.buf.free(self.slot)
            self.slot = None


class Rect(_Shape):
    """A rectangle with a border inside its edge (0: none). Same interface as
    views.Box where used: position, color / border_color, opacity, delete()."""

    kind = RECT

    def __init__(self, x: float, y: float, w: float, h: float, border: float, fill, border_color,
                 canvas: Canvas, layer) -> None:
        super().__init__(canvas, layer)
        self._set("rect", (x, y, w, h))
        self._set("border", border)
        self.set_colors(fill, border_color)

    def set_colors(self, fill, border, fill_on=None, border_on=None) -> None:
        """Off colors, and (optionally different) on colors."""
        f = self.buf.f
        f["fill"][self.slot] = _rgba(fill)
        f["edge"][self.slot] = _rgba(border)
        f["fill_on"][self.slot] = _rgba(fill if fill_on is None else fill_on)
        f["edge_on"][self.slot] = _rgba(border if border_on is None else border_on)
        self.buf.mark(self.slot)

    @property
    def position(self) -> Point:
        x, y, _, _ = self._get("rect")
        return float(x), float(y)

    @position.setter
    def position(self, xy: Point) -> None:
        self.buf.f["rect"][self.slot, :2] = xy
        self.buf.mark(self.slot)

    @property
    def x(self) -> float:
        return self.position[0]

    @property
    def y(self) -> float:
        return self.position[1]

    @property
    def width(self) -> float:
        return float(self._get("rect")[2])

    @width.setter
    def width(self, value: float) -> None:
        self.buf.f["rect"][self.slot, 2] = value
        self.buf.mark(self.slot)

    @property
    def height(self) -> float:
        return float(self._get("rect")[3])

    @height.setter
    def height(self, value: float) -> None:
        self.buf.f["rect"][self.slot, 3] = value
        self.buf.mark(self.slot)


class Dot(_Shape):
    """A filled circle: position, radius, color (or set_colors(off, on)), state, opacity."""

    kind = DOT

    def __init__(self, x: float, y: float, radius: float, color, canvas: Canvas, layer) -> None:
        super().__init__(canvas, layer)
        self._set("center", (x, y))
        self._set("radius", radius)
        self.set_colors(color, color)

    def set_colors(self, off, on) -> None:
        f = self.buf.f
        f["color"][self.slot] = _rgba(off)
        f["color_on"][self.slot] = _rgba(on)
        self.buf.mark(self.slot)

    @property
    def color(self) -> tuple:
        return tuple(int(c) for c in self._get("color"))

    @color.setter
    def color(self, value) -> None:
        self.set_colors(value, value)

    @property
    def position(self) -> Point:
        x, y = self._get("center")
        return float(x), float(y)

    @position.setter
    def position(self, xy: Point) -> None:
        self._set("center", xy)

    @property
    def radius(self) -> float:
        return float(self._get("radius"))

    @radius.setter
    def radius(self, value: float) -> None:
        self._set("radius", value)


class Segment(_Shape):
    """A straight thick line from a to b, blending from one color to another,
    with an optional round cap on either end."""

    kind = SEGMENT

    def __init__(self, thickness: float, color, canvas: Canvas, layer) -> None:
        super().__init__(canvas, layer)
        self._set("radius", thickness / 2)
        self.set_colors(color, color)

    def place(self, a: Point, b: Point, cap_a: bool = False, cap_b: bool = False) -> None:
        f, s = self.buf.f, self.slot
        f["a"][s] = a
        f["b"][s] = b
        f["flags"][s, 2:] = (255 if cap_a else 0, 255 if cap_b else 0)
        self.buf.mark(s)

    def set_colors(self, start, end, start_on=None, end_on=None) -> None:
        """Colors at a and b, off -- and on, if different."""
        f, s = self.buf.f, self.slot
        f["ca"][s] = _rgba(start)
        f["cb"][s] = _rgba(end)
        f["ca_on"][s] = _rgba(start if start_on is None else start_on)
        f["cb_on"][s] = _rgba(end if end_on is None else end_on)
        self.buf.mark(s)


class WireDot(Segment):
    """A dot drawn by the segment program: a zero-length segment with both caps.
    Junction dots are these, so they draw in their wire's place in the layer instead
    of above every wire. Same interface as Dot."""

    def __init__(self, x: float, y: float, radius: float, color, canvas: Canvas, layer) -> None:
        super().__init__(2 * radius, color, canvas, layer)
        self.position = (x, y)

    @property
    def position(self) -> Point:
        x, y = self._get("a")
        return float(x), float(y)

    @position.setter
    def position(self, xy: Point) -> None:
        self.place(xy, xy, True, True)

    @property
    def color(self) -> tuple:
        return tuple(int(c) for c in self._get("ca"))

    @color.setter
    def color(self, value) -> None:
        self.set_colors(value, value)

    def set_pair(self, off, on) -> None:
        self.set_colors(off, off, on, on)
