"""Who gets which color: wire gradients, and the glow of IN/OUT parts and pins.

Color sources are IN/OUT parts with a color set, and wires with a color set
(both chosen from T.WIRE_COLORS; the Recolor menu). Everything else inherits:

  - A wire's gradient has a stop at each end that touches a source, plus its
    own color in the middle if it has one:
        [src end color] -> [own color] -> [dst end color]
    Stops present are spread evenly along the wire (one stop: a solid wire).
    A wire with no stops at all is "neutral": the classic grey / red look.
  - A pin end's color is its part's color (only IN/OUT parts have one). A
    junction end's color is the parent wire's gradient at that spot, so color
    flows down branches. Parents are older wires, so one pass in creation order
    settles everything.
  - A part with no color of its own glows in the color of the wire at its pin,
    and every pin dot lights up in the color of the wire end on it.

Tinting keeps each theme color's saturation and brightness and swaps in the
tint's hue, so a blue switch looks like the red one, only blue (and red
tints reproduce the classic theme exactly).

Gradients blend in OKLCH: around the hue wheel (the short way), with even
lightness, so blue -> yellow runs through teal and green instead of grey.

After an edit only the region it can reach is repainted: the changed wires, the
wires on changed parts' pins, everything hanging off those (colors flow down
branches), and the parts at their ends. The color math is cached: a board only
ever uses a handful of palette colors.
"""

from __future__ import annotations

import colorsys
import math
from collections.abc import Iterable
from functools import lru_cache
from typing import TYPE_CHECKING

from ..sim import Pin
from . import theme as T

if TYPE_CHECKING:
    from .editor import Editor

Rgb = tuple[int, int, int]
Pair = tuple[Rgb, Rgb]           # (off, on)
Stops = list[tuple[float, Pair]]  # (fraction of the wire's length, color), ascending


def color_pair(name: str | None) -> Pair | None:
    """A palette color as (off, on); None for no color / a name this version doesn't know."""
    return T.WIRE_COLORS.get(name) if name else None


def part_color(part) -> str | None:
    return part.props.get("color") if part.type.look.lit else None


def paint(editor: Editor, parts: Iterable[int] | None = None, wires: Iterable[int] | None = None) -> None:
    """Recompute wire gradients and part tints: everywhere, or around the parts and wires
    (uids) that changed -- gone ones included, their uids are just skipped."""
    c = editor.circuit
    if parts is None and wires is None:
        todo, affected = c.wires, list(editor.part_views)
    else:
        seed_parts = {c.part_by_uid[uid] for uid in parts or () if uid in c.part_by_uid}
        seeds = {c.wire_by_uid[uid] for uid in wires or () if uid in c.wire_by_uid}
        seeds.update(w for part in seed_parts for pin in part.pins for w in c.ends_on(pin))
        todo = sorted(seeds.union(c.descendants(*seeds)), key=lambda w: w.uid)  # parents first
        affected = seed_parts | {e.part for w in todo for e in w.ends if isinstance(e, Pin)}
    for wire in todo:
        view = editor.wire_views.get(wire)
        if view is None:
            continue
        ends = []
        for end, pos in ((wire.src, view.src), (wire.dst, view.dst)):
            if isinstance(end, Pin):
                ends.append(color_pair(part_color(end.part)))
            else:
                parent = editor.wire_views.get(end)
                ends.append(parent.color_at(pos) if parent is not None else None)
        colors = [c for c in (ends[0], color_pair(view.color), ends[1]) if c is not None]
        n = len(colors)
        view.set_stops([(i / (n - 1) if n > 1 else 0.0, c) for i, c in enumerate(colors)])
    for part in affected:
        view = editor.part_views.get(part)
        if view is None:
            continue
        own = color_pair(part_color(part))
        pins = [own[1] if own else _pin_tint(editor, pin) for pin in part.pins]
        view.set_tints(pins, own[1] if own else next((t for t in pins if t), None))


def _pin_tint(editor: Editor, pin: Pin) -> Rgb | None:
    """The color at the end of the oldest wire on `pin` that has one."""
    for wire in editor.circuit.wires_at(pin):  # creation order
        view = editor.wire_views.get(wire)
        if view is not None and view.stops:
            return sample(view.stops, 0.0 if wire.src is pin else 1.0)[1]
    return None


# ---- color math -----------------------------------------------------------------


def sample(stops: Stops, f: float) -> Pair:
    """The gradient's color at fraction `f` of the wire."""
    if f <= stops[0][0]:
        return stops[0][1]
    for (f0, c0), (f1, c1) in zip(stops, stops[1:]):
        if f <= f1:
            t = 0.0 if f1 == f0 else (f - f0) / (f1 - f0)
            return mix(c0[0], c1[0], t), mix(c0[1], c1[1], t)
    return stops[-1][1]


@lru_cache(maxsize=1 << 16)
def mix(a: Rgb, b: Rgb, t: float) -> Rgb:
    """Blend two colors in OKLCH (see the module docstring)."""
    if a == b or t <= 0:
        return a
    if t >= 1:
        return b
    (l1, a1, b1), (l2, a2, b2) = _oklab(a), _oklab(b)
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    h1, h2 = math.atan2(b1, a1), math.atan2(b2, a2)
    if c1 < 1e-4:  # a grey has no hue of its own: take the other one's
        h1 = h2
    if c2 < 1e-4:
        h2 = h1
    dh = (h2 - h1 + math.pi) % (2 * math.pi) - math.pi  # the short way around
    L, C, h = l1 + t * (l2 - l1), c1 + t * (c2 - c1), h1 + t * dh
    return _from_oklab((L, C * math.cos(h), C * math.sin(h)))


@lru_cache(maxsize=1 << 12)
def with_hue(color: tuple, tint: Rgb | None) -> tuple:
    """`color` with `tint`'s hue (alpha, if any, kept)."""
    if tint is None:
        return color
    h = colorsys.rgb_to_hsv(*(c / 255 for c in tint[:3]))[0]
    _, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in color[:3]))
    return (*(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v)), *color[3:])


def _lin(c: float) -> float:
    c /= 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _srgb(c: float) -> int:
    c = max(0.0, min(1.0, c))
    c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
    return round(c * 255)


def _oklab(rgb: Rgb) -> tuple[float, float, float]:
    r, g, b = (_lin(c) for c in rgb[:3])
    l = math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def _from_oklab(lab) -> Rgb:
    L, a, b = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (_srgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
            _srgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
            _srgb(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s))
