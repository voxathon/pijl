"""The display parts. Their looks say everything the editor draws: a body size
(Look.size), no title, and a face of marks (Look.face) that follow pins, or, for
HEX, a face() hook that decodes. No pyglet here, so headless pijl knows them too.

They have no eval: they drive nothing, and the simulation never calls them.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from pijl.logic import X, Z, where
from pijl.parts import Look, Mark, PartType


def _digit(w: float, h: float, pins: tuple) -> tuple[Mark, ...]:
    """Segments a..g (and dp, if there are 8 pins) of a slanted digit on a w x h body.
    pins: the pin each mark follows (None: the face() hook lights it)."""
    r, gap, slant = 5.0, 7.0, 0.08
    shift = 0.0 if len(pins) == 8 else 7.0  # (no point to make room for: centered)
    x0, x1 = 28.0 + shift, w - 40.0 + shift
    y0, y1 = 30.0, h - 30.0
    ym = (y0 + y1) / 2

    def at(x, y):
        return x + (y - ym) * slant, y

    def seg(p, q, pin):
        (ax, ay), (bx, by) = at(*p), at(*q)
        d = math.hypot(bx - ax, by - ay)
        ux, uy = (bx - ax) / d, (by - ay) / d
        return Mark((ax + ux * gap, ay + uy * gap), (bx - ux * gap, by - uy * gap), r, pin)

    ends = (
        ((x0, y1), (x1, y1)),  # a
        ((x1, y1), (x1, ym)),  # b
        ((x1, ym), (x1, y0)),  # c
        ((x0, y0), (x1, y0)),  # d
        ((x0, ym), (x0, y0)),  # e
        ((x0, y1), (x0, ym)),  # f
        ((x0, ym), (x1, ym)),  # g
    )
    marks = tuple(seg(p, q, pin) for (p, q), pin in zip(ends, pins))
    if len(pins) == 8:
        marks += (Mark(at(x1 + 14.0, y0), radius=r + 1.0, pin=pins[7]),)
    return marks


DIGIT = Look(body="LED_OFF", swatch="LED_ON", size=(120, 180), titled=False)


class SevenSegment(PartType):
    """A seven-segment digit: each segment, and the decimal point, follows its own pin.
    Segments a..g run clockwise from the top, with g in the middle."""

    kind = "7SEG"
    ins = ("a", "b", "c", "d", "e", "f", "g", "dp")
    category = "DISPLAYS"
    look = replace(DIGIT, face=_digit(120, 180, ins))


# Hex digits 0..F as segments a..g lit: (16 x 7)
GLYPHS = (
    "abcdef", "bc", "abdeg", "abcdg", "bcfg", "acdfg", "acdefg", "abc",
    "abcdefg", "abcdfg", "abcefg", "cdefg", "adef", "bcdeg", "adefg", "aefg",
)  # fmt: skip
DECODE = np.array([[s in g for s in "abcdefg"] for g in GLYPHS])


class HexDigit(PartType):
    """A digit that decodes four bits itself (8 on top, 1 at the bottom) and shows 0..F.
    If any input isn't a known 0 or 1, every segment shows X (or Z, if none is driven)."""

    kind = "HEX"
    ins = ("8", "4", "2", "1")
    category = "DISPLAYS"
    look = replace(DIGIT, face=_digit(120, 180, (None,) * 7))

    def face(self, ctx, *bits):
        known = np.logical_and.reduce([b.known for b in bits])
        undriven = np.logical_and.reduce([b.isz for b in bits])
        value = sum(b.is1.astype(np.intp) << (3 - i) for i, b in enumerate(bits))
        unknown = where(undriven, Z, X)
        return tuple(where(known, DECODE[value, s], unknown) for s in range(7))


class LedBar(PartType):
    """Eight LEDs in a column, one beside each pin (7 on top, 0 at the bottom)."""

    kind = "BAR"
    ins = ("7", "6", "5", "4", "3", "2", "1", "0")
    category = "DISPLAYS"
    look = Look(
        body="LED_OFF",
        swatch="LED_ON",
        size=(60, 180),
        titled=False,
        face=tuple(
            Mark((18.0, y), (42.0, y), 6.0, pin)
            for pin, y in zip(ins, (90.0 + (3.5 - i) * 20.0 for i in range(8)))
        ),
    )


PARTS: tuple[PartType, ...] = (SevenSegment(), HexDigit(), LedBar())
