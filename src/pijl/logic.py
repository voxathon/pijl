"""Four-state logic values: 0, 1, X (unknown) and Z (not driven). No pyglet in here.

Every value is a two-bit code: bit 0 means "could be 0", bit 1 "could be 1".

    Z = 00   nothing drives it
    0 = 01
    1 = 10
    X = 11   could be either

That makes the two hot paths plain bitwise work:
  - A net's value is the OR of its drivers: Z drivers add nothing, a 0 and a 1
    together make X, and no drivers at all leave Z (see resolve).
  - A gate treats a Z input as X, then works out which outputs are still
    possible: AND can be 1 only if both inputs can, and can be 0 if either can.

The codes are an engine detail: the code 1 means the *value* 0. Part scripts only
ever see Logic arrays and Levels, whose & | ^ ~ do four-state logic, so
`lambda a, b: ~(a & b)` is a correct four-state NAND as written.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np

CODE = np.uint8


class Level(IntEnum):
    """One four-state value. Truthy only when it's 1."""

    Z = 0
    ZERO = 1
    ONE = 2
    X = 3

    def __bool__(self) -> bool:
        return self is Level.ONE

    def __str__(self) -> str:
        return "Z01X"[self]

    def __repr__(self) -> str:
        return f"Level.{self.name}"

    # IntEnum would do integer bitwise ops on the codes: route them through Logic.
    def __and__(self, other):
        return _level(Logic(self) & other)

    def __or__(self, other):
        return _level(Logic(self) | other)

    def __xor__(self, other):
        return _level(Logic(self) ^ other)

    def __invert__(self):
        return _level(~Logic(self))

    __rand__, __ror__, __rxor__ = __and__, __or__, __xor__


Z, ZERO, ONE, X = Level.Z, Level.ZERO, Level.ONE, Level.X


def _level(v: Logic | Any):
    return Level(int(v.codes)) if isinstance(v, Logic) and v.codes.ndim == 0 else v


def codes(value: Any) -> np.ndarray:
    """Codes for anything a script may hand over: Logic arrays and Levels as they are;
    bools, ints and arrays of them as 0 / 1 by truthiness."""
    if isinstance(value, Logic):
        return value.codes
    if isinstance(value, Level):
        return np.asarray(value.value, CODE)
    if isinstance(value, (list, tuple)) and any(
        isinstance(v, (Level, Logic)) for v in value
    ):
        return np.array(
            [codes(v) for v in value], CODE
        )  # (numpy would see a Level as its int code)
    return (np.asarray(value).astype(bool) + CODE(1)).astype(CODE)


class Logic:
    """An array of four-state values (or a single one, with shape ()).

    Inside, the two bits are two bool arrays, "planes": `_lo` (could be 0) and `_hi`
    (could be 1). numpy's bool ops are several times faster than its uint8 ones, and
    in planes AND is two ops and NOT is free (swap them). The engine hands over and
    takes back codes; the planes are split / joined only at that boundary.
    """

    __slots__ = ("_codes", "_lo", "_hi")
    __array_ufunc__ = (
        None  # numpy must not treat this as a number array: ours are the only ops
    )

    def __init__(self, value: Any = ()) -> None:
        if isinstance(value, Logic):
            self._codes, self._lo, self._hi = value._codes, value._lo, value._hi
        else:
            self._codes, self._lo, self._hi = codes(value), None, None

    @classmethod
    def of_codes(cls, c: np.ndarray) -> Logic:
        """Wrap an array that already holds codes, as is (for the engine)."""
        out = cls.__new__(cls)
        out._codes, out._lo, out._hi = c, None, None
        return out

    @classmethod
    def _of_planes(cls, lo: np.ndarray, hi: np.ndarray) -> Logic:
        out = cls.__new__(cls)
        out._codes, out._lo, out._hi = None, lo, hi
        return out

    @classmethod
    def full(cls, n: int, level: Level) -> Logic:
        return cls.of_codes(np.full(n, level.value, CODE))

    @property
    def codes(self) -> np.ndarray:
        if self._codes is None:
            self._codes = self._lo.view(CODE) | (self._hi.view(CODE) << 1)
        return self._codes

    def _planes(self) -> tuple[np.ndarray, np.ndarray]:
        if self._lo is None:
            c = self._codes
            self._lo, self._hi = (c & 1).view(bool), (c >> 1).view(bool)
        return self._lo, self._hi

    # ---- four-state operators --------------------------------------------------

    def __and__(self, other) -> Logic:
        (a0, a1), (b0, b1) = _gate_in(self), _gate_in(other)
        return Logic._of_planes(
            a0 | b0, a1 & b1
        )  # 1 only if both can be; 0 if either can

    def __or__(self, other) -> Logic:
        (a0, a1), (b0, b1) = _gate_in(self), _gate_in(other)
        return Logic._of_planes(a0 & b0, a1 | b1)

    def __xor__(self, other) -> Logic:
        (a0, a1), (b0, b1) = _gate_in(self), _gate_in(other)
        return Logic._of_planes((a0 & b0) | (a1 & b1), (a1 & b0) | (a0 & b1))

    def __invert__(self) -> Logic:
        lo, hi = _gate_in(self)
        return Logic._of_planes(hi, lo)

    __rand__, __ror__, __rxor__ = __and__, __or__, __xor__

    # ---- asking what's there ---------------------------------------------------

    @property
    def is1(self) -> np.ndarray:
        lo, hi = self._planes()
        return hi & ~lo

    @property
    def is0(self) -> np.ndarray:
        lo, hi = self._planes()
        return lo & ~hi

    @property
    def isx(self) -> np.ndarray:
        lo, hi = self._planes()
        return lo & hi

    @property
    def isz(self) -> np.ndarray:
        lo, hi = self._planes()
        return ~(lo | hi)

    @property
    def known(self) -> np.ndarray:
        """0 or 1 (not X, not Z)."""
        lo, hi = self._planes()
        return lo ^ hi

    # ---- array-ish ---------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.codes.shape

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, i):
        return _level(Logic.of_codes(self.codes[i]))

    def __iter__(self):
        return (Level(int(c)) for c in self.codes)

    def __bool__(self) -> bool:
        raise TypeError(
            "a Logic array has no single truth value: use .is1 / .is0 / .known"
        )

    def __repr__(self) -> str:
        if self.codes.ndim == 0:
            return f"Logic({Level(int(self.codes))})"
        return f"Logic({''.join('Z01X'[c] for c in self.codes.tolist())})"


def _gate_in(v) -> tuple[np.ndarray, np.ndarray]:
    """Planes as a gate input: Z reads as X."""
    if isinstance(v, Logic):
        fresh = v._codes is None  # made by an operator: those never hold Z
        lo, hi = v._planes()
        if fresh:
            return lo, hi
    elif isinstance(v, (Level, list, tuple)):
        lo, hi = Logic(v)._planes()
    else:  # bools: known values
        on = np.asarray(v).astype(bool)
        return ~on, on
    z = ~(lo | hi)
    return lo | z, hi | z


def where(cond, a, b) -> Logic:
    """cond ? a : b, per element. An unknown cond (X / Z) gives a where a and b agree
    on a known value, else X (like a real mux). The chosen value passes as it is, Z
    included, like a switch: where(en, a, Z) is a tri-state buffer."""
    c = codes(cond)
    a, b = np.broadcast_arrays(codes(a), codes(b), c)[:2]
    agree = (a == b) & ((a == ZERO) | (a == ONE))
    out = np.where(c == ONE, a, np.where(c == ZERO, b, np.where(agree, a, CODE(X))))
    return Logic.of_codes(out.astype(CODE))


# ---- nets --------------------------------------------------------------------------


def resolve(drivers: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Per group of driver codes (groups begin at `starts`, like np.add.reduceat): the
    value on the net. Z drivers count for nothing; disagreeing drivers give X."""
    return np.bitwise_or.reduceat(drivers, starts)


def fights(drivers: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Per group: does one driver say 0 while another says 1? (A lone X driver only
    passes an X along; that isn't a fight.)"""
    return np.bitwise_or.reduceat(np.where(drivers == X, CODE(0), drivers), starts) == X
