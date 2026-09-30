"""Settings and actions: what a part script lets the user change and do from the
part's context menu. Plain data; the editor builds the menu rows from them.

    settings = {
        "period":  Number(250, 1, 10_000, unit="ms", log=True),
        "running": Toggle(True),
        "mode":    Choice(("square", "pulse"), "square", labels=("Square", "Pulse")),
        "note":    Text("", max_len=24),
    }
    actions = {"reset": Action("Reset phase")}

A setting's key is the prop it's stored under (part.props[key]), and its default
is the prop's default: don't also list it in PartType.props. Every value the engine
writes has been through the setting's parse(): typed, clamped, snapped. Props that
aren't settings are the script's own data and never show up in a menu.

No pyglet in here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

SCALARS = (str, int, float, bool)  # what a Choice may offer (JSON scalars)


def _title(key: str) -> str:
    return key.replace("_", " ").capitalize()


def _parse_default(s: Setting):
    try:
        return s.parse(s.default)
    except ValueError as e:
        raise ValueError(
            f"the default {s.default!r} isn't a value it allows ({e})"
        ) from None


def _same(a, b) -> bool:
    """Equal and of the same type: True isn't 1, and 1.0 isn't 1."""
    return type(a) is type(b) and a == b


@dataclass(frozen=True)
class Setting:
    """Base of every setting. `label`: what the menu calls it (default: from the key).
    `hint`: a help line under the value while it's being edited."""

    label: str = field(default="", kw_only=True)
    hint: str = field(default="", kw_only=True)
    # (every subclass has a `default` field; it's not declared here because a field
    # declared in a base class keeps the base's place in the constructor's arguments)

    def parse(self, value) -> Any:
        """The value to store, or ValueError. Called on every write and on load."""
        raise NotImplementedError

    def check(self) -> None:
        """Called once by the registry. ValueError if the declaration itself is bad."""
        if not _same(_parse_default(self), self.default):
            raise ValueError(f"the default {self.default!r} isn't a value it allows")

    @property
    def initial(self):
        """What a new instance gets: the default, as parse() would store it."""
        return self.default

    def title(self, key: str) -> str:
        return self.label or _title(key)

    def show(self, value) -> str:
        """The value as the menu shows it."""
        return str(value)


@dataclass(frozen=True)
class Choice(Setting):
    """One of a fixed set of values: a submenu with the current one checked."""

    values: tuple = ()
    default: Any = None
    labels: tuple[str, ...] | None = (
        None  # what the menu shows per value (default: str(value))
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        if self.labels is not None:
            object.__setattr__(self, "labels", tuple(self.labels))

    def check(self) -> None:
        if not self.values:
            raise ValueError("a Choice needs values")
        if not all(type(v) in SCALARS for v in self.values):
            raise ValueError("Choice values must be str, int, float or bool")
        if len({(type(v), v) for v in self.values}) != len(self.values):
            raise ValueError("Choice values must be unique")
        if self.labels is not None and (
            len(self.labels) != len(self.values)
            or not all(isinstance(s, str) for s in self.labels)
        ):
            raise ValueError("Choice labels must be one string per value")
        super().check()

    def parse(self, value):
        for v in self.values:
            if _same(v, value):
                return v
        raise ValueError(f"{value!r} isn't one of {list(self.values)}")

    def show(self, value) -> str:
        if self.labels is not None:
            for v, text in zip(self.values, self.labels):
                if _same(v, value):
                    return text
        return str(value)


@dataclass(frozen=True)
class Toggle(Setting):
    """On or off: a menu row that flips it."""

    default: bool = False

    def parse(self, value) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"{value!r} isn't true or false")
        return value

    def show(self, value) -> str:
        return "on" if value else "off"


@dataclass(frozen=True)
class Text(Setting):
    """A line of text, typed into a prompt. Unprintable characters are dropped and
    it's cut to max_len."""

    default: str = ""
    max_len: int = 32

    def check(self) -> None:
        if not isinstance(self.max_len, int) or self.max_len < 1:
            raise ValueError("Text max_len must be a positive int")
        super().check()

    def parse(self, value) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{value!r} isn't text")
        return "".join(c for c in value if c.isprintable())[: self.max_len]

    def show(self, value) -> str:
        return value or "(empty)"


@dataclass(frozen=True)
class Number(Setting):
    """A number between min and max, edited in a popover with a slider and a typed field.

    Whole numbers (int) if default, min, max and step are all ints, otherwise floats.
    Values are clamped into [min, max] and snapped to min + k * step (step: 1 for
    ints, none -- continuous -- for floats). `log`: the slider is logarithmic (min
    must be > 0). `slider=False`: just the typed field. `live`: the part's changed
    hook also runs while the slider drags (at most once a frame), not only when the
    edit is done -- see PartType.changed."""

    default: int | float = 0
    min: int | float = 0
    max: int | float = 1
    step: int | float | None = None
    unit: str = ""
    slider: bool = True
    log: bool = False
    live: bool = False

    @property
    def whole(self) -> bool:
        nums = (self.default, self.min, self.max) + (
            (self.step,) if self.step is not None else ()
        )
        return all(type(n) is int for n in nums)

    def check(self) -> None:
        for name in ("default", "min", "max") + (
            ("step",) if self.step is not None else ()
        ):
            n = getattr(self, name)
            if type(n) not in (int, float) or not math.isfinite(n):
                raise ValueError(f"Number {name} must be a finite int or float")
        if not self.min < self.max:
            raise ValueError("Number needs min < max")
        if self.step is not None and self.step <= 0:
            raise ValueError("Number step must be > 0")
        if self.log and self.min <= 0:
            raise ValueError("a log Number needs min > 0")
        if (
            _parse_default(self) != self.default
        ):  # (an int default of a float Number is fine)
            raise ValueError(f"the default {self.default!r} isn't a value it allows")

    @property
    def initial(self):
        return self.parse(self.default)

    def parse(self, value) -> int | float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{value!r} isn't a number")
        v = min(max(value, self.min), self.max)
        step = self.step if self.step is not None else (1 if self.whole else None)
        if step is not None:
            v = min(self.min + round((v - self.min) / step) * step, self.max)
        if self.whole:
            return int(round(v))
        return float(f"{float(v):.12g}")  # no 0.30000000000000004 from snapping

    def fraction(self, value) -> float:
        """Where `value` sits on the slider, 0 (min) to 1 (max)."""
        lo, hi, v = self.min, self.max, min(max(value, self.min), self.max)
        if self.log:
            return (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
        return (v - lo) / (hi - lo)

    def at(self, fraction: float) -> int | float:
        """The value at a spot on the slider (0..1), parsed (snapped to step)."""
        f = min(max(fraction, 0.0), 1.0)
        lo, hi = self.min, self.max
        return self.parse(lo * (hi / lo) ** f if self.log else lo + f * (hi - lo))

    def parse_text(self, text: str) -> int | float:
        """Typed input -> a value (ValueError if it isn't a number)."""
        text = (
            text.strip().removesuffix(self.unit).strip() if self.unit else text.strip()
        )
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"{text!r} isn't a number") from None
        if value.is_integer() and self.whole:
            value = int(value)
        return self.parse(value)

    def show(self, value) -> str:
        text = str(value) if type(value) is int else f"{value:g}"
        return f"{text} {self.unit}" if self.unit else text


@dataclass(frozen=True)
class Action:
    """A menu row that runs the part's action(ctx, name) hook on the clicked (or
    selected) parts. `danger`: drawn in red, like Delete."""

    label: str = ""
    danger: bool = False

    def title(self, name: str) -> str:
        return self.label or _title(name)
