"""The part-script contract: what a part script provides, and what the engine promises it.

A part script is a .py file in a project's parts folder (see registry.py for how
they're found). It declares `API = 2` and a `register(reg)` function that calls
`reg.add(...)` with PartTypes -- as many as it likes:

    from pijl.parts import part

    API = 2

    def register(reg):
        reg.add(part("NAND", ins=("a", "b"), outs=("out",), eval=lambda a, b: ~(a & b)))

`part(...)` is shorthand for a pure PartType. Anything else subclasses PartType
and overrides whichever hooks it needs; every hook is optional.

What the engine promises:
  - eval runs once per kind per step, on arrays: element i of every input array
    (and of the result) belongs to ctx.parts[i]. Inputs are pijl.logic.Logic
    arrays: four-state values (0, 1, X, Z) whose & | ^ ~ do four-state logic, so
    use those, not `and`/`or`/`not`. Return a tuple with one value per output
    pin (just the value if there's one pin); each value is a Logic array, a
    Level (X, Z, ...), a bool, or an array or list of those with one entry per
    instance. Scalars are copied to every instance. Plain bools mean 0 / 1;
    return Z to not drive the net (a tri-state output).
  - API 1 scripts (written before four-state logic) still load: they get plain
    bool arrays (X and Z read as 0), and if a pure one has any input that isn't
    a known 0 or 1, all its outputs are X.
  - Every hook runs on the main thread. Start your own threads if you need them;
    just don't touch the UI from them.
  - A hook that raises disables its kind (in that circuit) and reports the error.
    Nothing else stops.
  - open() runs when a real instance appears -- never for the ghosts that follow
    the cursor before placement -- and every opened instance gets a close().
  - Parts that aren't `pure` are never skipped, cached or reordered; only live
    (opened) instances of them are evaluated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..sim.circuit import Part

API = 2  # bump when this contract changes in a way old scripts can't follow
SUPPORTED_APIS = (1, 2)  # what the loader still accepts (see _evaluate in sim/circuit.py)

LABEL_SIDES = ("below", "left", "right")


@dataclass(frozen=True)
class Look:
    """How the editor draws a part. Colors are names from ui/theme.py."""
    narrow: bool = False                # IO-width body with smaller text
    label: str = "below"                # where the user's label goes: below, left, right
    lit: tuple[str, str] | None = None  # body (off, on) colors, following the part's first pin
    swatch: str = "PART_BODY"           # its color in the part picker
    body: str = "PART_BODY"             # body (fill, border) colors, unless `lit` says otherwise
    pin_labels: bool = True             # name tags next to the pins (Tab picks hidden / hover / always)


class PartType:
    """One kind of part. Class attributes are the defaults; instances may set their own."""
    kind: str = ""                   # stable id: save files refer to parts by it
    title: str = ""                  # what the body says, if not the kind
    ins: tuple[str, ...] = ()        # input pin names, top to bottom
    outs: tuple[str, ...] = ()       # output pin names, top to bottom
    props: dict[str, Any] = {}       # per-instance settings (JSON values); each instance gets a copy
    choices: dict[str, tuple] = {}   # props the user can set from the part's context menu:
                                     # prop name -> the values to offer (each prop needs a default)
    pure: bool = False               # outputs depend only on inputs: the engine may optimize it
    category: str = ""               # default collection in the part picker ("" = loose)
    look: Look = Look()
    port: str | None = None          # "in" / "out": macro ports. Engine-only, see ports.py
    weak: tuple[str, ...] = ()       # outputs that only drive a net nobody else drives (pulls).
                                     # Of those on a net, the highest props["priority"] wins
    joins: tuple[tuple[str, ...], ...] = ()  # pin groups that are one net straight through the
                                     # part (an inline pull: (("in", "out"),)). They show the
                                     # net's value; a joined output's eval value drives that net
    api: int = API                   # the API its script was written for (set by the loader)

    def eval(self, ctx: Ctx, *ins):
        """Every step: input arrays in, output arrays (or scalars) out."""

    def open(self, part: Part) -> None:
        """A real instance appeared."""

    def close(self, part: Part) -> None:
        """An opened instance is going away (deleted, undone, app closing)."""

    def frame(self, ctx: Ctx) -> None:
        """Once per frame, outside the step loop, for the live instances."""

    def click(self, part: Part) -> None:
        """The user clicked a placed instance. Overriding this makes the part clickable
        (a click then no longer selects it)."""

    def has(self, hook: str) -> bool:
        """Does this type override `hook`? Hooks it doesn't override are never called."""
        return getattr(type(self), hook) is not getattr(PartType, hook)

    def __repr__(self) -> str:
        return f"<PartType {self.kind}>"


@dataclass
class Ctx:
    """What a hook sees: a batch of instances of one kind."""
    parts: list[Part]
    tick: int = 0  # simulation steps since the circuit was created
    time: float = field(default_factory=time.monotonic)

    @property
    def n(self) -> int:
        return len(self.parts)

    @property
    def props(self) -> list[dict[str, Any]]:
        return [p.props for p in self.parts]

    @property
    def state(self) -> list[dict[str, Any]]:
        """Per-instance scratch space, kept for as long as the instance lives."""
        return [p.state for p in self.parts]


class FunctionPart(PartType):
    """A pure part whose eval is a plain function of its inputs. Made by part()."""
    pure = True

    def __init__(self, kind: str, ins, outs, fn: Callable, category: str, look: Look) -> None:
        self.kind, self.ins, self.outs = kind, tuple(ins), tuple(outs)
        self.fn, self.category, self.look = fn, category, look

    def eval(self, ctx: Ctx, *ins):
        return self.fn(*ins)


def part(kind: str, ins=(), outs=(), eval: Callable = None, *, category: str = "",
         look: Look = Look()) -> PartType:
    """A pure part from a function: `eval` takes one array per input pin."""
    return FunctionPart(kind, ins, outs, eval, category, look)
