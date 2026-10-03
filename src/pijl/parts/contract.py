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
  - A `pure` part's eval may be skipped when its inputs didn't change, and one with
    up to 4 inputs may be called once, on every combination of 0 / 1 / X / Z at
    once, and turned into a lookup table (sim/lut.py). So a pure eval must depend
    on its inputs only: not on ctx.parts, props, ctx.tick or time. If it raises on
    some combination, it just isn't tabulated.
  - Settings (see settings.py) are props the user edits from the context menu,
    on one part or on a selection of parts of the same kind. Every value written
    has been through the setting's parse(). changed() runs once per finished edit
    (and on undo / redo), for all the edited parts at once; while a slider drags
    the props change but changed() waits, unless the setting says live=True.
    eval sees the new props on the next step either way.
  - face() is the only hook that's only called by the editor (never headless):
    once a frame, for the live instances on screen. It decides how the marks of
    the part's face (Look.face) that don't follow a pin are lit, so it may
    depend on anything; it must not change props or the circuit.
  - action() runs once per click on one of the part's actions, for all the
    clicked parts at once (live ones only). Prop changes it makes are undoable;
    anything else it does (part.state, the outside world) isn't.
  - Wide pins (buses): `widths = {"d": 8}` makes pin "d" 8 lanes wide (1 to
    MAX_WIDTH). In eval (and face) a wide pin's array has shape (n, 8), lane 0
    the least significant bit: & | ^ ~ work lane by lane, and pijl.logic.ints /
    Logic.of_ints turn lanes into numbers and back. Return a wide output as Logic
    (n, w) or (w,) -- or a Level, for every lane; plain numbers aren't taken
    (is 5 one lane or bits?): use Logic.of_ints. A width may also be the name of a
    prop (`{"d": "width"}`): each instance is as wide as its prop says (1 if it
    has no such prop), read when the instance is made.
  - A part's pins can differ from instance to instance: layout(props) says what an
    instance with these props has (pin names, widths, joins); by default the class
    attributes (ins, outs, widths, joins). Override it for a part whose pins follow
    its settings (SPLIT: as many pins as its pattern says). Joins may name lanes:
    "bus[0:4]" (lanes 0 to 3), "bus[5]". The engine reads pins only through
    layout() (see layout_of), when an instance is made; a settings edit that changes
    it rebuilds the instance (the editor drops wires that no longer fit). Instances
    of one kind with different layouts are evaluated in separate batches, and eval
    gets one array per input pin of *that* layout. Kinds with a wide pin, or a
    layout other than their class attributes, aren't tabulated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .settings import Action, Setting

if TYPE_CHECKING:
    from ..sim.circuit import Part

API = 2  # bump when this contract changes in a way old scripts can't follow
SUPPORTED_APIS = (
    1,
    2,
)  # what the loader still accepts (see _evaluate in sim/circuit.py)

LABEL_SIDES = ("below", "left", "right")


MAX_WIDTH = 64  # lanes in a pin, at most (so a bus's value fits a uint64)

GRID_STEP = 20  # Look.size is in these steps (2 x the editor's grid), so pins land on the grid


@dataclass(frozen=True)
class Mark:
    """One shape on a part's face (Look.face): a capsule `radius` thick from `a` to `b`,
    or a dot at `a` if there's no `b`. Points are in world units from the body's
    bottom left corner, y up. It follows `pin` (an input or output name): lit while
    that pin is 1, and patterned like a pin while it's X or Z. A mark without a pin
    shows what the part type's face() hook says."""

    a: tuple[float, float]
    b: tuple[float, float] | None = None
    radius: float = 5.0
    pin: str | None = None


@dataclass(frozen=True)
class Look:
    """How the editor draws a part. Colors are names from ui/theme.py."""

    narrow: bool = False  # IO-width body with smaller text
    label: str = "below"  # where the user's label goes: below, left, right
    lit: tuple[str, str] | None = (
        None  # body (off, on) colors, following the part's first pin
    )
    swatch: str = "PART_BODY"  # its color in the part picker
    body: str = "PART_BODY"  # body (fill, border) colors, unless `lit` says otherwise
    pin_labels: bool = (
        True  # name tags next to the pins (Tab picks hidden / hover / always)
    )
    size: tuple[int, int] | None = (
        None  # body (width, height) in multiples of GRID_STEP; None: from pins and title
    )
    titled: bool = True  # the title on the body (off for a part whose face says it all)
    face: tuple[Mark, ...] = ()  # shapes drawn on the body, over it (see Mark)
    face_colors: tuple[str, str] = ("FACE_OFF", "FACE_ON")  # marks' (off, on) colors.
    # A part with a face can be recolored: its props["color"] tints them
    cells: str | None = None  # a pin whose lanes, when it's a bus, the body shows as a
    # column of cells (lane 0 at the top, like pins; the body grows to fit), each lit by its
    # lane, in face_colors. A type with click_cell() gets a click on one


class PartType:
    """One kind of part. Class attributes are the defaults; instances may set their own."""

    kind: str = ""  # stable id: save files refer to parts by it
    title: str = ""  # what the body says, if not the kind
    ins: tuple[str, ...] = ()  # input pin names, top to bottom
    outs: tuple[str, ...] = ()  # output pin names, top to bottom
    props: dict[
        str, Any
    ] = {}  # per-instance data (JSON values); each instance gets a copy
    settings: dict[
        str, Setting
    ] = {}  # props the user can set from the context menu (settings.py);
    # prop name -> Setting, in menu order. Not also in `props`
    actions: dict[
        str, Action
    ] = {}  # context menu rows that call action(); name -> Action
    pure: bool = False  # outputs depend only on inputs: the engine may optimize it
    category: str = ""  # default collection in the part picker ("" = loose)
    look: Look = Look()
    port: str | None = None  # "in" / "out": macro ports. Engine-only, see ports.py
    weak: tuple[
        str, ...
    ] = ()  # outputs that only drive a net nobody else drives (pulls).
    # Of those on a net, the highest props["priority"] wins
    joins: tuple[
        tuple[str, ...], ...
    ] = ()  # pin groups that are one net straight through the
    # part (an inline pull: (("in", "out"),)). They show the
    # net's value; a joined output's eval value drives that net
    widths: dict[
        str, int | str
    ] = {}  # pin name -> lanes (a bus), or the name of the prop that says; others: 1
    api: int = API  # the API its script was written for (set by the loader)

    def eval(self, ctx: Ctx, *ins):
        """Every step: input arrays in, output arrays (or scalars) out."""

    def open(self, part: Part) -> None:
        """A real instance appeared."""

    def close(self, part: Part) -> None:
        """An opened instance is going away (deleted, undone, app closing)."""

    def frame(self, ctx: Ctx) -> None:
        """Once per frame, outside the step loop, for the live instances."""

    def changed(self, ctx: Ctx, key: str, old: list) -> None:
        """The user (or an undo / redo) changed setting `key` of ctx.parts; old[i] is
        what ctx.parts[i] had before. The new values are already in the props."""

    def action(self, ctx: Ctx, name: str) -> None:
        """The user picked action `name` (a key of `actions`) for ctx.parts."""

    def face(self, ctx: Ctx, *ins):
        """Once a frame, in the editor, for the parts it shows: input arrays in (as in
        eval), and one value per face mark without a pin out, in Look.face order (just
        the value if there's one). Values are as eval's. Only the look changes."""

    def click(self, part: Part) -> None:
        """The user clicked a placed instance. Overriding this makes the part clickable
        (a click then no longer selects it)."""

    def click_cell(self, part: Part, lane: int) -> None:
        """The user clicked cell `lane` of a placed instance (see Look.cells). A click on
        the body anywhere else goes to click(), if the type has it."""

    def layout(self, props: dict[str, Any]) -> dict[str, Any]:
        """The pins of an instance with these props, as plain data: "ins" and "outs"
        (names), "widths" (name -> lanes; or one width per pin, ins then outs) and
        "joins" (groups of pins, or lane ranges of pins, that are one net straight
        through the part). Keys left out: the class attributes. By default it's the
        class attributes, a width naming a prop read from `props`."""
        widths = {
            name: props.get(w, 1) if isinstance(w, str) else w for name, w in self.widths.items()
        }
        return {"ins": self.ins, "outs": self.outs, "widths": widths, "joins": self.joins}

    def has(self, hook: str) -> bool:
        """Does this type override `hook`? Hooks it doesn't override are never called."""
        return getattr(type(self), hook) is not getattr(PartType, hook)

    def __repr__(self) -> str:
        return f"<PartType {self.kind}>"


@dataclass(frozen=True)
class Layout:
    """An instance's pins, checked and normalized (layout_of): what the engine builds."""

    ins: tuple[str, ...]
    outs: tuple[str, ...]
    widths: tuple[int, ...]  # per pin, ins then outs
    # groups of lane ranges that are one net: (pin index (ins, then outs), first lane, lanes)
    joins: tuple[tuple[tuple[int, int, int], ...], ...] = ()

    @property
    def n_in(self) -> int:
        return len(self.ins)

    @property
    def wide(self) -> bool:
        return any(w != 1 for w in self.widths)


def layout_of(t: PartType, props: dict[str, Any]) -> Layout:
    """t.layout(props), checked: the one place pins are read from. ValueError if it
    isn't a layout (names that aren't strings, widths outside 1 to MAX_WIDTH, joins
    of unknown pins, lanes or unequal widths, ...)."""
    raw = t.layout(props)
    if not isinstance(raw, dict):
        raise ValueError(f"{t.kind}: layout() gave {type(raw).__name__}, not a dict")
    ins, outs = tuple(raw.get("ins", t.ins)), tuple(raw.get("outs", t.outs))
    names = ins + outs
    if not all(isinstance(p, str) for p in names):
        raise ValueError(f"{t.kind}: pin names must be strings")
    index: dict[str, int] = {}
    for k, name in enumerate(names):
        index[name] = -1 if name in index else k  # (-1: there are two)

    def pin(name: str, what: str) -> int:
        k = index.get(name)
        if k is None:
            raise ValueError(f"{t.kind}: {what} names {name!r}, which isn't a pin")
        if k < 0:
            raise ValueError(f"{t.kind}: {what} names {name!r}, but two pins are called that")
        return k

    given = raw.get("widths", t.widths)
    if isinstance(given, dict):
        widths = [1] * len(names)
        for name, w in given.items():
            widths[pin(name, "widths")] = w
    else:
        widths = list(given)
        if len(widths) != len(names):
            raise ValueError(f"{t.kind}: {len(widths)} widths for {len(names)} pins")
    for name, w in zip(names, widths):
        if type(w) is not int or not 1 <= w <= MAX_WIDTH:
            raise ValueError(f"{t.kind}: pin {name!r} can't be {w!r} lanes wide")
    joins, seen = [], set()
    for group in raw.get("joins", t.joins):
        members = []
        for ref in group:
            if not isinstance(ref, str):
                raise ValueError(f"{t.kind}: joins name pins (strings), not {ref!r}")
            name, first, n = _lanes(ref)
            k = pin(name, "joins")
            if n is None:
                first, n = 0, widths[k]
            if first < 0 or n < 1 or first + n > widths[k]:
                raise ValueError(f"{t.kind}: {ref!r}: {name!r} has lanes 0 to {widths[k] - 1}")
            if k >= len(ins) and t.has("eval") and n != widths[k]:
                raise ValueError(f"{t.kind}: {ref!r}: a part with an eval joins whole outputs")
            lanes = {(k, first + i) for i in range(n)}
            if lanes & seen:
                raise ValueError(f"{t.kind}: joins name {ref!r} twice")
            seen |= lanes
            members.append((k, first, n))
        if len({n for _, _, n in members}) > 1:
            raise ValueError(f"{t.kind}: joined pins must be equally wide ({group!r})")
        if len(members) > 1:
            joins.append(tuple(members))
    return Layout(ins, outs, tuple(widths), tuple(joins))


def _lanes(ref: str) -> tuple[str, int, int | None]:
    """"bus[0:4]" -> ("bus", 0, 4); "bus[5]" -> ("bus", 5, 1); "bus" -> ("bus", 0, None)."""
    if not ref.endswith("]") or "[" not in ref:
        return ref, 0, None
    name, _, inside = ref[:-1].rpartition("[")
    try:
        if ":" in inside:
            a, b = inside.split(":")
            return name, int(a), int(b) - int(a)
        return name, int(inside), 1
    except ValueError:
        raise ValueError(f"{ref!r}: lanes are written [i] or [first:end]") from None


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

    def __init__(
        self, kind: str, ins, outs, fn: Callable, category: str, look: Look
    ) -> None:
        self.kind, self.ins, self.outs = kind, tuple(ins), tuple(outs)
        self.fn, self.category, self.look = fn, category, look

    def eval(self, ctx: Ctx, *ins):
        return self.fn(*ins)


def part(
    kind: str,
    ins=(),
    outs=(),
    eval: Callable = None,
    *,
    category: str = "",
    look: Look = Look(),
) -> PartType:
    """A pure part from a function: `eval` takes one array per input pin."""
    return FunctionPart(kind, ins, outs, eval, category, look)
