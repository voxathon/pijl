"""The netlist language: parse a script, resolve it against the parts in a box.

No pyglet here: resolve() takes the parts and a few callables, so it runs headless.

    # a comment
    CLK.q -> R*.clk              one to many: fan out
    A*.q -> B*.d                 as many as many: pairwise, in name order
    ALU.a*, ALU.b* -> regs/R*.q  a list on either side; regs/ is another box
    <NAND>*.a -> X.q             <kind> picks parts by kind (a macro's name for macros)
    chain ADD*: cout -> cin      ADD0.cout -> ADD1.cin, ADD1.cout -> ADD2.cin, ...
    net DATA: R*.q, ALU.a        all of them one net: any mix of outputs and inputs
    bus DATA: R*.q, ALU.a        the same, drawn as a straight spine with taps

A net's name is optional (net: A.q, B.a). Pins that are wired already join the net
through the wires they have. All the pins of a net must be equally wide.

Statements end at a newline or ';'. A part's name is its label, or its kind when it
has none; without a box/ in front, it's looked for in the box, then (if nothing
there matches) on the whole board. A pin goes by its name or by in1, in2, ...,
out1, ... counted from the top. Case doesn't matter. Names and pins take * and ?
wildcards; anything with spaces or odd characters goes in quotes ("my reg".q).
Parts are ordered by name, numbers counted as numbers (R2 before R10), then top to
bottom, left to right.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Callable, Iterable

MACRO = "macro:"


class NetlistError(Exception):
    def __init__(self, line: int, msg: str) -> None:
        super().__init__(f"line {line}: {msg}")
        self.line, self.msg = line, msg


# ---- parsing ------------------------------------------------------------------------


@dataclass(frozen=True)
class End:
    """box/<kind>label.pin -- every part but the pin optional-ish (label or kind needed)."""

    box: str | None
    kind: str | None
    label: str | None
    pin: str

    def __str__(self) -> str:
        box = f"{self.box}/" if self.box else ""
        kind = f"<{self.kind}>" if self.kind else ""
        return f"{box}{kind}{self.label or ''}.{self.pin}"


@dataclass(frozen=True)
class Connect:
    line: int
    left: tuple[End, ...]
    right: tuple[End, ...]


@dataclass(frozen=True)
class Chain:
    line: int
    parts: End  # (pin unused: "")
    out: str
    into: str


@dataclass(frozen=True)
class NetStatement:
    line: int
    kind: str  # "net" or "bus"
    name: str | None
    ends: tuple[End, ...]


@dataclass
class NetPlan:
    """One net or bus to make: its pins, in the order the script names them."""

    kind: str
    name: str | None
    line: int
    pins: list[Any]


@dataclass
class Plan:
    """What a script asks for: (output, input) pairs, and nets."""

    pairs: list[tuple[Any, Any]]
    nets: list[NetPlan]

    def __len__(self) -> int:
        return len(self.pairs) + len(self.nets)


_TOKEN = re.compile(
    r"""\s*(?:
        (?P<arrow>->)
      | (?P<quoted>"[^"]*")
      | (?P<kind><[^<>]*>)
      | (?P<name>[A-Za-z0-9_*?+\-]+)
      | (?P<punct>[.,/:])
      | (?P<bad>\S)
    )""",
    re.VERBOSE,
)


def _tokens(text: str, line: int) -> list[tuple[str, str]]:
    out, i, glued = [], 0, -1  # glued: where the last name ended
    while i < len(text):
        m = _TOKEN.match(text, i)
        if m is None or m.end() == i:  # (only trailing space left)
            break
        i = m.end()
        kind = m.lastgroup
        value = m.group(kind)
        start = m.start(kind)
        if kind == "bad":
            raise NetlistError(line, f"unexpected {value!r}")
        if kind == "quoted":
            kind, value = "name", value[1:-1]
        elif kind == "kind":
            value = value[1:-1].strip()
        elif kind == "name" and value.endswith("-") and text.startswith(">", i):
            # "a->b": the name doesn't swallow the arrow's '-'
            kind, value, i = "name", value[:-1], i - 1
        elif kind == "punct":
            kind = value
        if kind == "name" and start == glued:  # ("my reg"* is one name)
            out[-1] = ("name", out[-1][1] + value)
        else:
            out.append((kind, value))
        glued = i if kind == "name" else -1
    return out


def statements(script: str) -> list[tuple[int, str]]:
    """(line number, text) per statement, comments and blanks gone."""
    out = []
    for n, raw in enumerate(script.splitlines(), 1):
        text = _strip_comment(raw)
        for part in text.split(";"):
            if part.strip():
                out.append((n, part.strip()))
    return out


def _strip_comment(raw: str) -> str:
    quoted = False
    for i, ch in enumerate(raw):
        if ch == '"':
            quoted = not quoted
        elif ch == "#" and not quoted:
            return raw[:i]
    return raw


def parse(script: str) -> list[Connect | Chain | NetStatement]:
    return [_statement(n, text) for n, text in statements(script)]


class _Reader:
    def __init__(self, toks: list[tuple[str, str]], line: int) -> None:
        self.toks, self.i, self.line = toks, 0, line

    def peek(self) -> str | None:
        return self.toks[self.i][0] if self.i < len(self.toks) else None

    def take(self, kind: str, what: str) -> str:
        if self.peek() != kind:
            got = self.toks[self.i][1] if self.i < len(self.toks) else "the end of the line"
            raise NetlistError(self.line, f"expected {what}, got {got!r}")
        value = self.toks[self.i][1]
        self.i += 1
        return value

    def done(self) -> None:
        if self.i < len(self.toks):
            raise NetlistError(self.line, f"unexpected {self.toks[self.i][1]!r}")


def _statement(line: int, text: str) -> Connect | Chain | NetStatement:
    r = _Reader(_tokens(text, line), line)
    t = r.toks
    if t[:1] in ([("name", "net")], [("name", "bus")]) and (
        t[1:2] == [(":", ":")] or (t[1:2] and t[1][0] == "name" and t[2:3] == [(":", ":")])
    ):
        kind = t[0][1]
        name = t[1][1] if t[1][0] == "name" else None
        r.i = 3 if name is not None else 2
        ends = _ends(r)
        r.done()
        return NetStatement(line, kind, name, ends)
    if r.toks[:1] == [("name", "chain")] and len(r.toks) > 1 and r.toks[1][0] != ".":
        r.i = 1
        parts = _part(r, allow_box=True)
        r.take(":", "':' after the parts to chain")
        out = r.take("name", "the pin to chain from")
        r.take("arrow", "'->'")
        into = r.take("name", "the pin to chain into")
        r.done()
        return Chain(line, parts, out, into)
    left = _ends(r)
    r.take("arrow", "'->'")
    right = _ends(r)
    r.done()
    return Connect(line, left, right)


def _ends(r: _Reader) -> tuple[End, ...]:
    ends = [_end(r)]
    while r.peek() == ",":
        r.i += 1
        ends.append(_end(r))
    return tuple(ends)


def _end(r: _Reader) -> End:
    part = _part(r, allow_box=True)
    r.take(".", "'.' and a pin name")
    pin = r.take("name", "a pin name")
    return End(part.box, part.kind, part.label, pin)


def _part(r: _Reader, allow_box: bool) -> End:
    box = kind = label = None
    if allow_box and r.peek() == "name" and r.toks[r.i + 1 : r.i + 2] == [("/", "/")]:
        box = r.take("name", "a box name")
        r.i += 1
    if r.peek() == "kind":
        kind = r.take("kind", "a kind")
    if r.peek() == "name":
        label = r.take("name", "a part name")
    if kind is None and label is None:
        r.take("name", "a part name")  # (raises, saying so)
    return End(box, kind, label, "")


# ---- resolving ----------------------------------------------------------------------


@dataclass
class Scope:
    """What a script sees. `parts`: the box's parts. `boxes(label)`: another box's
    parts (raise KeyError, with a message, when there's no such box or several).
    `pos(part)`: (x, y), for ordering parts that share a name. `wired(pin)`: an
    input that already has a wire. `board()`: every part on the board, where names
    that match nothing in the box are looked for (default: just the box's)."""

    parts: list[Any]
    boxes: Callable[[str], list[Any]]
    pos: Callable[[Any], tuple[float, float]]
    wired: Callable[[Any], bool]
    board: Callable[[], list[Any]] | None = None
    _sorted: dict = field(default_factory=dict)

    def sorted(self, box: str | None | tuple, line: int) -> list[Any]:
        """`box`: None for this box, BOARD for the whole board, or another box's label."""
        if box not in self._sorted:
            if box is None:
                parts = self.parts
            elif box is BOARD:
                parts = self.board() if self.board is not None else self.parts
            else:
                try:
                    parts = self.boxes(box)
                except KeyError as e:
                    raise NetlistError(line, e.args[0] if e.args else f"no box {box!r}")
            self._sorted[box] = sorted(parts, key=self._order)
        return self._sorted[box]

    def _order(self, part) -> tuple:
        x, y = self.pos(part)
        return (_natural(name_of(part)), -y, x)


BOARD = ("board",)  # (Scope.sorted: not a label, which is a str)


def name_of(part) -> str:
    return part.label or kind_of(part)


def kind_of(part) -> str:
    k = part.kind
    if k.startswith(MACRO):  # (a macro: what it's called, not its id)
        return getattr(part.type, "title", None) or k[len(MACRO) :]
    return k


def _natural(s: str) -> tuple:
    return tuple(int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", s))


def _match(name: str, pattern: str) -> bool:
    return fnmatchcase(name.casefold(), pattern.casefold())


def _parts(scope: Scope, end: End, line: int) -> list[Any]:
    """The parts `end` names: in its box; for a name without box/, in this box, or on
    the whole board if nothing in the box matches."""
    for where in _wheres(end):
        found = _parts_in(scope, end, where, line)
        if found:
            return found
    raise _no_part(end, line)


def _wheres(end: End) -> tuple:
    return (end.box,) if end.box is not None else (None, BOARD)


def _parts_in(scope: Scope, end: End, where, line: int) -> list[Any]:
    return [
        p
        for p in scope.sorted(where, line)
        if (end.label is None or _match(name_of(p), end.label))
        and (end.kind is None or _match(kind_of(p), end.kind))
    ]


def _no_part(end: End, line: int) -> NetlistError:
    where = f"in the box {end.box!r}" if end.box is not None else "in the box or on the board"
    return NetlistError(line, f"no part matches {str(end).rsplit('.', 1)[0]!r} {where}")


def _pin_names(part) -> list[tuple[Any, str, str]]:
    """(pin, its name, its default name) for every pin: in1, in2, ... and out1, ...
    count from the top, for pins that have no name worth typing."""
    lay = part.layout
    return [
        (pin, name, f"{side}{i}")
        for side, names, pins in (("in", lay.ins, part.inputs), ("out", lay.outs, part.outputs))
        for i, (name, pin) in enumerate(zip(names, pins), 1)
    ]


def _pins(part, pattern: str) -> list[Any]:
    return [pin for pin, name, alias in _pin_names(part) if _match(name, pattern) or _match(alias, pattern)]


def _no_pin(part, pattern: str, line: int, n: int = 0) -> NetlistError:
    have = ", ".join(
        alias if name == alias or name.isdigit() else f"{name} ({alias})"
        for _, name, alias in _pin_names(part)
    )
    what = "no pin of {} matches" if not n else f"{n} pins of {{}} match"
    return NetlistError(
        line, f"{what.format(name_of(part))} {pattern!r}: it has {have or 'no pins'}"
    )


def _expand(scope: Scope, ends: Iterable[End], line: int) -> list[Any]:
    out = []
    for end in ends:
        # (in the box, then on the board: the first place with matching pins, not just
        # parts -- "*.clk" shouldn't stop at a box holding only the clock)
        first = None
        for where in _wheres(end):
            parts = _parts_in(scope, end, where, line)
            pins = [pin for p in parts for pin in _pins(p, end.pin)]
            if pins:
                break
            first = first or parts
        else:
            if not first:
                raise _no_part(end, line)
            raise _no_pin(first[0], end.pin, line)
        out += pins
    return out


def pin_name(pin) -> str:
    lay = pin.part.layout
    names = lay.ins if pin.is_input else lay.outs
    return f"{name_of(pin.part)}.{names[pin.index]}"


def resolve(script: str, scope: Scope) -> Plan:
    """The (output pin, input pin) pairs and the nets the script asks for, checked
    against each other and the board: everything or nothing. NetlistError on the
    first problem."""
    pairs: list[tuple[Any, Any, int]] = []
    nets: list[NetPlan] = []
    for st in parse(script):
        if isinstance(st, NetStatement):
            pins = list(dict.fromkeys(_expand(scope, st.ends, st.line)))
            nets.append(NetPlan(st.kind, st.name, st.line, pins))
            continue
        if isinstance(st, Chain):
            parts = _parts(scope, st.parts, st.line)
            if len(parts) < 2:
                raise NetlistError(st.line, "a chain needs at least two parts")
            for a, b in zip(parts, parts[1:]):
                pairs.append((_one(a, st.out, st.line), _one(b, st.into, st.line), st.line))
            continue
        left = _expand(scope, st.left, st.line)
        right = _expand(scope, st.right, st.line)
        if len(left) == 1:
            left = left * len(right)
        elif len(right) == 1:
            right = right * len(left)
        elif len(left) != len(right):
            raise NetlistError(
                st.line,
                f"{len(left)} pins on the left, {len(right)} on the right: "
                "give one side a single pin, or both the same number",
            )
        pairs += [(a, b, st.line) for a, b in zip(left, right)]
    fed: dict = {}  # input -> the line that gives it a new wire
    return Plan(_check(pairs, scope, fed), [_check_net(n, scope, fed) for n in nets])


def _one(part, pattern: str, line: int):
    pins = _pins(part, pattern)
    if len(pins) != 1:
        raise _no_pin(part, pattern, line, len(pins))
    return pins[0]


def _check(pairs: list[tuple[Any, Any, int]], scope: Scope, fed: dict) -> list[tuple[Any, Any]]:
    out, seen = [], set()
    for a, b, line in pairs:
        if a.is_input and not b.is_input:
            a, b = b, a
        what = f"{pin_name(a)} -> {pin_name(b)}"
        if a.is_input == b.is_input:
            side = "inputs" if a.is_input else "outputs"
            raise NetlistError(line, f"{what}: both are {side}")
        if a.part is b.part:
            raise NetlistError(line, f"{what}: a part can't be wired to itself")
        if a.width != b.width:
            raise NetlistError(line, f"{what}: {a.width} lanes into {b.width}")
        if (a, b) in seen:
            continue  # (asked for twice: once is plenty)
        if b in fed:
            raise NetlistError(line, f"{pin_name(b)} is fed twice (also on line {fed[b]})")
        if scope.wired(b):
            raise NetlistError(line, f"{pin_name(b)} is wired already: delete that wire first")
        seen.add((a, b))
        fed[b] = line
        out.append((a, b))
    return out


def _check_net(net: NetPlan, scope: Scope, fed: dict) -> NetPlan:
    what = f"{net.kind} {net.name}" if net.name else f"this {net.kind}"
    if len(net.pins) < 2:
        raise NetlistError(net.line, f"{what} has only {pin_name(net.pins[0])}: a net needs two pins")
    widths = {p.width for p in net.pins}
    if len(widths) > 1:
        by = ", ".join(f"{pin_name(p)} {p.width}" for p in net.pins[:6])
        raise NetlistError(net.line, f"{what} mixes widths ({by}{', ...' if len(net.pins) > 6 else ''})")
    for p in net.pins:
        # (a wired pin joins through its wires: only an unwired input gets a new one)
        if p.is_input and not scope.wired(p):
            if p in fed:
                raise NetlistError(net.line, f"{pin_name(p)} is fed twice (also on line {fed[p]})")
            fed[p] = net.line
    return net
