"""Pure logic simulation. No pygame/pyglet/UI imports allowed in here.

Model:
  - A Part has input pins and output pins.
  - A Wire joins two endpoints. An endpoint is a Pin, or another Wire (a
    junction / branch: "attached somewhere along that wire").
  - Everything joined by wires forms a *net*. A net's output pins drive it,
    its input pins read it:
        no drivers        -> reads 0   (future Z: floating)
        drivers agree     -> that value
        drivers disagree  -> CONFLICT: reads 0 for now, flagged so the UI can
                             show it (future X). See Circuit.step.
  - Circuit.step() advances time by one tick: every part computes its outputs
    from its *current* inputs, then every net carries its value to its readers.
    So each gate costs one tick of delay, and feedback loops (latches) work
    without infinite recursion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

# kind -> (n_inputs, n_outputs, logic function or None)
# A logic function takes a list of input bools and returns a list of output bools.
LogicFn = Callable[[list[bool]], list[bool]]

BUILTINS: dict[str, tuple[int, int, LogicFn | None]] = {
    "IN":   (0, 1, None),  # switch: output is driven by the user, see Part.toggle()
    "OUT":  (1, 0, None),  # LED: just displays its input
    "NAND": (2, 1, lambda i: [not (i[0] and i[1])]),
    "AND":  (2, 1, lambda i: [i[0] and i[1]]),
    "OR":   (2, 1, lambda i: [i[0] or i[1]]),
    "NOT":  (1, 1, lambda i: [not i[0]]),
}


@dataclass(eq=False)
class Pin:
    part: Part
    index: int
    is_input: bool
    state: bool = False

    def __repr__(self) -> str:
        side = "in" if self.is_input else "out"
        return f"<Pin {self.part.kind}.{side}{self.index}={int(self.state)}>"


@dataclass(eq=False)
class Part:
    kind: str
    inputs: list[Pin] = field(default_factory=list)
    outputs: list[Pin] = field(default_factory=list)
    # User-given name. Lives in the model (not the UI) because it's circuit data:
    # when a board is saved as a macro, IN/OUT labels become its pin names.
    label: str = ""
    # Stable identity within a circuit. Survives undo/redo (a part deleted and
    # restored comes back with the same uid) and is what snapshots/save files use
    # to refer to parts, since Python object identity doesn't survive either.
    uid: int = 0

    @property
    def pins(self) -> list[Pin]:
        return self.inputs + self.outputs

    def toggle(self) -> None:
        """Only meaningful for IN switches."""
        if self.kind == "IN":
            self.outputs[0].state = not self.outputs[0].state


Endpoint = Union[Pin, "Wire"]


@dataclass(eq=False)
class Wire:
    # Two ends, each a Pin or another Wire. Normalized by Circuit.connect: an
    # output pin is always `src`, an input pin always `dst` -- so a plain
    # pin-to-pin wire reads src=output, dst=input like before junctions existed.
    src: Endpoint
    dst: Endpoint
    uid: int = 0  # stable identity, like Part.uid (wires can be endpoints of wires)

    @property
    def ends(self) -> tuple[Endpoint, Endpoint]:
        return self.src, self.dst


class Circuit:
    def __init__(self) -> None:
        self.parts: list[Part] = []
        self.wires: list[Wire] = []  # creation order: a wire always comes after the wires it attaches to
        self._next_uid = 1
        self._next_wire_uid = 1
        # Nets are derived from the wiring and cached until the wiring changes.
        self._nets_dirty = True
        self._nets: list[tuple[list[Pin], list[Pin]]] = []  # (drivers, readers) per net
        self._net_of_wire: dict[Wire, int] = {}
        self.net_value: list[bool] = []
        self.net_conflict: list[bool] = []

    # ---- editing -------------------------------------------------------

    def add_part(self, kind: str, uid: int | None = None) -> Part:
        """`uid` recreates a specific part (undo, loading); normally leave it None."""
        n_in, n_out, _ = BUILTINS[kind]
        if uid is None:
            uid = self._next_uid
        self._next_uid = max(self._next_uid, uid + 1)
        part = Part(kind, uid=uid)
        part.inputs = [Pin(part, i, True) for i in range(n_in)]
        part.outputs = [Pin(part, i, False) for i in range(n_out)]
        self.parts.append(part)
        return part

    def remove_part(self, part: Part) -> list[Wire]:
        """Removes the part, every wire touching it, and every wire hanging off
        those. Returns all removed wires."""
        removed: list[Wire] = []
        for w in [w for w in self.wires if any(isinstance(e, Pin) and e.part is part for e in w.ends)]:
            if w in self.wires:  # may already be gone as a branch of an earlier one
                removed += self.remove_wire(w)
        self.parts.remove(part)
        self._nets_dirty = True
        return removed

    def can_connect(self, a: Endpoint, b: Endpoint) -> bool:
        if a is b:
            return False
        if isinstance(a, Pin) and isinstance(b, Pin):
            return a.is_input != b.is_input and a.part is not b.part
        if isinstance(b, Pin):
            a, b = b, a
        if isinstance(a, Pin):  # pin + wire
            if a in b.ends:
                return False  # already attached right there
            if a.is_input:
                # Wiring into an input replaces its current wire (and that wire's
                # branches). Attaching to one of those would saw off our own branch.
                doomed = set()
                for w in self.wires_at(a):
                    doomed |= {w, *self.descendants(w)}
                return b not in doomed
        return True  # wire + wire: joins two nets

    def connect(self, a: Endpoint, b: Endpoint, uid: int | None = None) -> tuple[Wire | None, list[Wire]]:
        """Connect two endpoints in either order. Returns (new_wire, replaced_wires).

        new_wire is None if the connection is invalid (see can_connect). An input
        pin takes one wire, so wiring into an already-wired input replaces the old
        wire, along with any branches hanging off it.
        """
        if not self.can_connect(a, b):
            return None, []
        # outputs are src, inputs are dst (see Wire)
        if (isinstance(b, Pin) and not b.is_input) or (isinstance(a, Pin) and a.is_input):
            a, b = b, a
        replaced: list[Wire] = []
        for end in (a, b):
            if isinstance(end, Pin) and end.is_input:
                for old in self.wires_at(end):
                    replaced += self.remove_wire(old)
        if uid is None:
            uid = self._next_wire_uid
        self._next_wire_uid = max(self._next_wire_uid, uid + 1)
        wire = Wire(a, b, uid)
        self.wires.append(wire)
        self._nets_dirty = True
        return wire, replaced

    def remove_wire(self, wire: Wire) -> list[Wire]:
        """Removes the wire and everything attached to it. Returns them, parents first."""
        removed = [wire, *self.descendants(wire)]
        dead = set(removed)
        self.wires = [w for w in self.wires if w not in dead]
        self._nets_dirty = True
        return removed

    def merge(self, keep: Wire, absorb: Wire) -> None:
        """Splice `absorb` (a wire attached to `keep`) onto keep's far end.

        keep.dst becomes absorb's other end, everything attached to absorb
        re-attaches to keep, and absorb disappears. keep's old dst is simply
        dropped -- the caller cut it off. Used by cut-deletion (see the editor):
        "delete from the junction onward" turns trunk + branch into one wire.

        Order invariants survive: keep was created before absorb (absorb attaches
        to it), so everything that now attaches to keep still comes after it.
        """
        far = absorb.dst if absorb.src is keep else absorb.src
        keep.dst = far
        for w in self.wires:
            if w.src is absorb:
                w.src = keep
            if w.dst is absorb:
                w.dst = keep
        self.wires.remove(absorb)
        self._nets_dirty = True

    def attachments(self, wire: Wire) -> list[Wire]:
        """Wires with an end on `wire` (branches, stubs, extra drivers)."""
        return [w for w in self.wires if wire in w.ends]

    def wires_at(self, pin: Pin) -> list[Wire]:
        return [w for w in self.wires if pin in w.ends]

    def descendants(self, wire: Wire) -> list[Wire]:
        """Wires attached to `wire`, wires attached to those, and so on (in creation order)."""
        found = {wire}
        for w in self.wires:  # creation order means parents are seen before children
            if w.src in found or w.dst in found:
                found.add(w)
        found.discard(wire)
        return [w for w in self.wires if w in found]

    # ---- nets -------------------------------------------------------------

    def _rebuild_nets(self) -> None:
        """Group pins and wires into nets (union-find over identities)."""
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for w in self.wires:
            for end in w.ends:
                parent[find(id(end))] = find(id(w))

        index: dict[int, int] = {}
        self._nets = []
        for w in self.wires:
            root = find(id(w))
            if root not in index:
                index[root] = len(self._nets)
                self._nets.append(([], []))
        self._net_of_wire = {w: index[find(id(w))] for w in self.wires}
        for part in self.parts:
            for pin in part.pins:
                if id(pin) in parent:
                    drivers, readers = self._nets[index[find(id(pin))]]
                    (readers if pin.is_input else drivers).append(pin)
                elif pin.is_input:
                    pin.state = False  # unconnected input reads 0
        self.net_value = [False] * len(self._nets)
        self.net_conflict = [False] * len(self._nets)
        self._nets_dirty = False

    def wire_state(self, wire: Wire) -> tuple[bool, bool]:
        """(value, conflict) of the net this wire belongs to, as of the last step."""
        if self._nets_dirty:
            self._rebuild_nets()
        i = self._net_of_wire[wire]
        return self.net_value[i], self.net_conflict[i]

    # ---- simulation ----------------------------------------------------

    def step(self) -> None:
        if self._nets_dirty:
            self._rebuild_nets()

        # Phase 1: every part computes outputs from current inputs.
        # Compute all first, then write, so evaluation order doesn't matter.
        results: list[tuple[Part, list[bool]]] = []
        for part in self.parts:
            fn = BUILTINS[part.kind][2]
            if fn is not None:
                results.append((part, fn([p.state for p in part.inputs])))
        for part, outs in results:
            for pin, value in zip(part.outputs, outs):
                pin.state = value

        # Phase 2: every net resolves its drivers and hands the value to its readers.
        for i, (drivers, readers) in enumerate(self._nets):
            if not drivers:
                value, conflict = False, False  # floating (future: Z)
            else:
                value = drivers[0].state
                conflict = any(d.state != value for d in drivers[1:])
                if conflict:
                    value = False  # placeholder until 4-state logic: X reads as 0
            self.net_value[i] = value
            self.net_conflict[i] = conflict
            for pin in readers:
                pin.state = value
