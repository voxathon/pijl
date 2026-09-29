"""Pure logic simulation. No pygame/pyglet/UI imports allowed in here.

Model:
  - A Chip has input pins and output pins.
  - A Wire copies the state of one output pin onto one input pin.
  - Circuit.step() advances time by one tick: every chip computes its outputs
    from its *current* inputs, then every wire carries those outputs along.
    So each gate costs one tick of delay, and feedback loops (latches) work
    without infinite recursion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# kind -> (n_inputs, n_outputs, logic function or None)
# A logic function takes a list of input bools and returns a list of output bools.
LogicFn = Callable[[list[bool]], list[bool]]

BUILTINS: dict[str, tuple[int, int, LogicFn | None]] = {
    "IN":   (0, 1, None),  # switch: output is driven by the user, see Chip.toggle()
    "OUT":  (1, 0, None),  # LED: just displays its input
    "NAND": (2, 1, lambda i: [not (i[0] and i[1])]),
    "AND":  (2, 1, lambda i: [i[0] and i[1]]),
    "OR":   (2, 1, lambda i: [i[0] or i[1]]),
    "NOT":  (1, 1, lambda i: [not i[0]]),
}


@dataclass(eq=False)
class Pin:
    chip: Chip
    index: int
    is_input: bool
    state: bool = False

    def __repr__(self) -> str:
        side = "in" if self.is_input else "out"
        return f"<Pin {self.chip.kind}.{side}{self.index}={int(self.state)}>"


@dataclass(eq=False)
class Chip:
    kind: str
    inputs: list[Pin] = field(default_factory=list)
    outputs: list[Pin] = field(default_factory=list)
    # User-given name. Lives in the model (not the UI) because it's circuit data:
    # when a board is packaged into a custom chip, IN/OUT labels become its pin names.
    label: str = ""
    # Stable identity within a circuit. Survives undo/redo (a chip deleted and
    # restored comes back with the same uid) and is what snapshots/save files use
    # to refer to chips, since Python object identity doesn't survive either.
    uid: int = 0

    @property
    def pins(self) -> list[Pin]:
        return self.inputs + self.outputs

    def toggle(self) -> None:
        """Only meaningful for IN switches."""
        if self.kind == "IN":
            self.outputs[0].state = not self.outputs[0].state


@dataclass(eq=False)
class Wire:
    src: Pin  # always an output pin
    dst: Pin  # always an input pin


class Circuit:
    def __init__(self) -> None:
        self.chips: list[Chip] = []
        self.wires: list[Wire] = []
        self._next_uid = 1

    # ---- editing -------------------------------------------------------

    def add_chip(self, kind: str, uid: int | None = None) -> Chip:
        """`uid` recreates a specific chip (undo, loading); normally leave it None."""
        n_in, n_out, _ = BUILTINS[kind]
        if uid is None:
            uid = self._next_uid
        self._next_uid = max(self._next_uid, uid + 1)
        chip = Chip(kind, uid=uid)
        chip.inputs = [Pin(chip, i, True) for i in range(n_in)]
        chip.outputs = [Pin(chip, i, False) for i in range(n_out)]
        self.chips.append(chip)
        return chip

    def remove_chip(self, chip: Chip) -> list[Wire]:
        """Removes the chip and every wire touching it. Returns removed wires."""
        pins = set(map(id, chip.pins))
        dead = [w for w in self.wires if id(w.src) in pins or id(w.dst) in pins]
        for w in dead:
            self.remove_wire(w)
        self.chips.remove(chip)
        return dead

    def connect(self, a: Pin, b: Pin) -> tuple[Wire | None, Wire | None]:
        """Connect two pins in either order. Returns (new_wire, replaced_wire).

        new_wire is None if the connection is invalid (in->in, out->out,
        same chip). An input pin can only have one driver, so wiring into an
        already-driven input replaces the old wire.
        """
        if a.is_input == b.is_input or a.chip is b.chip:
            return None, None
        src, dst = (b, a) if a.is_input else (a, b)
        replaced = self.wire_into(dst)
        if replaced is not None:
            self.remove_wire(replaced)
        wire = Wire(src, dst)
        self.wires.append(wire)
        return wire, replaced

    def remove_wire(self, wire: Wire) -> None:
        self.wires.remove(wire)
        wire.dst.state = False

    def wire_into(self, pin: Pin) -> Wire | None:
        return next((w for w in self.wires if w.dst is pin), None)

    # ---- simulation ----------------------------------------------------

    def step(self) -> None:
        # Phase 1: every chip computes outputs from current inputs.
        # Compute all first, then write, so evaluation order doesn't matter.
        results: list[tuple[Chip, list[bool]]] = []
        for chip in self.chips:
            fn = BUILTINS[chip.kind][2]
            if fn is not None:
                results.append((chip, fn([p.state for p in chip.inputs])))
        for chip, outs in results:
            for pin, value in zip(chip.outputs, outs):
                pin.state = value

        # Phase 2: wires carry output states onto input pins.
        for w in self.wires:
            w.dst.state = w.src.state
