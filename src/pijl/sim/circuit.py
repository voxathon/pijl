"""Pure logic simulation. No pygame/pyglet/UI imports allowed in here.

Model:
  - A Part has input pins and output pins. What it does is up to its PartType
    (see pijl.parts): the circuit only knows the contract, never specific kinds.
  - A Wire joins two endpoints. An endpoint is a Pin, or another Wire (a
    junction / branch: "attached somewhere along that wire").
  - Values are four-state: 0, 1, X (unknown) and Z (not driven); see pijl.logic.
  - Everything joined by wires forms a *net*. A net's output pins drive it,
    its input pins read it:
        no drivers        -> Z (floating; a gate reads it as X)
        drivers agree     -> that value (Z drivers don't count)
        drivers disagree  -> X, and flagged as a conflict so the UI can show it
    Weak outputs (PartType.weak: pull-ups, pull-downs) only count when the strong
    drivers leave the net at Z. Of those, only the ones with the highest
    props["priority"] on the net count; if they disagree, that's a conflict too.
  - New parts power up with their outputs at X (parts without an eval, like the
    IN switch, at 0) -- or, with settling (below), at random 0s and 1s.
  - Circuit.step() advances time by one tick: every part computes its outputs
    from its *current* inputs, then every net carries its value to its readers.
    So each gate costs one tick of delay, and feedback loops (latches) work
    without infinite recursion. Parts are evaluated a whole kind at a time, on
    arrays (PartType.eval), which is what lets big boards go fast later. Which code
    does that is the engine config's choice, made when the circuit is built
    (sim/config.py: the steppers in sim/plain.py and sim/dirty.py); every choice
    gives the same result, tick for tick.
  - Parts are *live* once placed. Ghosts (following the cursor before a click)
    aren't: they're never opened, and only pure parts among them are evaluated.
  - Joined pins (PartType.joins) are one net straight through their part: they
    show the net's value and never drive it. A joined output's value drives that
    net from a hidden pin (Part.drives), so an inline pull-up is one net with a
    weak driver on it, not two nets with a gate in between.
  - A macro instance is never evaluated: its body is added as *hidden* parts and
    wires (owned by the instance, not in `parts` / `wires`, which are only what's
    on the board), recursively. Its pins are joined straight into the nets of the
    body's IN / OUT ports, so the inside and outside of a macro are one net: no
    delay at the boundary, and wrapping something in a macro can't change timing.
  - Settling (`settle_ticks`): new parts are a little random for a while. A part
    powers up (opens) with random 0s and 1s on its outputs, and for its first
    ticks it only takes its newly computed outputs about half the time. Without
    it, a latch that appears all at once -- opened, pasted or placed inside a
    macro -- stays X forever (nothing ever tells it which way to fall), and a
    symmetric one started at 0s and 1s flips between both unstable states
    forever. Real hardware settles on noise; this is the noise. Only the power-on
    value is random: an X with a real cause (a floating input, a fight) stays X
    and doesn't flicker. Seeded, so the same actions give the same results. Off
    (0) unless asked for.

Speed: every pin's state lives in one numpy array of logic codes (Pin.state reads
and writes its slot), so a step is array work, not a loop over pins. Each part kind
keeps index arrays of its instances' input and output pins; nets are index arrays of
their drivers (a net with one just copies it; the rest are sorted by net and
OR-reduced, see logic.resolve) and of their readers. Wires have slots too, with their ends kept in arrays as wiring changes, so finding the nets
again (scipy's connected components) doesn't loop over wires in Python either.
What changed for the UI (take_changes) is a diff of the state arrays against what
it was shown last.

Buses: a pin can be several lanes wide (PartType.widths). A pin of width w is w
pin slots in a row (its first, the "head", is what a Pin handle stands for), and a
wire of width w is w wire slots in a row, lane i joining lane i of each end. So a
bus is just w ordinary nets: nothing past the wiring knows about widths, except
eval, which gets a wide pin's lanes as one (n, w) array. Wires only join ends of
the same width (can_connect). A part's pin layout -- its widths -- is its *shape*;
instances of one kind may differ (a width read from props), and are evaluated a
shape at a time.

Parts and pins are tables too: a part is a row by part slot (type, uid, owner, its
first pin slot, ...; see Circuit._new_rows), a pin a row of _PinStates. Part and
Pin objects are handles on rows, made when something asks for one and then kept
(one per slot, so `is` works). The editor asks for every part on the board; the
parts inside macros mostly never get one. A macro type's first instance is built
part by part and recorded (_Blueprint); later instances are stamped from that, a
few array writes for the whole nested body.
"""

from __future__ import annotations

import time
import traceback
import copy
from dataclasses import dataclass
from types import MappingProxyType, MethodType
from collections.abc import Sequence
from typing import Any, Callable, Iterable, Mapping, Union

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..logic import CODE, ONE, X, Z, ZERO, Level, Logic, codes, fights, resolve
from ..parts import MAX_WIDTH, Ctx, Layout, PartType, Registry, builtin_registry, fresh_props, layout_of
from ..parts.registry import copy_props, flat, layout_props
from .config import EngineConfig
from .config import default as default_config

_FAILED = object()  # what Circuit._guard returns when the hook raised


class _Drop:
    """Where pokes go when the stepper doesn't want them."""

    __slots__ = ()

    def append(self, _) -> None:
        pass

class _PinStates:
    """Every pin of a circuit, by slot: its state (a logic code), whether it reads its
    net, whether it's a weak driver, and what it is (its part's slot, its index, its
    side, whether it's passive). A Pin object is a handle on one slot, made when
    something asks for it (see handle): the pins inside macros mostly never get one."""

    def __init__(self, circuit: Circuit) -> None:
        self.circuit = circuit
        self.n = 0  # slots handed out (never reused)
        self.states = np.zeros(1024, CODE)
        self.reader = np.zeros(
            1024, bool
        )  # an input or a pass-through pin: reads its net, never drives
        self.alive = np.zeros(1024, bool)  # its part is still in the circuit
        self.weak = np.zeros(1024, bool)  # a weak output (see PartType.weak)
        self.part = np.zeros(1024, np.int32)  # its part's slot
        self.index = np.zeros(1024, np.int32)  # its index on its side
        self.is_input = np.zeros(1024, bool)
        self.passive = np.zeros(1024, bool)  # see Pin.passive
        self.lane = np.zeros(1024, np.int16)  # its lane in its pin (0: the head)
        self.width = np.ones(1024, np.int16)  # its pin's lanes (unused slots: 1)
        # Pin handles by (head) slot, made on first use; None again once the pin is gone
        self.pins: list[Pin | None] = []

    _COLS = (
        "states", "reader", "alive", "weak", "part", "index", "is_input", "passive", "lane", "width",
    )  # fmt: skip

    def new(
        self, part: int, index: int, is_input: bool, initial: Level = Z, weak: bool = False,
        width: int = 1,
    ) -> int:
        """One more pin, `width` lanes (no handle yet); its (head) slot."""
        slot = self.n
        end = slot + width
        self._room(end)
        self.n = end
        if width == 1:
            self.pins.append(None)
            self.states[slot], self.reader[slot], self.alive[slot] = initial, is_input, True
            self.weak[slot], self.passive[slot] = weak, False
            self.part[slot], self.index[slot], self.is_input[slot] = part, index, is_input
            return slot  # (lane 0 of 1: what _room left there)
        self.pins.extend([None] * width)
        self.states[slot:end] = initial
        self.reader[slot:end] = is_input
        self.alive[slot:end] = True
        self.weak[slot:end] = weak
        self.part[slot:end], self.index[slot:end], self.is_input[slot:end] = part, index, is_input
        self.passive[slot:end] = False
        self.lane[slot:end] = np.arange(width)
        self.width[slot:end] = width
        return slot

    def new_block(
        self,
        part: np.ndarray,
        index: np.ndarray,
        is_input: np.ndarray,
        initial: np.ndarray,
        weak: np.ndarray,
        reader: np.ndarray | None = None,
        passive: np.ndarray | None = None,
        lane: np.ndarray | None = None,
        width: np.ndarray | None = None,
    ) -> None:
        """new() for many pin slots at once: the next len(part) slots, in order, lane by
        lane. (`reader`, `passive`, `lane`, `width`: when they aren't just is_input,
        False, 0 and 1.)"""
        first, k = self.n, len(part)
        end = first + k
        self._room(end)
        self.n = end
        self.pins.extend([None] * k)
        self.states[first:end] = initial
        self.reader[first:end] = is_input if reader is None else reader
        self.alive[first:end] = True
        self.weak[first:end] = weak
        self.part[first:end] = part
        self.index[first:end] = index
        self.is_input[first:end] = is_input
        self.passive[first:end] = False if passive is None else passive
        self.lane[first:end] = 0 if lane is None else lane
        self.width[first:end] = 1 if width is None else width

    def handle(self, slot: int) -> Pin:
        pin = self.pins[slot]
        if pin is None:
            pin = self.pins[slot] = Pin.__new__(Pin)
            pin._store, pin.slot, pin._part = self, slot, None
            pin.index, pin.is_input = int(self.index[slot]), bool(self.is_input[slot])
            pin.width = int(self.width[slot])
        return pin

    def kill(self, slots: list[int]) -> None:
        """These pin slots' parts are gone (every lane: give them all)."""
        self.alive[slots] = False
        for slot in slots:
            self.pins[slot] = None

    def lanes(self, head: int) -> np.ndarray:
        """A pin's lane slots, by its head."""
        return np.arange(head, head + int(self.width[head]))

    def _room(self, n: int) -> None:
        """Arrays long enough for n pins. (New slots: lane 0 of a one-lane pin.)"""
        while n > len(self.states):
            for name in self._COLS:
                old = getattr(self, name)
                more = np.ones_like(old) if name == "width" else np.zeros_like(old)
                setattr(self, name, np.concatenate((old, more)))


class _WireSlots:
    """Every wire of a circuit (board and hidden), by slot: its two ends, as pin slots
    or wire slots, its uid, and whether it's still there. Board wires are Wire objects
    from the start; a hidden one (inside a macro) gets one only if something asks."""

    def __init__(self, pins: _PinStates) -> None:
        self.pins = pins
        self.wires: list[Wire | None] = []  # by slot; None: gone, or hidden and not asked for
        self.alive = np.zeros(256, bool)
        self.board = np.zeros(256, bool)  # on the board (not inside a macro)
        self.end_is_wire = np.zeros((256, 2), bool)
        self.end_slot = np.zeros((256, 2), np.int32)
        self.uid = np.zeros(256, np.int64)
        self.lane = np.zeros(256, np.int16)  # its lane in its wire (0: the head)
        self.width = np.ones(256, np.int16)  # its wire's lanes (unused slots: 1)

    _COLS = ("alive", "board", "end_is_wire", "end_slot", "uid", "lane", "width")

    def _room(self, end: int) -> None:
        """Arrays long enough for `end` slots. (New slots: lane 0 of a one-lane wire.)"""
        while end > len(self.alive):
            for name in self._COLS:
                old = getattr(self, name)
                more = np.ones_like(old) if name == "width" else np.zeros_like(old)
                setattr(self, name, np.concatenate((old, more)))

    def _new_slots(self, width: int) -> int:
        """`width` slots in a row, for one wire; the first (its head)."""
        slot = len(self.wires)
        end = slot + width
        if end > len(self.alive):
            self._room(end)
        if width == 1:
            self.wires.append(None)
            self.alive[slot] = True  # (lane 0 of 1: what _room left there)
            return slot
        self.wires.extend([None] * width)
        self.alive[slot:end] = True
        self.lane[slot:end] = np.arange(width)
        self.width[slot:end] = width
        return slot

    def add(self, wire: Wire, board: bool) -> None:
        w = wire.width
        wire.slot = slot = self._new_slots(w)
        self.wires[slot] = wire
        if w == 1:
            self.board[slot], self.uid[slot] = board, wire.uid
        else:
            self.board[slot : slot + w], self.uid[slot : slot + w] = board, wire.uid
        self.set_ends(wire)

    def add_hidden(self, uid: int, ends: list[tuple[bool, int]], width: int = 1) -> int:
        """A wire inside a macro, by its ends (is it a wire?, head slot; slot -1: a free
        end); no object. Its head slot."""
        slot = self._new_slots(width)
        if width == 1:
            self.board[slot], self.uid[slot] = False, uid
            for side, (is_wire, at) in enumerate(ends):
                self.end_is_wire[slot, side] = is_wire
                self.end_slot[slot, side] = slot if is_wire and at < 0 else at
            return slot
        end = slot + width
        self.board[slot:end], self.uid[slot:end] = False, uid
        lanes = np.arange(width)
        for side, (is_wire, at) in enumerate(ends):
            self.end_is_wire[slot:end, side] = is_wire
            self.end_slot[slot:end, side] = (slot if is_wire and at < 0 else at) + lanes
        return slot

    def add_hidden_block(
        self, end_is_wire: np.ndarray, end_slot: np.ndarray, uid: np.ndarray,
        lane: np.ndarray, width: np.ndarray,
    ) -> None:
        """add_hidden for many wire slots at once (lanes included): the next len(uid)."""
        first, k = len(self.wires), len(uid)
        end = first + k
        self._room(end)
        self.wires.extend([None] * k)
        self.alive[first:end] = True
        self.board[first:end] = False
        self.end_is_wire[first:end] = end_is_wire
        self.end_slot[first:end] = end_slot
        self.uid[first:end] = uid
        self.lane[first:end] = lane
        self.width[first:end] = width

    def handle(self, slot: int) -> Wire:
        """The Wire of this slot (made now for a hidden one)."""
        wire = self.wires[slot]
        if wire is None:
            ends = [
                (FREE if e == slot else self.handle(int(e))) if w else self.pins.handle(int(e))
                for w, e in zip(self.end_is_wire[slot].tolist(), self.end_slot[slot].tolist())
            ]
            wire = Wire(ends[0], ends[1], int(self.uid[slot]), slot, int(self.width[slot]))
            wire.src, wire.dst = (wire if e is FREE else e for e in ends)
            self.wires[slot] = wire
        return wire

    def kill(self, slots: list[int]) -> None:
        """These wires are gone (by head slot: their lanes go too)."""
        width = self.width
        for slot in slots:
            w = width[slot]
            if w == 1:
                self.alive[slot] = False
            else:
                self.alive[slot : slot + int(w)] = False
            self.wires[slot] = None

    def set_ends(self, wire: Wire) -> None:
        """Lane i of each end is the end's lane i (a free end: the wire's own)."""
        slot, w = wire.slot, wire.width
        if w == 1:
            for side, end in enumerate(wire.ends):
                self.end_is_wire[slot, side] = isinstance(end, Wire)
                self.end_slot[slot, side] = end.slot
            return
        lanes = np.arange(w)
        for side, end in enumerate(wire.ends):
            self.end_is_wire[slot : slot + w, side] = isinstance(end, Wire)
            self.end_slot[slot : slot + w, side] = end.slot + lanes


class Pin:
    """A handle on one pin slot (see _PinStates): one per slot, so `is` works. What
    never changes about a pin (index, side, its part) it keeps itself: the editor
    asks all the time."""

    # width: how many lanes (1, or a bus: see PartType.widths)
    __slots__ = ("_store", "slot", "index", "is_input", "width", "_part")

    @property
    def part(self) -> Part:
        part = self._part
        if part is None:
            s = self._store
            part = self._part = s.circuit._handle(int(s.part[self.slot]))
        return part

    @property
    def state(self) -> Level | Logic:
        """A Level; a wide pin's is a Logic array of its lanes (lane 0 first)."""
        s = self._store
        w = self.width
        if w == 1:
            return Level(int(s.states[self.slot]))
        return Logic.of_codes(s.states[self.slot : self.slot + w].copy())

    @state.setter
    def state(self, value) -> None:
        """A Level, or a bool / 0 / 1. A wide pin takes a Level (every lane), a Logic
        array or list of its lanes, or a number (its bits: see Logic.of_ints)."""
        s = self._store
        w = self.width
        if w == 1:
            s.states[self.slot] = codes(value)
            s.circuit._poked.append(self.slot)
            return
        lanes = slice(self.slot, self.slot + w)
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, Level)):
            s.states[lanes] = Logic.of_ints(value, w).codes
        else:
            s.states[lanes] = np.broadcast_to(codes(value), (w,))
        s.circuit._poked.append(np.arange(lanes.start, lanes.stop))

    @property
    def passive(self) -> bool:
        """Pass-through: a macro instance's pins, its body's port pins, and joined outputs
        (PartType.joins). They join nets (see Part.links) but never drive them; they just
        show the net's value."""
        return bool(self._store.passive[self.slot])

    @passive.setter
    def passive(self, value: bool) -> None:
        s = self._store
        s.passive[self.slot] = value
        s.reader[self.slot] = s.is_input[self.slot] or value

    def __repr__(self) -> str:
        side = "in" if self.is_input else "out"
        return f"<Pin {self.part.kind}.{side}{self.index}={self.state}>"


_NO_INNER: Mapping[int, Part] = MappingProxyType({})  # (shared: read-only)


class Part:
    """A handle on one part slot: the part's data is in its circuit's part table
    (Circuit._rows ...), so a part costs a row, not an object -- the parts inside a
    macro mostly never get one (see Circuit._handle). One handle per slot, so `is`,
    dict keys and sets work as with objects. What part scripts see: kind, type, uid,
    label, props, state, live, inputs / outputs (pins)."""

    # Once a part has a handle, the handle holds its type, uid, kind, label and props
    # itself (the editor reads them all the time; the row's label and props are left
    # as they were). `label`: the user-given name. It lives in the model (not the
    # UI) because it's circuit data: when a board is saved as a macro, IN/OUT labels
    # become its pin names.
    __slots__ = ("_c", "slot", "_ins", "_outs", "type", "uid", "kind", "label", "_props", "_state")

    @property
    def props(self) -> dict[str, Any]:
        """Its settings (PartType.props)."""
        props = self._props
        if props is None:  # (inside a macro: a copy of the body's, made when first asked)
            c, slot = self._c, self.slot
            body = c._types[c._type_id[c._owner[slot]]].body
            props = self._props = copy.deepcopy(body.parts[self.uid][4])
        return props

    @props.setter
    def props(self, value: dict[str, Any]) -> None:
        self._props = value

    @property
    def state(self) -> dict[str, Any]:
        """Scratch space for its PartType's hooks (made when first asked for; hooks get
        the handle, so it lives there -- and stays with it after the part is gone)."""
        if self._state is None:
            self._state = {}
        return self._state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = value

    @property
    def live(self) -> bool:
        """Opened: placed for real, not a ghost."""
        return bool(self._c._live[self.slot])

    @live.setter
    def live(self, value: bool) -> None:
        self._c._live[self.slot] = value

    @property
    def owner(self) -> Part | None:
        """The macro instance it's inside (hidden parts), else None."""
        o = int(self._c._owner[self.slot])
        return None if o < 0 else self._c._handle(o)

    @property
    def inputs(self) -> list[Pin]:
        if self._ins is None:
            self._pin_lists()
        return self._ins

    @property
    def outputs(self) -> list[Pin]:
        if self._outs is None:
            self._pin_lists()
        return self._outs

    @property
    def layout(self) -> Layout:
        """Its pins: names, widths and joins (see PartType.layout). Pin names are
        layout.ins[i] / layout.outs[i] -- for most kinds the type's ins / outs."""
        c = self._c
        return c._shapes[c._shape[self.slot]].layout

    def _pin_lists(self) -> None:
        c = self._c
        store = c._pins
        sh = c._shapes[c._shape[self.slot]]
        handles, p0, n_in = store.pins, int(c._pin0[self.slot]), sh.n_in
        off, widths = sh.offs, sh.widths
        out = []
        for k in range(sh.n_pins):
            s = p0 + off[k]
            pin = handles[s]
            if pin is None:  # (what store.handle does, knowing the answers already)
                pin = handles[s] = Pin.__new__(Pin)
                pin._store, pin.slot, pin._part = store, s, self
                pin.index, pin.is_input = (k, True) if k < n_in else (k - n_in, False)
                pin.width = widths[k]
            out.append(pin)
        self._ins, self._outs = out[:n_in], out[n_in:]

    @property
    def circuit(self) -> Circuit:
        return self._c

    @property
    def pins(self) -> list[Pin]:
        if self._ins is None:
            self._pin_lists()
        return self._ins + self._outs

    @property
    def drives(self) -> list[Pin]:
        """Per output: the pin its eval value goes to. The output itself, or for a joined
        output (which only shows its net) a hidden pin that drives the net. Not in
        `pins`."""
        extra = self._c._drives.get(self.slot)
        if extra is None:
            return self.outputs
        return [self._c._pins.handle(s) for s in extra]

    @property
    def inner(self) -> Mapping[int, Part]:
        """A macro instance's body parts, by their uid in the body."""
        c = self._c
        inner = c._inner.get(self.slot)
        if inner is None:
            return _NO_INNER
        return {u: c._handle(s) for u, s in zip(c._uid[inner].tolist(), inner.tolist())}

    @property
    def inner_wires(self) -> list[Wire]:
        slots = self._c._inner_wires.get(self.slot, ())
        return [self._c._wire_slots.handle(s) for s in list(slots)]

    @property
    def links(self) -> list[tuple[Pin, Pin]]:
        """Pins joined into one net: instance pin <-> port pin, and joins (PartType.joins)."""
        h = self._c._pins.handle
        return [(h(a), h(b)) for a, b in np.asarray(self._c._links.get(self.slot, ())).tolist()]

    def __repr__(self) -> str:
        return f"Part({self.kind!r}, uid={self.uid}, label={self.label!r}, live={self.live})"


class _Handles(Sequence):
    """Parts by slot, as handles made when an item is asked for: ctx.parts for a batch
    (an eval that never looks at its parts makes none)."""

    __slots__ = ("_c", "slots")

    def __init__(self, circuit: Circuit, slots: np.ndarray) -> None:
        self._c, self.slots = circuit, slots

    def __len__(self) -> int:
        return len(self.slots)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self._c._handle(s) for s in self.slots[i].tolist()]
        return self._c._handle(int(self.slots[i]))

    def __iter__(self):
        h = self._c._handle
        return (h(s) for s in self.slots.tolist())

    def __eq__(self, other) -> bool:
        return list(self) == list(other)


Endpoint = Union[Pin, "Wire"]


class _Free:
    """Stands in for "nothing" as an end given to Circuit.connect: the new wire's end
    becomes the wire itself (see Wire)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "FREE"


FREE = _Free()


@dataclass(eq=False, slots=True)
class Wire:
    # Two ends, each a Pin or another Wire. Normalized by Circuit.connect: an
    # output pin is always `src`, an input pin always `dst` -- so a plain
    # pin-to-pin wire reads src=output, dst=input like before junctions existed.
    # A free end (attached to nothing) is the wire itself: in the nets that's a
    # wire touching only itself, which joins nothing (see is_free).
    src: Endpoint
    dst: Endpoint
    uid: int = 0  # stable identity, like Part.uid (wires can be endpoints of wires)
    slot: int = -1  # its place in the circuit's wire arrays (see _WireSlots): its head
    width: int = 1  # lanes: a bus is wider than 1 (its slots are slot, slot + 1, ...)

    @property
    def ends(self) -> tuple[Endpoint, Endpoint]:
        return self.src, self.dst

    def is_free(self, end: Endpoint) -> bool:
        """Is that end of this wire attached to nothing?"""
        return end is self


class Circuit:
    def __init__(
        self,
        registry: Registry | None = None,
        settle_ticks: int = 0,
        seed: int = 0,
        config: EngineConfig | None = None,
    ) -> None:
        """`registry`: anything with get(kind) / `kind in` -- a Registry, or a
        macros.Catalog to have macros too. `config`: the engine's code paths (see
        sim/config.py; default: config.default(), as decided at startup)."""
        self.config = config or default_config()
        self._bind()
        self.registry = registry or builtin_registry()
        self._parts: dict[
            Part, None
        ] = {}  # what's on the board (macro insides are hidden_parts)
        self.settle_ticks = settle_ticks
        self.rng = np.random.default_rng(seed)
        self._pins = _PinStates(self)
        self._wire_slots = _WireSlots(self._pins)
        # The part table (see _new_rows): columns by part slot ...
        self._settle = np.zeros(256, np.int32)  # ticks of settling jitter left
        self._type_id = np.zeros(256, np.int32)  # index into _types
        self._uid = np.zeros(256, np.int64)
        self._live = np.zeros(256, bool)
        self._alive = np.zeros(256, bool)
        self._owner = np.full(256, -1, np.int32)
        self._pin0 = np.zeros(256, np.int32)
        self._shape = np.zeros(256, np.int32)  # index into _shapes: its pin layout
        self._label = np.full(256, None, object)
        self._props = np.full(256, None, object)
        self._handles = np.full(256, None, object)
        # ... and what few parts have, by slot
        self._inner: dict[int, np.ndarray] = {}  # macro instance -> its body's part slots
        self._inner_wires: dict[int, np.ndarray] = {}  # macro instance -> its wire slots
        self._links: dict[int, np.ndarray] = {}  # pin slot pairs joined into one net (k x 2)
        self._drives: dict[int, list[int]] = {}  # joined parts: per output, its driver pin
        self._types: list[PartType] = []
        self._type_ids: dict[PartType, int] = {}
        self._shapes: list[_Shape] = []  # pin layouts: a type and its pins' widths
        self._shape_ids: dict[tuple, int] = {}  # (type, widths or None) -> shape id
        self._sh_cols = (np.zeros(0, np.intp),) * 3  # per shape: pins, lanes, input lanes
        self._blueprints: dict[PartType, _Blueprint] = {}  # macro types: see _make
        self._linked: dict[int, None] = {}  # slots of parts with links (macros, joins)
        self._n_part_slots = 0
        self._settling = 0  # how many parts are still settling
        self.tick = 0
        self.faults: dict[str, str] = {}  # kind -> why it's disabled (a hook raised)
        self.errors: list[str] = []  # new fault messages, for the UI to pick up
        self._kinds_dirty = True
        self._kinds: dict[PartType, list[Part]] = {}  # instances grouped by type
        self._batches_dirty = True
        self._batches: list[_Batch] = []  # what step() evaluates, kind by kind
        self._wires: dict[
            Wire, None
        ] = {}  # creation order: a wire always comes after the wires it attaches to
        # pin or wire -> the board wires with an end on it: the wire itself when there's
        # one (most pins), else a list (see _link; read it with ends_on)
        self._at: dict[Endpoint, Wire | list[Wire]] = {}
        self.part_by_uid: dict[
            int, Part
        ] = {}  # board parts (hidden ones have their own uid spaces)
        self.wire_by_uid: dict[int, Wire] = {}
        self.revision = 0  # bumped by every edit of the board's parts and wiring
        self._next_uid = 1
        self._next_wire_uid = 1
        # Nets are derived from the wiring and cached until the wiring changes.
        self._nets_dirty = True
        self._nets_version = 0  # bumped when they're rebuilt (see nets_version)
        # Driver pin slots. Nets with one driver just copy it; the others are reduced.
        self._solo = np.empty(0, np.intp)  # the only driver of its net ...
        self._solo_net = np.empty(0, np.intp)  # ... and that net
        self._drivers = np.empty(
            0, np.intp
        )  # drivers of nets with several, grouped by net
        self._driven = np.empty(0, np.intp)  # those nets, ascending ...
        self._drv_starts = np.empty(
            0, np.intp
        )  # ... and where each one's drivers start
        self._readers = np.empty(0, np.intp)  # reader pin slots ...
        self._reader_net = np.empty(0, np.intp)  # ... and their nets
        self._weak = np.empty(
            0, np.intp
        )  # the weak drivers that count (top priority), by net ...
        self._weak_nets = np.empty(0, np.intp)  # ... the nets that have some ...
        self._weak_starts = np.empty(0, np.intp)  # ... and where each one's start
        self._wire_net = np.zeros(0, np.intp)  # per wire slot: its net (-1: gone)
        self.net_value = np.zeros(0, CODE)  # logic codes (see pijl.logic)
        self.net_conflict = np.zeros(0, bool)  # drivers fighting: one says 0, another 1
        # For the dirty-set stepper (sim/dirty.py): what the next step must run
        self._full = True  # the next step runs everything (something changed wholesale)
        self._dirty = np.empty(0, np.intp)  # else: the part slots it must run
        self._quiet = False  # the last step changed no pin
        self._batch_of = np.zeros(0, np.intp)  # per part slot: its batch (-1: none) ...
        self._batch_pos = np.zeros(0, np.intp)  # ... and its place in it
        self._n_pure = 0  # parts in pure batches
        self._lut_plan = None  # sim/lut.py's units, made from the batches
        # Nets by number, for carrying only some (sim/dirty.py); -1 where none
        self._pin_net = np.zeros(0, np.intp)  # per pin slot
        self._solo_of = np.zeros(0, np.intp)  # per net: its only driver
        self._multi_at = np.zeros(0, np.intp)  # per net: its group in _drivers
        self._drv_count = np.zeros(0, np.intp)  # per group: how many drivers
        self._weak_at = np.zeros(0, np.intp)  # per net: its group in _weak
        self._weak_count = np.zeros(0, np.intp)
        self._rd_sorted = np.zeros(0, np.intp)  # reader pin slots grouped by net ...
        self._rd_start = np.zeros(1, np.intp)  # ... where each net's start (n_nets + 1)
        self._rd_count = np.zeros(0, np.intp)  # ... and how many it has
        # What the UI was shown last (take_changes), to tell it what changed since.
        self._changed_all = True
        self._shown = np.zeros(0, bool)
        self._shown_value = self._shown_conflict = np.zeros(0, bool)

    def _bind(self) -> None:
        """Take on the code paths self.config names: step() and run_until_stable() come
        from its stepper, and the stepper runs the parts through its evaluator. Pokes (pin slots written from outside step(): Pin.state,
        write_pins) are only kept for a stepper that wants them."""
        from . import batches, dirty, lut, plain  # (they import this module)

        stepper = {"adaptive": dirty, "off": plain}[self.config.dirty]
        evaluator = {"batches": batches, "lut": lut}[self.config.eval]
        self._run_all = MethodType(evaluator.run_all, self)
        self._run_all_tracked = MethodType(evaluator.run_all_tracked, self)
        self._run_some = MethodType(evaluator.run_some, self)
        self._costs = evaluator.COSTS
        self.step: Callable[[], None] = MethodType(stepper.step, self)
        # run_until_stable(limit): step until a step changes no pin and nothing is
        # settling any more, at most `limit` steps that change something (plus the one
        # that shows nothing does). Returns how many changed something (0: it already
        # was stable), or None if it never got there (it oscillates, or just needs
        # more). A part that isn't pure may change the world without changing a pin; it
        # counts as stable all the same.
        self.run_until_stable: Callable[[int], int | None] = MethodType(stepper.run_until_stable, self)
        self._poked: list | _Drop = [] if stepper.TAKES_POKES else _Drop()

    @property
    def parts(self) -> list[Part]:
        """The board's parts, in the order they were added (a copy)."""
        return list(self._parts)

    @property
    def wires(self) -> list[Wire]:
        """The board's wires in creation order: parents before children (a copy)."""
        return list(self._wires)

    # ---- editing -------------------------------------------------------

    def add_part(self, kind: str, uid: int | None = None, live: bool = True) -> Part:
        """`uid` recreates a specific part (undo, loading); normally leave it None.
        `live=False` makes a ghost: call open_part once it's placed for real.
        KeyError if there's no such kind (or it's a macro that can't be loaded)."""
        return self.add_parts([self.registry.get(kind)], [uid], live)[0]

    def add_parts(
        self,
        types: list[PartType],
        uids: list[int | None],
        live: bool = True,
        props: list[dict | None] | None = None,
    ) -> list[Part]:
        """add_part for many parts at once, by type (registry.get(kind)): pins, settling
        and the rest set up a whole type at a time rather than part by part. `props`:
        each part's (copied; None: the defaults). Give them here rather than setting
        them afterwards when they decide pin widths (PartType.widths)."""
        given = []
        for uid in uids:
            if uid is None:
                uid = self._next_uid
            self._next_uid = max(self._next_uid, uid + 1)
            given.append(uid)
        if props is not None:
            props = [None if p is None else copy_props(p) for p in props]
        parts = [self._handle(s) for s in self._make_many(types, given, props=props)]
        for part in parts:
            self._parts[part] = None
            self.part_by_uid[part.uid] = part
        self.revision += 1
        if live:
            self.open_parts(parts)
        return parts

    # ---- the part table ----------------------------------------------------
    # A part is a row, by part slot (never reused); Part objects are handles on rows,
    # made by _handle when something asks. Columns: type, uid, live, alive, owner (the
    # macro instance it's inside, -1 on the board), its first pin slot (its pins' lanes
    # are the next slots: see its shape), its shape (_Shape), label, props (None inside a macro until
    # asked for: then a copy of the body's). What few parts have is kept by slot in
    # dicts, as small int arrays: a macro's body parts and wires (_inner,
    # _inner_wires), links (pin slot pairs), joined outputs' driver pins (_drives).

    def _new_rows(self, k: int) -> range:
        first = self._n_part_slots
        end = first + k
        if end > len(self._settle):
            cap = len(self._settle)
            while cap < end:
                cap *= 2
            for name in self._ROW_COLS:
                old = getattr(self, name)
                fill = None if old.dtype == object else (-1 if name == "_owner" else 0)
                new = np.full(cap, fill, old.dtype)
                new[: len(old)] = old
                setattr(self, name, new)
        self._n_part_slots = end
        return range(first, end)

    _ROW_COLS = (
        "_settle", "_type_id", "_uid", "_live", "_alive", "_owner", "_pin0", "_shape",
        "_label", "_props", "_handles",
    )  # fmt: skip

    def _type_of(self, t: PartType) -> int:
        tid = self._type_ids.get(t)
        if tid is None:
            tid = self._type_ids[t] = len(self._types)
            self._types.append(t)
        return tid

    def _handle(self, slot: int) -> Part:
        """The Part for this slot (one per slot, made on first use)."""
        part = self._handles[slot]
        if part is None:
            part = self._handles[slot] = Part.__new__(Part)
            part._c, part.slot, part._ins, part._outs = self, slot, None, None
            t = part.type = self._types[self._type_id[slot]]
            part.uid, part.kind = int(self._uid[slot]), t.kind
            part.label, part._props, part._state = self._label[slot], self._props[slot], None
        return part

    def _make_many(
        self,
        types: list[PartType],
        uids: list[int],
        owner: int = -1,
        props: list[dict | None] | None = None,
    ) -> list[int]:
        """_make for each (type, uid), in order: the same rows and pin slots as making
        them one by one (their slots). Plain types (no joins, no body) are added a run
        at a time: rows and pins as arrays. `props`: each one's (None: the defaults);
        on the board they become the part's own, inside a macro they're only read."""
        out: list[int] = []
        i, n = 0, len(types)
        if props is None:
            props = [None] * n
        while i < n:
            t = types[i]
            if _one_by_one(t):
                out.append(self._make(t, uids[i], owner, props[i]))
                i += 1
                continue
            j = i
            while j < n and not _one_by_one(types[j]):
                j += 1
            out += self._make_plain(types[i:j], uids[i:j], owner, props[i:j])
            i = j
        if out:
            self._kinds_dirty = self._batches_dirty = self._nets_dirty = True
        return out

    def _make_plain(
        self, types: list[PartType], uids: list[int], owner: int, props: list[dict | None]
    ) -> list[int]:
        """Rows and pins for a run of plain parts, as arrays (see _make_many)."""
        store = self._pins
        rows = self._new_rows(len(types))
        # types whose widths read no props: one shape each
        plain = {t: self._shape_of(t, None) for t in dict.fromkeys(types) if layout_props(t) == ()}
        if len(plain) == len(dict.fromkeys(types)):
            sids = [plain[t] for t in types]
        else:
            known: dict = {}  # type, or (type, the props its widths read) -> shape id
            sids = []
            for t, given in zip(types, props):
                sid = plain.get(t)
                if sid is None:
                    keys = layout_props(t)
                    if keys is None:  # (its own layout(): ask every time)
                        sid = self._shape_of(t, given)
                    else:
                        key = t if given is None else (t, *(given.get(k) for k in keys))
                        sid = known.get(key)
                        if sid is None:
                            sid = known[key] = self._shape_of(t, given)
                sids.append(sid)
        shapes = self._shapes
        board = owner < 0
        if board:  # (inside a macro, props were only read)
            made = [shapes[sid].props() if given is None else given for sid, given in zip(sids, props)]
        # Every column of every new pin slot is one gather from the shapes' own columns
        # (a few shapes, many parts), not a concatenation of one small array per part.
        sid_arr = np.array(sids, np.intp)
        one = len(set(sids)) == 1  # (one shape: adding a part, or a run of one kind)
        if one:
            uniq, inv = sid_arr[:1], np.zeros(len(sids), np.intp)
        else:
            uniq, inv = np.unique(sid_arr, return_inverse=True)
        used = [shapes[u] for u in uniq.tolist()]
        lanes = np.array([sh.n_lanes for sh in used], np.intp)
        counts = lanes[inv]
        sl = slice(rows.start, rows.stop)
        self._type_id[sl] = np.array([sh.type_id for sh in used], np.int32)[inv]
        self._shape[sl] = sid_arr
        self._uid[sl] = uids
        self._live[sl] = False
        self._alive[sl] = True
        self._owner[sl] = owner
        self._pin0[sl] = store.n + np.cumsum(counts) - counts
        self._label[sl] = [""] * len(sids)
        if board:  # (on the board: props now; inside a macro, see _expand)
            self._props[sl] = _objects(made)
        if one:
            col = lambda name: np.tile(getattr(used[0], name), len(sids))  # noqa: E731
        elif sids:
            ends = np.cumsum(counts)
            at = np.repeat((np.cumsum(lanes) - lanes)[inv] - (ends - counts), counts) + np.arange(int(ends[-1]))
            col = lambda name: np.concatenate([getattr(sh, name) for sh in used])[at]  # noqa: E731
        if sids:
            store.new_block(
                np.repeat(np.arange(rows.start, rows.stop, dtype=np.int32), counts),
                col("index"), col("is_input"), col("initial"), col("weak"),
                lane=col("lane"), width=col("width"),
            )
        return list(rows)

    def _shape_of(self, t: PartType, props: dict | None) -> int:
        """The shape (pin layout) of an instance of t with these props (None: t's
        defaults), as an id into _shapes: from layout_of(t, props). If that raises,
        the kind is faulted and the instance gets the class attributes' pins, each one
        lane wide."""
        try:
            lay = layout_of(t, fresh_props(t) if props is None else props)
        except Exception as e:
            if t.kind not in self.faults:
                self.faults[t.kind] = msg = f"{t.kind}.layout: {e}"
                self.errors.append(msg)
            lay = Layout(t.ins, t.outs, (1,) * (len(t.ins) + len(t.outs)))
        key = (t, lay)
        sid = self._shape_ids.get(key)
        if sid is None:
            sid = self._shape_ids[key] = len(self._shapes)
            self._shapes.append(_Shape(t, lay, self._type_of(t)))
        return sid

    def layout_for(self, t: PartType, props: dict) -> Layout:
        """The pins an instance of t with these props gets (see layout_of: if that
        fails, the kind is faulted and it gets one-lane class attribute pins)."""
        return self._shapes[self._shape_of(t, props)].layout

    def reshaped(self, part: Part, props: dict) -> bool:
        """Would `part` have other pins with these props (a settings edit: see
        PartType.layout)? Then the editor rebuilds it."""
        return self._shape_of(part.type, props) != int(self._shape[part.slot])

    def _shape_cols(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per shape id: (pins, lanes, input lanes)."""
        if len(self._sh_cols[0]) != len(self._shapes):
            self._sh_cols = tuple(
                np.array(col, np.intp)
                for col in zip(
                    *((sh.n_pins, sh.n_lanes, sh.in_lanes) for sh in self._shapes)
                )
            )
        return self._sh_cols

    def _lanes_of(self, slots: np.ndarray) -> np.ndarray:
        """Every lane slot of these parts' pins, part by part (not their drives)."""
        lanes = self._shape_cols()[1][self._shape[slots]]
        first = self._pin0[slots].astype(np.intp)
        return np.repeat(first - (np.cumsum(lanes) - lanes), lanes) + np.arange(int(lanes.sum()))

    def _make(self, t: PartType, uid: int, owner: int = -1, props: dict | None = None) -> int:
        """One part of any type (a macro's body is built too); its slot. A macro type's
        first instance is built part by part and recorded (_Blueprint); the rest are
        stamped from that: the same rows, pins and wires, as arrays."""
        body = getattr(t, "body", None)
        if body is not None:
            bp = self._blueprints.get(t)
            if bp is not None and bp.current(self.registry):
                return self._stamp(bp, t, uid, owner, props)
            marks = (self._n_part_slots, self._pins.n, len(self._wire_slots.wires))
        slot = self._make_one(t, uid, owner, props)
        if body is not None:
            self._blueprints[t] = _Blueprint(self, *marks)
        return slot

    def _make_one(self, t: PartType, uid: int, owner: int, props: dict | None = None) -> int:
        slot = self._new_rows(1)[0]
        store = self._pins
        if props is None:
            props = fresh_props(t)
        sid = self._shape_of(t, props)
        sh = self._shapes[sid]
        widths = sh.widths
        self._type_id[slot] = self._type_of(t)
        self._shape[slot] = sid
        self._uid[slot], self._live[slot], self._alive[slot] = uid, False, True
        self._owner[slot], self._label[slot] = owner, ""
        self._props[slot] = props
        power_on = X if t.has("eval") else ZERO  # (the IN switch starts off)
        joined = {k for group in sh.joins for k, _, _ in group}
        n_in = sh.n_in
        self._pin0[slot] = store.n
        for i in range(n_in):
            store.new(slot, i, True, width=widths[i])
        for i, name in enumerate(sh.layout.outs):
            store.new(
                slot, i, False, power_on, name in t.weak and n_in + i not in joined, widths[n_in + i]
            )
        if joined:
            self._join(slot, t, sh, power_on)
        if getattr(t, "body", None) is not None:
            self._expand(slot, t)
        self._kinds_dirty = self._batches_dirty = self._nets_dirty = True
        return slot

    def _stamp(
        self, bp: _Blueprint, t: PartType, uid: int, owner: int, props: dict | None = None
    ) -> int:
        """Another instance of a macro, from its blueprint: rows, pins and wires at the
        next slots, offset from the recorded ones."""
        store, ws = self._pins, self._wire_slots
        r0, p0, w0 = self._n_part_slots, store.n, len(ws.wires)
        rows = self._new_rows(len(bp.type_id))
        sl = slice(rows.start, rows.stop)
        self._type_id[sl] = bp.type_id
        self._uid[sl] = bp.uid
        self._uid[r0] = uid
        self._live[sl] = False
        self._alive[sl] = True
        self._owner[sl] = bp.owner + r0
        self._owner[r0] = owner
        self._pin0[sl] = bp.pin0 + p0
        self._shape[sl] = bp.shape
        self._label[sl] = bp.label
        self._props[sl] = None  # (Part.props copies the body's)
        self._props[r0] = fresh_props(t) if props is None else props
        store.new_block(
            bp.pin_part + r0, bp.pin_index, bp.pin_is_input, bp.pin_states, bp.pin_weak,
            bp.pin_reader, bp.pin_passive, bp.pin_lane, bp.pin_width,
        )
        ws.add_hidden_block(
            bp.end_is_wire, np.where(bp.end_is_wire, bp.end_slot + w0, bp.end_slot + p0), bp.wire_uid,
            bp.wire_lane, bp.wire_width,
        )
        for rel, inner, wires in bp.inner:
            self._inner[r0 + rel] = inner + r0
            self._inner_wires[r0 + rel] = wires + w0
        for rel, links in bp.links:
            self._links[r0 + rel] = links + p0
            self._linked[r0 + rel] = None
        for rel, drives in bp.drives:
            self._drives[r0 + rel] = [d + p0 for d in drives]
        self._kinds_dirty = self._batches_dirty = self._nets_dirty = True
        return r0

    def _expand(self, inst: int, t: PartType) -> None:
        """Build a macro instance's body as hidden parts + wires, and join its pins
        to the body's ports (see the module docstring). Rows and pins only: the
        body's parts get handles when something asks (Part.inner, hooks)."""
        body = t.body
        kinds = list(body.parts.items())
        slots = self._make_many(
            [self.registry.get(d[0]) for _, d in kinds], [u for u, _ in kinds], inst,
            [d[4] for _, d in kinds],
        )
        inner = dict(zip((u for u, _ in kinds), slots))  # (while building it)
        for (_uid, (_kind, label, _x, _y, _props)), s in zip(kinds, slots):
            if label:
                self._label[s] = label
        self._props[np.array(slots, np.intp)] = None  # (Part.props copies the body's)
        store, ws = self._pins, self._wire_slots
        widths = getattr(body, "wire_widths", {})
        # per body part: its first pin slot and shape, as plain Python for the loop below
        # (two dicts, not one of pairs: a pair per part is that many more objects for
        # the garbage collector to walk)
        first_of = dict(zip(inner, self._pin0[slots].tolist()))
        shapes = self._shapes
        shape_of = dict(zip(inner, (shapes[s] for s in self._shape[slots].tolist())))
        wires: dict[int, int] = {}  # uid -> head slot
        mine = []
        for uid in sorted(body.wires):  # parents first
            width = widths.get(uid, 1)
            ends = []
            for ref in body.wires[uid][:2]:
                if ref[0] == "w":
                    if ref[1] == uid:
                        ends.append((True, -1))
                        continue
                    slot = wires.get(ref[1], -1)  # (-1: left out, so is this one)
                    w = widths.get(ref[1], 1) if slot >= 0 else -1
                    ends.append((True, slot))
                else:
                    _, puid, is_input, index = ref
                    sh = shape_of[puid]
                    k = index if is_input else sh.n_in + index
                    ends.append((False, first_of[puid] + sh.offs[k]))
                    w = sh.widths[k]
                if w != width:
                    break  # (ends of another width: a stale body. Left out, like a dead end)
            else:
                slot = wires[uid] = ws.add_hidden(uid, ends, width)
                mine.append(slot)
        self._inner[inst] = np.array(slots, np.int32)
        self._inner_wires[inst] = np.array(mine, np.int32)
        links = []
        first, mine_sh = int(self._pin0[inst]), self._shapes[self._shape[inst]]
        n_in = mine_sh.n_in
        for i, port in enumerate(t.in_ids):  # instance input i <-> its IN's output
            p0, sh = first_of[port], shape_of[port]
            links += _pairs(first + mine_sh.offs[i], mine_sh.widths[i], p0 + sh.offs[sh.n_in], sh.widths[sh.n_in])
        for i, port in enumerate(t.out_ids):  # its OUT's input <-> instance output i
            p0, sh = first_of[port], shape_of[port]
            k = n_in + i
            links += _pairs(p0 + sh.offs[0], sh.widths[0], first + mine_sh.offs[k], mine_sh.widths[k])
        links = self._links[inst] = np.array(links, np.int32).reshape(-1, 2)
        store.passive[links.ravel()] = True
        store.reader[links.ravel()] = True
        self._linked[inst] = None

    def _in_pin(self, part: int, i: int) -> int:
        """Input i's (head) pin slot."""
        return int(self._pin0[part]) + self._shapes[self._shape[part]].offs[i]

    def _out_pin(self, part: int, i: int) -> int:
        """Output i's (head) pin slot."""
        sh = self._shapes[self._shape[part]]
        return int(self._pin0[part]) + sh.offs[sh.n_in + i]

    def _join(self, slot: int, t: PartType, sh: _Shape, power_on: Level) -> None:
        """Make each of the layout's join groups one net: link its lanes and turn its
        outputs' into pass-through ones. A part with an eval gets a hidden pin for each
        joined output, to drive the net with what eval says (its joins are of whole
        pins: see layout_of); one without just joins (a splitter)."""
        store = self._pins
        p0, n_in = int(self._pin0[slot]), sh.n_in
        heads = [p0 + off for off in sh.offs]
        evals = t.has("eval")
        drives = heads[n_in:]
        links = []
        for group in sh.joins:
            lanes = [list(range(heads[k] + first, heads[k] + first + n)) for k, first, n in group]
            for (k, _first, _n), mine in zip(group, lanes):
                if k < n_in:
                    continue
                store.passive[mine] = True  # (an output: it only shows the net now)
                store.reader[mine] = True
                if evals:
                    i = k - n_in
                    name = sh.layout.outs[i]
                    drives[i] = store.new(slot, i, False, power_on, name in t.weak, sh.widths[k])
                    lanes.append(list(range(drives[i], drives[i] + sh.widths[k])))
            links += [(a, b) for other in lanes[1:] for a, b in zip(lanes[0], other)]
        if evals:
            self._drives[slot] = drives
        self._links[slot] = np.array(links, np.int32).reshape(-1, 2)
        self._linked[slot] = None

    def _tree(self, slot: int) -> list[int]:
        """The part plus, for a macro instance, everything inside it (any depth): slots."""
        out, todo = [], [slot]
        inner = self._inner
        while todo:
            s = todo.pop()
            out.append(s)
            kids = inner.get(s)
            if kids is not None:
                todo += kids.tolist()
        return out

    @property
    def hidden_parts(self) -> list[Part]:
        """The parts inside the board's macros (handles: made now)."""
        n = self._n_part_slots
        return [
            self._handle(s)
            for s in np.flatnonzero(self._alive[:n] & (self._owner[:n] >= 0)).tolist()
        ]

    @property
    def hidden_count(self) -> int:
        n = self._n_part_slots
        return int(np.count_nonzero(self._alive[:n] & (self._owner[:n] >= 0)))

    @property
    def hidden_wires(self) -> list[Wire]:
        ws = self._wire_slots
        n = len(ws.wires)
        heads = ws.alive[:n] & ~ws.board[:n] & (ws.lane[:n] == 0)
        return [ws.handle(s) for s in np.flatnonzero(heads).tolist()]

    def open_part(self, part: Part) -> None:
        """Make a ghost live (placed for real): open hooks, and settling starts.
        A macro instance opens everything inside it too."""
        self.open_parts([part])

    def open_parts(self, parts: Iterable[Part]) -> None:
        """open_part for many parts at once."""
        live = self._live
        opening = [
            s
            for part in parts
            for s in (self._tree(part.slot) if part.slot in self._inner else (part.slot,))
            if not live[s]
        ]
        if not opening:
            return
        at = np.array(opening, np.intp)
        live[at] = True
        self._batches_dirty = True
        types = self._types
        if self.settle_ticks:
            self._settle[at] = self.settle_ticks
            self._settling += len(opening)
            noisy = self._drives_of(at[[types[t].has("eval") for t in self._type_id[at].tolist()]])
            if noisy.size:  # power-on noise
                self._pins.states[noisy] = self.rng.integers(
                    ZERO, ONE + 1, len(noisy), dtype=CODE
                )
        for s in opening:
            t = types[self._type_id[s]]
            if t.has("open") and t.kind not in self.faults:
                p = self._handle(s)
                self._guard(t, "open", lambda: t.open(p))

    def _drives_of(self, slots: np.ndarray) -> np.ndarray:
        """_drive_slots of many parts, in order, as one array."""
        if not slots.size:
            return np.empty(0, np.intp)
        if self._drives and any(s in self._drives for s in slots.tolist()):
            return np.array([d for s in slots.tolist() for d in self._drive_slots(s)], np.intp)
        _, lanes, in_lanes = self._shape_cols()
        sh = self._shape[slots]
        n_out = lanes[sh] - in_lanes[sh]  # (outputs' lanes come after the inputs')
        first = self._pin0[slots].astype(np.intp) + in_lanes[sh]
        return np.repeat(first - (np.cumsum(n_out) - n_out), n_out) + np.arange(int(n_out.sum()))

    def _drive_slots(self, slot: int) -> list[int]:
        """Where a part's outputs' eval values go (see Part.drives), as pin slots: every
        lane."""
        extra = self._drives.get(slot)
        if extra is not None:
            return [s for head in extra for s in self._pins.lanes(head).tolist()]
        sh = self._shapes[self._shape[slot]]
        first = int(self._pin0[slot]) + sh.in_lanes
        return list(range(first, int(self._pin0[slot]) + sh.n_lanes))

    def close_part(self, part: Part) -> None:
        self._close(self._tree(part.slot))

    def _close(self, slots: list[int]) -> None:
        """Close these parts (slots, macro insides included), the live ones, in order."""
        at = np.array(slots, np.intp)
        closing = at[self._live[at]]
        if not closing.size:
            return
        self._live[closing] = False
        self._batches_dirty = True
        types = self._types
        for s, tid in zip(closing.tolist(), self._type_id[closing].tolist()):
            t = types[tid]
            if t.has("close"):
                p = self._handle(s)
                self._guard(t, "close", lambda: t.close(p))

    def close_all(self) -> None:
        """The circuit is going away (app closing): close every live part."""
        for part in self.parts:
            self.close_part(part)

    def click_cell(self, part: Part, lane: int) -> bool:
        """The user clicked cell `lane` of a placed part (Look.cells). False if its
        type doesn't take cell clicks."""
        t = part.type
        if not t.has("click_cell") or not part.live:
            return False
        if t.kind not in self.faults:
            self._guard(t, "click_cell", lambda: t.click_cell(part, lane))
        return True

    def click(self, part: Part) -> bool:
        """The user clicked a placed part. False if its type doesn't take clicks."""
        t = part.type
        if not t.has("click") or not part.live:
            return False
        if t.kind not in self.faults:
            self._guard(
                t, "click", lambda: t.click(part)
            )  # (IN flips its output pin right here)
        return True

    def remove_part(self, part: Part) -> list[Wire]:
        """Removes the part, every wire touching it, and every wire hanging off
        those. Returns all removed wires (parents first)."""
        return self.remove_parts([part])

    def remove_parts(self, parts: Iterable[Part]) -> list[Wire]:
        """remove_part for many parts at once."""
        parts = list(parts)
        if not parts:
            return []
        inner = self._inner
        tree = [s for part in parts for s in (self._tree(part.slot) if part.slot in inner else (part.slot,))]
        self._close(tree)
        removed = self.remove_wires(
            {w for part in parts for pin in part.pins for w in self.ends_on(pin)}
        )
        for part in parts:
            del self._parts[part]
            del self.part_by_uid[part.uid]
        self.revision += 1
        at = np.array(tree, np.intp)
        self._pins.kill(self._lanes_of(at).tolist())
        self._wire_slots.kill([w for s in tree for w in np.asarray(self._inner_wires.get(s, ())).tolist()])
        self._settle[at] = 0
        self._alive[at] = False
        # (the rows stay dead: let go of what they held -- a handle someone still has
        # keeps its own copy of what it needs)
        self._handles[at] = self._label[at] = self._props[at] = None
        for s in tree:
            for d in (self._linked, self._inner, self._inner_wires, self._links, self._drives):
                d.pop(s, None)
        self._nets_dirty = self._kinds_dirty = self._batches_dirty = True
        return removed

    def props_changed(self, part: Part) -> None:
        """Call after changing a board part's props: some of them (a pull's priority)
        decide how its nets resolve."""
        if part.type.weak:
            self._nets_dirty = True

    # ---- settings and actions (see pijl/parts/settings.py) ---------------------

    def set_setting(
        self, parts: list[Part], key: str, value, notify: bool = True
    ) -> list:
        """Set setting `key` of `parts` (all of one kind) to `value`, through the setting's
        parse() (ValueError if it doesn't parse). Returns the old values, one per part.
        `notify=False` while a slider drags: the changed hook waits for the caller's
        settings_changed() when the edit is done (unless the setting is live)."""
        t = _one_type(parts)
        s = t.settings[key]
        value = s.parse(value)
        old = [p.props.get(key) for p in parts]
        for p in parts:
            p.props[key] = value
        if t.weak:
            self._nets_dirty = True
        if notify or getattr(s, "live", False):
            self.settings_changed(parts, key, old)
        return old

    def put_setting(
        self, parts: list[Part], key: str, values: list, notify: bool = True
    ) -> None:
        """Give each part its own value back (values[i] for parts[i]): an edit taken back
        with Esc. `notify=False` when the changed hook never heard of the edit."""
        old = [p.props.get(key) for p in parts]
        for p, v in zip(parts, values):
            p.props[key] = v
        if parts and parts[0].type.weak:
            self._nets_dirty = True
        if notify:
            self.settings_changed(parts, key, old)

    def settings_changed(self, parts: list[Part], key: str, old: list) -> None:
        """Tell the parts' type (all one kind) that setting `key` changed from `old`
        (one value per part): one changed() call for the parts whose value differs."""
        if not parts:
            return
        t = parts[0].type
        if t.weak:
            self._nets_dirty = True
        if not t.has("changed") or t.kind in self.faults:
            return
        moved = [(p, o) for p, o in zip(parts, old) if p.props.get(key) != o]
        if moved:
            ctx = Ctx([p for p, _ in moved], self.tick)
            self._guard(
                t, "changed", lambda: t.changed(ctx, key, [o for _, o in moved])
            )

    def run_action(self, parts: list[Part], name: str) -> None:
        """The user picked action `name` for `parts` (all one kind): one action() call,
        for the live ones."""
        t = _one_type(parts)
        if name not in t.actions:
            raise KeyError(f"{t.kind} has no action {name!r}")
        live = [p for p in parts if p.live]
        if live and t.kind not in self.faults:
            self._guard(t, "action", lambda: t.action(Ctx(live, self.tick), name))

    def can_connect(self, a: Endpoint, b: Endpoint) -> bool:
        """`a` or `b` may be FREE (a free end: see Wire). Ends must be equally wide."""
        if a is FREE or b is FREE:
            return True
        if a is b or a.width != b.width:
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

    def connect(
        self,
        a: Endpoint,
        b: Endpoint,
        uid: int | None = None,
        check: bool = True,
        width: int | None = None,
    ) -> tuple[Wire | None, list[Wire]]:
        """Connect two endpoints in either order. Returns (new_wire, replaced_wires).
        Either may be FREE: that end of the new wire is attached to nothing.

        new_wire is None if the connection is invalid (see can_connect). An input
        pin takes one wire, so wiring into an already-wired input replaces the old
        wire, along with any branches hanging off it.

        `check=False` skips can_connect: for rebuilding wiring that existed before
        (undo, paste). Some of it can't be drawn by hand -- cut-deletion can splice a
        wire that runs from a part back into itself -- but it must come back as it was.

        The wire is as wide as its ends (`width` if both are FREE; default 1). Ends of
        different widths (or not `width`, if given) never connect, checked or not.
        """
        if check and not self.can_connect(a, b):
            return None, []
        wa = width if a is FREE else a.width
        wb = wa if b is FREE else b.width
        if wa is None:
            wa = wb
        if wa != wb or (width is not None and wa != width):
            return None, []
        width = wa
        if width is None:
            width = 1
        elif width != 1 and not 1 <= width <= MAX_WIDTH:
            raise ValueError(f"a wire can't be {width} lanes wide")
        # outputs are src, inputs are dst (see Wire)
        if (isinstance(b, Pin) and not b.is_input) or (
            isinstance(a, Pin) and a.is_input
        ):
            a, b = b, a
        replaced: list[Wire] = []
        for end in (a, b):
            if isinstance(end, Pin) and end.is_input:
                for old in self.wires_at(end):
                    replaced += self.remove_wire(old)
        if uid is None:
            uid = self._next_wire_uid
        self._next_wire_uid = max(self._next_wire_uid, uid + 1)
        wire = Wire(a, b, uid, width=width)
        wire.src, wire.dst = (wire if e is FREE else e for e in (a, b))
        self._wire_slots.add(wire, board=True)
        self._wires[wire] = None
        self.wire_by_uid[uid] = wire
        self._link(wire)
        self._nets_dirty = True
        self.revision += 1
        return wire, replaced

    def _link(self, wire: Wire) -> None:
        at = self._at
        for end in wire.ends:
            if end is wire:
                continue  # a free end: nothing to be found by
            had = at.get(end)
            if had is None:
                at[end] = wire
            elif type(had) is list:
                had.append(wire)
            else:
                at[end] = [had, wire]

    def _unlink(self, wire: Wire) -> None:
        at = self._at
        for end in wire.ends:
            if end is wire:
                continue
            had = at[end]
            if type(had) is list:
                had.remove(wire)
                if len(had) == 1:
                    at[end] = had[0]
            else:  # (it was the only one)
                del at[end]

    def remove_wire(self, wire: Wire) -> list[Wire]:
        """Removes the wire and everything attached to it. Returns them, parents first."""
        return self.remove_wires([wire])

    def remove_wires(self, wires: Iterable[Wire]) -> list[Wire]:
        """remove_wire for many wires at once (ones already gone are skipped)."""
        wires = [w for w in wires if w in self._wires]
        if not wires:
            return []
        removed = _by_uid({*wires, *self.descendants(*wires)})
        for w in removed:
            del self._wires[w]
            del self.wire_by_uid[w.uid]
            self._unlink(w)
        self._wire_slots.kill([w.slot for w in removed])
        self._nets_dirty = True
        self.revision += 1
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
        slots = self._wire_slots
        self._unlink(keep)
        keep.dst = keep if far is absorb else far  # (absorb's far end was free: so is keep's)
        for w in list(self.ends_on(absorb)):
            self._unlink(w)
            if w.src is absorb:
                w.src = keep
            if w.dst is absorb:
                w.dst = keep
            self._link(w)
            slots.set_ends(w)
        self._unlink(absorb)
        del self._wires[absorb]
        del self.wire_by_uid[absorb.uid]
        slots.kill([absorb.slot])
        self._link(keep)
        slots.set_ends(keep)
        self._nets_dirty = True
        self.revision += 1

    def detach(self, wire: Wire, side: str) -> None:
        """Free one end ("src" or "dst") of a wire: it stays where it is, attached to
        nothing (see Wire). Its branches stay on it."""
        self._unlink(wire)
        setattr(wire, side, wire)
        self._link(wire)
        self._wire_slots.set_ends(wire)
        self._nets_dirty = True
        self.revision += 1

    def attach(self, wire: Wire, side: str, end: Endpoint) -> None:
        """Undo a detach: plug a free end back onto what it was on. (No checks: for
        anything else, wire anew.)"""
        self._unlink(wire)
        setattr(wire, side, end)
        self._link(wire)
        self._wire_slots.set_ends(wire)
        self._nets_dirty = True
        self.revision += 1

    def attachments(self, wire: Wire) -> list[Wire]:
        """Wires with an end on `wire` (branches, stubs, extra drivers)."""
        return _by_uid(self.ends_on(wire))

    def wires_at(self, pin: Pin) -> list[Wire]:
        return _by_uid(self.ends_on(pin))

    def ends_on(self, end: Endpoint) -> tuple[Wire, ...] | list[Wire]:
        """The wires with an end on this pin or wire, in no particular order (fast; don't modify)."""
        had = self._at.get(end)
        if had is None:
            return ()
        return had if type(had) is list else (had,)

    def with_descendants(self, slots: np.ndarray) -> np.ndarray:
        """descendants(), by wire slot and with the given ones too: every wire attached
        to these, to those, and so on, as sorted slots. With arrays: no walk per wire."""
        ws = self._wire_slots
        n = len(ws.wires)
        found = np.zeros(n, bool)
        found[slots] = True
        # the wires with an end on another wire, and those ends (heads: lane 0 on lane 0)
        kids = np.flatnonzero(ws.alive[:n] & (ws.lane[:n] == 0) & ws.end_is_wire[:n].any(axis=1))
        is_wire = ws.end_is_wire[kids]
        on = np.where(is_wire, ws.end_slot[kids], 0)
        new = found
        while kids.size:
            hit = ((new[on] & is_wire).any(axis=1)) & ~found[kids]
            if not hit.any():
                break
            new = np.zeros(n, bool)
            new[kids[hit]] = True
            found |= new
        return np.flatnonzero(found)

    def descendants(self, *wires: Wire) -> list[Wire]:
        """Wires attached to these, wires attached to those, and so on (in creation order)."""
        found: set[Wire] = set()
        todo = list(wires)
        while todo:
            for w in self.ends_on(todo.pop()):
                if w not in found:
                    found.add(w)
                    todo.append(w)
        return _by_uid(found.difference(wires))

    # ---- nets -------------------------------------------------------------

    def _rebuild_nets(self) -> None:
        """Group pins and wires into nets: the connected components of a graph whose
        nodes are pin slots and wires, and whose edges are wire ends and macro links."""
        store, ws = self._pins, self._wire_slots
        n_pins, n_wires = store.n, len(ws.wires)
        live = np.flatnonzero(
            ws.alive[:n_wires]
        )  # wire slots; a wire's node is n_pins + its slot
        ends = np.where(
            ws.end_is_wire[live], n_pins + ws.end_slot[live], ws.end_slot[live]
        )
        links = [self._links[s] for s in self._linked]  # macro pins <-> ports, joins
        pairs = np.concatenate(links).astype(np.intp) if links else np.empty((0, 2), np.intp)
        link_a, link_b = pairs[:, 0], pairs[:, 1]
        a = np.concatenate((n_pins + live, n_pins + live, link_a))
        b = np.concatenate((ends[:, 0], ends[:, 1], link_b))
        n = n_pins + n_wires
        member = np.zeros(
            n, bool
        )  # in some net: a wire, or a pin with a wire or link on it
        member[a] = member[b] = True
        member[:n_pins] &= store.alive[:n_pins]
        graph = coo_matrix((np.ones(len(a), bool), (a, b)), shape=(n, n))
        _, label = connected_components(graph, directed=False)
        _, net = np.unique(
            label[member], return_inverse=True
        )  # net numbers: 0, 1, 2, ...
        net_of = np.full(n, -1, np.intp)
        net_of[member] = net
        n_nets = int(net.max()) + 1 if net.size else 0

        in_net = member[:n_pins]
        reader = store.reader[:n_pins]
        weak = store.weak[:n_pins]
        drivers, driven, starts = _grouped(
            np.flatnonzero(in_net & ~reader & ~weak), net_of
        )
        counts = np.diff(np.append(starts, len(drivers)))  # drivers per driven net
        solo = counts == 1
        self._solo, self._solo_net = drivers[starts[solo]], driven[solo]
        self._drivers, self._driven, self._drv_starts = _grouped(
            drivers[np.repeat(~solo, counts)], net_of
        )
        self._readers = np.flatnonzero(in_net & reader)
        self._reader_net = net_of[self._readers]
        store.states[:n_pins][store.alive[:n_pins] & reader & ~in_net] = (
            Z  # unconnected: floating
        )

        # Weak drivers: only the highest priority ones on each net count. Priorities only
        # change by editing (props_changed), so they're settled here, not every step.
        pulls = np.flatnonzero(in_net & weak)
        if pulls.size:
            prio = np.array(
                [_priority(self._handle(int(store.part[i]))) for i in pulls.tolist()],
                np.int64,
            )
            top = np.full(n_nets, np.iinfo(np.int64).min)
            np.maximum.at(top, net_of[pulls], prio)
            pulls = pulls[prio == top[net_of[pulls]]]
        self._weak, self._weak_nets, self._weak_starts = _grouped(pulls, net_of)

        self._wire_net = net_of[n_pins:]
        self.net_value = np.zeros(n_nets, CODE)
        self.net_conflict = np.zeros(n_nets, bool)

        # The same, by net number: what sim/dirty.py looks up
        self._pin_net = net_of[:n_pins].copy()
        self._solo_of = np.full(n_nets, -1, np.intp)
        self._solo_of[self._solo_net] = self._solo
        self._multi_at, self._drv_count = _groups_by_net(
            n_nets, self._driven, self._drv_starts, len(self._drivers)
        )
        self._weak_at, self._weak_count = _groups_by_net(
            n_nets, self._weak_nets, self._weak_starts, len(self._weak)
        )
        order = np.argsort(self._reader_net, kind="stable")
        self._rd_sorted = self._readers[order]
        self._rd_start = np.searchsorted(self._reader_net[order], np.arange(n_nets + 1))
        self._rd_count = np.diff(self._rd_start)
        self._full = True
        self._nets_dirty = False
        self._nets_version += 1
        self._changed_all = True  # net numbers mean something else now; pins were reset
        self._carry()  # readers see their (new) nets now, not a tick later with stale values

    def wire_lanes(self, wire: Wire) -> Logic:
        """Every lane of a wire's value (lane 0 first), as of the last step: a bus's
        nets, as a Logic array (a lane with no net reads Z)."""
        lanes, _ = self.lanes_of_wires(np.array([wire.slot], np.intp))
        nets = self.wire_nets(lanes)
        codes = np.full(len(lanes), Z, CODE)
        if len(self.net_value):
            codes[nets >= 0] = self.net_value[nets[nets >= 0]]
        return Logic.of_codes(codes)

    def wire_state(self, wire: Wire) -> tuple[Level, bool]:
        """(value, conflict) of the net this wire belongs to, as of the last step."""
        if self._nets_dirty:
            self._rebuild_nets()
        i = self._wire_net[wire.slot]
        return Level(int(self.net_value[i])), bool(self.net_conflict[i])

    @property
    def pin_count(self) -> int:
        """Pin slots handed out so far (they're never reused)."""
        return self._pins.n

    def pin_slots_of(self, parts: list[Part]) -> tuple[np.ndarray, np.ndarray]:
        """The pins of these parts as slots, without making Pin objects: (all of them,
        part by part in pins order, and how many each part has)."""
        return self._pins_of(np.fromiter((p.slot for p in parts), np.intp, len(parts)), True)

    def _pins_of(self, slots: np.ndarray, counts_too: bool = False):
        """pin_slots_of, by part slot (just the pins, unless `counts_too`)."""
        pins, lanes, _ = self._shape_cols()
        sh = self._shape[slots]
        counts = pins[sh]
        flat = self._lanes_of(slots)
        if int(lanes[sh].sum()) != len(flat) or len(flat) != int(counts.sum()):
            flat = flat[self._pins.lane[flat] == 0]  # (buses: just their heads)
        return (flat, counts) if counts_too else flat

    def lanes_of_pins(self, heads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The lane slots of these pins (by head slot), pin after pin, and how many
        each has: for showing a bus as a whole."""
        w = self._pins.width[heads].astype(np.intp)
        return np.repeat(heads - (np.cumsum(w) - w), w) + np.arange(int(w.sum())), w

    def lanes_of_wires(self, heads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """lanes_of_pins, for wires (wire slots): their lane slots, and how many each has."""
        w = self._wire_slots.width[heads].astype(np.intp)
        return np.repeat(heads - (np.cumsum(w) - w), w) + np.arange(int(w.sum())), w

    def pin_codes(self, slots: np.ndarray) -> np.ndarray:
        """The logic codes of these pin slots: Pin.state for many pins at once."""
        return self._pins.states[slots]

    @property
    def nets_version(self) -> int:
        """Changes whenever net numbers do (the wiring changed and nets were rebuilt):
        what wire_nets gave before is stale then."""
        if self._nets_dirty:
            self._rebuild_nets()
        return self._nets_version

    def wire_nets(self, slots: np.ndarray) -> np.ndarray:
        """The nets of these wire slots (-1: none, e.g. gone). Valid until nets_version
        changes; index net_value / net_conflict with them (see ui/sync.py)."""
        if self._nets_dirty:
            self._rebuild_nets()
        known = slots < len(self._wire_net)
        net = np.full(len(slots), -1, np.intp)
        net[known] = self._wire_net[slots[known]]
        return net

    def wire_states(
        self, slots: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """wire_state for many wire slots at once: (value codes, conflicts, has a net)."""
        net = self.wire_nets(slots)
        has = net >= 0
        if not len(self.net_value):
            return np.zeros(len(slots), CODE), np.zeros(len(slots), bool), has
        net = np.maximum(net, 0)
        return self.net_value[net], self.net_conflict[net], has

    def take_changes(self) -> tuple[bool, set[Part], list[Wire]]:
        """What changed since the last call: (everything?, parts whose pins changed,
        board wires whose net changed). With everything=True, the rest is empty."""
        if self._nets_dirty:
            self._rebuild_nets()
        states = self._pins.states[: self._pins.n]
        if (
            self._changed_all
            or len(self._shown) != len(states)
            or len(self._shown_value) != len(self.net_value)
        ):
            result = True, set(), []
        else:
            store = self._pins
            changed_pins = np.flatnonzero(states != self._shown)
            changed_pins = changed_pins[store.alive[changed_pins]]
            parts = {self._handle(s) for s in np.unique(store.part[changed_pins]).tolist()}
            changed = (self.net_value != self._shown_value) | (
                self.net_conflict != self._shown_conflict
            )
            wires = []
            if changed.any():  # the board wires on those nets
                ws, net = self._wire_slots, self._wire_net
                n = len(net)
                on = np.flatnonzero(
                    ws.alive[:n]
                    & ws.board[:n]
                    & (ws.lane[:n] == 0)
                    & changed[np.maximum(net, 0)]
                    & (net >= 0)
                )
                wires = [ws.wires[i] for i in on.tolist()]
            result = False, parts, wires
        self._shown = states.copy()
        self._shown_value, self._shown_conflict = (
            self.net_value.copy(),
            self.net_conflict.copy(),
        )
        self._changed_all = False
        return result

    # ---- simulation ----------------------------------------------------

    # step() and run_until_stable() are bound in __init__, from the stepper the engine
    # config names: sim/plain.py (dirty=off) or sim/dirty.py (dirty=adaptive).

    def write_pins(self, slots: np.ndarray, codes: np.ndarray) -> None:
        """Pin.state for many pins at once (driving inputs from outside): the fast way."""
        self._pins.states[slots] = codes
        self._poked.append(slots)

    def _resolve(self) -> tuple[np.ndarray, np.ndarray]:
        """Every net's value from its drivers, and whether they fight. The value is the
        OR of the drivers' codes (logic.resolve): none or only Z -> Z, disagreeing -> X.
        Nets the strong drivers leave at Z go to their pulls."""
        states = self._pins.states
        value = np.zeros(len(self.net_value), CODE)
        conflict = np.zeros(len(self.net_value), bool)
        value[self._solo_net] = states[self._solo]
        if self._drivers.size:
            d = states[self._drivers]
            value[self._driven] = resolve(d, self._drv_starts)
            conflict[self._driven] = fights(d, self._drv_starts)
        if self._weak.size:
            w = states[self._weak]
            nets = self._weak_nets
            free = value[nets] == Z
            value[nets[free]] = resolve(w, self._weak_starts)[free]
            conflict[nets[free]] = fights(w, self._weak_starts)[free]
        return value, conflict

    def _carry(self) -> None:
        """Resolve every net and hand every reader its net's value."""
        value, conflict = self._resolve()
        self.net_value, self.net_conflict = value, conflict
        self._pins.states[self._readers] = value[self._reader_net]

    def frame(self) -> None:
        """Once per frame (not per step): the frame hook of every live part that has one."""
        now = time.monotonic()
        for t, slots in self._by_kind().items():
            if t.has("frame") and t.kind not in self.faults:
                live = slots[self._live[slots]]
                if live.size:
                    ctx = Ctx(_Handles(self, live), self.tick, now)
                    self._guard(t, "frame", lambda: t.frame(ctx))

    def face_codes(self, t: PartType, slots: np.ndarray) -> np.ndarray:
        """What t.face() says for these parts (slots, all of type t): logic codes, one
        row per part and a column per face mark without a pin. Parts that aren't live
        show 0, and a faulted kind (or a raising hook) shows X."""
        k = sum(m.pin is None for m in t.look.face)
        out = np.full((len(slots), k), ZERO, CODE)
        n = self._n_part_slots
        ok = slots < n
        ok[ok] = self._alive[slots[ok]] & self._live[slots[ok]]
        live = slots[ok]
        if not live.size:
            return out
        if t.kind in self.faults:
            out[ok] = X
            return out
        rows = np.flatnonzero(ok)
        sids = self._shape[live]
        for sid in _in_order(sids):
            m = sids == sid
            sub, sh = live[m], self._shapes[sid]
            first = self._pin0[sub].astype(np.intp)
            ins = [
                Logic.of_codes(self._pins.states[_lanes(first + sh.off[i], sh.widths[i])])
                for i in range(sh.n_in)
            ]
            ctx = Ctx(_Handles(self, sub), self.tick)
            raw = self._guard(t, "face", lambda: _values(t.face(ctx, *ins), k, len(sub), "face"))
            out[rows[m]] = X if raw is _FAILED else np.column_stack(raw)
        return out

    # ---- part types --------------------------------------------------------

    def _eval_batches(self) -> list[_Batch]:
        """Per kind with an eval: the instances step() evaluates (live ones only, unless
        pure) and index arrays of their pins."""
        if self._batches_dirty:
            self._batches = []
            for t, slots in self._by_kind().items():
                if not t.has("eval"):
                    continue
                if not t.pure:
                    slots = slots[self._live[slots]]
                if not slots.size:
                    continue
                sids = self._shape[slots]
                shapes = _in_order(sids)
                for sid in shapes:  # (one batch per shape: its pins, a pin's lanes one array)
                    sub = slots if len(shapes) == 1 else slots[sids == sid]
                    sh = self._shapes[sid]
                    w, n_in, n_out = sh.widths, sh.n_in, sh.n_pins - sh.n_in
                    first = self._pin0[sub].astype(np.intp)
                    if sh.joins:  # (joined outputs drive from their own pins: Part.drives)
                        drives = [self._drives[s] for s in sub.tolist()]
                        heads = [np.array([d[i] for d in drives], np.intp) for i in range(n_out)]
                    else:
                        heads = [first + sh.off[n_in + i] for i in range(n_out)]
                    self._batches.append(
                        _Batch(
                            t,
                            _Handles(self, sub),
                            [_lanes(first + sh.off[i], w[i]) for i in range(n_in)],
                            [_lanes(h, w[n_in + i]) for i, h in enumerate(heads)],
                            sub,
                            w[n_in:],
                            not sh.tabulable,
                        )
                    )
            n = self._n_part_slots
            self._batch_of = np.full(n, -1, np.intp)
            self._batch_pos = np.zeros(n, np.intp)
            self._n_pure = 0
            for b, batch in enumerate(self._batches):
                self._batch_of[batch.slots] = b
                self._batch_pos[batch.slots] = np.arange(len(batch.slots))
                if batch.type.pure:
                    self._n_pure += len(batch.slots)
            self._batches_dirty = False
            self._full = True
        return self._batches

    def _by_kind(self) -> dict[PartType, np.ndarray]:
        """The parts there are, by type, as slots: the board's in the order they were
        added, then the hidden ones in the order they were made (the order the batches,
        and with them settling's random draws, go in)."""
        if self._kinds_dirty:
            n = self._n_part_slots
            alive = self._alive[:n]
            board = np.array([p.slot for p in self._parts], np.intp)
            hidden = np.flatnonzero(alive & (self._owner[:n] >= 0))
            order = np.concatenate((board, hidden))
            tids = self._type_id[order]
            self._kinds = {}
            if order.size:
                # types in order of first appearance; each one's slots in that order
                uniq, first = np.unique(tids, return_index=True)
                by = np.argsort(tids, kind="stable")
                bounds = np.searchsorted(tids[by], uniq)
                groups = np.split(order[by], bounds[1:])
                for k in np.argsort(first, kind="stable").tolist():
                    self._kinds[self._types[uniq[k]]] = groups[k]
            self._kinds_dirty = False
        return self._kinds

    def _guard(self, t: PartType, hook: str, fn: Callable[[], Any]) -> Any:
        """Run one of t's hooks. If it raises, t is disabled in this circuit (its
        outputs go X and its hooks stop being called) and the error is kept."""
        try:
            return fn()
        except Exception:
            if t.kind not in self.faults:
                detail = traceback.format_exc(limit=-1).strip().splitlines()[-1]
                self.faults[t.kind] = msg = f"{t.kind}.{hook}: {detail}"
                self.errors.append(msg)
                self._changed_all = True
                for s in self._by_kind().get(t, np.empty(0, np.intp)).tolist():
                    self._pins.states[self._drive_slots(s)] = X
                self._full = True
                self._batches_dirty = True  # (evaluators leave faulted kinds out)
            return _FAILED


@dataclass
class _Batch:
    type: PartType
    parts: Sequence[Part]  # (handles, made when asked for: _Handles)
    ins: list[np.ndarray]  # per input pin: the instances' pin slots ((n, lanes) if wide)
    outs: list[np.ndarray]  # per output pin: where its values go (Part.drives); as ins
    slots: np.ndarray  # the instances' part slots
    out_widths: tuple[int, ...] = ()  # per output pin: its lanes (() : the type's, 1 each)
    no_table: bool = False  # a bus, or pins other than the type's: no lookup table


def _lanes(heads: np.ndarray, width: int) -> np.ndarray:
    """Pin slots of these pins' lanes: the heads as they are for one lane, else (n, width)."""
    return heads if width == 1 else heads[:, None] + np.arange(width)


def _pairs(a: int, wa: int, b: int, wb: int) -> list[tuple[int, int]]:
    """_lane_pairs, the widths given."""
    if wa != wb:
        return []
    return [(a, b)] if wa == 1 else [(a + i, b + i) for i in range(wa)]


def _lane_pairs(store: _PinStates, a: int, b: int) -> list[tuple[int, int]]:
    """Link pins a and b (heads) lane by lane. Pins of different widths join nothing."""
    w = store.width[a]
    if store.width[b] != w:
        return []
    return [(a, b)] if w == 1 else [(a + i, b + i) for i in range(int(w))]


def _one_by_one(t: PartType) -> bool:
    """Made one at a time (_make), not a run at once: a macro, or a part with joins
    (or its own layout(), which may have some)."""
    return bool(t.joins) or getattr(t, "body", None) is not None or layout_props(t) is None


def _in_order(ids: np.ndarray) -> list[int]:
    """The distinct values, in order of first appearance."""
    uniq, first = np.unique(ids, return_index=True)
    return uniq[np.argsort(first)].tolist()


def _by_uid(wires) -> list[Wire]:
    """Creation order: uids grow as wires are made, and a wire's parents are always older
    (a merge keeps the older wire), so this also puts parents before children."""
    return sorted(wires, key=lambda w: w.uid)


def _groups_by_net(
    n_nets: int, nets: np.ndarray, starts: np.ndarray, total: int
) -> tuple[np.ndarray, np.ndarray]:
    """For _grouped's output: per net, its group (-1: none); per group, its size."""
    at = np.full(n_nets, -1, np.intp)
    at[nets] = np.arange(len(nets))
    return at, np.diff(np.append(starts, total)).astype(np.intp)


def _grouped(
    slots: np.ndarray, net_of: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pin slots sorted by net, plus the nets (ascending) and where each one's pins start:
    the shape logic.resolve takes."""
    slots = slots[np.argsort(net_of[slots], kind="stable")]
    nets, starts = np.unique(net_of[slots], return_index=True)
    return slots, nets, starts


class _Blueprint:
    """A macro instance as built (rows r0.., pins p0.., wires w0.. up to now): every
    column relative to those, so the next instance is the same arrays plus offsets
    (Circuit._stamp). Taken right after the first one is built, before any step: its
    pins still hold their power-on codes."""

    def __init__(self, c: Circuit, r0: int, p0: int, w0: int) -> None:
        r1, p1, w1 = c._n_part_slots, c._pins.n, len(c._wire_slots.wires)
        rows, pins, wires = slice(r0, r1), slice(p0, p1), slice(w0, w1)
        self.type_id = c._type_id[rows].copy()
        self.uid = c._uid[rows].copy()
        self.owner = c._owner[rows] - r0
        self.pin0 = c._pin0[rows] - p0
        self.shape = c._shape[rows].copy()
        self.label = c._label[rows].copy()
        st = c._pins
        self.pin_part = st.part[pins] - r0
        self.pin_index = st.index[pins].copy()
        self.pin_is_input = st.is_input[pins].copy()
        self.pin_states = st.states[pins].copy()
        self.pin_weak = st.weak[pins].copy()
        self.pin_reader = st.reader[pins].copy()
        self.pin_passive = st.passive[pins].copy()
        self.pin_lane = st.lane[pins].copy()
        self.pin_width = st.width[pins].copy()
        ws = c._wire_slots
        self.end_is_wire = ws.end_is_wire[wires].copy()
        self.end_slot = np.where(self.end_is_wire, ws.end_slot[wires] - w0, ws.end_slot[wires] - p0)
        self.wire_uid = ws.uid[wires].copy()
        self.wire_lane = ws.lane[wires].copy()
        self.wire_width = ws.width[wires].copy()
        # (few: the macro instances inside, joined parts)
        self.inner = [
            (s - r0, c._inner[s] - r0, c._inner_wires[s] - w0) for s in range(r0, r1) if s in c._inner
        ]
        self.links = [(s - r0, c._links[s] - p0) for s in range(r0, r1) if s in c._links]
        self.drives = [
            (s - r0, [d - p0 for d in c._drives[s]]) for s in range(r0, r1) if s in c._drives
        ]
        # the types inside, by kind: still what the registry says? (see current)
        self.kinds = {c._types[i].kind: c._types[i] for i in set(self.type_id[1:].tolist())}

    def current(self, registry) -> bool:
        """Is every kind inside still the type it was (a macro inside edited since)?"""
        try:
            return all(registry.get(k) is t for k, t in self.kinds.items())
        except KeyError:
            return False


class _Shape:
    """A pin layout: a type and its Layout (pin names, widths, joins). Lane slots go
    input by input, then output by output, each pin's lanes in a row. Also what every
    new plain part (no joins, no body) of the shape starts as, lane by lane: its pins'
    indices, sides, power-on codes and weak flags; and its props."""

    def __init__(self, t: PartType, lay: Layout, type_id: int) -> None:
        n_in, n_out = len(lay.ins), len(lay.outs)
        self.type, self.type_id, self.layout = t, type_id, lay
        self.n_pins = n_in + n_out
        self.widths = lay.widths
        self.joins = lay.joins
        self.wide = lay.wide
        # an eval can be tabulated (sim/lut.py) only as the type's own, one-lane pins
        self.tabulable = not lay.wide and (lay.ins, lay.outs) == (t.ins, t.outs)
        w = np.array(self.widths, np.intp).reshape(-1)
        self.off = np.cumsum(w) - w  # per pin (ins, then outs): its head, from the first
        self.offs = tuple(self.off.tolist())  # (the same, for one at a time)
        self.n_in = n_in
        self.n_lanes = int(w.sum())
        self.in_lanes = int(w[:n_in].sum())
        pin = np.repeat(np.arange(self.n_pins), w)  # per lane: its pin
        self.lane = (np.arange(self.n_lanes) - self.off[pin]).astype(np.int16)
        self.width = w[pin].astype(np.int16)
        self.is_input = pin < n_in
        self.index = np.where(self.is_input, pin, pin - n_in).astype(np.int32)
        power_on = X if t.has("eval") else ZERO  # (the IN switch starts off)
        self.initial = np.where(self.is_input, CODE(Z), CODE(power_on)).astype(CODE)
        weak = np.array([False] * n_in + [name in t.weak for name in lay.outs], bool)
        self.weak = weak[pin]
        self.props = _props_maker(t)


def _props_maker(t: PartType) -> Callable[[], dict]:
    """What makes a new instance's props (fresh_props): a plain copy of one fresh set
    when every value in it is immutable, so no two parts can end up sharing one."""
    template = fresh_props(t)
    return template.copy if flat(template) else lambda: fresh_props(t)


def _objects(values: list) -> np.ndarray:
    """An object array of these values, each one an item (a dict stays one)."""
    out = np.empty(len(values), object)
    for i, v in enumerate(values):
        out[i] = v
    return out


def _one_type(parts: list[Part]) -> PartType:
    types = {p.type for p in parts}
    if len(types) != 1:
        raise ValueError("settings and actions work on parts of one kind at a time")
    return types.pop()


def _priority(part: Part) -> int:
    try:
        return int(part.props.get("priority", 0))
    except (TypeError, ValueError):
        return 0


def _evaluate(
    t: PartType, ctx: Ctx, ins: list[np.ndarray], out_widths: tuple[int, ...] = ()
) -> list[np.ndarray]:
    """Run t.eval on the input codes; the output codes, one array of ctx.n per pin.

    API 2 scripts get Logic arrays. API 1 scripts were written for plain bools, so
    they get bools (X and Z read as 0), and a pure one's outputs are X wherever any
    of its inputs isn't a known 0 or 1: it can't know what its function does with X.
    """
    if t.api >= 2:
        raw = t.eval(ctx, *(Logic.of_codes(c) for c in ins))
        k = len(out_widths) if out_widths else len(t.outs)
        return _values(raw, k, ctx.n, "eval", out_widths)
    outs = _outputs(t, t.eval(ctx, *(c == ONE for c in ins)), ctx.n)
    if t.pure and ins:
        unknown = np.logical_or.reduce([(c != ZERO) & (c != ONE) for c in ins])
        if unknown.any():
            outs = [np.where(unknown, CODE(X), o) for o in outs]
    return outs


def _outputs(t: PartType, raw: Any, n: int) -> list[np.ndarray]:
    """Normalize what eval returned: one array of n logic codes per output pin."""
    return _values(raw, len(t.outs), n, "eval")


def _values(
    raw: Any, k: int, n: int, hook: str, widths: tuple[int, ...] = ()
) -> list[np.ndarray]:
    """k values (one per output pin, or face mark) of n instances each, as codes: (n,)
    each, or (n, w) for an output w lanes wide (widths: per value; () : all 1)."""
    if k == 0:
        return []
    if k == 1 and not (isinstance(raw, tuple) and len(raw) == 1):
        raw = (
            raw,
        )  # one value: anything but a 1-tuple is it (a scalar, list or array)
    if not isinstance(raw, tuple) or len(raw) != k:
        raise ValueError(f"{hook} returned {raw!r}; expected {k} value(s)")
    if not widths or all(w == 1 for w in widths):
        return [np.broadcast_to(codes(v), (n,)) for v in raw]
    return [
        np.broadcast_to(codes(v), (n,)) if w == 1 else _wide(v, n, w, hook)
        for v, w in zip(raw, widths)
    ]


def _wide(v: Any, n: int, w: int, hook: str) -> np.ndarray:
    """A wide output's value as (n, w) codes. Numbers aren't taken: 5 could be one lane
    or bits (Logic.of_ints says which)."""
    if not isinstance(v, (Logic, Level)) and np.asarray(v).dtype.kind not in "b":
        if not (isinstance(v, (list, tuple)) and all(isinstance(x, (Logic, Level)) for x in v)):
            raise ValueError(
                f"{hook}: a {w}-lane output takes Logic (or Levels, or bools), not "
                f"{type(v).__name__}: use Logic.of_ints for numbers"
            )
    c = codes(v)
    return np.broadcast_to(c, (n, w))
