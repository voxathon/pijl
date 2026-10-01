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
    arrays (PartType.eval), which is what lets big boards go fast later.
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
from types import MappingProxyType
from collections.abc import Sequence
from typing import Any, Callable, Iterable, Mapping, Union

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..logic import CODE, ONE, X, Z, ZERO, Level, Logic, codes, fights, resolve
from ..parts import Ctx, PartType, Registry, builtin_registry, fresh_props
from ..parts.registry import flat

_FAILED = object()  # what Circuit._guard returns when the hook raised


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
        # Pin handles by slot, made on first use; None again once the pin is gone
        self.pins: list[Pin | None] = []

    _COLS = ("states", "reader", "alive", "weak", "part", "index", "is_input", "passive")

    def new(
        self, part: int, index: int, is_input: bool, initial: Level = Z, weak: bool = False
    ) -> int:
        """One more pin (no handle yet); its slot."""
        slot = self.n
        self._room(slot + 1)
        self.n += 1
        self.pins.append(None)
        self.states[slot] = initial
        self.reader[slot] = is_input
        self.alive[slot] = True
        self.weak[slot] = weak
        self.part[slot], self.index[slot], self.is_input[slot] = part, index, is_input
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
    ) -> None:
        """new() for many pins at once: the next len(part) slots, in order. (`reader`,
        `passive`: when they aren't just is_input and False.)"""
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
        if passive is not None:
            self.passive[first:end] = passive

    def handle(self, slot: int) -> Pin:
        pin = self.pins[slot]
        if pin is None:
            pin = self.pins[slot] = Pin.__new__(Pin)
            pin._store, pin.slot, pin._part = self, slot, None
            pin.index, pin.is_input = int(self.index[slot]), bool(self.is_input[slot])
        return pin

    def kill(self, slots: list[int]) -> None:
        """These pins' parts are gone."""
        self.alive[slots] = False
        for slot in slots:
            self.pins[slot] = None

    def _room(self, n: int) -> None:
        """Arrays long enough for n pins."""
        while n > len(self.states):
            for name in self._COLS:
                old = getattr(self, name)
                setattr(self, name, np.concatenate((old, np.zeros_like(old))))


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

    _COLS = ("alive", "board", "end_is_wire", "end_slot", "uid")

    def _new_slot(self) -> int:
        slot = len(self.wires)
        if slot == len(self.alive):
            for name in self._COLS:
                old = getattr(self, name)
                setattr(self, name, np.concatenate((old, np.zeros_like(old))))
        self.wires.append(None)
        self.alive[slot] = True
        return slot

    def add(self, wire: Wire, board: bool) -> None:
        wire.slot = slot = self._new_slot()
        self.wires[slot] = wire
        self.board[slot], self.uid[slot] = board, wire.uid
        self.set_ends(wire)

    def add_hidden(self, uid: int, ends: list[tuple[bool, int]]) -> int:
        """A wire inside a macro, by its ends (is it a wire?, slot); no object. Its slot."""
        slot = self._new_slot()
        self.board[slot], self.uid[slot] = False, uid
        for side, (is_wire, end) in enumerate(ends):
            self.end_is_wire[slot, side], self.end_slot[slot, side] = is_wire, end
        return slot

    def add_hidden_block(self, end_is_wire: np.ndarray, end_slot: np.ndarray, uid: np.ndarray) -> None:
        """add_hidden for many wires at once: the next len(uid) slots."""
        first, k = len(self.wires), len(uid)
        end = first + k
        while end > len(self.alive):
            for name in self._COLS:
                old = getattr(self, name)
                setattr(self, name, np.concatenate((old, np.zeros_like(old))))
        self.wires.extend([None] * k)
        self.alive[first:end] = True
        self.board[first:end] = False
        self.end_is_wire[first:end] = end_is_wire
        self.end_slot[first:end] = end_slot
        self.uid[first:end] = uid

    def handle(self, slot: int) -> Wire:
        """The Wire of this slot (made now for a hidden one)."""
        wire = self.wires[slot]
        if wire is None:
            ends = [
                self.handle(int(e)) if w else self.pins.handle(int(e))
                for w, e in zip(self.end_is_wire[slot].tolist(), self.end_slot[slot].tolist())
            ]
            wire = self.wires[slot] = Wire(ends[0], ends[1], int(self.uid[slot]), slot)
        return wire

    def kill(self, slots: list[int]) -> None:
        """These wires are gone."""
        self.alive[slots] = False
        for slot in slots:
            self.wires[slot] = None

    def set_ends(self, wire: Wire) -> None:
        for side, end in enumerate(wire.ends):
            self.end_is_wire[wire.slot, side] = isinstance(end, Wire)
            self.end_slot[wire.slot, side] = end.slot


class Pin:
    """A handle on one pin slot (see _PinStates): one per slot, so `is` works. What
    never changes about a pin (index, side, its part) it keeps itself: the editor
    asks all the time."""

    __slots__ = ("_store", "slot", "index", "is_input", "_part")

    @property
    def part(self) -> Part:
        part = self._part
        if part is None:
            s = self._store
            part = self._part = s.circuit._handle(int(s.part[self.slot]))
        return part

    @property
    def state(self) -> Level:
        return Level(int(self._store.states[self.slot]))

    @state.setter
    def state(self, value) -> None:
        """A Level, or a bool / 0 / 1."""
        self._store.states[self.slot] = codes(value)

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
    __slots__ = ("_c", "slot", "_ins", "_outs", "type", "uid", "kind", "label", "_props")

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
        """Scratch space for its PartType's hooks (made when first asked for)."""
        return self._c._state.setdefault(self.slot, {})

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._c._state[self.slot] = value

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

    def _pin_lists(self) -> None:
        c, t = self._c, self.type
        store = c._pins
        handles, p0, n_in = store.pins, int(c._pin0[self.slot]), len(t.ins)
        out = []
        for k in range(n_in + len(t.outs)):
            s = p0 + k
            pin = handles[s]
            if pin is None:  # (what store.handle does, knowing the answers already)
                pin = handles[s] = Pin.__new__(Pin)
                pin._store, pin.slot, pin._part = store, s, self
                pin.index, pin.is_input = (k, True) if k < n_in else (k - n_in, False)
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


@dataclass(eq=False, slots=True)
class Wire:
    # Two ends, each a Pin or another Wire. Normalized by Circuit.connect: an
    # output pin is always `src`, an input pin always `dst` -- so a plain
    # pin-to-pin wire reads src=output, dst=input like before junctions existed.
    src: Endpoint
    dst: Endpoint
    uid: int = 0  # stable identity, like Part.uid (wires can be endpoints of wires)
    slot: int = -1  # its place in the circuit's wire arrays (see _WireSlots)

    @property
    def ends(self) -> tuple[Endpoint, Endpoint]:
        return self.src, self.dst


class Circuit:
    def __init__(
        self, registry: Registry | None = None, settle_ticks: int = 0, seed: int = 0
    ) -> None:
        """`registry`: anything with get(kind) / `kind in` -- a Registry, or a
        macros.Catalog to have macros too."""
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
        self._label = np.full(256, None, object)
        self._props = np.full(256, None, object)
        self._handles = np.full(256, None, object)
        # ... and what few parts have, by slot
        self._inner: dict[int, np.ndarray] = {}  # macro instance -> its body's part slots
        self._inner_wires: dict[int, np.ndarray] = {}  # macro instance -> its wire slots
        self._links: dict[int, np.ndarray] = {}  # pin slot pairs joined into one net (k x 2)
        self._drives: dict[int, list[int]] = {}  # joined parts: per output, its driver pin
        self._state: dict[int, dict] = {}  # Part.state
        self._types: list[PartType] = []
        self._type_pins = np.zeros(0, np.intp)  # per type: its pin count (pin_slots_of)
        self._type_ids: dict[PartType, int] = {}
        self._templates: dict[PartType, _Template] = {}
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
        # What the UI was shown last (take_changes), to tell it what changed since.
        self._changed_all = True
        self._shown = np.zeros(0, bool)
        self._shown_value = self._shown_conflict = np.zeros(0, bool)

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
        self, types: list[PartType], uids: list[int | None], live: bool = True
    ) -> list[Part]:
        """add_part for many parts at once, by type (registry.get(kind)): pins, settling
        and the rest set up a whole type at a time rather than part by part."""
        given = []
        for uid in uids:
            if uid is None:
                uid = self._next_uid
            self._next_uid = max(self._next_uid, uid + 1)
            given.append(uid)
        parts = [self._handle(s) for s in self._make_many(types, given)]
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
    # macro instance it's inside, -1 on the board), its first pin slot (its pins are
    # the next len(ins) + len(outs) slots), label, props (None inside a macro until
    # asked for: then a copy of the body's). What few parts have is kept by slot in
    # dicts, as small int arrays: a macro's body parts and wires (_inner,
    # _inner_wires), links (pin slot pairs), joined outputs' driver pins (_drives);
    # and hook scratch space (_state).

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
        "_settle", "_type_id", "_uid", "_live", "_alive", "_owner", "_pin0",
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
            part.label, part._props = self._label[slot], self._props[slot]
        return part

    def _make_many(
        self, types: list[PartType], uids: list[int], owner: int = -1
    ) -> list[int]:
        """_make for each (type, uid), in order: the same rows and pin slots as making
        them one by one (their slots). Plain types (no joins, no body) are added a run
        at a time: rows and pins as arrays."""
        out: list[int] = []
        i, n = 0, len(types)
        while i < n:
            t = types[i]
            if t.joins or getattr(t, "body", None) is not None:
                out.append(self._make(t, uids[i], owner))
                i += 1
                continue
            j = i
            while j < n and not (types[j].joins or getattr(types[j], "body", None) is not None):
                j += 1
            out += self._make_plain(types[i:j], uids[i:j], owner)
            i = j
        if out:
            self._kinds_dirty = self._batches_dirty = self._nets_dirty = True
        return out

    def _make_plain(self, types: list[PartType], uids: list[int], owner: int) -> list[int]:
        """Rows and pins for a run of plain parts, as arrays (see _make_many)."""
        store = self._pins
        rows = self._new_rows(len(types))
        templates: dict[PartType, _Template] = {}
        tms = []
        for t in types:
            tm = templates.get(t)
            if tm is None:
                tm = templates[t] = self._templates.get(t) or self._template(t)
            tms.append(tm)
        counts = np.fromiter((len(tm.sides) for tm in tms), np.intp, len(tms))
        sl = slice(rows.start, rows.stop)
        self._type_id[sl] = [tm.type_id for tm in tms]
        self._uid[sl] = uids
        self._live[sl] = False
        self._alive[sl] = True
        self._owner[sl] = owner
        self._pin0[sl] = store.n + np.cumsum(counts) - counts
        self._label[sl] = [""] * len(tms)
        if owner < 0:  # (on the board: props now; inside a macro, see _expand)
            self._props[sl] = _objects([tm.props() for tm in tms])
        store.new_block(
            np.repeat(np.arange(rows.start, rows.stop, dtype=np.int32), counts),
            np.concatenate([tm.index for tm in tms]) if tms else np.empty(0, np.int32),
            np.concatenate([tm.is_input for tm in tms]) if tms else np.empty(0, bool),
            np.concatenate([tm.initial for tm in tms]) if tms else np.empty(0, CODE),
            np.concatenate([tm.weak for tm in tms]) if tms else np.empty(0, bool),
        )
        return list(rows)

    def _template(self, t: PartType) -> _Template:
        tm = self._templates[t] = _Template(t, self._type_of(t))
        return tm

    def _make(self, t: PartType, uid: int, owner: int = -1) -> int:
        """One part of any type (a macro's body is built too); its slot. A macro type's
        first instance is built part by part and recorded (_Blueprint); the rest are
        stamped from that: the same rows, pins and wires, as arrays."""
        body = getattr(t, "body", None)
        if body is not None:
            bp = self._blueprints.get(t)
            if bp is not None and bp.current(self.registry):
                return self._stamp(bp, t, uid, owner)
            marks = (self._n_part_slots, self._pins.n, len(self._wire_slots.wires))
        slot = self._make_one(t, uid, owner)
        if body is not None:
            self._blueprints[t] = _Blueprint(self, *marks)
        return slot

    def _make_one(self, t: PartType, uid: int, owner: int) -> int:
        slot = self._new_rows(1)[0]
        store = self._pins
        self._type_id[slot] = self._type_of(t)
        self._uid[slot], self._live[slot], self._alive[slot] = uid, False, True
        self._owner[slot], self._label[slot] = owner, ""
        self._props[slot] = fresh_props(t)
        power_on = X if t.has("eval") else ZERO  # (the IN switch starts off)
        joined = {name for group in t.joins for name in group}
        self._pin0[slot] = store.n
        for i in range(len(t.ins)):
            store.new(slot, i, True)
        for i, name in enumerate(t.outs):
            store.new(slot, i, False, power_on, name in t.weak and name not in joined)
        if joined:
            self._join(slot, t, power_on)
        if getattr(t, "body", None) is not None:
            self._expand(slot, t)
        self._kinds_dirty = self._batches_dirty = self._nets_dirty = True
        return slot

    def _stamp(self, bp: _Blueprint, t: PartType, uid: int, owner: int) -> int:
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
        self._label[sl] = bp.label
        self._props[sl] = None  # (Part.props copies the body's)
        self._props[r0] = fresh_props(t)
        store.new_block(
            bp.pin_part + r0, bp.pin_index, bp.pin_is_input, bp.pin_states, bp.pin_weak,
            bp.pin_reader, bp.pin_passive,
        )
        ws.add_hidden_block(
            bp.end_is_wire, np.where(bp.end_is_wire, bp.end_slot + w0, bp.end_slot + p0), bp.wire_uid
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
        slots = self._make_many([self.registry.get(d[0]) for _, d in kinds], [u for u, _ in kinds], inst)
        inner = dict(zip((u for u, _ in kinds), slots))  # (while building it)
        for (_uid, (_kind, label, _x, _y, _props)), s in zip(kinds, slots):
            if label:
                self._label[s] = label
        self._props[np.array(slots, np.intp)] = None  # (Part.props copies the body's)
        store = self._pins
        wires: dict[int, int] = {}
        mine = []
        for uid in sorted(body.wires):  # parents first
            ends = []
            for ref in body.wires[uid][:2]:
                if ref[0] == "w":
                    ends.append((True, wires[ref[1]]))
                else:
                    _, puid, is_input, index = ref
                    p = inner[puid]
                    first = int(self._pin0[p])
                    ends.append((False, first + (0 if is_input else len(self._types[self._type_id[p]].ins)) + index))
            wires[uid] = self._wire_slots.add_hidden(uid, ends)
            mine.append(wires[uid])
        self._inner[inst] = np.array(slots, np.int32)
        self._inner_wires[inst] = np.array(mine, np.int32)
        p0, n_in = int(self._pin0[inst]), len(t.ins)
        links = [(p0 + i, self._out_pin(inner[port], 0)) for i, port in enumerate(t.in_ids)]
        links += [(self._in_pin(inner[port], 0), p0 + n_in + i) for i, port in enumerate(t.out_ids)]
        links = self._links[inst] = np.array(links, np.int32).reshape(-1, 2)
        store.passive[links.ravel()] = True
        store.reader[links.ravel()] = True
        self._linked[inst] = None

    def _in_pin(self, part: int, i: int) -> int:
        return int(self._pin0[part]) + i

    def _out_pin(self, part: int, i: int) -> int:
        return int(self._pin0[part]) + len(self._types[self._type_id[part]].ins) + i

    def _join(self, slot: int, t: PartType, power_on: Level) -> None:
        """Make each of the type's join groups one net: link its pins, turn its outputs into
        pass-through pins, and give each of those a hidden pin to drive the net from."""
        store = self._pins
        p0, n_in = int(self._pin0[slot]), len(t.ins)
        drives = [p0 + n_in + i for i in range(len(t.outs))]
        named = {name: p0 + i for i, name in enumerate(t.ins)} | {
            name: p0 + n_in + i for i, name in enumerate(t.outs)
        }
        links = []
        for group in t.joins:
            pins = [named[name] for name in group]
            for i, name in enumerate(t.outs):
                if name in group:
                    out = p0 + n_in + i
                    store.passive[out] = True
                    store.reader[out] = True
                    drives[i] = store.new(slot, i, False, power_on, name in t.weak)
                    pins.append(drives[i])
            links += [(pins[0], p) for p in pins[1:]]
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
        return [ws.handle(s) for s in np.flatnonzero(ws.alive[:n] & ~ws.board[:n]).tolist()]

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
        types = self._types
        tids = self._type_id[slots]
        n_in = np.array([len(types[t].ins) for t in range(len(types))], np.intp)[tids]
        n_out = np.array([len(types[t].outs) for t in range(len(types))], np.intp)[tids]
        first = self._pin0[slots].astype(np.intp) + n_in
        return np.repeat(first - (np.cumsum(n_out) - n_out), n_out) + np.arange(int(n_out.sum()))

    def _drive_slots(self, slot: int) -> list[int]:
        """Where a part's outputs' eval values go (see Part.drives), as pin slots."""
        extra = self._drives.get(slot)
        if extra is not None:
            return extra
        t = self._types[self._type_id[slot]]
        first = int(self._pin0[slot]) + len(t.ins)
        return list(range(first, first + len(t.outs)))

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
        self._pins.kill(self._pins_of(at).tolist())
        self._wire_slots.kill([w for s in tree for w in np.asarray(self._inner_wires.get(s, ())).tolist()])
        self._settle[at] = 0
        self._alive[at] = False
        for s in tree:
            self._linked.pop(s, None)
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

    def connect(
        self, a: Endpoint, b: Endpoint, uid: int | None = None, check: bool = True
    ) -> tuple[Wire | None, list[Wire]]:
        """Connect two endpoints in either order. Returns (new_wire, replaced_wires).

        new_wire is None if the connection is invalid (see can_connect). An input
        pin takes one wire, so wiring into an already-wired input replaces the old
        wire, along with any branches hanging off it.

        `check=False` skips can_connect: for rebuilding wiring that existed before
        (undo, paste). Some of it can't be drawn by hand -- cut-deletion can splice a
        wire that runs from a part back into itself -- but it must come back as it was.
        """
        if check and not self.can_connect(a, b):
            return None, []
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
        wire = Wire(a, b, uid)
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
        keep.dst = far
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
        # the wires with an end on another wire, and those ends
        kids = np.flatnonzero(ws.alive[:n] & ws.end_is_wire[:n].any(axis=1))
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
        self._nets_dirty = False
        self._nets_version += 1
        self._changed_all = True  # net numbers mean something else now; pins were reset
        self._carry()  # readers see their (new) nets now, not a tick later with stale values

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
        if len(self._type_pins) != len(self._types):
            self._type_pins = np.array([len(t.ins) + len(t.outs) for t in self._types], np.intp)
        counts = self._type_pins[self._type_id[slots]]
        first = self._pin0[slots].astype(np.intp)
        flat = np.repeat(first - (np.cumsum(counts) - counts), counts) + np.arange(int(counts.sum()))
        return (flat, counts) if counts_too else flat

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

    def step(self) -> None:
        if self._nets_dirty:
            self._rebuild_nets()
        states = self._pins.states

        # Phase 1: every part computes outputs from current inputs, one kind at a time.
        # Compute all first, then write, so evaluation order doesn't matter.
        now = time.monotonic()
        results: list[tuple[_Batch, list[np.ndarray]]] = []
        for batch in self._eval_batches():
            t = batch.type
            if t.kind in self.faults:
                continue
            ctx = Ctx(batch.parts, self.tick, now)
            outs = self._guard(
                t, "eval", lambda: _evaluate(t, ctx, [states[idx] for idx in batch.ins])
            )
            if outs is not _FAILED:
                results.append((batch, outs))
        for batch, outs in results:
            if (
                self._settling
            ):  # settling parts take their new outputs only half the time
                keep = (self._settle[batch.slots] <= 0) | (
                    self.rng.random(len(batch.parts)) < 0.5
                )
                for idx, values in zip(batch.outs, outs):
                    states[idx[keep]] = values[keep]
            else:
                for idx, values in zip(batch.outs, outs):
                    states[idx] = values

        # Phase 2: every net resolves its drivers and hands the value to its readers.
        self._carry()
        self.tick += 1
        if self._settling:
            s = self._settle[: self._n_part_slots]
            s[s > 0] -= 1
            self._settling = int(np.count_nonzero(s))

    def _carry(self) -> None:
        """Resolve every net from its drivers and hand the value to its readers. The
        value is the OR of the drivers' codes (logic.resolve): none or only Z -> Z,
        disagreeing -> X. Nets the strong drivers leave at Z go to their pulls."""
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
        self.net_value, self.net_conflict = value, conflict
        states[self._readers] = value[self._reader_net]

    def frame(self) -> None:
        """Once per frame (not per step): the frame hook of every live part that has one."""
        now = time.monotonic()
        for t, slots in self._by_kind().items():
            if t.has("frame") and t.kind not in self.faults:
                live = slots[self._live[slots]]
                if live.size:
                    ctx = Ctx(_Handles(self, live), self.tick, now)
                    self._guard(t, "frame", lambda: t.frame(ctx))

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
                first = self._pin0[slots].astype(np.intp)
                n_in = len(t.ins)
                if t.joins:  # (joined outputs drive from their own pins: Part.drives)
                    drives = [self._drives[s] for s in slots.tolist()]
                    outs = [np.array([d[i] for d in drives], np.intp) for i in range(len(t.outs))]
                else:
                    outs = [first + n_in + i for i in range(len(t.outs))]
                self._batches.append(
                    _Batch(
                        t,
                        _Handles(self, slots),
                        [first + i for i in range(n_in)],
                        outs,
                        slots,
                    )
                )
            self._batches_dirty = False
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
            return _FAILED


@dataclass
class _Batch:
    type: PartType
    parts: Sequence[Part]  # (handles, made when asked for: _Handles)
    ins: list[np.ndarray]  # per input pin: the instances' pin slots
    outs: list[np.ndarray]  # per output pin: where its values go (Part.drives)
    slots: np.ndarray  # the instances' part slots


def _by_uid(wires) -> list[Wire]:
    """Creation order: uids grow as wires are made, and a wire's parents are always older
    (a merge keeps the older wire), so this also puts parents before children."""
    return sorted(wires, key=lambda w: w.uid)


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
        self.label = c._label[rows].copy()
        st = c._pins
        self.pin_part = st.part[pins] - r0
        self.pin_index = st.index[pins].copy()
        self.pin_is_input = st.is_input[pins].copy()
        self.pin_states = st.states[pins].copy()
        self.pin_weak = st.weak[pins].copy()
        self.pin_reader = st.reader[pins].copy()
        self.pin_passive = st.passive[pins].copy()
        ws = c._wire_slots
        self.end_is_wire = ws.end_is_wire[wires].copy()
        self.end_slot = np.where(self.end_is_wire, ws.end_slot[wires] - w0, ws.end_slot[wires] - p0)
        self.wire_uid = ws.uid[wires].copy()
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


class _Template:
    """What every new part of a plain type (no joins, no body) starts as: its type's
    id, its pins' indices and sides, their power-on codes and weak flags, its props."""

    def __init__(self, t: PartType, type_id: int) -> None:
        n_in, n_out = len(t.ins), len(t.outs)
        self.type_id = type_id
        self.sides = [(True, i) for i in range(n_in)] + [
            (False, i) for i in range(n_out)
        ]
        self.index = np.array([i for _, i in self.sides], np.int32)
        self.is_input = np.array([side for side, _ in self.sides], bool)
        power_on = X if t.has("eval") else ZERO  # (the IN switch starts off)
        self.initial = np.array([int(Z)] * n_in + [int(power_on)] * n_out, CODE)
        self.weak = np.array([False] * n_in + [name in t.weak for name in t.outs], bool)
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


def _evaluate(t: PartType, ctx: Ctx, ins: list[np.ndarray]) -> list[np.ndarray]:
    """Run t.eval on the input codes; the output codes, one array of ctx.n per pin.

    API 2 scripts get Logic arrays. API 1 scripts were written for plain bools, so
    they get bools (X and Z read as 0), and a pure one's outputs are X wherever any
    of its inputs isn't a known 0 or 1: it can't know what its function does with X.
    """
    if t.api >= 2:
        return _outputs(t, t.eval(ctx, *(Logic.of_codes(c) for c in ins)), ctx.n)
    outs = _outputs(t, t.eval(ctx, *(c == ONE for c in ins)), ctx.n)
    if t.pure and ins:
        unknown = np.logical_or.reduce([(c != ZERO) & (c != ONE) for c in ins])
        if unknown.any():
            outs = [np.where(unknown, CODE(X), o) for o in outs]
    return outs


def _outputs(t: PartType, raw: Any, n: int) -> list[np.ndarray]:
    """Normalize what eval returned: one array of n logic codes per output pin."""
    k = len(t.outs)
    if k == 0:
        return []
    if k == 1 and not (isinstance(raw, tuple) and len(raw) == 1):
        raw = (
            raw,
        )  # one output: anything but a 1-tuple is its value (a scalar, list or array)
    if not isinstance(raw, tuple) or len(raw) != k:
        raise ValueError(f"eval returned {raw!r}; expected {k} output value(s)")
    return [np.broadcast_to(codes(v), (n,)) for v in raw]
