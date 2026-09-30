"""Pure logic simulation. No pygame/pyglet/UI imports allowed in here.

Model:
  - A Part has input pins and output pins. What it does is up to its PartType
    (see pijl.parts): the circuit only knows the contract, never specific kinds.
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
    without infinite recursion. Parts are evaluated a whole kind at a time, on
    arrays (PartType.eval), which is what lets big boards go fast later.
  - Parts are *live* once placed. Ghosts (following the cursor before a click)
    aren't: they're never opened, and only pure parts among them are evaluated.
  - A macro instance is never evaluated: its body is added as *hidden* parts and
    wires (owned by the instance, not in `parts` / `wires`, which are only what's
    on the board), recursively. Its pins are joined straight into the nets of the
    body's IN / OUT ports, so the inside and outside of a macro are one net: no
    delay at the boundary, and wrapping something in a macro can't change timing.
  - Settling (`settle_ticks`): new parts are a little random for a while. For its
    first ticks, each part only takes its newly computed outputs about half the
    time. Without it, anything symmetric that appears all at once -- a latch
    being opened, pasted or placed inside a macro -- flips between both of its
    unstable states forever, since in a one-tick-per-gate world nothing ever
    breaks the tie. Real hardware settles on noise; this is the noise. Seeded,
    so the same actions give the same results. Off (0) unless asked for.

Speed: every pin's state lives in one numpy array (Pin.state reads and writes its
slot), so a step is array work, not a loop over pins. Each part kind keeps index
arrays of its instances' input and output pins; nets are index arrays of their
drivers (sorted by net, reduced with min / max) and of their readers. Wires have
slots too, with their ends kept in arrays as wiring changes, so finding the nets
again (scipy's connected components) doesn't loop over wires in Python either.
What changed for the UI (take_changes) is a diff of the state arrays against what
it was shown last.
"""

from __future__ import annotations

import time
import traceback
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Union

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..parts import Ctx, PartType, Registry, builtin_registry, fresh_props

_FAILED = object()  # what Circuit._guard returns when the hook raised


class _PinStates:
    """Every pin of a circuit, by slot: its state, and whether it reads its net."""

    def __init__(self) -> None:
        self.states = np.zeros(1024, bool)
        self.reader = np.zeros(1024, bool)  # an input or a pass-through pin: reads its net, never drives
        self.alive = np.zeros(1024, bool)   # its part is still in the circuit
        self.pins: list[Pin] = []

    def add(self, pin: Pin) -> int:
        slot = len(self.pins)
        if slot == len(self.states):
            for name in ("states", "reader", "alive"):
                old = getattr(self, name)
                setattr(self, name, np.concatenate((old, np.zeros(len(old), bool))))
        self.pins.append(pin)
        self.reader[slot] = pin.is_input
        self.alive[slot] = True
        return slot


class _WireSlots:
    """Every wire of a circuit (board and hidden), by slot: its two ends, as pin slots
    or wire slots, and whether it's still there."""

    def __init__(self) -> None:
        self.wires: list[Wire] = []
        self.alive = np.zeros(256, bool)
        self.board = np.zeros(256, bool)         # on the board (not inside a macro)
        self.end_is_wire = np.zeros((256, 2), bool)
        self.end_slot = np.zeros((256, 2), np.intp)

    def add(self, wire: Wire, board: bool) -> None:
        wire.slot = slot = len(self.wires)
        if slot == len(self.alive):
            for name in ("alive", "board", "end_is_wire", "end_slot"):
                old = getattr(self, name)
                setattr(self, name, np.concatenate((old, np.zeros_like(old))))
        self.wires.append(wire)
        self.alive[slot], self.board[slot] = True, board
        self.set_ends(wire)

    def set_ends(self, wire: Wire) -> None:
        for side, end in enumerate(wire.ends):
            self.end_is_wire[wire.slot, side] = isinstance(end, Wire)
            self.end_slot[wire.slot, side] = end.slot


class Pin:
    __slots__ = ("part", "index", "is_input", "slot", "_passive", "_store")

    def __init__(self, part: Part, index: int, is_input: bool, store: _PinStates) -> None:
        self.part, self.index, self.is_input = part, index, is_input
        self._passive = False
        self._store = store
        self.slot = store.add(self)

    @property
    def state(self) -> bool:
        return bool(self._store.states[self.slot])

    @state.setter
    def state(self, value: bool) -> None:
        self._store.states[self.slot] = value

    @property
    def passive(self) -> bool:
        """Pass-through: a macro instance's pins and its body's port pins. They join nets
        (see Part.links) but never drive them; they just show the net's value."""
        return self._passive

    @passive.setter
    def passive(self, value: bool) -> None:
        self._passive = value
        self._store.reader[self.slot] = self.is_input or value

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
    type: PartType | None = field(default=None, repr=False)
    props: dict[str, Any] = field(default_factory=dict)  # this instance's settings (see PartType.props)
    state: dict[str, Any] = field(default_factory=dict)  # scratch space for its PartType's hooks
    live: bool = False  # opened: placed for real, not a ghost
    slot: int = -1      # its place in the circuit's per-part arrays (settling)
    # Macro instances: the body's parts by their uid in the body, the body's wires,
    # and which pins are joined (instance pin <-> port pin). Hidden parts: `owner`.
    owner: Part | None = field(default=None, repr=False)
    inner: dict[int, Part] = field(default_factory=dict, repr=False)
    inner_wires: list[Wire] = field(default_factory=list, repr=False)
    links: list[tuple[Pin, Pin]] = field(default_factory=list, repr=False)

    @property
    def pins(self) -> list[Pin]:
        return self.inputs + self.outputs


Endpoint = Union[Pin, "Wire"]


@dataclass(eq=False)
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
    def __init__(self, registry: Registry | None = None, settle_ticks: int = 0, seed: int = 0) -> None:
        """`registry`: anything with get(kind) / `kind in` -- a Registry, or a
        macros.Catalog to have macros too."""
        self.registry = registry or builtin_registry()
        self._parts: dict[Part, None] = {}  # what's on the board (macro insides are hidden_parts)
        self.hidden_parts: list[Part] = []
        self.hidden_wires: list[Wire] = []
        self.settle_ticks = settle_ticks
        self.rng = np.random.default_rng(seed)
        self._pins = _PinStates()
        self._wire_slots = _WireSlots()
        self._macros: dict[Part, None] = {}  # instances with links (their pins <-> their ports)
        self._settle = np.zeros(256, np.int32)  # per part slot: ticks of settling jitter left
        self._n_part_slots = 0
        self._settling = 0  # how many parts are still settling
        self.tick = 0
        self.faults: dict[str, str] = {}  # kind -> why it's disabled (a hook raised)
        self.errors: list[str] = []       # new fault messages, for the UI to pick up
        self._kinds_dirty = True
        self._kinds: dict[PartType, list[Part]] = {}  # instances grouped by type
        self._batches_dirty = True
        self._batches: list[_Batch] = []  # what step() evaluates, kind by kind
        self._wires: dict[Wire, None] = {}  # creation order: a wire always comes after the wires it attaches to
        self._at: dict[Endpoint, list[Wire]] = {}  # pin or wire -> the board wires with an end on it
        self.part_by_uid: dict[int, Part] = {}  # board parts (hidden ones have their own uid spaces)
        self.wire_by_uid: dict[int, Wire] = {}
        self.revision = 0  # bumped by every edit of the board's parts and wiring
        self._next_uid = 1
        self._next_wire_uid = 1
        # Nets are derived from the wiring and cached until the wiring changes.
        self._nets_dirty = True
        self._drivers = np.empty(0, np.intp)   # driver pin slots, grouped by net
        self._driven = np.empty(0, np.intp)    # the nets that have drivers, ascending ...
        self._drv_starts = np.empty(0, np.intp)  # ... and where each one's drivers start
        self._readers = np.empty(0, np.intp)   # reader pin slots ...
        self._reader_net = np.empty(0, np.intp)  # ... and their nets
        self._wire_net = np.zeros(0, np.intp)  # per wire slot: its net (-1: gone)
        self.net_value = np.zeros(0, bool)
        self.net_conflict = np.zeros(0, bool)
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
        t = self.registry.get(kind)
        if uid is None:
            uid = self._next_uid
        self._next_uid = max(self._next_uid, uid + 1)
        part = self._make(t, uid)
        self._parts[part] = None
        self.part_by_uid[uid] = part
        self.revision += 1
        if live:
            self.open_part(part)
        return part

    def _make(self, t: PartType, uid: int, owner: Part | None = None) -> Part:
        part = Part(t.kind, uid=uid, type=t, props=fresh_props(t), owner=owner, slot=self._n_part_slots)
        self._n_part_slots += 1
        if part.slot == len(self._settle):
            self._settle = np.concatenate((self._settle, np.zeros(len(self._settle), np.int32)))
        part.inputs = [Pin(part, i, True, self._pins) for i in range(len(t.ins))]
        part.outputs = [Pin(part, i, False, self._pins) for i in range(len(t.outs))]
        if getattr(t, "body", None) is not None:
            self._expand(part)
        self._kinds_dirty = self._batches_dirty = self._nets_dirty = True
        return part

    def _expand(self, inst: Part) -> None:
        """Build a macro instance's body as hidden parts + wires, and join its pins
        to the body's ports (see the module docstring)."""
        t = inst.type
        body = t.body
        for uid, (kind, label, _x, _y, props) in body.parts.items():
            p = self._make(self.registry.get(kind), uid, owner=inst)
            p.label, p.props = label, copy.deepcopy(props)
            inst.inner[uid] = p
            self.hidden_parts.append(p)
        wires: dict[int, Wire] = {}
        for uid in sorted(body.wires):  # parents first
            src, dst = (self._end(ref, inst, wires) for ref in body.wires[uid][:2])
            w = wires[uid] = Wire(src, dst, uid)
            self._wire_slots.add(w, board=False)
            inst.inner_wires.append(w)
            self.hidden_wires.append(w)
        for pin, port_uid in zip(inst.inputs, t.in_ids):
            inst.links.append((pin, inst.inner[port_uid].outputs[0]))
        for pin, port_uid in zip(inst.outputs, t.out_ids):
            inst.links.append((inst.inner[port_uid].inputs[0], pin))
        for a, b in inst.links:
            a.passive = b.passive = True
        self._macros[inst] = None

    @staticmethod
    def _end(ref: tuple, inst: Part, wires: dict[int, Wire]) -> Endpoint:
        if ref[0] == "w":
            return wires[ref[1]]
        _, uid, is_input, index = ref
        part = inst.inner[uid]
        return (part.inputs if is_input else part.outputs)[index]

    def _tree(self, part: Part) -> list[Part]:
        """The part plus, for a macro instance, everything inside it (any depth)."""
        out, todo = [], [part]
        while todo:
            p = todo.pop()
            out.append(p)
            todo += p.inner.values()
        return out

    def open_part(self, part: Part) -> None:
        """Make a ghost live (placed for real): open hooks, and settling starts.
        A macro instance opens everything inside it too."""
        for p in self._tree(part):
            if p.live:
                continue
            p.live = True
            self._batches_dirty = True
            if self.settle_ticks:
                self._settle[p.slot] = self.settle_ticks
                self._settling += 1
            t = p.type
            if t.has("open") and t.kind not in self.faults:
                self._guard(t, "open", lambda: t.open(p))

    def close_part(self, part: Part) -> None:
        for p in self._tree(part):
            if not p.live:
                continue
            p.live = False
            self._batches_dirty = True
            t = p.type
            if t.has("close"):
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
            self._guard(t, "click", lambda: t.click(part))  # (IN flips its output pin right here)
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
        for part in parts:
            self.close_part(part)
        removed = self.remove_wires({w for part in parts for pin in part.pins for w in self._at.get(pin, ())})
        for part in parts:
            del self._parts[part]
            del self.part_by_uid[part.uid]
        self.revision += 1
        tree = [p for part in parts for p in self._tree(part)]
        self._pins.alive[[pin.slot for p in tree for pin in p.pins]] = False
        self._wire_slots.alive[[w.slot for p in tree for w in p.inner_wires]] = False
        self._settle[[p.slot for p in tree]] = 0
        for p in tree:
            self._macros.pop(p, None)
        if any(part.inner for part in parts):  # macro instances: their insides go too
            dead = set(tree)
            dead_wires = {w for p in dead for w in p.inner_wires}
            self.hidden_parts = [p for p in self.hidden_parts if p not in dead]
            self.hidden_wires = [w for w in self.hidden_wires if w not in dead_wires]
        self._nets_dirty = self._kinds_dirty = self._batches_dirty = True
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

    def connect(self, a: Endpoint, b: Endpoint, uid: int | None = None,
                check: bool = True) -> tuple[Wire | None, list[Wire]]:
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
        self._wire_slots.add(wire, board=True)
        self._wires[wire] = None
        self.wire_by_uid[uid] = wire
        self._link(wire)
        self._nets_dirty = True
        self.revision += 1
        return wire, replaced

    def _link(self, wire: Wire) -> None:
        for end in wire.ends:
            self._at.setdefault(end, []).append(wire)

    def _unlink(self, wire: Wire) -> None:
        for end in wire.ends:
            attached = self._at[end]
            attached.remove(wire)
            if not attached:
                del self._at[end]

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
        self._wire_slots.alive[[w.slot for w in removed]] = False
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
        for w in list(self._at.get(absorb, ())):
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
        slots.alive[absorb.slot] = False
        self._link(keep)
        slots.set_ends(keep)
        self._nets_dirty = True
        self.revision += 1

    def attachments(self, wire: Wire) -> list[Wire]:
        """Wires with an end on `wire` (branches, stubs, extra drivers)."""
        return _by_uid(self._at.get(wire, ()))

    def wires_at(self, pin: Pin) -> list[Wire]:
        return _by_uid(self._at.get(pin, ()))

    def ends_on(self, end: Endpoint) -> tuple[Wire, ...] | list[Wire]:
        """The wires with an end on this pin or wire, in no particular order (fast; don't modify)."""
        return self._at.get(end, ())

    def descendants(self, *wires: Wire) -> list[Wire]:
        """Wires attached to these, wires attached to those, and so on (in creation order)."""
        found: set[Wire] = set()
        todo = list(wires)
        while todo:
            for w in self._at.get(todo.pop(), ()):
                if w not in found:
                    found.add(w)
                    todo.append(w)
        return _by_uid(found.difference(wires))

    # ---- nets -------------------------------------------------------------

    def _rebuild_nets(self) -> None:
        """Group pins and wires into nets: the connected components of a graph whose
        nodes are pin slots and wires, and whose edges are wire ends and macro links."""
        store, ws = self._pins, self._wire_slots
        n_pins, n_wires = len(store.pins), len(ws.wires)
        live = np.flatnonzero(ws.alive[:n_wires])  # wire slots; a wire's node is n_pins + its slot
        ends = np.where(ws.end_is_wire[live], n_pins + ws.end_slot[live], ws.end_slot[live])
        links = [(x.slot, y.slot) for inst in self._macros for x, y in inst.links]  # macro pins <-> ports
        link_a, link_b = (np.array(side, np.intp) for side in zip(*links)) if links else (np.empty(0, np.intp),) * 2
        a = np.concatenate((n_pins + live, n_pins + live, link_a))
        b = np.concatenate((ends[:, 0], ends[:, 1], link_b))
        n = n_pins + n_wires
        member = np.zeros(n, bool)  # in some net: a wire, or a pin with a wire or link on it
        member[a] = member[b] = True
        member[:n_pins] &= store.alive[:n_pins]
        graph = coo_matrix((np.ones(len(a), bool), (a, b)), shape=(n, n))
        _, label = connected_components(graph, directed=False)
        _, net = np.unique(label[member], return_inverse=True)  # net numbers: 0, 1, 2, ...
        net_of = np.full(n, -1, np.intp)
        net_of[member] = net
        n_nets = int(net.max()) + 1 if net.size else 0

        in_net = member[:n_pins]
        reader = store.reader[:n_pins]
        drivers = np.flatnonzero(in_net & ~reader)
        order = np.argsort(net_of[drivers], kind="stable")
        self._drivers = drivers[order]
        driver_net = net_of[self._drivers]
        self._driven, self._drv_starts = np.unique(driver_net, return_index=True)
        self._readers = np.flatnonzero(in_net & reader)
        self._reader_net = net_of[self._readers]
        store.states[:n_pins][store.alive[:n_pins] & reader & ~in_net] = False  # unconnected: reads 0

        self._wire_net = net_of[n_pins:]
        self.net_value = np.zeros(n_nets, bool)
        self.net_conflict = np.zeros(n_nets, bool)
        self._nets_dirty = False
        self._changed_all = True  # net numbers mean something else now; pins were reset

    def wire_state(self, wire: Wire) -> tuple[bool, bool]:
        """(value, conflict) of the net this wire belongs to, as of the last step."""
        if self._nets_dirty:
            self._rebuild_nets()
        i = self._wire_net[wire.slot]
        return bool(self.net_value[i]), bool(self.net_conflict[i])

    def take_changes(self) -> tuple[bool, set[Part], list[Wire]]:
        """What changed since the last call: (everything?, parts whose pins changed,
        board wires whose net changed). With everything=True, the rest is empty."""
        if self._nets_dirty:
            self._rebuild_nets()
        states = self._pins.states[:len(self._pins.pins)]
        if self._changed_all or len(self._shown) != len(states) or len(self._shown_value) != len(self.net_value):
            result = True, set(), []
        else:
            pins = self._pins.pins
            parts = {pins[i].part for i in np.flatnonzero(states != self._shown).tolist()}
            changed = (self.net_value != self._shown_value) | (self.net_conflict != self._shown_conflict)
            wires = []
            if changed.any():  # the board wires on those nets
                ws, net = self._wire_slots, self._wire_net
                n = len(net)
                on = np.flatnonzero(ws.alive[:n] & ws.board[:n] & changed[np.maximum(net, 0)] & (net >= 0))
                wires = [ws.wires[i] for i in on.tolist()]
            result = False, parts, wires
        self._shown = states.copy()
        self._shown_value, self._shown_conflict = self.net_value.copy(), self.net_conflict.copy()
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
            ins = [states[idx] for idx in batch.ins]
            ctx = Ctx(batch.parts, self.tick, now)
            outs = self._guard(t, "eval", lambda: _outputs(t, t.eval(ctx, *ins), len(batch.parts)))
            if outs is not _FAILED:
                results.append((batch, outs))
        for batch, outs in results:
            if self._settling:  # settling parts take their new outputs only half the time
                keep = (self._settle[batch.slots] <= 0) | (self.rng.random(len(batch.parts)) < 0.5)
                for idx, values in zip(batch.outs, outs):
                    states[idx[keep]] = values[keep]
            else:
                for idx, values in zip(batch.outs, outs):
                    states[idx] = values

        # Phase 2: every net resolves its drivers and hands the value to its readers.
        # No drivers: 0 (future: Z). Drivers disagree: a conflict, which reads 0 until
        # 4-state logic (X). So the value is the drivers' minimum, and a conflict is
        # minimum != maximum.
        value = np.zeros(len(self.net_value), bool)
        conflict = np.zeros(len(self.net_value), bool)
        if self._drivers.size:
            d = states[self._drivers]
            low = np.minimum.reduceat(d, self._drv_starts)
            value[self._driven] = low
            conflict[self._driven] = low != np.maximum.reduceat(d, self._drv_starts)
        self.net_value, self.net_conflict = value, conflict
        states[self._readers] = value[self._reader_net]
        self.tick += 1
        if self._settling:
            s = self._settle[:self._n_part_slots]
            s[s > 0] -= 1
            self._settling = int(np.count_nonzero(s))

    def frame(self) -> None:
        """Once per frame (not per step): the frame hook of every live part that has one."""
        now = time.monotonic()
        for t, group in self._by_kind().items():
            if t.has("frame") and t.kind not in self.faults:
                live = [p for p in group if p.live]
                if live:
                    self._guard(t, "frame", lambda: t.frame(Ctx(live, self.tick, now)))

    # ---- part types --------------------------------------------------------

    def _eval_batches(self) -> list[_Batch]:
        """Per kind with an eval: the instances step() evaluates (live ones only, unless
        pure) and index arrays of their pins."""
        if self._batches_dirty:
            self._batches = []
            for t, group in self._by_kind().items():
                if not t.has("eval"):
                    continue
                if not t.pure:
                    group = [p for p in group if p.live]
                if group:
                    self._batches.append(_Batch(
                        t, group,
                        [np.array([p.inputs[i].slot for p in group], np.intp) for i in range(len(t.ins))],
                        [np.array([p.outputs[i].slot for p in group], np.intp) for i in range(len(t.outs))],
                        np.array([p.slot for p in group], np.intp)))
            self._batches_dirty = False
        return self._batches

    def _by_kind(self) -> dict[PartType, list[Part]]:
        if self._kinds_dirty:
            self._kinds = {}
            for part in self.parts + self.hidden_parts:
                self._kinds.setdefault(part.type, []).append(part)
            self._kinds_dirty = False
        return self._kinds

    def _guard(self, t: PartType, hook: str, fn: Callable[[], Any]) -> Any:
        """Run one of t's hooks. If it raises, t is disabled in this circuit (its
        outputs drop to 0 and its hooks stop being called) and the error is kept."""
        try:
            return fn()
        except Exception:
            if t.kind not in self.faults:
                detail = traceback.format_exc(limit=-1).strip().splitlines()[-1]
                self.faults[t.kind] = msg = f"{t.kind}.{hook}: {detail}"
                self.errors.append(msg)
                self._changed_all = True
                for part in self._by_kind().get(t, ()):
                    for pin in part.outputs:
                        pin.state = False
            return _FAILED


@dataclass
class _Batch:
    type: PartType
    parts: list[Part]
    ins: list[np.ndarray]   # per input pin: the instances' pin slots
    outs: list[np.ndarray]  # per output pin
    slots: np.ndarray       # the instances' part slots


def _by_uid(wires) -> list[Wire]:
    """Creation order: uids grow as wires are made, and a wire's parents are always older
    (a merge keeps the older wire), so this also puts parents before children."""
    return sorted(wires, key=lambda w: w.uid)


def _outputs(t: PartType, raw: Any, n: int) -> list[np.ndarray]:
    """Normalize what eval returned: one bool array of n per output pin."""
    k = len(t.outs)
    if k == 0:
        return []
    if k == 1 and not (isinstance(raw, tuple) and len(raw) == 1):
        raw = (raw,)  # one output: anything but a 1-tuple is its value (a scalar, list or array)
    if not isinstance(raw, tuple) or len(raw) != k:
        raise ValueError(f"eval returned {raw!r}; expected {k} output value(s)")
    return [np.broadcast_to(np.asarray(v).astype(bool), (n,)) for v in raw]
