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
"""

from __future__ import annotations

import random
import time
import traceback
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Union

import numpy as np

from ..parts import Ctx, PartType, Registry, builtin_registry, fresh_props

_FAILED = object()  # what Circuit._guard returns when the hook raised


@dataclass(eq=False)
class Pin:
    part: Part
    index: int
    is_input: bool
    state: bool = False
    # Pass-through: a macro instance's pins and its body's port pins. They join nets
    # (see Part.links) but never drive them; they just show the net's value.
    passive: bool = False

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
    settle: int = 0     # ticks of settling jitter left (see the module docstring)
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
        self.rng = random.Random(seed)
        self._settling: list[Part] = []
        self.tick = 0
        self.faults: dict[str, str] = {}  # kind -> why it's disabled (a hook raised)
        self.errors: list[str] = []       # new fault messages, for the UI to pick up
        self._kinds_dirty = True
        self._kinds: dict[PartType, list[Part]] = {}  # instances grouped by type
        self._wires: dict[Wire, None] = {}  # creation order: a wire always comes after the wires it attaches to
        self._at: dict[Endpoint, list[Wire]] = {}  # pin or wire -> the board wires with an end on it
        self.revision = 0  # bumped by every edit of the board's parts and wiring
        self._next_uid = 1
        self._next_wire_uid = 1
        # Nets are derived from the wiring and cached until the wiring changes.
        self._nets_dirty = True
        self._nets: list[tuple[list[Pin], list[Pin]]] = []  # (drivers, readers) per net
        self._net_of_wire: dict[Wire, int] = {}
        self.net_value: list[bool] = []
        self.net_conflict: list[bool] = []
        self._wires_of_net: list[list[Wire]] = []  # board wires only
        # What changed since the UI last asked (take_changes), so it can redraw just that.
        self._changed_all = True
        self._changed_parts: set[Part] = set()
        self._changed_nets: set[int] = set()

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
        self.revision += 1
        if live:
            self.open_part(part)
        return part

    def _make(self, t: PartType, uid: int, owner: Part | None = None) -> Part:
        part = Part(t.kind, uid=uid, type=t, props=fresh_props(t), owner=owner)
        part.inputs = [Pin(part, i, True) for i in range(len(t.ins))]
        part.outputs = [Pin(part, i, False) for i in range(len(t.outs))]
        if getattr(t, "body", None) is not None:
            self._expand(part)
        self._kinds_dirty = self._nets_dirty = True
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
            inst.inner_wires.append(w)
            self.hidden_wires.append(w)
        for pin, port_uid in zip(inst.inputs, t.in_ids):
            inst.links.append((pin, inst.inner[port_uid].outputs[0]))
        for pin, port_uid in zip(inst.outputs, t.out_ids):
            inst.links.append((inst.inner[port_uid].inputs[0], pin))
        for a, b in inst.links:
            a.passive = b.passive = True

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
            if self.settle_ticks:
                p.settle = self.settle_ticks
                self._settling.append(p)
            t = p.type
            if t.has("open") and t.kind not in self.faults:
                self._guard(t, "open", lambda: t.open(p))
                self._changed_parts.add(p)  # hooks may set pins directly

    def close_part(self, part: Part) -> None:
        for p in self._tree(part):
            if not p.live:
                continue
            p.live = False
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
            self._guard(t, "click", lambda: t.click(part))
            self._changed_parts.add(part)  # e.g. IN flips its output pin right here
        return True

    def remove_part(self, part: Part) -> list[Wire]:
        """Removes the part, every wire touching it, and every wire hanging off
        those. Returns all removed wires."""
        self.close_part(part)
        removed: list[Wire] = []
        for w in _by_uid({w for pin in part.pins for w in self._at.get(pin, ())}):
            if w in self._wires:  # may already be gone as a branch of an earlier one
                removed += self.remove_wire(w)
        del self._parts[part]
        self.revision += 1
        if part.inner:  # a macro instance: its insides go too
            dead = set(self._tree(part))
            dead_wires = {w for p in dead for w in p.inner_wires}
            self.hidden_parts = [p for p in self.hidden_parts if p not in dead]
            self.hidden_wires = [w for w in self.hidden_wires if w not in dead_wires]
        self._nets_dirty = self._kinds_dirty = True
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
        self._wires[wire] = None
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
        removed = [wire, *self.descendants(wire)]
        for w in removed:
            del self._wires[w]
            self._unlink(w)
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
        self._unlink(keep)
        keep.dst = far
        for w in list(self._at.get(absorb, ())):
            self._unlink(w)
            if w.src is absorb:
                w.src = keep
            if w.dst is absorb:
                w.dst = keep
            self._link(w)
        self._unlink(absorb)
        del self._wires[absorb]
        self._link(keep)
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
        """Group pins and wires into nets (union-find over identities)."""
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        wires = self.wires + self.hidden_wires
        parts = self.parts + self.hidden_parts
        for w in wires:
            for end in w.ends:
                parent[find(id(end))] = find(id(w))
        for part in parts:
            for a, b in part.links:  # macro pins <-> their ports: one net
                parent[find(id(a))] = find(id(b))

        index: dict[int, int] = {}
        self._nets = []
        for part in parts:
            for pin in part.pins:
                if id(pin) in parent and find(id(pin)) not in index:
                    index[find(id(pin))] = len(self._nets)
                    self._nets.append(([], []))
        for w in wires:
            if find(id(w)) not in index:
                index[find(id(w))] = len(self._nets)
                self._nets.append(([], []))
        self._net_of_wire = {w: index[find(id(w))] for w in wires}
        self._wires_of_net = [[] for _ in self._nets]
        for w in self.wires:
            self._wires_of_net[self._net_of_wire[w]].append(w)
        for part in parts:
            for pin in part.pins:
                if id(pin) in parent:
                    drivers, readers = self._nets[index[find(id(pin))]]
                    # passive (pass-through) pins only show the value
                    (readers if pin.is_input or pin.passive else drivers).append(pin)
                elif pin.is_input or pin.passive:
                    pin.state = False  # unconnected: reads 0
        self.net_value = [False] * len(self._nets)
        self.net_conflict = [False] * len(self._nets)
        self._nets_dirty = False
        self._changed_all = True  # net numbers mean something else now; pins were reset

    def wire_state(self, wire: Wire) -> tuple[bool, bool]:
        """(value, conflict) of the net this wire belongs to, as of the last step."""
        if self._nets_dirty:
            self._rebuild_nets()
        i = self._net_of_wire[wire]
        return self.net_value[i], self.net_conflict[i]

    def take_changes(self) -> tuple[bool, set[Part], list[Wire]]:
        """What changed since the last call: (everything?, parts whose pins changed,
        board wires whose net changed). With everything=True, the rest is empty."""
        if self._nets_dirty:
            self._rebuild_nets()
        if self._changed_all:
            result = True, set(), []
        else:
            result = False, self._changed_parts, [w for i in self._changed_nets for w in self._wires_of_net[i]]
        self._changed_all = False
        self._changed_parts, self._changed_nets = set(), set()
        return result

    # ---- simulation ----------------------------------------------------

    def step(self) -> None:
        if self._nets_dirty:
            self._rebuild_nets()

        # Phase 1: every part computes outputs from current inputs, one kind at a time.
        # Compute all first, then write, so evaluation order doesn't matter.
        now = time.monotonic()
        results: list[tuple[list[Part], list[list[bool]]]] = []
        for t, group in self._by_kind().items():
            if not t.has("eval") or t.kind in self.faults:
                continue
            if not t.pure:
                group = [p for p in group if p.live]
            if not group:
                continue
            n = len(group)
            ins = [np.fromiter((p.inputs[i].state for p in group), bool, n) for i in range(len(t.ins))]
            ctx = Ctx(group, self.tick, now)
            outs = self._guard(t, "eval", lambda: _outputs(t, t.eval(ctx, *ins), n))
            if outs is not _FAILED:
                results.append((group, outs))
        rng = self.rng
        changed = self._changed_parts
        for group, outs in results:
            if self._settling:  # settling parts take their new outputs only half the time
                keep = [p.settle <= 0 or rng.random() < 0.5 for p in group]
            else:
                keep = None
            for i, values in enumerate(outs):
                for j, (part, value) in enumerate(zip(group, values)):
                    pin = part.outputs[i]
                    if pin.state != value and (keep is None or keep[j]):
                        pin.state = value
                        changed.add(part)

        # Phase 2: every net resolves its drivers and hands the value to its readers.
        for i, (drivers, readers) in enumerate(self._nets):
            if not drivers:
                value, conflict = False, False  # floating (future: Z)
            else:
                value = drivers[0].state
                conflict = any(d.state != value for d in drivers[1:])
                if conflict:
                    value = False  # placeholder until 4-state logic: X reads as 0
            if value != self.net_value[i] or conflict != self.net_conflict[i]:
                self.net_value[i] = value
                self.net_conflict[i] = conflict
                self._changed_nets.add(i)
            for pin in readers:
                if pin.state != value:
                    pin.state = value
                    changed.add(pin.part)
        self.tick += 1
        if self._settling:
            for p in self._settling:
                p.settle -= 1
            self._settling = [p for p in self._settling if p.settle > 0]

    def frame(self) -> None:
        """Once per frame (not per step): the frame hook of every live part that has one."""
        now = time.monotonic()
        for t, group in self._by_kind().items():
            if t.has("frame") and t.kind not in self.faults:
                live = [p for p in group if p.live]
                if live:
                    self._guard(t, "frame", lambda: t.frame(Ctx(live, self.tick, now)))
                    self._changed_parts.update(live)

    # ---- part types --------------------------------------------------------

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


def _by_uid(wires) -> list[Wire]:
    """Creation order: uids grow as wires are made, and a wire's parents are always older
    (a merge keeps the older wire), so this also puts parents before children."""
    return sorted(wires, key=lambda w: w.uid)


def _outputs(t: PartType, raw: Any, n: int) -> list[list[bool]]:
    """Normalize what eval returned: one list of n bools per output pin."""
    k = len(t.outs)
    if k == 0:
        return []
    if k == 1 and not (isinstance(raw, tuple) and len(raw) == 1):
        raw = (raw,)  # one output: anything but a 1-tuple is its value (a scalar, list or array)
    if not isinstance(raw, tuple) or len(raw) != k:
        raise ValueError(f"eval returned {raw!r}; expected {k} output value(s)")
    return [np.broadcast_to(np.asarray(v).astype(bool), (n,)).tolist() for v in raw]
