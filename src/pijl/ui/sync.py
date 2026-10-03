"""Sim state -> the views' state bytes, for every view at once.

Pins, lit bodies and wires show their logic level through one byte per shape (see
sdf_shapes.SHOW_*). Writing those view by view meant a few Python calls per pin and
wire: tens of milliseconds a frame on a board where most of them toggle every tick
(rings of inverters), far more than the sim itself.

So each instance buffer records, per slot, which circuit pin or wire that shape
shows (InstanceBuffer.pin_src / wire_src), set when the shapes are made and cleared
when they're freed. Edits never walk the views: making or freeing shapes updates the
sources as it goes. What only changes with edits is worked out once and kept: the
slots showing something, the pins they show, and the nets of the wires (until the
buffer's `gen` or the circuit's nets_version moves). Each frame is then a gather
from the sim's arrays and one scatter into the buffer's state bytes, uploaded whole
(see canvas.py): comparing first would cost about what writing does.
"""

from __future__ import annotations

import weakref

import numpy as np

from ..logic import ONE, X, Z
from .canvas import Canvas, InstanceBuffer
from .sdf_shapes import SEGMENT, SHOW_BY_CODE, SHOW_FIGHT, SHOW_X, SHOW_Z, partial

_NONE = np.empty(0, np.int32)  # a buffer that shows no pins / wires

# by logic code, and by code + 4 when the net's drivers fight
_WIRE_SHOW = np.concatenate((SHOW_BY_CODE, np.full(4, SHOW_FIGHT, np.uint8)))


def _bus_show(codes: np.ndarray, starts: np.ndarray, n: np.ndarray, fights=None) -> np.ndarray:
    """State bytes for buses, from their lanes' codes (each bus's from `starts`, `n` of
    them): a fight on any lane shows as one; else any X -- or some lanes Z and some
    not -- as X; all Z as Z; else lit as far as its lanes are 1 (sdf_shapes.partial)."""
    if not len(starts):
        return np.zeros(0, np.uint8)
    ones = np.add.reduceat((codes == ONE).astype(np.intp), starts)
    xs = np.add.reduceat((codes == X).astype(np.intp), starts)
    zs = np.add.reduceat((codes == Z).astype(np.intp), starts)
    out = partial(ones, n)
    out[(xs > 0) | ((zs > 0) & (zs < n))] = SHOW_X
    out[zs == n] = SHOW_Z
    if fights is not None:
        out[fights] = SHOW_FIGHT
    return out


class ViewSync:
    def __init__(self) -> None:
        # per buffer: (key, shape slots, the pins they show) / (key, shape slots, nets)
        self._pins: dict[int, tuple] = {}
        self._wires: dict[int, tuple] = {}
        self._lanes: dict[int, tuple] = {}  # (key, shape slots, the lanes they show)
        # per part table: (key, [(part type, part slots, mark slots n x k)])
        self._faces: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def __call__(self, circuit, canvas: Canvas, table=None) -> None:
        """Show the circuit's current pin and net states on the canvas's shapes, and
        (given the canvas's part table) light the face marks face() hooks decide."""
        for buf in canvas.buffers():
            shapes, pins, bus = self._pin_slots(buf, circuit)
            if shapes.size:
                buf.set_state(shapes, SHOW_BY_CODE[circuit.pin_codes(pins)])
            if bus is not None:  # buses: all their lanes at once
                shapes, lanes, starts, n = bus
                buf.set_state(shapes, _bus_show(circuit.pin_codes(lanes), starts, n))
            shapes, nets, bus = self._wire_slots(buf, circuit)
            if shapes.size:
                fight = circuit.net_conflict[nets].view(np.uint8) << 2
                buf.set_state(shapes, _WIRE_SHOW[circuit.net_value[nets] | fight])
            if bus is not None:
                shapes, nets, starts, n = bus
                fights = np.logical_or.reduceat(circuit.net_conflict[nets], starts)
                buf.set_state(shapes, _bus_show(circuit.net_value[nets], starts, n, fights))
            if buf.lane_src is not None:  # bit cells: one lane each
                shapes, lanes = self._lane_slots(buf, circuit)
                if shapes.size:
                    buf.set_state(shapes, SHOW_BY_CODE[circuit.pin_codes(lanes)])
        if table is not None and table.hooked:
            buf = canvas.buffer(SEGMENT, table.layers.bodies)
            for t, parts, marks in self._face_groups(table):
                codes = circuit.face_codes(t, parts)
                buf.set_state(marks.ravel(), SHOW_BY_CODE[codes].ravel())

    def _face_groups(self, table) -> list[tuple]:
        """The views with face() marks, by part type: what changes only with edits."""
        from .views import _face_layout
        cached = self._faces.get(table)
        if cached is None or cached[0] != table.face_gen:
            groups: dict = {}
            for row in table.hooked:
                part = table.view[row].part
                t = part.type
                parts, marks = groups.setdefault(t, ([], []))
                parts.append(part.slot)
                marks.append(table.face_slots(row)[_face_layout(part.layout, t.look).hooked])
            cached = self._faces[table] = (
                table.face_gen,
                [(t, np.array(p, np.intp), np.stack(m)) for t, (p, m) in groups.items()],
            )
        return cached[1]

    def _pin_slots(self, buf: InstanceBuffer, circuit) -> tuple:
        """(shapes, the one-lane pins they show, buses): buses is None, or (shapes,
        their lane slots, where each one's lanes start, how many it has)."""
        key = (buf.gen, id(circuit), circuit.pin_count)
        cached = self._pins.get(id(buf))
        if cached is None or cached[0] != key:
            src = buf.pin_src if buf.pin_src is not None else _NONE
            shapes = np.flatnonzero(src[: buf.end] >= 0)
            pins = src[shapes]
            ok = pins < circuit.pin_count  # (a stale view of a previous circuit: skip it)
            # (native ints: indexing with int32 would convert them every frame)
            shapes, pins = shapes[ok], pins[ok].astype(np.intp)
            lanes, n = circuit.lanes_of_pins(pins)
            bus = None
            wide = n > 1
            if wide.any():
                lanes, n = circuit.lanes_of_pins(pins[wide])
                bus = (shapes[wide], lanes, np.cumsum(n) - n, n)
                shapes, pins = shapes[~wide], pins[~wide]
            cached = self._pins[id(buf)] = (key, shapes, pins, bus)
        return cached[1], cached[2], cached[3]

    def _lane_slots(self, buf: InstanceBuffer, circuit) -> tuple[np.ndarray, np.ndarray]:
        key = (buf.gen, id(circuit), circuit.pin_count)
        cached = self._lanes.get(id(buf))
        if cached is None or cached[0] != key:
            src = buf.lane_src
            shapes = np.flatnonzero(src[: buf.end] >= 0)
            lanes = src[shapes]
            ok = lanes < circuit.pin_count
            cached = self._lanes[id(buf)] = (key, shapes[ok], lanes[ok].astype(np.intp))
        return cached[1], cached[2]

    def _wire_slots(self, buf: InstanceBuffer, circuit) -> tuple:
        """(shapes, the nets of the one-lane wires they show, buses): buses is None,
        or (shapes, their lanes' nets, where each one's start, how many it has)."""
        key = (buf.gen, id(circuit), circuit.nets_version)
        cached = self._wires.get(id(buf))
        if cached is None or cached[0] != key:
            src = buf.wire_src if buf.wire_src is not None else _NONE
            shapes = np.flatnonzero(src[: buf.end] >= 0)
            wires = src[shapes].astype(np.intp)
            nets = circuit.wire_nets(wires)
            has = nets >= 0  # (no net: left as it is)
            if not len(circuit.net_value):
                has[:] = False
            shapes, wires, nets = shapes[has], wires[has], nets[has]
            bus = None
            _, n = circuit.lanes_of_wires(wires)
            wide = n > 1
            if wide.any():
                lanes, n = circuit.lanes_of_wires(wires[wide])
                lane_nets = circuit.wire_nets(lanes)
                if (lane_nets >= 0).all():
                    bus = (shapes[wide], lane_nets, np.cumsum(n) - n, n)
                shapes, nets = shapes[~wide], nets[~wide]
            cached = self._wires[id(buf)] = (key, shapes, nets, bus)
        return cached[1], cached[2], cached[3]
