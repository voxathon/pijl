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

import numpy as np

from .canvas import Canvas, InstanceBuffer
from .sdf_shapes import SHOW_BY_CODE, SHOW_FIGHT

_NONE = np.empty(0, np.int32)  # a buffer that shows no pins / wires

# by logic code, and by code + 4 when the net's drivers fight
_WIRE_SHOW = np.concatenate((SHOW_BY_CODE, np.full(4, SHOW_FIGHT, np.uint8)))


class ViewSync:
    def __init__(self) -> None:
        # per buffer: (key, shape slots, the pins they show) / (key, shape slots, nets)
        self._pins: dict[int, tuple] = {}
        self._wires: dict[int, tuple] = {}

    def __call__(self, circuit, canvas: Canvas) -> None:
        """Show the circuit's current pin and net states on the canvas's shapes."""
        for buf in canvas.buffers():
            shapes, pins = self._pin_slots(buf, circuit)
            if shapes.size:
                buf.set_state(shapes, SHOW_BY_CODE[circuit.pin_codes(pins)])
            shapes, nets = self._wire_slots(buf, circuit)
            if shapes.size:
                fight = circuit.net_conflict[nets].view(np.uint8) << 2
                buf.set_state(shapes, _WIRE_SHOW[circuit.net_value[nets] | fight])

    def _pin_slots(self, buf: InstanceBuffer, circuit) -> tuple[np.ndarray, np.ndarray]:
        key = (buf.gen, id(circuit), circuit.pin_count)
        cached = self._pins.get(id(buf))
        if cached is None or cached[0] != key:
            src = buf.pin_src if buf.pin_src is not None else _NONE
            shapes = np.flatnonzero(src[: buf.end] >= 0)
            pins = src[shapes]
            ok = pins < circuit.pin_count  # (a stale view of a previous circuit: skip it)
            # (native ints: indexing with int32 would convert them every frame)
            cached = self._pins[id(buf)] = (key, shapes[ok], pins[ok].astype(np.intp))
        return cached[1], cached[2]

    def _wire_slots(self, buf: InstanceBuffer, circuit) -> tuple[np.ndarray, np.ndarray]:
        key = (buf.gen, id(circuit), circuit.nets_version)
        cached = self._wires.get(id(buf))
        if cached is None or cached[0] != key:
            src = buf.wire_src if buf.wire_src is not None else _NONE
            shapes = np.flatnonzero(src[: buf.end] >= 0)
            nets = circuit.wire_nets(src[shapes])
            has = nets >= 0  # (no net: left as it is)
            if not len(circuit.net_value):
                has[:] = False
            cached = self._wires[id(buf)] = (key, shapes[has], nets[has])
        return cached[1], cached[2]
