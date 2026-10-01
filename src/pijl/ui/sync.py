"""Sim state -> the views' state bytes, for every view at once.

Pins, lit bodies and wires show their logic level through one byte per shape (see
sdf_shapes.SHOW_*). Writing those view by view meant a few Python calls per pin and
wire: tens of milliseconds a frame on a board where most of them toggle every tick
(rings of inverters), far more than the sim itself.

So each instance buffer records, per slot, which circuit pin or wire that shape
shows (InstanceBuffer.pin_src / wire_src), set when the shapes are made and cleared
when they're freed. Each frame is then a gather from the sim's arrays, a compare
against the bytes already there, and a scatter of the ones that differ. Edits never
walk the views: making or freeing shapes updates the sources as it goes, and the
slots in use are found again (one numpy call per buffer) only when they changed.
"""

from __future__ import annotations

import numpy as np

from .canvas import Canvas, InstanceBuffer
from .sdf_shapes import SHOW_BY_CODE, SHOW_FIGHT


class ViewSync:
    def __init__(self) -> None:
        # per buffer: (its gen then, slots showing a pin, slots showing a wire)
        self._active: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}

    def __call__(self, circuit, canvas: Canvas) -> None:
        """Show the circuit's current pin and net states on the canvas's shapes."""
        n_pins = circuit.pin_count
        for buf in canvas.buffers():
            pin_slots, wire_slots = self._slots(buf)
            if pin_slots.size:
                pins = buf.pin_src[pin_slots]
                ok = pins < n_pins  # (a stale view of a previous circuit: skip it)
                _apply(buf, pin_slots[ok], SHOW_BY_CODE[circuit.pin_codes(pins[ok])])
            if wire_slots.size:
                value, conflict, has = circuit.wire_states(buf.wire_src[wire_slots])
                want = np.where(conflict, SHOW_FIGHT, SHOW_BY_CODE[value])
                _apply(buf, wire_slots[has], want[has])

    def _slots(self, buf: InstanceBuffer) -> tuple[np.ndarray, np.ndarray]:
        cached = self._active.get(id(buf))
        if cached is None or cached[0] != buf.gen:
            end = buf.end
            cached = self._active[id(buf)] = (
                buf.gen,
                np.flatnonzero(buf.pin_src[:end] >= 0),
                np.flatnonzero(buf.wire_src[:end] >= 0),
            )
        return cached[1], cached[2]


def _apply(buf: InstanceBuffer, shapes: np.ndarray, want: np.ndarray) -> None:
    """Set these shapes' state bytes, marking only the ones that change."""
    if not shapes.size:
        return
    flags = buf.f["flags"]
    diff = flags[shapes, 0] != want
    if diff.any():
        changed = shapes[diff]
        flags[changed, 0] = want[diff]
        buf.mark_many(changed)
