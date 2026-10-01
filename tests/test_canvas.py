import numpy as np

from pijl.ui.canvas import InstanceBuffer, Kind


def _buffer(monkeypatch, capacity: int = 2) -> InstanceBuffer:
    """An InstanceBuffer without GL: only its slot bookkeeping and numpy side."""
    monkeypatch.setattr(InstanceBuffer, "_make_gl", lambda self: None)
    kind = Kind("test", 0, "", "", np.dtype([("x", "f4"), ("flags", "u1", 4)]))
    kind._program = object()
    return InstanceBuffer(kind, capacity)


def test_slots_grow_and_are_reused_lowest_first(monkeypatch):
    buf = _buffer(monkeypatch)
    assert [buf.alloc() for _ in range(5)] == [
        0,
        1,
        2,
        3,
        4,
    ]  # grew past its capacity of 2
    assert len(buf.data) >= 5 and buf.top == 5 and buf.realloc
    buf.free(4)
    assert buf.top == 4  # nothing drawn past the last slot in use
    buf.free(1)
    assert buf.top == 4
    assert buf.alloc() == 1 and buf.alloc() == 4 and buf.top == 5


def test_freed_slots_are_zeroed_and_marked(monkeypatch):
    buf = _buffer(monkeypatch)
    a, b = buf.alloc(), buf.alloc()
    buf.f["x"][b] = 7.0
    buf.f["flags"][b] = (1, 2, 3, 4)
    buf.realloc = buf.any_dirty = False
    buf.dirty[:] = False
    buf.free(b)
    assert (
        buf.f["x"][b] == 0 and not buf.f["flags"][b].any()
    )  # a zero instance draws nothing
    assert buf.dirty[b] and buf.any_dirty and buf.top == 1


def test_fields_survive_growing(monkeypatch):
    buf = _buffer(monkeypatch)
    s = buf.alloc()
    buf.f["x"][s] = 3.5
    for _ in range(10):
        buf.alloc()
    assert buf.f["x"][s] == 3.5  # `f` points at the new arrays


def test_sources_are_forgotten_when_freed_and_kept_when_growing(monkeypatch):
    buf = _buffer(monkeypatch)
    a, b = buf.alloc(), buf.alloc()
    gen = buf.gen
    buf.show_pins([a, b], [10, 11])
    assert buf.gen > gen
    buf.free(a)
    assert buf.pin_src[a] == -1 and buf.pin_src[b] == 11
    for _ in range(5):  # grows past the capacity of 2
        buf.alloc()
    assert (
        buf.pin_src[b] == 11
        and (buf.pin_src[2:] == -1).all()
        and buf.wire_src is None  # (never asked for: not made)
    )


def test_view_sync_writes_what_the_sim_says(monkeypatch):
    from pijl.logic import ONE, ZERO
    from pijl.sim import Circuit
    from pijl.ui.sdf_shapes import SHOW_OFF, SHOW_ON
    from pijl.ui.sync import ViewSync

    c = Circuit()
    a, g = c.add_part("IN"), c.add_part("NOT")
    w, _ = c.connect(a.outputs[0], g.inputs[0])
    buf = _buffer(monkeypatch, capacity=8)
    pin_shape, wire_shape, other = buf.alloc(), buf.alloc(), buf.alloc()
    buf.show_pins([pin_shape], [g.outputs[0].slot])
    buf.show_wires([wire_shape], [w.slot])

    class Canvas:
        def buffers(self):
            return [buf]

    sync = ViewSync()
    a.outputs[0].state = ZERO
    for _ in range(3):
        c.step()
    sync(c, Canvas())
    flags = buf.state
    assert (
        flags[pin_shape] == SHOW_ON
        and flags[wire_shape] == SHOW_OFF
        and flags[other] == 0
    )
    a.outputs[0].state = ONE
    for _ in range(3):
        c.step()
    sync(c, Canvas())
    assert flags[pin_shape] == SHOW_OFF and flags[wire_shape] == SHOW_ON


def test_view_sync_follows_rewiring_and_shows_fights(monkeypatch):
    from pijl.sim import Circuit
    from pijl.ui.sdf_shapes import SHOW_FIGHT, SHOW_OFF, SHOW_ON
    from pijl.ui.sync import ViewSync

    c = Circuit()
    a, b, led = c.add_part("IN"), c.add_part("IN"), c.add_part("OUT")
    w, _ = c.connect(a.outputs[0], led.inputs[0])
    buf = _buffer(monkeypatch, capacity=8)
    shape = buf.alloc()
    buf.show_wires([shape], [w.slot])

    class Canvas:
        def buffers(self):
            return [buf]

    sync = ViewSync()
    a.outputs[0].state, b.outputs[0].state = True, False
    for _ in range(3):
        c.step()
    sync(c, Canvas())
    assert buf.state[shape] == SHOW_ON
    # A second driver joins w's net: net numbers change, though no shape did.
    c.connect(b.outputs[0], w)
    for _ in range(3):
        c.step()
    sync(c, Canvas())
    assert buf.state[shape] == SHOW_FIGHT
    a.outputs[0].state = False
    for _ in range(3):
        c.step()
    sync(c, Canvas())
    assert buf.state[shape] == SHOW_OFF
