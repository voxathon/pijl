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
