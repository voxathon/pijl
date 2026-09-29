from pijl.sim import Circuit
from pijl.ui.document import History, Snapshot


def snap(n: int) -> Snapshot:
    return Snapshot({i: ("NOT", "", float(i), 0.0) for i in range(n)}, {})


def test_uids_are_unique_and_restorable():
    c = Circuit()
    a, b = c.add_part("IN"), c.add_part("NOT")
    assert (a.uid, b.uid) == (1, 2)
    c.remove_part(b)
    again = c.add_part("NOT", uid=2)  # undo recreates the same uid
    assert again.uid == 2
    assert c.add_part("AND").uid == 3  # and fresh uids never collide with restored ones
    c.add_part("OR", uid=10)
    assert c.add_part("OR").uid == 11


def test_commit_ignores_no_ops():
    h = History(snap(1))
    assert not h.commit(snap(1))
    assert h.undo_stack == []


def test_undo_redo_roundtrip():
    h = History(snap(0))
    for n in (1, 2, 3):
        h.commit(snap(n))
    assert h.undo() == snap(2)
    assert h.undo() == snap(1)
    assert h.redo() == snap(2)
    assert h.current == snap(2)


def test_new_commit_clears_redo():
    h = History(snap(0))
    h.commit(snap(1))
    h.undo()
    h.commit(snap(5))
    assert h.redo() is None
    assert h.undo() == snap(0)


def test_undo_past_start_and_redo_past_end_are_no_ops():
    h = History(snap(0))
    assert h.undo() is None and h.redo() is None
    assert h.current == snap(0)


def test_limit_drops_oldest():
    h = History(snap(0), limit=3)
    for n in range(1, 10):
        h.commit(snap(n))
    assert len(h.undo_stack) == 3
    assert h.undo_stack[0] == snap(6)
