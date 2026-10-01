from pijl.sim import Circuit
from pijl.ui.document import History, Snapshot


def snap(n: int) -> Snapshot:
    return Snapshot({i: ("NOT", "", float(i), 0.0, {}) for i in range(n)}, {})


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
    assert h.undo() == (
        ({}, {2: ("NOT", "", 2.0, 0.0, {})}),
        ({}, {}),
        ({}, {}),
    )  # the step, per section: (before, after); uid 2 wasn't there before
    assert h.current == snap(2)
    h.undo()
    assert h.current == snap(1)
    h.redo()
    assert h.current == snap(2)


def test_new_commit_clears_redo():
    h = History(snap(0))
    h.commit(snap(1))
    h.undo()
    h.commit(snap(5))
    assert h.redo() is None
    h.undo()
    assert h.current == snap(0)


def test_undo_past_start_and_redo_past_end_are_no_ops():
    h = History(snap(0))
    assert h.undo() is None and h.redo() is None
    assert h.current == snap(0)


def test_limit_drops_oldest():
    h = History(snap(0), limit=3)
    for n in range(1, 10):
        h.commit(snap(n))
    assert len(h.undo_stack) == 3
    for _ in range(3):
        h.undo()
    assert h.current == snap(6) and h.undo() is None


def test_record_takes_only_what_changed():
    h = History(snap(3))
    moved = ("NOT", "", 9.0, 9.0, {})
    assert h.record({1: moved, 7: None}, {}, {})  # 7 was never there: not a change
    assert h.undo_stack[0][0] == (
        ({1: ("NOT", "", 1.0, 0.0, {})}, {1: moved}),
        ({}, {}),
        ({}, {}),
    )
    assert h.current.parts[1] == moved
    assert not h.record({1: moved}, {}, {})


def test_states_name_points_in_the_timeline():
    h = History(snap(0))
    saved = h.state
    h.commit(snap(1))
    assert h.state != saved
    h.undo()
    assert h.state == saved  # back where it was saved: nothing unsaved
    h.redo()
    h.commit(snap(2))
    assert len({saved, h.state}) == 2


def test_amend_folds_into_the_last_step():
    h = History(snap(1))
    a = ("NOT", "", 5.0, 0.0, {})
    b = ("NOT", "", 6.0, 0.0, {})
    h.record({0: a}, {}, {})
    first = h.state
    h.amend({0: b}, {}, {})
    assert len(h.undo_stack) == 1 and h.state != first
    h.undo()
    assert h.current == snap(1)  # both tweaks undone at once
    h.redo()
    assert h.current.parts[0] == b


def test_amend_drops_what_ends_up_as_it_started():
    h = History(snap(2))
    moved = ("NOT", "", 9.0, 0.0, {})
    new = ("AND", "", 3.0, 3.0, {})
    h.record({0: moved, 5: new}, {}, {})  # move 0, add 5
    h.amend({0: snap(2).parts[0], 1: moved}, {}, {})  # 0 back where it was, move 1
    h.amend({5: None}, {}, {})  # and 5 gone again
    (step, _, _), = h.undo_stack
    assert step[0] == ({1: snap(2).parts[1]}, {1: moved})  # only 1 is left in it
    h.undo()
    assert h.current == snap(2)
    h.redo()
    assert h.current.parts == {0: snap(2).parts[0], 1: moved}
