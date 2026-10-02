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
        ({}, {2: ("NOT", "", 2.0, 0.0, {})}, ()),
        ({}, {}, ()),
        ({}, {}, ()),
    )  # the step, per section: (before, after, moves); uid 2 wasn't there before
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


def test_size_cap_drops_oldest_but_keeps_the_newest():
    h = History(snap(0), max_values=10)
    h.commit(snap(4))  # 4 values
    h.commit(snap(8))  # 4 more: 8, fits
    assert len(h.undo_stack) == 2
    h.commit(snap(12))  # 12: the oldest goes
    assert len(h.undo_stack) == 2
    h.commit(snap(40))  # 28 on its own: over, but the newest stays
    assert len(h.undo_stack) == 1
    h.undo()
    assert h.current == snap(12) and h.undo() is None


def test_record_takes_only_what_changed():
    h = History(snap(3))
    moved = ("NOT", "", 9.0, 9.0, {})
    assert h.record({1: moved, 7: None}, {}, {})  # 7 was never there: not a change
    assert h.undo_stack[0][0] == (
        ({1: ("NOT", "", 1.0, 0.0, {})}, {1: moved}, ()),
        ({}, {}, ()),
        ({}, {}, ()),
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
    assert step[0] == ({1: snap(2).parts[1]}, {1: moved}, ())  # only 1 is left in it
    h.undo()
    assert h.current == snap(2)
    h.redo()
    assert h.current.parts == {0: snap(2).parts[0], 1: moved}


def moved_by(base: Snapshot, dx: float, dy: float, uids=None) -> dict:
    return {
        u: (k, label, x + dx, y + dy, props)
        for u, (k, label, x, y, props) in base.parts.items()
        if uids is None or u in uids
    }


def test_a_big_move_is_stored_as_one_delta_and_undone_exactly():
    base = snap(50)
    h = History(base)
    h.record(moved_by(base, 40.0, -20.0), {}, {})
    (step, _, _), = h.undo_stack
    before, after, moves = step[0]
    assert not before and not after  # no copies of the parts...
    ((uids, dx, dy),) = moves  # ...just which ones and by how much
    assert sorted(uids.tolist()) == list(range(50)) and (dx, dy) == (40.0, -20.0)
    assert h.current.parts == moved_by(base, 40.0, -20.0)
    h.undo()
    assert h.current == base
    h.redo()
    assert h.current.parts == moved_by(base, 40.0, -20.0)


def test_moves_that_floats_cant_undo_exactly_keep_their_values():
    # Off-grid floats: x + d - d isn't always x. Only entries whose move is exact both
    # ways may be stored as a delta; the rest keep their values, so undo stays exact.
    base = Snapshot({i: ("NOT", "", 0.1 * i + 1e-3, 0.7 * i, {}) for i in range(400)}, {})
    h = History(base)
    after = moved_by(base, 0.3, 0.1)
    h.record(after, {}, {})
    before_vals, _, moves = h.undo_stack[0][0][0]
    assert moves and before_vals  # some of each
    for uids, dx, dy in moves:
        for u in uids.tolist():
            x0, y0 = base.parts[u][2:4]
            x1, y1 = after[u][2:4]
            assert (x0 + dx, y0 + dy) == (x1, y1) and (x1 - dx, y1 - dy) == (x0, y0)
    for _ in range(3):
        h.undo()
        assert h.current == base  # bit for bit
        h.redo()
        assert h.current.parts == after


def test_wires_with_bends_and_junctions_move_as_deltas():
    wires = {
        i: (("p", 1, False, 0), ("w", 99), ((10.0 * i, 5.0), (10.0 * i, 9.0)), None, (3.0, 4.0))
        for i in range(10)
    }
    base = Snapshot({}, wires)
    h = History(base)
    shifted = {
        u: (s, d, tuple((x + 20.0, y) for x, y in b), sp, (dp[0] + 20.0, dp[1]))
        for u, (s, d, b, sp, dp) in wires.items()
    }
    h.record({}, shifted, {})
    before, after, moves = h.undo_stack[0][0][1]
    assert not before and not after and len(moves) == 1
    h.undo()
    assert h.current == base
    h.redo()
    assert h.current.wires == shifted


def test_amend_merges_into_a_step_that_moved():
    base = snap(30)
    h = History(base)
    h.record(moved_by(base, 20.0, 0.0), {}, {})  # a Ctrl+scroll notch...
    h.amend(moved_by(base, 40.0, 0.0), {}, {})  # ...and another: one step
    assert len(h.undo_stack) == 1
    ((uids, dx, dy),) = h.undo_stack[0][0][0][2]
    assert (dx, dy) == (40.0, 0.0)  # (from where the step started)
    h.amend(moved_by(base, 40.0, 0.0, uids={0}) | {1: base.parts[1]}, {}, {})
    h.undo()
    assert h.current == base
    h.redo()
    assert h.current.parts == moved_by(base, 40.0, 0.0) | {1: base.parts[1]}


def test_int_positions_come_back_as_ints():
    base = Snapshot({i: ("NOT", "", 20 * i, 40, {}) for i in range(20)}, {})  # ints
    h = History(base)
    h.record(moved_by(base, 20.0, 0.0), {}, {})
    h.undo()
    assert all(type(d[2]) is int for d in h.current.parts.values())
