from pijl.ui.spatial import SpatialIndex


def test_rect_query_and_move():
    h = SpatialIndex(capacity=2)
    h.put_rect("a", 0, 0, 5, 5)
    h.put_rect("b", 100, 100, 105, 105)
    h.put_rect("c", 200, 0, 205, 5)  # (grows)
    assert h.near(2, 2, 1) == {"a"}
    assert h.query(-50, -50, 300, 300) == {"a", "b", "c"}
    h.put_rect("a", 100, 0, 105, 5)  # moved: gone from where it was
    assert h.near(2, 2, 1) == set() and h.near(101, 1, 1) == {"a"}


def test_polyline_is_boxed_in_pieces():
    h = SpatialIndex()
    h.put_polyline("w", [(0, 0), (1000, 1000)])  # a diagonal
    assert "w" in h.near(555, 555, 1)
    assert "w" not in h.near(50, 950, 1)  # inside its bounding box, far from the line


def test_remove_and_reuse():
    h = SpatialIndex()
    h.put_polyline("w", [(0, 0), (30, 0), (30, 30)])
    h.remove("w")
    assert len(h) == 0 and h.query(-1e9, -1e9, 1e9, 1e9) == set()
    h.put_polyline("v", [(0, 0), (30, 0)])
    h.put_polyline("v", [(0, 0), (500, 0), (500, 500)])  # more pieces, then fewer
    h.put_polyline("v", [(0, 0), (10, 0)])
    assert h.near(400, 0, 1) == set() and h.near(5, 0, 1) == {"v"}


def test_shift_moves_everything_given():
    h = SpatialIndex()
    h.put_rect("a", 0, 0, 1, 1)
    h.put_polyline("w", [(0, 0), (300, 0)])
    h.put_rect("still", 0, 0, 1, 1)
    h.shift(["a", "w"], 1000, 50)
    assert h.near(0.5, 0.5, 0.1) == {"still"}
    assert h.near(1000.5, 50, 0.1) == {"a", "w"} and h.near(1250, 50, 1) == {"w"}


def test_huge_query():
    h = SpatialIndex()
    h.put_rect("a", 0, 0, 1, 1)
    h.put_rect("b", 1e6, 1e6, 1e6 + 1, 1e6 + 1)
    assert h.query(-1e7, -1e7, 1e7, 1e7) == {"a", "b"}
    assert h.query(-1e7, -1e7, 10, 10) == {"a"}
