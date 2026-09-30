from pijl.ui.spatial import SpatialHash


def test_rect_query_and_move():
    h = SpatialHash(cell=10)
    h.put_rect("a", 0, 0, 5, 5)
    h.put_rect("b", 100, 100, 105, 105)
    assert h.near(2, 2, 1) == {"a"}
    assert h.query(-50, -50, 200, 200) == {"a", "b"}
    h.put_rect("a", 100, 0, 105, 5)  # moved: gone from its old cells
    assert h.near(2, 2, 1) == set() and h.near(101, 1, 1) == {"a"}


def test_polyline_claims_only_cells_along_it():
    h = SpatialHash(cell=10)
    h.put_polyline("w", [(0, 0), (100, 100)])  # a diagonal
    assert "w" in h.near(55, 55, 1)
    assert "w" not in h.near(5, 95, 1)  # inside its bounding box, far from the line


def test_remove_and_empty_cells():
    h = SpatialHash(cell=10)
    h.put_polyline("w", [(0, 0), (30, 0), (30, 30)])
    h.remove("w")
    assert h.cells == {} and len(h) == 0


def test_huge_query_walks_occupied_cells():
    h = SpatialHash(cell=10)
    h.put_rect("a", 0, 0, 1, 1)
    h.put_rect("b", 1e6, 1e6, 1e6 + 1, 1e6 + 1)
    assert h.query(-1e7, -1e7, 1e7, 1e7) == {"a", "b"}
    assert h.query(-1e7, -1e7, 10, 10) == {"a"}
