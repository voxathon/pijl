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


def test_bulk_boxes_match_one_line_at_a_time():
    import random

    from pijl.ui.spatial import polyline_boxes, polylines_boxes

    rng = random.Random(1)
    lines = [
        [
            (rng.uniform(-3e3, 3e3), rng.uniform(-3e3, 3e3))
            for _ in range(rng.randint(1, 5))
        ]
        for _ in range(200)
    ]
    boxes, counts = polylines_boxes(lines)
    assert counts.tolist() == [len(polyline_boxes(pts)) for pts in lines]
    single = [b for pts in lines for b in polyline_boxes(pts)]
    assert abs(boxes - single).max() < 1e-9


def test_a_straight_run_is_one_box_a_diagonal_many():
    from pijl.ui.spatial import polyline_boxes

    assert len(polyline_boxes([(0, 0), (5000, 0)])) == 1
    assert (
        len(polyline_boxes([(0, 0), (5000, 30)])) == 1
    )  # barely off axis: still tight
    assert len(polyline_boxes([(0, 0), (1000, 1000)])) > 10


def test_put_polylines_replaces_what_was_there():
    h = SpatialIndex()
    h.put_polyline("w", [(0, 0), (500, 0)])
    h.put_rect("r", 1000, 1000, 1001, 1001)
    h.put_polylines([("w", [(0, 100), (10, 100)]), ("v", [(0, 0), (30, 30)])])
    assert h.near(400, 0, 1) == set()  # w's old boxes are gone
    assert h.near(5, 100, 1) == {"w"} and h.near(15, 15, 1) == {"v"}
    assert h.near(1000.5, 1000.5, 0.1) == {"r"}
