from pijl.ui.views import project_onto
from pijl.ui.wire_edit import _anchor_of, _at

L_SHAPE = [(0, 0), (100, 0), (100, 100)]


def test_junction_on_a_corner_anchors_to_that_vertex_exactly():
    k, t = _anchor_of(L_SHAPE, (100, 0))
    assert t in (0.0, 1.0)
    moved = [(0, 0), (150, 40), (100, 100)]  # the corner got dragged
    assert _at(moved, k, t) == (150, 40)


def test_junction_mid_segment_keeps_its_fraction():
    k, t = _anchor_of(L_SHAPE, (100, 25))
    assert (k, t) == (1, 0.25)
    stretched = [(0, 0), (100, 0), (100, 200)]
    p = _at(stretched, k, t)
    assert p == (100, 50)
    assert (
        project_onto(stretched, p) == p
    )  # on the line bit for bit: refresh won't nudge it
