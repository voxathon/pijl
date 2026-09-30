from pijl.ui import theme as T
from pijl.ui.camera import MIN_LEVEL, Camera


def test_fit_level_fits():
    level = Camera.fit_level(10000, 1000, 1000, 1000)
    assert 2 ** (level / 8) * 10000 <= 1000 < 2 ** ((level + 1) / 8) * 10000


def test_zoom_out_stops_at_min_level():
    cam = Camera()
    cam.min_level = MIN_LEVEL - 10  # a big board
    cam.set_level(-100, 0, 0)
    assert cam.level == MIN_LEVEL - 10


def test_shrunk_floor_never_pushes_the_camera_in():
    cam = Camera()
    cam.min_level = MIN_LEVEL - 10
    cam.set_level(MIN_LEVEL - 10, 0, 0)
    cam.min_level = MIN_LEVEL  # the board got smaller
    assert not cam.set_level(cam.level - 1, 0, 0)  # can't go further out ...
    assert cam.level == MIN_LEVEL - 10  # ... but stays where it was
    assert cam.set_level(cam.level + 1, 0, 0)  # and zooming in still works
