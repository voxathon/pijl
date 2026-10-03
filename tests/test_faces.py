"""Part faces: Look.size, Look.titled, Look.face (marks) and the face() hook. The
registry's checks and Circuit.face_codes run anywhere; the editor half needs GL (a
hidden window, see hidden_editor.py) and is skipped where there's none."""

from pathlib import Path

import numpy as np
import pytest
from hidden_editor import hidden_editor

from pijl.logic import ONE, ZERO, X
from pijl.parts import GRID_STEP, TEMPLATES, Look, Mark, PartType, load
from pijl.parts.registry import Registry
from pijl.sim import Circuit

FACES = Path(__file__).parent / "part_scripts" / "faces"


def test_grid_step_matches_the_editor():
    from pijl.ui import theme as T

    assert GRID_STEP == 2 * T.GRID


@pytest.mark.parametrize(
    "look, hook, error",
    [
        (Look(size=(50, 80)), False, "multiples of 20"),
        (Look(size=[60, 80]), False, "multiples of 20"),
        (Look(face=[Mark((1.0, 2.0), pin="a")]), False, "tuple of Marks"),
        (Look(face=((1, 2),)), False, "not a Mark"),
        (Look(face=(Mark([1.0, 2.0], pin="a"),)), False, "(x, y) tuples"),
        (Look(face=(Mark((1.0, 2.0), radius=0, pin="a"),)), False, "radius"),
        (Look(face=(Mark((1.0, 2.0), pin="nope"),)), False, "no pin called 'nope'"),
        (Look(face=(Mark((1.0, 2.0)),)), False, "need a face() hook"),
        (Look(face=(Mark((1.0, 2.0), pin="a"),)), True, "no face marks without a pin"),
        (Look(face_colors=("FACE_OFF",)), False, "two theme color names"),
    ],
)
def test_bad_faces_are_refused(look, hook, error):
    class P(PartType):
        kind = "P"
        ins = ("a",)

    if hook:
        P.face = lambda self, ctx, a: a
    P.look = look
    with pytest.raises((ValueError, TypeError), match=error.replace("(", r"\(").replace(")", r"\)")):
        Registry().add(P())


def test_face_codes_run_the_hook_for_live_parts():
    c = Circuit(load(TEMPLATES, FACES))
    assert not c.registry.errors
    sw = c.add_part("IN")
    lamp = c.add_part("LAMP2")
    ghost = c.add_part("LAMP2", live=False)
    c.connect(sw.outputs[0], lamp.inputs[0])
    t = c.registry.get("LAMP2")
    slots = np.array([lamp.slot, ghost.slot], np.intp)
    for _ in range(3):
        c.step()
    assert c.face_codes(t, slots).tolist() == [[ONE], [ZERO]]  # (~0; a ghost shows 0)
    sw.outputs[0].state = True
    for _ in range(3):
        c.step()
    assert c.face_codes(t, slots).tolist() == [[ZERO], [ZERO]]
    x = c.add_part("X_FACE")
    assert c.face_codes(c.registry.get("X_FACE"), np.array([x.slot])).tolist() == [[X, ONE]]


def test_a_raising_face_hook_faults_its_kind():
    c = Circuit(load(TEMPLATES, FACES))
    p = c.add_part("BROKEN_FACE")
    t = c.registry.get("BROKEN_FACE")
    assert c.face_codes(t, np.array([p.slot])).tolist() == [[X]]
    assert "BROKEN_FACE.face" in c.faults["BROKEN_FACE"]
    assert c.face_codes(t, np.array([p.slot])).tolist() == [[X]]  # (not called again)


# ---- the editor ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def ed(tmp_path_factory):
    editor = hidden_editor(tmp_path_factory)
    editor.parts.load_folder(FACES)
    editor._clear_board()
    editor._reset_history(None)
    yield editor
    editor.close()


def marks(ed):
    from pijl.ui.sdf_shapes import SEGMENT

    return ed.world.buffer(SEGMENT, ed.layers.bodies)


def test_a_face_is_drawn_lit_moved_tinted_and_freed(ed):
    from pyglet.window import key

    from pijl.ui import theme as T
    from pijl.ui.paint import with_hue
    from pijl.ui.sdf_shapes import SHOW_OFF, SHOW_ON, SHOW_X, SHOW_Z, _rgba

    buf = marks(ed)
    before = int(buf.used.sum())
    lamp = ed.add_part("LAMP2", 100, 0)
    assert (lamp.w, lamp.h) == (60, 80) and lamp.title == ""
    t = ed.part_table
    slots = t.face_slots(lamp.row)
    assert len(slots) == 2 and int(buf.used.sum()) == before + 2
    assert buf.f["a"][slots].tolist() == [[110.0, 40.0], [145.0, 40.0]]
    assert buf.f["b"][slots[1]].tolist() == [145.0, 40.0]  # (a dot)
    ed.update(1 / 60)
    assert buf.state[slots].tolist() == [SHOW_Z, SHOW_X]  # (a floats: Z, and ~Z is X)
    sw = ed.add_part("IN", 0, 0)
    ed.connect(sw.part.outputs[0], lamp.part.inputs[0])
    sw.part.outputs[0].state = True
    ed._record()
    for _ in range(3):
        ed.update(1 / 60)
    assert buf.state[slots].tolist() == [SHOW_ON, SHOW_OFF]
    # moved with the body, and with a drag's lift
    lamp.move_to(140, 20)
    assert buf.f["a"][slots].tolist() == [[150.0, 60.0], [185.0, 60.0]]
    lamp.set_lifted(True)
    assert buf.f["lift"][slots].tolist() == [1.0, 1.0]
    lamp.set_lifted(False)
    lamp.set_ghost(True)
    assert buf.f["flags"][slots, 1].tolist() == [T.GHOST_OPACITY] * 2
    lamp.set_ghost(False)
    # recolored: tinted
    ed._set_part_color(lamp, "blue")
    ed._record()  # (painting goes with recording)
    tint = T.WIRE_COLORS["blue"][1]
    assert tuple(buf.f["ca_on"][slots[0]]) == _rgba(with_hue(T.FACE_ON, tint))
    # deleted: freed; undone: back
    ed.remove_part(lamp)
    ed._record()
    ed.update(1 / 60)
    assert int(buf.used.sum()) == before
    assert not t.hooked
    ed.dispatch_event("on_key_press", key.Z, key.MOD_CTRL)
    ed.update(1 / 60)
    assert int(buf.used.sum()) == before + 2 and len(t.hooked) == 1
    ed._clear_board()
    ed.update(1 / 60)
    assert int(marks(ed).used.sum()) == 0
