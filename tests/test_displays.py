"""The displays bundled mod: its parts reach every registry, and HEX decodes. How faces
are drawn is core's (test_faces.py); the editor half here needs GL (a hidden window,
see hidden_editor.py) and is skipped where there's none."""

import sys
from pathlib import Path

import numpy as np
import pytest
from hidden_editor import hidden_editor

from pijl import mods
from pijl.logic import ONE, ZERO, X, Z
from pijl.parts import TEMPLATES, registry
from pijl.sim import Circuit

SHIPPED = Path(mods.__file__).parent / "bundled"


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    """The mod, enabled and loaded; the registry patch is taken back afterwards."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mods, "BUNDLED", SHIPPED)
        mp.setenv("PIJL_DATA", str(tmp_path_factory.mktemp("data")))
        mp.setenv("PIJL_MODS", "")
        mp.setenv("PIJL_SAFE", "")
        mp.setattr(registry.Registry, "load_folder", registry.Registry.load_folder)
        mods.plan()
        mods.enable("displays")
        rep = mods.load()
        assert [m.name for m in rep.loaded] == ["displays"] and not rep.problems, rep.problems
        yield sys.modules["pijl_mods.displays"]
        mods._report = None
        mods._hooks.clear()
        mods._tracebacks.clear()
        for name in [n for n in sys.modules if n == mods.PACKAGE or n.startswith(mods.PACKAGE + ".")]:
            del sys.modules[name]


def test_every_registry_gets_the_displays(loaded, tmp_path):
    reg = registry.load(TEMPLATES)  # (not builtin_registry: cached, maybe from before)
    assert {"7SEG", "HEX", "BAR"} <= {t.kind for t in reg} and not reg.errors
    # a project's own part of the same name wins
    (tmp_path / "mine.py").write_text(
        "from pijl.parts import part\nAPI = 2\n"
        "def register(reg):\n    reg.add(part('HEX', ins=('a',), outs=('b',), eval=lambda a: a))\n"
    )
    reg = registry.load(tmp_path)
    assert reg.get("HEX").ins == ("a",) and "7SEG" in reg and not reg.errors


def test_hex_decodes(loaded):
    c = Circuit(registry.load(TEMPLATES))
    t = c.registry.get("HEX")
    digits = [c.add_part("HEX") for _ in range(17)]
    for value, d in enumerate(digits[:16]):
        for i, pin in enumerate(d.inputs):
            sw = c.add_part("IN")
            sw.outputs[0].state = bool(value >> (3 - i) & 1)
            c.connect(sw.outputs[0], pin)
    half = c.add_part("HEX")  # one input driven: the rest float, so X
    c.connect(c.add_part("IN").outputs[0], half.inputs[0])
    for _ in range(3):
        c.step()
    got = c.face_codes(t, np.array([d.slot for d in [*digits, half]], np.intp))
    lit = loaded.parts.DECODE
    assert (got[:16] == np.where(lit, ONE, ZERO)).all()
    assert (got[16] == Z).all() and (got[17] == X).all()


def test_the_editor_draws_them(loaded, tmp_path_factory):
    from pijl.ui.sdf_shapes import SEGMENT

    ed = hidden_editor(tmp_path_factory)
    try:
        buf = ed.world.buffer(SEGMENT, ed.layers.bodies)
        before = int(buf.used.sum())
        views = [ed.add_part(k, 200 * i, 0) for i, k in enumerate(("7SEG", "HEX", "BAR"))]
        ed.update(1 / 60)
        assert int(buf.used.sum()) == before + 8 + 7 + 8
        assert [(v.w, v.h) for v in views] == [(120, 180), (120, 180), (60, 180)]
    finally:
        ed.close()
