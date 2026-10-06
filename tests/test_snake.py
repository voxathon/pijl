"""The snake bundled mod: it takes over the editor's run, and the rules hold. The
window itself needs GL and isn't opened here."""

import random
import sys
from collections import deque
from pathlib import Path

import pytest

from pijl import mods

SHIPPED = Path(mods.__file__).parent / "bundled"


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    """The mod, enabled and loaded; editor.run is put back afterwards."""
    import pijl.ui.editor as editor

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mods, "BUNDLED", SHIPPED)
        mp.setenv("PIJL_DATA", str(tmp_path_factory.mktemp("data")))
        mp.setenv("PIJL_MODS", "")
        mp.setenv("PIJL_SAFE", "")
        mp.setattr(editor, "run", editor.run)
        mods.plan()
        mods.enable("snake")
        rep = mods.load()
        assert [m.name for m in rep.loaded] == ["snake"] and not rep.problems, rep.problems
        yield sys.modules["pijl_mods.snake"]
        mods._report = None
        mods._hooks.clear()
        mods._tracebacks.clear()
        for name in [n for n in sys.modules if n == mods.PACKAGE or n.startswith(mods.PACKAGE + ".")]:
            del sys.modules[name]


def test_it_replaces_the_editor(loaded):
    import pijl.ui.editor as editor

    assert editor.run is loaded.run


def test_it_eats_grows_and_speeds_up(loaded):
    g = loaded.Game(random.Random(1))
    hx, hy = g.body[0]
    g.food = (hx + 1, hy)
    g.step()
    assert g.score == 1 and len(g.body) == 4 and g.body[0] == (hx + 1, hy)
    assert g.eaten in loaded.FOOD and g.food not in g.body
    assert g.hz > loaded.START_HZ


def test_no_turning_back_and_turns_queue(loaded):
    g = loaded.Game(random.Random(1))
    g.steer("left")  # (heading right: ignored)
    assert not g.turns
    g.steer("up")
    g.steer("left")  # (after up, left is fine)
    g.food = None
    g.step()
    g.step()
    assert g.heading == (-1, 0)


def test_walls_and_itself_kill(loaded):
    g = loaded.Game(random.Random(1))
    g.food = None
    while not g.dead:
        g.step()
    assert g.body[0][0] == loaded.COLS - 1  # (ran into the right wall)

    g.reset()
    g.food = None
    g.body = deque([(5, 5), (6, 5), (6, 4), (5, 4), (4, 4)])
    g.heading = (0, -1)  # (straight down into its own body)
    g.step()
    assert g.dead


def test_the_tail_moves_out_of_the_way(loaded):
    g = loaded.Game(random.Random(1))
    g.food = None
    g.body = deque([(5, 5), (6, 5), (6, 4), (5, 4)])
    g.heading = (0, -1)  # (into the cell the tail is leaving)
    g.step()
    assert not g.dead


def test_pause_holds_still(loaded):
    g = loaded.Game(random.Random(1))
    g.paused = True
    before = list(g.body)
    g.step()
    assert list(g.body) == before
