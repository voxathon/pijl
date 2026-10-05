"""Mod rows in the right-click menus (ui/menus.py). Needs GL (see hidden_editor.py)."""

import pytest
from hidden_editor import hidden_editor


@pytest.fixture(scope="module")
def ed(tmp_path_factory):
    editor = hidden_editor(tmp_path_factory)
    yield editor
    editor.close()


@pytest.fixture
def menus(monkeypatch):
    from pijl.ui import menus

    monkeypatch.setattr(menus, "PROVIDERS", {k: [] for k in menus.KINDS})
    return menus


def opened(ed, monkeypatch, open_it):
    """The rows of the menu `open_it` opens."""
    got = []
    monkeypatch.setattr(ed, "_open_menu", lambda x, y, items: got.append(items))
    open_it()
    (items,) = got
    return [i.text for i in items]


def test_every_menu_takes_mod_rows(ed, menus, monkeypatch):
    from pijl.ui.menu import MenuItem

    ed._cancel()
    ed._clear_board()
    seen = {}
    for kind in menus.KINDS:

        @menus.items(kind)
        def rows(editor, target, kind=kind):
            seen[kind] = target
            return [MenuItem(f"mod {kind}", lambda: None)]

    a = ed.add_part("NOT", 0, 0)
    b = ed.add_part("NOT", 200, 0)
    w = ed.wire_views[ed.connect(a.part.outputs[0], b.part.inputs[0])]
    box = ed.add_box(("b", -100, -200, 500, 400, None, {}))

    part = opened(ed, monkeypatch, lambda: ed._item_menu(0, 0, a.x + 5, a.y + 5, 0))
    assert part[-2:] == ["mod part", "Delete"] and seen["part"] == [a]
    mid = w.points[0][0] + 20, w.points[0][1]
    wire = opened(ed, monkeypatch, lambda: ed._item_menu(0, 0, *mid, 0))
    assert wire[-2:] == ["mod wire", "Delete"] and seen["wire"] is w
    board = opened(ed, monkeypatch, lambda: ed._board_menu(0, 0, 50, -150))
    assert "mod board" in board and "mod box" in board  # (inside the box)
    assert seen["board"] == (50, -150) and seen["box"] is box
    rows = [i.text for i in ed._box_items(box)]
    assert rows.index("mod box") < rows.index("Remove box (keep what's in it)")


def test_a_broken_provider_is_reported_and_skipped(ed, menus):
    from pijl.ui.menu import MenuItem

    @menus.items("box")
    def broken(editor, box):
        raise RuntimeError("oops")

    @menus.items("box")
    def fine(editor, box):
        return [MenuItem("fine", lambda: None)]

    box = ed.add_box(("b", 0, 0, 100, 100, None, {}))
    assert "fine" in [i.text for i in ed._box_items(box)]
    assert "menu rows failed: RuntimeError: oops" in ed.status.text
    with pytest.raises(ValueError):
        menus.items("nope")
