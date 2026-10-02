"""What the editor logs (pijl.edit, pijl.files): edits, undo / redo, saves. Needs GL
(a hidden window, see hidden_editor.py); skipped where there's none."""

import logging

import pytest
from hidden_editor import hidden_editor


@pytest.fixture(scope="module")
def ed(tmp_path_factory):
    editor = hidden_editor(tmp_path_factory)
    editor._clear_board()
    editor._reset_history(None)
    yield editor
    editor.close()


def test_edits_undo_redo_and_saves_are_logged(ed, caplog):
    caplog.set_level(logging.DEBUG, logger="pijl")
    a = ed.add_part("NOT", 0, 0)
    b = ed.add_part("NOT", 200, 0)
    ed.connect(a.part.outputs[0], b.part.inputs[0])
    ed._record()
    ed._undo()
    ed._redo()
    assert ed._write("logged")
    lines = [(r.name, r.getMessage()) for r in caplog.records]
    assert ("pijl.edit", "edit: +2 parts, +1 wire") in lines
    assert ("pijl.edit", "undo: +2 parts, +1 wire") in lines
    assert ("pijl.edit", "redo: +2 parts, +1 wire") in lines
    assert any(n == "pijl.files" and m.startswith("saved logged [logged]: 2 parts") for n, m in lines)
