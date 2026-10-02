"""Selecting in every in-place text field: the prompt, a collection's name, a part's
label, the number popover (the search field has its own, test_picker_search.py).
Needs GL (a hidden window, see hidden_editor.py); skipped where there's none."""

import pytest
from hidden_editor import hidden_editor


@pytest.fixture(scope="module")
def ed(tmp_path_factory):
    editor = hidden_editor(tmp_path_factory)
    yield editor
    editor.close()


@pytest.fixture
def fresh(ed, monkeypatch):
    # the system clipboard is shared with every other program (and racy): a fake one
    clip = [""]
    monkeypatch.setattr(
        ed, "set_clipboard_text", lambda text: clip.__setitem__(0, text)
    )
    monkeypatch.setattr(ed, "get_clipboard_text", lambda: clip[0])
    ed._cancel()
    ed._clear_board()
    ed.text_mouse.forget()
    if not ed.picker.open:
        ed.picker.toggle()
    return ed


def key_press(ed, symbol, modifiers=0):
    ed.dispatch_event("on_key_press", symbol, modifiers)


def press(ed, x, y, modifiers=0):
    from pyglet.window import mouse

    ed.dispatch_event("on_mouse_press", x, y, mouse.LEFT, modifiers)
    ed.dispatch_event("on_mouse_release", x, y, mouse.LEFT, modifiers)


def test_prompt_starts_selected_and_selects_like_any_field(fresh):
    from pyglet.window import key

    from pijl.ui.editor import Mode

    ed = fresh
    ed._save_as()
    assert ed.mode is Mode.PROMPT
    edit = ed.prompt.edit
    if edit.text:  # prefilled: all selected, typing replaces it
        assert edit.selected_text == edit.text
    ed.dispatch_event("on_text", "adder")
    assert edit.text == "adder"
    ed.dispatch_event("on_text_motion_select", key.MOTION_LEFT)
    ed.dispatch_event("on_text_motion_select", key.MOTION_LEFT)
    assert edit.selected_text == "er"
    assert ed.prompt.cursor.sel.visible
    key_press(ed, key.X, key.MOD_CTRL)  # cut...
    assert edit.text == "add"
    key_press(ed, key.V, key.MOD_CTRL)  # ...and paste back twice
    key_press(ed, key.V, key.MOD_CTRL)
    assert edit.text == "adderer"
    # a click in the field puts the caret there (and doesn't close the prompt)
    c = ed.prompt.cursor
    press(ed, c.x0 + c.width("ad"), ed.prompt.field_text.y)
    assert ed.mode is Mode.PROMPT
    assert (edit.caret, edit.selection) == (2, None)
    key_press(ed, key.A, key.MOD_CTRL)
    assert edit.selected_text == "adderer"


def test_collection_names_select_and_stay_upper_case(fresh):
    from pyglet.window import key

    ed = fresh
    ed._start_rename(ed.library.new_collection(), fresh=True)
    edit = ed.picker.edit
    ed.dispatch_event("on_text", "logic")
    assert edit.text == "LOGIC"
    key_press(ed, key.A, key.MOD_CTRL)
    ed.set_clipboard_text("misc ß")
    key_press(ed, key.V, key.MOD_CTRL)  # replaces the selection, made upper case too
    assert edit.text == "MISC SS"
    assert edit.caret == len(edit.text)
    ed.dispatch_event("on_text_motion_select", key.MOTION_PREVIOUS_WORD)
    assert edit.selected_text == "SS"
    w = ed.picker.widgets[("section", ed.picker.renaming)]
    assert w.cursor.sel.visible
    key_press(ed, key.ESCAPE)


def test_part_labels_select_and_take_clicks(fresh):
    from pyglet.window import key

    from pijl.ui.editor import Mode

    ed = fresh
    ed._start_placing("AND")
    ed._commit_placing(again=False)
    view = next(iter(ed.part_views.values()))
    ed._start_edit(view)
    assert ed.mode is Mode.EDITING_LABEL
    ed.dispatch_event("on_text", "carry")
    ed.dispatch_event("on_text_motion_select", key.MOTION_BEGINNING_OF_LINE)
    assert ed.edit.selected_text == "carry"
    assert ed.edit_sel.visible and ed.edit_sel.width > 0
    # a click on the label's text puts the caret between the letters there
    name = view.name
    x, y = ed.camera.world_to_screen(
        (name.caret_x(1) + name.caret_x(2)) / 2 + 0.01, name.y
    )
    press(ed, x, y)
    assert ed.mode is Mode.EDITING_LABEL
    assert (ed.edit.caret, ed.edit.selection) == (2, None)
    key_press(ed, key.ENTER)
    assert view.part.label == "carry"


def test_number_popover_value_starts_selected(ed):
    import pyglet
    from pyglet.window import key

    from pijl.parts import Number
    from pijl.ui.popover import NumberPopover

    pop = NumberPopover(
        pyglet.graphics.Batch(), 800, 600, (100, 500), "Delay", Number(5, 0, 10), 5, ""
    )  # (drawn in the editor's GL context)
    assert pop.edit.selected_text == pop.edit.text != ""
    pop.type_text("7")
    assert pop.text == "7"
    pop.type_text("25")
    pop.motion(key.MOTION_LEFT, select=True)
    assert pop.edit.selected_text == "5"
    assert pop.cursor.sel.visible
    pop.delete()
