"""LineEdit: typing, caret motions, and the selection (anchor .. caret)."""

from pyglet.window import key

from pijl.ui.line_edit import LineEdit


def test_shift_motions_select_and_typing_replaces_the_selection():
    e = LineEdit("hello world", 40)
    for _ in range(5):
        e.motion(key.MOTION_LEFT, select=True)
    assert e.selected_text == "world"
    e.insert("there")
    assert (e.text, e.caret, e.selection) == ("hello there", 11, None)


def test_select_all_then_backspace_empties():
    e = LineEdit("abc", 40)
    e.select_all()
    e.motion(key.MOTION_BACKSPACE)
    assert (e.text, e.caret) == ("", 0)


def test_shift_backspace_still_just_deletes():
    e = LineEdit("abc", 40)
    e.motion(key.MOTION_BACKSPACE, select=True)
    assert (e.text, e.selection) == ("ab", None)


def test_arrow_without_shift_goes_to_that_end_of_the_selection():
    e = LineEdit("abcdef", 40)
    e.set_caret(1)
    e.set_caret(4, extend=True)
    e.motion(key.MOTION_LEFT)
    assert (e.caret, e.selection) == (1, None)
    e.set_caret(4, extend=True)
    e.motion(key.MOTION_RIGHT)
    assert (e.caret, e.selection) == (4, None)


def test_word_motions():
    e = LineEdit("one two  three", 40)
    e.motion(key.MOTION_PREVIOUS_WORD)
    assert e.caret == 9
    e.motion(key.MOTION_PREVIOUS_WORD, select=True)
    assert e.selected_text == "two  "
    e.set_caret(0)
    e.motion(key.MOTION_NEXT_WORD)
    assert e.caret == 4


def test_replacing_a_selection_respects_max_len():
    e = LineEdit("abcd", 5)
    e.set_caret(0)
    e.set_caret(2, extend=True)
    e.insert("wxyz")
    assert e.text == "wxycd"
