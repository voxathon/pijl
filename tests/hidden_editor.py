"""The real editor for the tests that drive it with synthetic events, in a window that
never shows. PIJL_HEADLESS_GL=1 makes pyglet render offscreen through EGL instead (Linux
only: no window system needed at all)."""

import os

import pytest


def hidden_editor(tmp_path_factory):
    """A fresh Editor on its own data root; skips the module where there's no GL."""
    os.environ["PIJL_DATA"] = str(tmp_path_factory.mktemp("pijl-data"))
    try:
        import pyglet

        if os.environ.get("PIJL_HEADLESS_GL"):
            pyglet.options["headless"] = True
        from pijl.ui.editor import Editor

        editor = Editor(visible=False)
    except Exception as e:  # (no display / GL)
        pytest.skip(f"no editor window here: {e}")
    editor._enable_event_queue = False  # dispatch synthetic events right away
    return editor
