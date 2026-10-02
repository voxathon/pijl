"""The editor's window and everything drawn in it.

Importing this package doesn't import the editor: modules here read theme.UI_SCALE
when they're first imported, so the launcher sets it before anything else is loaded.
"""


def __getattr__(name: str):
    if name in ("Editor", "run"):
        from . import editor

        return getattr(editor, name)
    raise AttributeError(name)


__all__ = ["Editor", "run"]
