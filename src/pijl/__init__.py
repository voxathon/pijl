import logging

__version__ = "0.3.7"  # (also in pyproject.toml; a test keeps them equal)

# pijl logs to the "pijl.*" loggers (pijl.app, .mods, .files, .edit, .ui, .sim) and
# attaches no handlers: a mod (or a program using pijl) decides where it goes. The
# NullHandler keeps Python's last-resort handler from printing to stderr meanwhile.
logging.getLogger("pijl").addHandler(logging.NullHandler())


def main() -> None:
    import sys

    from .cli import main

    sys.exit(main())
