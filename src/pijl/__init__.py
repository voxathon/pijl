__version__ = "0.3.1"  # (also in pyproject.toml; a test keeps them equal)


def main() -> None:
    import sys

    from .cli import main

    sys.exit(main())
