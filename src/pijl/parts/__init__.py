"""Part types: what kinds of parts exist and how each one behaves.

Everything except IN/OUT (see ports.py) comes from part scripts. The ones that
ship with pijl live in templates/ and are copied into every new project; the
project's copy is what gets loaded (see project.py). No pyglet in here.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from .contract import API, GRID_STEP, MAX_WIDTH, Ctx, Layout, Look, Mark, PartType, layout_of, part
from .registry import Registry, check_props, fresh_props, load
from .settings import Action, Choice, Number, Setting, Text, Toggle

TEMPLATES = Path(__file__).parent / "templates"


@cache
def builtin_registry() -> Registry:
    """The ports plus the shipped templates, loaded straight from the package.
    For tests and anything else that has no project."""
    return load(TEMPLATES)


__all__ = [
    "API",
    "Action",
    "Choice",
    "Ctx",
    "GRID_STEP",
    "Layout",
    "Look",
    "MAX_WIDTH",
    "Mark",
    "Number",
    "PartType",
    "Registry",
    "Setting",
    "TEMPLATES",
    "Text",
    "Toggle",
    "builtin_registry",
    "check_props",
    "fresh_props",
    "layout_of",
    "load",
    "part",
]
