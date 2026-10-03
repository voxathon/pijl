"""Displays & Stuff: a bundled mod that adds display parts.

    7SEG    a seven-segment digit: pins a..g and dp, one per segment
    HEX     a digit that decodes four bits (8, 4, 2, 1) and shows 0..F
    BAR     eight LEDs in a column, one per pin

They're in the part picker under DISPLAYS. Recolor one from its right-click menu.

Everything they look like is in parts.py, through the part contract (Look.size,
Look.face, the face() hook): this mod draws nothing itself. What it does patch is
the registry: the parts are added after the project's own part scripts, so a
project that defines a part with the same name keeps its own (see MODDING.md,
"Adding part types from a mod"). Boards saved with them need this mod to open
them with the displays in place.
"""

from __future__ import annotations

from pijl.mods import after_import

from .parts import PARTS


@after_import("pijl.parts.registry")
def _(m):
    old = m.Registry.load_folder

    def load_folder(self, root):
        old(self, root)
        for t in PARTS:
            if t.kind not in self:  # (the project's own part of that name wins)
                self.add(t)

    m.Registry.load_folder = load_folder
