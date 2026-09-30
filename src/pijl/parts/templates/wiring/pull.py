"""PULLUP / PULLDOWN: what a net reads when nothing else drives it (all Z).

Any other driver wins over a pull. Several pulls on one net: only the ones with the
highest `priority` count, and if those disagree it's a conflict (X). Raise one's
priority to settle it.
"""

from pijl.logic import ONE, ZERO
from pijl.parts import PartType

API = 2


class Pull(PartType):
    outs = ("out",)
    weak = ("out",)
    props = {"priority": 0}
    pure = True
    category = "WIRING"

    def __init__(self, kind, level):
        self.kind, self.level = kind, level

    def eval(self, ctx):
        return self.level


def register(reg):
    reg.add(Pull("PULLUP", ONE))
    reg.add(Pull("PULLDOWN", ZERO))
