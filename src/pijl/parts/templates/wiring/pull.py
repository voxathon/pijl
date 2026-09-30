"""PULLUP / PULLDOWN: what a net reads when nothing else drives it (all Z).

A pull sits inline: `in` and `out` are the same net straight through it (no delay,
either direction), and the pull drives that net weakly. Any other driver wins over
a pull. Several pulls on one net: only the ones with the highest `priority` count,
and if those disagree it's a conflict (X). Raise one's priority to settle it
(right-click it: Priority).
"""

from pijl.logic import ONE, ZERO
from pijl.parts import PartType

API = 2


class Pull(PartType):
    ins = ("in",)
    outs = ("out",)
    joins = (("in", "out"),)  # one net through the part...
    weak = ("out",)           # ...which it drives weakly
    props = {"priority": 0}
    choices = {"priority": tuple(range(10))}  # (the part's context menu)
    pure = True
    category = "WIRING"

    def __init__(self, kind, level):
        self.kind, self.level = kind, level

    def eval(self, ctx, net):  # (net: what the net reads; a pull doesn't care)
        return self.level


def register(reg):
    reg.add(Pull("PULLUP", ONE))
    reg.add(Pull("PULLDOWN", ZERO))
