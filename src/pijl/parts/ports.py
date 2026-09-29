"""IN and OUT: the two parts the engine itself defines.

Inside a macro they are its pins (the flattener removes them and joins the nets
on either side), so their meaning can't be up to a script. On the top-level
board they're a switch and an LED. No script may define a part with a `port`.
"""

from __future__ import annotations

from .contract import Look, PartType


class Switch(PartType):
    kind = "IN"
    outs = ("out",)
    port = "in"
    category = "I/O"
    look = Look(narrow=True, label="left", lit=("SWITCH_OFF", "SWITCH_ON"), swatch="SWITCH_ON")

    def click(self, part) -> None:
        # No eval: the output keeps whatever the user last set it to.
        pin = part.outputs[0]
        pin.state = not pin.state


class Led(PartType):
    kind = "OUT"
    ins = ("in",)
    port = "out"
    category = "I/O"
    look = Look(narrow=True, label="right", lit=("LED_OFF", "LED_ON"), swatch="LED_ON")


PORTS: tuple[PartType, ...] = (Switch(), Led())
