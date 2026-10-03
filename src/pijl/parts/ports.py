"""IN and OUT: the two parts the engine itself defines.

Inside a macro they are its pins (the flattener removes them and joins the nets
on either side), so their meaning can't be up to a script. On the top-level
board they're a switch and an LED. No script may define a part with a `port`.

Both can be a bus: their width setting is how many lanes (and so is the macro pin
they make).
"""

from __future__ import annotations

from ..logic import ONE, ZERO, Logic
from .contract import MAX_WIDTH, Look, PartType
from .settings import Number


WIDTH = Number(1, 1, MAX_WIDTH, hint="lanes: 2 or more is a bus")


class Switch(PartType):
    kind = "IN"
    outs = ("out",)
    settings = {"width": WIDTH}
    widths = {"out": "width"}
    port = "in"
    category = "I/O"
    look = Look(
        narrow=True,
        label="left",
        lit=("SWITCH_OFF", "SWITCH_ON"),
        swatch="SWITCH_ON",
        pin_labels=False,
        cells="out",  # a bus: a switch per lane
    )  # one pin: a tag would only say "out"

    def click(self, part) -> None:
        # No eval: the output keeps whatever the user last set it to.
        pin = part.outputs[0]
        if pin.width == 1:
            pin.state = not pin.state

    def click_cell(self, part, lane: int) -> None:
        pin = part.outputs[0]
        lanes = pin.state.codes.copy()
        lanes[lane] = int(ZERO) if lanes[lane] == int(ONE) else int(ONE)  # (X, Z: on)
        pin.state = Logic.of_codes(lanes)


class Led(PartType):
    kind = "OUT"
    ins = ("in",)
    settings = {"width": WIDTH}
    widths = {"in": "width"}
    port = "out"
    category = "I/O"
    look = Look(
        narrow=True,
        label="right",
        lit=("LED_OFF", "LED_ON"),
        swatch="LED_ON",
        pin_labels=False,
        cells="in",  # a bus: a light per lane
    )


PORTS: tuple[PartType, ...] = (Switch(), Led())
