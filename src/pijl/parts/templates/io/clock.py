"""CLK: a clock. Its output is low for the first half of every period and high for
the second, so circuits get half a period to settle before each rising edge.

The period counts simulation ticks (one tick = one gate delay), not seconds, so a
clock runs the same in the editor, headless and in a bench. All clocks with the
same period are in phase: they follow the circuit's tick count. Odd periods spend
the extra tick low. Right-click one to set its period.
"""

import numpy as np

from pijl.parts import Look, Number, PartType

API = 2


class Clock(PartType):
    kind = "CLK"
    outs = ("clk",)
    settings = {
        "period": Number(
            32,
            2,
            1_000_000,
            unit="ticks",
            log=True,
            hint="a full cycle: low for half of it, then high",
        )
    }
    category = "I/O"
    look = Look(
        narrow=True,
        label="left",
        lit=("LED_OFF", "LED_ON"),
        swatch="LED_ON",
        pin_labels=False,
    )

    def eval(self, ctx):
        period = np.fromiter((p["period"] for p in ctx.props), np.int64, ctx.n)
        return ctx.tick % period >= (period + 1) // 2


def register(reg):
    reg.add(Clock())
