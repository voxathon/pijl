"""Exercises the non-pure hooks: open/close, frame, props and per-instance state."""

from pijl.parts import PartType

API = 1


class Counter(PartType):
    kind = "COUNTER"
    outs = ("odd",)
    props = {"step": 1}
    category = "TEST"

    def __init__(self):
        self.log = []  # (hook, uid): read by the tests

    def open(self, part):
        self.log.append(("open", part.uid))
        part.state["count"] = 0

    def close(self, part):
        self.log.append(("close", part.uid))

    def frame(self, ctx):
        for props, state in zip(ctx.props, ctx.state):
            state["count"] += props["step"]

    def eval(self, ctx, *ins):
        return [s["count"] % 2 == 1 for s in ctx.state]


def register(reg):
    reg.add(Counter)
