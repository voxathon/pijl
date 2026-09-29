"""A stand-in for a bridge to the outside world: a worker thread feeds eval."""

import threading

from pijl.parts import PartType

API = 1


class Bridge(PartType):
    kind = "BRIDGE"
    outs = ("rx",)

    def open(self, part):
        part.state["value"] = False
        part.state["stop"] = stop = threading.Event()

        def work():
            part.state["value"] = True  # "the remote end answered"
            stop.wait()

        part.state["thread"] = t = threading.Thread(target=work, daemon=True)
        t.start()

    def close(self, part):
        part.state["stop"].set()
        part.state["thread"].join(timeout=1)

    def eval(self, ctx):
        return [s["value"] for s in ctx.state]


def register(reg):
    reg.add(Bridge)
