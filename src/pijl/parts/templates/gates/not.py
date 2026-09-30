"""NOT: flips its input."""

from pijl.parts import part

API = 2


def register(reg):
    reg.add(part("NOT", ins=("a",), outs=("out",), eval=lambda a: ~a, category="GATES"))
