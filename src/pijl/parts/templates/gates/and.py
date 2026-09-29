"""AND: 1 only when both inputs are 1."""

from pijl.parts import part

API = 1


def register(reg):
    reg.add(part("AND", ins=("a", "b"), outs=("out",), eval=lambda a, b: a & b, category="GATES"))
