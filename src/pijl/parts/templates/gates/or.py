"""OR: 1 when either input is 1."""

from pijl.parts import part

API = 2


def register(reg):
    reg.add(part("OR", ins=("a", "b"), outs=("out",), eval=lambda a, b: a | b, category="GATES"))
