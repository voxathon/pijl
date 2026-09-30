"""BUF: passes its input on, one tick later. A floating input comes out as X."""

from pijl.parts import part

API = 2


def register(reg):
    reg.add(part("BUF", ins=("a",), outs=("out",), eval=lambda a: a & 1, category="GATES"))
