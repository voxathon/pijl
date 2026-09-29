from .._common import invert
from pijl.parts import part

API = 1


def register(reg):
    reg.add(part("INV", ins=("a",), outs=("out",), eval=invert))
