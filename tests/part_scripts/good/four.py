"""API 2: a gate that sees X itself (compare with AND16 in misc.py, an API 1 script)."""

from pijl.parts import part

API = 2


def register(reg):
    reg.add(part("AND4S", ins=("a", "b"), outs=("out",), eval=lambda a, b: a & b))
