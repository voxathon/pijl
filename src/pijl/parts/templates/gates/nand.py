"""NAND: 0 only when both inputs are 1. Every other gate can be built from this one."""

from pijl.parts import part

API = 2


def register(reg):
    reg.add(part("NAND", ins=("a", "b"), outs=("out",), eval=lambda a, b: ~(a & b), category="GATES"))
