from functools import reduce
from operator import and_

from pijl.parts import part

API = 1


def register(reg):
    reg.add(part("HIGH", outs=("out",), eval=lambda: True))  # scalar: copied to every instance
    reg.add(part("AND16", ins=tuple(f"i{n}" for n in range(16)), outs=("out",),
                 eval=lambda *ins: reduce(and_, ins)))  # 16 inputs: nothing gets enumerated
    reg.add(part("SPLIT", ins=("a",), outs=("same", "flipped"), eval=lambda a: (a, ~a)))
    reg.add(part("BOOM", ins=("a",), outs=("out",), eval=lambda a: 1 / 0))
