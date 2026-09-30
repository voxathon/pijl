"""XOR: 1 when its inputs differ."""

from pijl.parts import part

API = 2


def register(reg):
    reg.add(
        part(
            "XOR",
            ins=("a", "b"),
            outs=("out",),
            eval=lambda a, b: a ^ b,
            category="GATES",
        )
    )
