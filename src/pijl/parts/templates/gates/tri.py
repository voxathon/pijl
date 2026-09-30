"""TRI: a tri-state buffer. With `en` at 1 it drives its input onto the net; at 0 it
lets go (Z), so several of them can share one bus. An unknown `en` gives X."""

from pijl.logic import Z, where
from pijl.parts import part

API = 2


def register(reg):
    reg.add(part("TRI", ins=("a", "en"), outs=("out",), eval=lambda a, en: where(en, a, Z), category="GATES"))
