from pijl.parts import part

API = 1


def register(reg):
    reg.add(part("FINE", outs=("out",), eval=lambda: False))
