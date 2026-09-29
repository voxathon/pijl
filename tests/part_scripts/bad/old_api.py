from pijl.parts import part

API = 0


def register(reg):
    reg.add(part("OLD", outs=("out",), eval=lambda: True))
