from pijl.parts import part

API = 1


def register(reg):
    reg.add(part("SINK", ins=("a",), eval=lambda a: None))  # pure, no outputs: pointless
