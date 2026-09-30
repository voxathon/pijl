from pijl.parts import part

API = 1


def register(reg):
    reg.add(part("HALF", outs=("out",), eval=lambda: True))
    reg.add(
        part("HALF", outs=("out",), eval=lambda: True)
    )  # duplicate: the whole script is skipped
