from pijl.parts import PartType

API = 1


class FakeIn(PartType):
    kind = "FAKEIN"
    outs = ("out",)
    port = "in"


def register(reg):
    reg.add(FakeIn)
