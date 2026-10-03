"""Fixture parts with faces (tests/test_faces.py)."""

from pijl.logic import X
from pijl.parts import Look, Mark, PartType

API = 2


class Lamp(PartType):
    """Two marks: one follows pin `a`, the other is `a` inverted, by the face hook."""

    kind = "LAMP2"
    ins = ("a", "b")
    look = Look(
        size=(60, 80),
        titled=False,
        face=(Mark((10.0, 40.0), (30.0, 40.0), 4.0, "a"), Mark((45.0, 40.0))),
    )

    def face(self, ctx, a, b):
        return ~a


class Broken(PartType):
    kind = "BROKEN_FACE"
    ins = ("a",)
    look = Look(face=(Mark((10.0, 10.0)),))

    def face(self, ctx, a):
        raise RuntimeError("nope")


class Unknown(PartType):
    kind = "X_FACE"
    look = Look(face=(Mark((10.0, 10.0)), Mark((20.0, 10.0))))

    def face(self, ctx):
        return X, True


def register(reg):
    reg.add(Lamp())
    reg.add(Broken())
    reg.add(Unknown())
