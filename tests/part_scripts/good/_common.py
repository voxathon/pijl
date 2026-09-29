"""A private helper: never loaded as a part script itself."""


def invert(x):
    return ~x


def register(reg):
    raise AssertionError("private modules must not be registered")
