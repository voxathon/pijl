"""SPLIT: a bus into groups of its lanes, or (flipped) groups of lanes into a bus.

One side is the bus, `width` lanes. The other side has a pin per group of lanes,
as the pattern says, from lane 0 up: "4,4" is lanes 0-3 and 4-7, "3,5" lanes 0-2
and 3-7, and "1x8" is eight pins of one lane each ("size x how many"). Blank: a
pin per lane. Lanes the pattern leaves over make one more pin; a pattern longer
than the bus stops at its last lane. The pins are named by their lanes ("0-3").

It's wiring, not a gate: each lane is one net straight through the part, either
way, with no delay. Flip: the bus is the output (a merger) instead of the input.
Right-click it to set the width, the pattern or the flip.
"""

from pijl.parts import MAX_WIDTH, Number, PartType, Text, Toggle

API = 2


class Pattern(Text):
    """A Text setting that has to read as a pattern (see sizes)."""

    def parse(self, value) -> str:
        text = super().parse(value).replace(" ", "")
        sizes(text, MAX_WIDTH)  # (ValueError if it isn't one)
        return text


def sizes(pattern: str, width: int) -> list[int]:
    """The pin sizes the pattern gives a bus `width` lanes wide (see the docstring)."""
    out: list[int] = []
    for item in filter(None, pattern.replace(" ", "").lower().split(",")):
        size, _, count = item.partition("x")
        try:
            size, count = int(size), int(count or 1)
        except ValueError:
            raise ValueError(f"{item!r}: write sizes like 4,4 or 1x8") from None
        if size < 1 or count < 1:
            raise ValueError(f"{item!r}: sizes and counts start at 1")
        out += [size] * min(count, MAX_WIDTH)
    kept, total = [], 0
    for size in out:  # (cut to the bus; what's left over is one more pin)
        if total >= width:
            break
        kept.append(min(size, width - total))
        total += kept[-1]
    if not out:
        kept, total = [1] * width, width
    if total < width:
        kept.append(width - total)
    return kept


class Split(PartType):
    kind = "SPLIT"
    category = "WIRING"
    settings = {
        "width": Number(8, 1, MAX_WIDTH, hint="lanes in the bus"),
        "pattern": Pattern(
            "", max_len=48, hint="pin sizes from lane 0: 4,4  3,5  1x8 (blank: one per lane)"
        ),
        "flip": Toggle(label="Flip (bus out: a merger)"),
    }

    def layout(self, props):
        width = props.get("width", 8)
        fans, widths, joins, lane = [], {"bus": width}, [], 0
        for size in sizes(props.get("pattern", ""), width):
            name = str(lane) if size == 1 else f"{lane}-{lane + size - 1}"
            fans.append(name)
            widths[name] = size
            joins.append((f"bus[{lane}:{lane + size}]", name))
            lane += size
        bus, fans = ("bus",), tuple(fans)
        ins, outs = (fans, bus) if props.get("flip") else (bus, fans)
        return {"ins": ins, "outs": outs, "widths": widths, "joins": tuple(joins)}


# (the class attributes: an instance with the default settings, for the part picker)
Split.ins = ("bus",)
Split.outs = tuple(str(i) for i in range(8))


def register(reg):
    reg.add(Split)
