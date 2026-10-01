"""World-space text that is crisp at every zoom and costs nothing to zoom.

How: each glyph is rasterized ONCE, big (64px em), and converted into a
signed distance field (SDF): every texel stores "how far am I from the glyph
outline" instead of "am I ink". Distances interpolate smoothly, so when the
GPU scales a glyph quad up or down, the fragment shader can threshold the
interpolated distance at 0.5 and get a sharp edge, anti-aliased to exactly
one screen pixel via fwidth(). Zooming is just the camera matrix; no text
is ever re-laid out.

Consolas is monospace, so layout is "advance the pen by a fixed amount".

Why not pyglet.text.Label in world space? It rasterizes at one size, so the
camera stretches a small bitmap into mush. Re-rasterizing labels per zoom
level works but costs ~0.27 ms per label per zoom step (measured) -- a
visible hitch with a few hundred parts.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyglet
from scipy.ndimage import distance_transform_edt

from .canvas import UNIFORMS, Canvas, Kind

FONT_NAME = "Consolas"
EM_PT = 48  # rasterization size in points
EM_PX = EM_PT * 96 / 72  # ... in pixels (64)
SPREAD = 8  # SDF range in atlas pixels on each side of the outline
SUPERSAMPLE = (
    4  # glyphs are rasterized this much bigger, then the SDF is averaged down;
)
# without it the SDF encodes the bitmap's pixel stairs and edges look jagged
CHARS = "".join(chr(c) for c in range(33, 127))  # printable ASCII except space
ATLAS_W = 1024
_FORMAT_VERSION = 2  # bump when the cache layout or SDF math changes

_VERTEX = f"""#version 150 core
in vec4 rect;   // x, y, width, height of the glyph's quad
in vec4 uv;     // u0, v0, u1, v1 in the atlas
in vec4 color;
in float lift;
out vec2 tex;
flat out vec4 c;
{UNIFORMS}
void main() {{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    tex = mix(uv.xy, uv.zw, corner);
    c = vec4(color.rgb, color.a * top_fade);
    gl_Position = window.projection * window.view * vec4(rect.xy + lift * lift_offset + corner * rect.zw, 0.0, 1.0);
}}
"""

_FRAGMENT = """#version 150 core
in vec2 tex;
flat in vec4 c;
out vec4 final_color;
uniform sampler2D sdf;
const float SPREAD = %(spread)s;
void main() {
    float d = texture(sdf, tex).r;      // 0.5 = on the outline, >0.5 = inside
    // How many atlas texels one screen pixel covers. Taken from the UV derivatives
    // rather than fwidth(d): at heavy minification d itself turns to noise.
    vec2 texel = tex * vec2(textureSize(sdf, 0));
    float texels_per_px = max(length(dFdx(texel)), length(dFdy(texel)));
    float w = texels_per_px / (2.0 * SPREAD);   // SDF value change per screen pixel
    float a = smoothstep(0.5 - 0.7 * w, 0.5 + 0.7 * w, d);
    // Fade out when glyphs get too small to read (em under ~5-10 screen px).
    a *= clamp((0.8 - w) / 0.4, 0.0, 1.0);
    final_color = vec4(c.rgb, c.a * a);
    if (final_color.a < 0.01) discard;
}
""" % {"spread": float(SPREAD)}

GLYPH = Kind(
    "glyph",
    0,
    _VERTEX,
    _FRAGMENT,
    np.dtype([("rect", "f4", 4), ("uv", "f4", 4), ("color", "u1", 4), ("lift", "f4")]),
    texture=lambda: _get_atlas().texture,
    positions=("rect",),
)


@dataclass
class _Glyph:
    left: float  # quad extents relative to the pen position on the baseline, source px
    bottom: float
    right: float
    top: float
    uv: tuple[float, float, float, float]  # u0, v0, u1, v1


class _Atlas:
    def __init__(self) -> None:
        atlas, table, self.advance, self.cap_height = _load_or_build()
        h = atlas.shape[0]
        self.glyphs = {
            ch: _Glyph(*row[:4], uv=tuple(row[4:]))
            for ch, row in zip(CHARS, table.tolist())
        }
        rgba = np.repeat(atlas[:, :, None], 4, axis=2)
        self.texture = pyglet.image.ImageData(
            ATLAS_W, h, "RGBA", rgba.tobytes()
        ).get_texture()
        # per character: its quad relative to the pen (source px) and its atlas uv, for numpy layout
        self.boxes = {
            ch: (g.left, g.bottom, g.right - g.left, g.top - g.bottom)
            for ch, g in self.glyphs.items()
        }


def _cache_path() -> Path:
    key = repr((FONT_NAME, EM_PT, SPREAD, SUPERSAMPLE, CHARS, ATLAS_W, _FORMAT_VERSION))
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache")
    return base / "pijl" / f"sdf_atlas_{digest}.npz"


def _load_or_build() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Building takes ~2 s (supersampled distance transforms), so cache it on disk.
    The cache file name hashes every parameter, so changing one rebuilds."""
    path = _cache_path()
    try:
        with np.load(path) as f:
            return f["atlas"], f["table"], float(f["advance"]), float(f["cap_height"])
    except (OSError, KeyError, ValueError):
        pass
    atlas, table, advance, cap_height = _build()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, atlas=atlas, table=table, advance=advance, cap_height=cap_height
        )
    except OSError:
        pass  # no cache, just slower next launch
    return atlas, table, advance, cap_height


def _build() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Returns (atlas uint8 [h, ATLAS_W], table float [len(CHARS), 8], advance, cap_height).

    Table rows: left, bottom, right, top (atlas px, relative to pen on baseline), u0, v0, u1, v1.
    """
    font = pyglet.font.load(FONT_NAME, EM_PT * SUPERSAMPLE)
    advance = 0.0
    cells: list[tuple[np.ndarray, tuple[float, float, float, float]]] = []

    for ch in CHARS:
        g = font.get_glyphs(ch)[0][0]
        advance = g.advance / SUPERSAMPLE  # monospace: identical for all
        img = g.get_image_data()
        alpha = np.frombuffer(img.get_data("RGBA", img.width * 4), np.uint8)
        alpha = alpha.reshape(img.height, img.width, 4)[:, :, 3]
        # Rows come as stored in the texture: bottom-up, except where the font renderer
        # stored them top-down and flipped the glyph's tex_coords instead (FreeType, so
        # Linux). Bottom-left vertex's v above the top-left's: flipped.
        if g.tex_coords[1] > g.tex_coords[10]:
            alpha = alpha[::-1]  # now bottom-up
        sdf = _to_sdf(alpha)
        left, bottom = (
            g.vertices[0] / SUPERSAMPLE - SPREAD,
            g.vertices[1] / SUPERSAMPLE - SPREAD,
        )
        cells.append((sdf, (left, bottom, left + sdf.shape[1], bottom + sdf.shape[0])))

    # Cap height (top of 'H' ink above the baseline) for vertical centering.
    h_sdf, h_box = cells[CHARS.index("H")]
    rows = np.nonzero((h_sdf >= 128).any(axis=1))[0]
    cap_height = rows.max() + 1 + h_box[1]

    # Shelf-pack cells into the atlas.
    x = y = shelf_h = 0
    placements = []
    for sdf, _ in cells:
        h, w = sdf.shape
        if x + w > ATLAS_W:
            x, y, shelf_h = 0, y + shelf_h + 1, 0
        placements.append((x, y))
        x += w + 1
        shelf_h = max(shelf_h, h)
    atlas_h = 1 << int(np.ceil(np.log2(y + shelf_h + 1)))
    atlas = np.zeros((atlas_h, ATLAS_W), np.uint8)
    table = np.zeros((len(CHARS), 8), np.float64)
    for i, ((sdf, box), (px, py)) in enumerate(zip(cells, placements)):
        h, w = sdf.shape
        atlas[py : py + h, px : px + w] = sdf
        table[i] = (
            *box,
            px / ATLAS_W,
            py / atlas_h,
            (px + w) / ATLAS_W,
            (py + h) / atlas_h,
        )
    return atlas, table, advance, float(cap_height)


def _to_sdf(alpha: np.ndarray) -> np.ndarray:
    """Supersampled coverage bitmap -> padded uint8 SDF at atlas resolution (128 = outline)."""
    ss = SUPERSAMPLE
    inside = np.pad(alpha > 127, SPREAD * ss)
    h, w = inside.shape
    inside = np.pad(inside, ((0, -h % ss), (0, -w % ss)))  # make divisible by ss
    dist = distance_transform_edt(inside) - distance_transform_edt(
        ~inside
    )  # + inside, - outside
    h, w = inside.shape
    dist = (
        dist.reshape(h // ss, ss, w // ss, ss).mean(axis=(1, 3)) / ss
    )  # average down, atlas px
    return np.clip(128 + dist * (127 / SPREAD), 0, 255).astype(np.uint8)


_atlas: _Atlas | None = None


def _get_atlas() -> _Atlas:
    """Built lazily: needs a GL context, i.e. a window must exist."""
    global _atlas
    if _atlas is None:
        _atlas = _Atlas()
    return _atlas


class SDFText:
    """Factory for world-space labels that all share one atlas and one instance buffer."""

    def __init__(self, canvas: Canvas, layer) -> None:
        self.atlas = _get_atlas()
        self.buf = canvas.buffer(GLYPH, layer)

    def measure(self, text: str, size: float) -> float:
        """Width `text` would have at `size` (monospace: just the character count)."""
        return len(text) * self.atlas.advance * size * 96 / 72 / EM_PX

    def width(self, text: str, size: float) -> float:
        """A label's width showing `text` at `size` (SDFLabel.width, before making it)."""
        scale = size * 96 / 72 / EM_PX
        return len(_layout(self.atlas, text, scale)[0]) * self.atlas.advance * scale

    def cap_height(self, size: float) -> float:
        """A label's cap_height at `size`, before making it."""
        return self.atlas.cap_height * (size * 96 / 72 / EM_PX)

    def label(
        self,
        text: str,
        x: float,
        y: float,
        size: float,
        color: tuple[int, int, int, int],
        anchor_x: str = "center",
    ) -> SDFLabel:
        return SDFLabel(self, text, x, y, size, color, anchor_x)

    def labels(self, specs: list[tuple]) -> list[SDFLabel]:
        """label() for each (text, x, y, size, color, anchor_x), all at once: the glyphs
        get the slots they would have got one label at a time, written per field."""
        slots, counts = self.place(specs)
        out, k = [], 0
        for spec, n in zip(specs, counts.tolist()):
            out.append(self.wrap(spec, slots[k : k + n] if n else NO_SLOTS))
            k += n
        return out

    def place(self, specs: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
        """The glyphs of each (text, x, y, size, color, anchor_x), all at once, with no
        label object: (their slots, how many each spec got -- in order, so spec i's
        are the next counts[i] slots). As labels() would place them: the glyphs get
        the slots they would have got one label at a time, written per field."""
        atlas = self.atlas
        rels, uvs, counts, offsets, colors = [], [], [], [], []
        for text, x, y, size, color, anchor_x in specs:
            scale = size * 96 / 72 / EM_PX
            shown, rel, uv = _layout(atlas, text, scale)
            counts.append(len(uv))
            if len(uv):
                rels.append(rel)
                uvs.append(uv)
                offsets.append((_left(x, len(shown) * atlas.advance * scale, anchor_x), y, 0.0, 0.0))
                colors.append(color)
        counts = np.array(counts, np.intp)
        total = int(counts.sum())
        if not total:
            return NO_SLOTS, counts
        counts_some = counts[counts > 0]
        buf = self.buf
        slots = buf.alloc_many(total)
        f = buf.f
        f["uv"][slots] = np.concatenate(uvs)
        f["color"][slots] = np.repeat(np.array(colors, np.uint8), counts_some, axis=0)
        f["lift"][slots] = 0.0
        # (float64, like move_to's rel + (left, y, 0, 0), then stored as float32)
        f["rect"][slots] = np.concatenate(rels).astype(np.float64) + np.repeat(
            np.array(offsets), counts_some, axis=0
        )
        buf.mark_many(slots)
        return slots, counts

    def wrap(self, spec: tuple, slots: np.ndarray) -> SDFLabel:
        """The label for one of place()'s specs and its glyphs."""
        text, x, y, size, color, anchor_x = spec
        label = SDFLabel.__new__(SDFLabel)
        label._init(self, x, y, size, color, anchor_x)
        label.text, label._rel, _ = _layout(self.atlas, text, label.scale)
        label.slots = slots
        return label

    def place_at(self, slots: np.ndarray, text: str, x: float, y: float, size: float, anchor_x: str) -> None:
        """Move glyphs placed for `text` (see place) to anchor (x, y): SDFLabel.move_to
        without the label."""
        scale = size * 96 / 72 / EM_PX
        shown, rel, _ = _layout(self.atlas, text, scale)
        left = _left(x, len(shown) * self.atlas.advance * scale, anchor_x)
        self.buf.f["rect"][slots] = rel + (left, y, 0.0, 0.0)
        self.buf.mark_many(slots)


def _left(x: float, width: float, anchor_x: str) -> float:
    """Where text `width` wide starts, anchored at x."""
    if anchor_x == "left":
        return x
    return x - width / 2 if anchor_x == "center" else x - width


_layouts: dict[tuple[str, float], tuple[str, np.ndarray, np.ndarray]] = {}


def _layout(
    atlas: _Atlas, text: str, scale: float
) -> tuple[str, np.ndarray, np.ndarray]:
    """(the text as shown, glyph quads relative to its anchor, glyph uvs), cached: a
    board shows the same few titles thousands of times. (Callers mustn't modify them.)"""
    key = (text, scale)
    hit = _layouts.get(key)
    if hit is None:
        shown = "".join(c for c in text if c == " " or c in atlas.glyphs)
        visible = [(i, c) for i, c in enumerate(shown) if c != " "]
        boxes = np.array([atlas.boxes[c] for _, c in visible], np.float32).reshape(
            -1, 4
        )
        pens = np.array([i for i, _ in visible], np.float32)
        rel = np.column_stack(
            (
                pens * atlas.advance + boxes[:, 0],
                boxes[:, 1] - atlas.cap_height / 2,
                boxes[:, 2],
                boxes[:, 3],
            )
        ).astype(np.float32) * np.float32(scale)
        uv = np.array([atlas.glyphs[c].uv for _, c in visible], np.float32).reshape(
            -1, 4
        )
        if len(_layouts) > 4096:
            _layouts.clear()
        hit = _layouts[key] = (shown, rel, uv)
    return hit


_colors: dict[tuple, tuple] = {}  # one tuple per distinct label color (see SDFLabel._init)
NO_SLOTS = np.empty(0, np.intp)  # what a label without glyphs holds (shared, never written)
NO_SLOTS.flags.writeable = False


class SDFLabel:
    """One line of text anchored at (x, y): vertically centered on the capitals,
    horizontally per `anchor_x` ("left", "center", "right"). `size` is in points
    at zoom 1, like pyglet's font_size. Characters outside the atlas are dropped.
    Each visible character is one glyph instance. (Slotted: a big board has one or
    two per part.)"""

    __slots__ = (
        "owner",
        "atlas",
        "buf",
        "scale",
        "anchor_x",
        "x",
        "y",
        "_color",
        "_lift",
        "slots",
        "_rel",
        "text",
    )

    def __init__(
        self,
        owner: SDFText,
        text: str,
        x: float,
        y: float,
        size: float,
        color: tuple[int, int, int, int],
        anchor_x: str = "center",
    ) -> None:
        self._init(owner, x, y, size, color, anchor_x)
        self.set_text(text)

    def _init(
        self,
        owner: SDFText,
        x: float,
        y: float,
        size: float,
        color: tuple[int, int, int, int],
        anchor_x: str,
    ) -> None:
        """Everything but the text (see set_text, SDFText.labels)."""
        self.owner = owner
        self.atlas = owner.atlas
        self.buf = owner.buf
        self.scale = size * 96 / 72 / EM_PX  # atlas px -> world units
        self.anchor_x = anchor_x
        self.x, self.y = x, y
        color = tuple(color)
        self._color = _colors.setdefault(color, color)
        self._lift = 0.0
        self.slots = NO_SLOTS
        # per glyph: its quad relative to (left edge, capitals' center), world units; moving is one add
        self._rel = np.empty((0, 4), np.float32)
        self.text = ""

    def set_text(self, text: str) -> None:
        self.text, self._rel, uv = _layout(self.atlas, text, self.scale)
        self._free()
        if len(uv):
            buf = self.buf
            self.slots = buf.alloc_many(len(uv))
            buf.f["uv"][self.slots] = uv
            buf.f["color"][self.slots] = self._color
            buf.f["lift"][self.slots] = self._lift
            buf.mark_many(self.slots)
        self.move_to(self.x, self.y)

    @property
    def width(self) -> float:
        return len(self.text) * self.atlas.advance * self.scale

    @property
    def cap_height(self) -> float:
        return self.atlas.cap_height * self.scale

    def caret_x(self, index: int) -> float:
        """World x of the gap before character `index` (monospace makes this trivial)."""
        return self._left() + index * self.atlas.advance * self.scale

    def _left(self) -> float:
        return _left(self.x, self.width, self.anchor_x)

    def move_to(self, x: float, y: float) -> None:
        self.x, self.y = x, y
        if not self.slots.size:
            return
        self.buf.f["rect"][self.slots] = self._rel + (self._left(), y, 0.0, 0.0)
        self.buf.mark_many(self.slots)

    @property
    def lifted(self) -> bool:
        return bool(self._lift)

    @lifted.setter
    def lifted(self, on: bool) -> None:
        """Drawn shifted by the canvas's offset (see canvas.py)."""
        self._lift = 1.0 if on else 0.0
        if self.slots.size:
            self.buf.f["lift"][self.slots] = self._lift
            self.buf.mark_many(self.slots)

    @property
    def opacity(self) -> int:
        return self._color[3]

    @opacity.setter
    def opacity(self, value: int) -> None:
        color = (*self._color[:3], value)
        self._color = _colors.setdefault(color, color)
        if self.slots.size:
            self.buf.f["color"][self.slots, 3] = value
            self.buf.mark_many(self.slots)

    def _free(self) -> None:
        for slot in self.slots.tolist():
            self.buf.free(slot)
        self.slots = NO_SLOTS

    def delete(self) -> None:
        self._free()
