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
visible hitch with a few hundred chips.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyglet
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from scipy.ndimage import distance_transform_edt

FONT_NAME = "Consolas"
EM_PT = 48               # rasterization size in points
EM_PX = EM_PT * 96 / 72  # ... in pixels (64)
SPREAD = 8               # SDF range in atlas pixels on each side of the outline
SUPERSAMPLE = 4          # glyphs are rasterized this much bigger, then the SDF is averaged down;
                         # without it the SDF encodes the bitmap's pixel stairs and edges look jagged
CHARS = "".join(chr(c) for c in range(33, 127))  # printable ASCII except space
ATLAS_W = 1024
_FORMAT_VERSION = 1  # bump when the cache layout or SDF math changes

_VERTEX = """#version 150 core
in vec2 position;
in vec2 tex_coords;
in vec4 colors;
out vec2 uv;
out vec4 color;
uniform WindowBlock { mat4 projection; mat4 view; } window;
void main() {
    gl_Position = window.projection * window.view * vec4(position, 0.0, 1.0);
    uv = tex_coords;
    color = colors;
}
"""

_FRAGMENT = """#version 150 core
in vec2 uv;
in vec4 color;
out vec4 final_color;
uniform sampler2D sdf;
const float SPREAD = %(spread)s;
void main() {
    float d = texture(sdf, uv).r;       // 0.5 = on the outline, >0.5 = inside
    // How many atlas texels one screen pixel covers. Taken from the UV derivatives
    // rather than fwidth(d): at heavy minification d itself turns to noise.
    vec2 texel = uv * vec2(textureSize(sdf, 0));
    float texels_per_px = max(length(dFdx(texel)), length(dFdy(texel)));
    float w = texels_per_px / (2.0 * SPREAD);   // SDF value change per screen pixel
    float a = smoothstep(0.5 - 0.7 * w, 0.5 + 0.7 * w, d);
    // Fade out when glyphs get too small to read (em under ~5-10 screen px).
    a *= clamp((0.8 - w) / 0.4, 0.0, 1.0);
    final_color = vec4(color.rgb, color.a * a);
    if (final_color.a < 0.01) discard;
}
""" % {"spread": float(SPREAD)}


@dataclass
class _Glyph:
    left: float    # quad extents relative to the pen position on the baseline, source px
    bottom: float
    right: float
    top: float
    uv: tuple[float, float, float, float]  # u0, v0, u1, v1


class _Atlas:
    def __init__(self) -> None:
        atlas, table, self.advance, self.cap_height = _load_or_build()
        h = atlas.shape[0]
        self.glyphs = {ch: _Glyph(*row[:4], uv=tuple(row[4:])) for ch, row in zip(CHARS, table.tolist())}
        rgba = np.repeat(atlas[:, :, None], 4, axis=2)
        self.texture = pyglet.image.ImageData(ATLAS_W, h, "RGBA", rgba.tobytes()).get_texture()
        self.program = ShaderProgram(Shader(_VERTEX, "vertex"), Shader(_FRAGMENT, "fragment"))


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
        np.savez_compressed(path, atlas=atlas, table=table, advance=advance, cap_height=cap_height)
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
        alpha = alpha.reshape(img.height, img.width, 4)[:, :, 3]  # rows bottom-up
        sdf = _to_sdf(alpha)
        left, bottom = g.vertices[0] / SUPERSAMPLE - SPREAD, g.vertices[1] / SUPERSAMPLE - SPREAD
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
        atlas[py:py + h, px:px + w] = sdf
        table[i] = (*box, px / ATLAS_W, py / atlas_h, (px + w) / ATLAS_W, (py + h) / atlas_h)
    return atlas, table, advance, float(cap_height)


def _to_sdf(alpha: np.ndarray) -> np.ndarray:
    """Supersampled coverage bitmap -> padded uint8 SDF at atlas resolution (128 = outline)."""
    ss = SUPERSAMPLE
    inside = np.pad(alpha > 127, SPREAD * ss)
    h, w = inside.shape
    inside = np.pad(inside, ((0, -h % ss), (0, -w % ss)))  # make divisible by ss
    dist = distance_transform_edt(inside) - distance_transform_edt(~inside)  # + inside, - outside
    h, w = inside.shape
    dist = dist.reshape(h // ss, ss, w // ss, ss).mean(axis=(1, 3)) / ss  # average down, atlas px
    return np.clip(128 + dist * (127 / SPREAD), 0, 255).astype(np.uint8)


_atlas: _Atlas | None = None


def _get_atlas() -> _Atlas:
    """Built lazily: needs a GL context, i.e. a window must exist."""
    global _atlas
    if _atlas is None:
        _atlas = _Atlas()
    return _atlas


class _SDFGroup(pyglet.graphics.Group):
    def __init__(self, atlas: _Atlas, order: int) -> None:
        super().__init__(order=order)
        self.atlas = atlas

    def set_state(self) -> None:
        self.atlas.program.use()
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(self.atlas.texture.target, self.atlas.texture.id)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def unset_state(self) -> None:
        gl.glDisable(gl.GL_BLEND)
        self.atlas.program.stop()

    __eq__ = object.__eq__
    __hash__ = object.__hash__


class SDFText:
    """Factory for world-space labels that all share one atlas and one draw call."""

    def __init__(self, batch: pyglet.graphics.Batch, order: int) -> None:
        self.batch = batch
        self.atlas = _get_atlas()
        self.group = _SDFGroup(self.atlas, order)

    def label(self, text: str, x: float, y: float, size: float,
              color: tuple[int, int, int, int], anchor_x: str = "center") -> SDFLabel:
        return SDFLabel(self, text, x, y, size, color, anchor_x)


class SDFLabel:
    """One line of text anchored at (x, y): vertically centered on the capitals,
    horizontally per `anchor_x` ("left", "center", "right"). `size` is in points
    at zoom 1, like pyglet's font_size. Characters outside the atlas are dropped."""

    def __init__(self, owner: SDFText, text: str, x: float, y: float, size: float,
                 color: tuple[int, int, int, int], anchor_x: str = "center") -> None:
        self.owner = owner
        self.atlas = owner.atlas
        self.scale = size * 96 / 72 / EM_PX  # atlas px -> world units
        self.anchor_x = anchor_x
        self.x, self.y = x, y
        self._color = tuple(color)
        self.vlist = None
        self.text = ""
        self.set_text(text)

    def set_text(self, text: str) -> None:
        self.text = "".join(c for c in text if c == " " or c in self.atlas.glyphs)
        if self.vlist is not None:
            self.vlist.delete()
            self.vlist = None
        chars = [c for c in self.text if c != " "]
        n = len(chars)
        if n:
            self.vlist = self.atlas.program.vertex_list_indexed(
                n * 4, gl.GL_TRIANGLES,
                [i * 4 + k for i in range(n) for k in (0, 1, 2, 0, 2, 3)],
                self.owner.batch, self.owner.group,
                position=("f", [0.0] * n * 8),
                tex_coords=("f", [c for ch in chars for c in _quad_uv(self.atlas.glyphs[ch].uv)]),
                colors=("Bn", self._color * n * 4),
            )
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
        return {"left": self.x, "center": self.x - self.width / 2, "right": self.x - self.width}[self.anchor_x]

    def move_to(self, x: float, y: float) -> None:
        self.x, self.y = x, y
        if self.vlist is None:
            return
        a, s = self.atlas, self.scale
        pen = self._left()
        base = y - a.cap_height * s / 2  # center capitals vertically
        pos: list[float] = []
        for ch in self.text:
            g = a.glyphs.get(ch)
            if g is not None:
                l, b, r, t = pen + g.left * s, base + g.bottom * s, pen + g.right * s, base + g.top * s
                pos += (l, b, r, b, r, t, l, t)
            pen += a.advance * s
        self.vlist.position[:] = pos

    @property
    def opacity(self) -> int:
        return self._color[3]

    @opacity.setter
    def opacity(self, value: int) -> None:
        self._color = (*self._color[:3], value)
        if self.vlist is not None:
            self.vlist.colors[:] = self._color * (len(self.vlist.colors) // 4)

    def delete(self) -> None:
        if self.vlist is not None:
            self.vlist.delete()
            self.vlist = None


def _quad_uv(uv: tuple[float, float, float, float]) -> tuple[float, ...]:
    u0, v0, u1, v1 = uv
    return (u0, v0, u1, v0, u1, v1, u0, v1)
