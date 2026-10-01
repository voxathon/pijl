"""World drawing without pyglet shapes: every shape is one *instance* in a numpy array.

pyglet's shapes cost a vertex list each -- allocation, several ctypes buffer writes
per property change, and a per-shape Python object tree. At tens of thousands of
parts that was most of the time spent creating, moving and recoloring things.

Here each shape kind (see sdf_shapes.py, sdf_text.py) has one InstanceBuffer per
layer: a numpy structured array with one record per shape, mirrored in one GL
buffer and drawn with a single instanced draw call. The quad's corners come from
gl_VertexID, so the only per-shape data is the record itself. Setting a property is
a numpy element write plus a dirty mark; dirty slots are uploaded once per frame.

Shapes carry both of their colors (off and on) plus a state flag, so a pin or wire
switching on/off is a one-byte write and the shader picks the color.

Every kind also has a `lift` field: lifted instances are drawn shifted by the canvas's
`offset` (a uniform). Dragging a selection lifts it once and then only changes the
offset per mouse move, instead of rewriting every shape (see Editor._begin_move).

Draw order: layers by their order; inside a layer, by slot. Freed slots are reused
lowest first (like pyglet's allocator), so "newer on top" is only roughly true.
"""

from __future__ import annotations

import ctypes
import heapq
import time
import weakref

import numpy as np
import pyglet
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram

GAP = 64  # dirty slots at most this far apart are uploaded as one run
MAX_RUNS = (
    32  # more runs than this: upload one range from the first dirty slot to the last
)

UNIFORMS = """uniform WindowBlock { mat4 projection; mat4 view; } window;
uniform vec2 lift_offset;  // added to the position of lifted instances (Canvas.offset)"""


class Kind:
    """A shape kind: its shaders and the layout of one instance. Programs are compiled
    on first use (that needs a GL context). `texture`: a function returning a texture
    to bind while drawing, if the kind needs one. `positions`: the fields whose first
    two numbers are a world position (what InstanceBuffer.shift moves)."""

    def __init__(
        self,
        name: str,
        rank: int,
        vertex: str,
        fragment: str,
        dtype: np.dtype,
        texture=None,
        positions: tuple[str, ...] = (),
    ) -> None:
        self.name, self.rank = name, rank
        self.vertex, self.fragment = vertex, fragment
        self.dtype = dtype
        self.texture = texture
        self.positions = positions
        self._program: ShaderProgram | None = None

    @property
    def program(self) -> ShaderProgram:
        if self._program is None:
            self._program = ShaderProgram(
                Shader(self.vertex, "vertex"), Shader(self.fragment, "fragment")
            )
        return self._program


class InstanceBuffer:
    """All instances of one kind in one layer. Shapes hold a slot; `f[field][slot]`
    is their data (re-read `f` on every access: growing replaces the arrays)."""

    def __init__(self, kind: Kind, capacity: int = 256) -> None:
        self.kind = kind
        self.program = kind.program
        self.dtype = kind.dtype
        self._zero = np.zeros((), kind.dtype)
        self.data = np.zeros(capacity, kind.dtype)
        self.used = np.zeros(capacity, bool)
        self.dirty = np.zeros(capacity, bool)
        self.f = {name: self.data[name] for name in kind.dtype.names}
        self.any_dirty = False
        self.realloc = True  # upload everything (new, or grown)
        self.free_slots: list[int] = []  # heap: lowest first
        self.end = 0  # slots below this were handed out at some point
        self.top = 0  # highest slot in use + 1: what gets drawn
        self._make_gl()

    def _make_gl(self) -> None:
        vbo = gl.GLuint()
        gl.glGenBuffers(1, ctypes.byref(vbo))
        self.vbo = vbo
        # One vertex array per GL context (window): the buffer is shared between
        # windows, vertex arrays aren't. Made on first draw in each (see _vao); a
        # closed window's goes with its context.
        self._vaos: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def _vao(self) -> gl.GLuint:
        """This context's vertex array over the buffer. (Growing the buffer re-specifies
        it under the same name, so these stay valid.)"""
        vao = self._vaos.get(gl.current_context)
        if vao is not None:
            return vao
        vao = self._vaos[gl.current_context] = gl.GLuint()
        gl.glGenVertexArrays(1, ctypes.byref(vao))
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        stride = self.dtype.itemsize
        for name in self.dtype.names:
            loc = gl.glGetAttribLocation(self.program.id, name.encode())
            if loc < 0:
                continue  # unused by the shader (optimized away)
            sub, offset = self.dtype.fields[name][:2]
            count = sub.shape[0] if sub.shape else 1
            if sub.base == np.float32:
                gltype, normalized = gl.GL_FLOAT, gl.GL_FALSE
            elif sub.base == np.uint8:
                gltype, normalized = gl.GL_UNSIGNED_BYTE, gl.GL_TRUE
            else:
                raise TypeError(
                    f"{self.kind.name}.{name}: unsupported field type {sub}"
                )
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(
                loc, count, gltype, normalized, stride, ctypes.c_void_p(offset)
            )
            gl.glVertexAttribDivisor(loc, 1)
        gl.glBindVertexArray(0)
        return vao

    # ---- slots ---------------------------------------------------------------

    def alloc(self) -> int:
        if self.free_slots:
            slot = heapq.heappop(self.free_slots)
        else:
            if self.end == len(self.data):
                self._grow()
            slot = self.end
            self.end += 1
        self.used[slot] = True
        self.top = max(self.top, slot + 1)
        return slot

    def alloc_many(self, n: int) -> np.ndarray:
        """n slots, as alloc() would hand them out one by one."""
        free = self.free_slots
        out = [heapq.heappop(free) for _ in range(min(n, len(free)))]
        rest = n - len(out)
        if rest:
            while self.end + rest > len(self.data):
                self._grow()
            out.extend(range(self.end, self.end + rest))
            self.end += rest
        slots = np.array(out, np.intp)
        if n:
            self.used[slots] = True
            self.top = max(self.top, int(slots.max()) + 1)
        return slots

    def free(self, slot: int) -> None:
        self.data[slot] = self._zero  # all-zero: a degenerate quad, draws nothing
        self.used[slot] = False
        self.mark(slot)
        heapq.heappush(self.free_slots, slot)
        while self.top and not self.used[self.top - 1]:
            self.top -= 1

    def free_many(self, slots: np.ndarray) -> None:
        """free() for many slots at once."""
        if not slots.size:
            return
        self.data[slots] = self._zero
        self.used[slots] = False
        self.mark_many(slots)
        self.free_slots.extend(slots.tolist())
        heapq.heapify(self.free_slots)
        in_use = np.flatnonzero(self.used[: self.top])
        self.top = int(in_use[-1]) + 1 if in_use.size else 0

    def _grow(self) -> None:
        n = len(self.data)
        for name in ("data", "used", "dirty"):
            old = getattr(self, name)
            new = np.zeros(2 * n, old.dtype)
            new[:n] = old
            setattr(self, name, new)
        self.f = {name: self.data[name] for name in self.dtype.names}
        self.realloc = True

    def mark(self, slot: int) -> None:
        self.dirty[slot] = True
        self.any_dirty = True

    def mark_many(self, slots) -> None:
        self.dirty[slots] = True
        self.any_dirty = True

    def shift(self, slots: np.ndarray, dx: float, dy: float) -> None:
        """Move these instances by (dx, dy), all at once."""
        for name in self.kind.positions:
            self.f[name][slots, :2] += (dx, dy)
        self.mark_many(slots)

    def set_lift(self, slots: np.ndarray, on: bool) -> None:
        self.f["lift"][slots] = 1.0 if on else 0.0
        self.mark_many(slots)

    # ---- GPU -----------------------------------------------------------------

    def _upload(self) -> None:
        if not (self.realloc or self.any_dirty):
            return
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        base, size = self.data.ctypes.data, self.dtype.itemsize
        if self.realloc:
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER, self.data.nbytes, base, gl.GL_DYNAMIC_DRAW
            )
        else:
            idx = np.flatnonzero(self.dirty[: self.end])
            if idx.size:
                breaks = np.flatnonzero(np.diff(idx) > GAP)
                starts = idx[np.r_[0, breaks + 1]]
                stops = idx[np.r_[breaks, idx.size - 1]] + 1
                if len(starts) > MAX_RUNS:
                    starts, stops = idx[:1], idx[-1:] + 1
                for lo, hi in zip(starts.tolist(), stops.tolist()):
                    gl.glBufferSubData(
                        gl.GL_ARRAY_BUFFER,
                        lo * size,
                        (hi - lo) * size,
                        base + lo * size,
                    )
        self.dirty[:] = False
        self.realloc = self.any_dirty = False

    def draw(self, offset: tuple[float, float] = (0.0, 0.0), now: float = 0.0) -> None:
        if not self.top:
            return
        self._upload()
        self.program.use()
        self.program["lift_offset"] = offset
        if (
            "time" in self.program.uniforms
        ):  # (only programs that animate something have it)
            self.program["time"] = now
        if self.kind.texture is not None:
            tex = self.kind.texture()
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(tex.target, tex.id)
        gl.glBindVertexArray(self._vao())
        gl.glDrawArraysInstanced(gl.GL_TRIANGLE_STRIP, 0, 4, self.top)
        gl.glBindVertexArray(0)
        self.program.stop()


class Canvas:
    """The world: instance buffers by layer, then a plain pyglet batch (`batch`) on top
    for the few things still made of pyglet shapes (wire-edit handles, the caret)."""

    def __init__(self, batch: pyglet.graphics.Batch) -> None:
        self.batch = batch
        self._buffers: dict[tuple[int, int, str], InstanceBuffer] = {}
        self.offset = (
            0.0,
            0.0,
        )  # where lifted instances are drawn, relative to their data

    def buffer(self, kind: Kind, layer) -> InstanceBuffer:
        """`layer`: a pyglet Group (its order counts) or an order number."""
        key = (getattr(layer, "order", layer), kind.rank, kind.name)
        buf = self._buffers.get(key)
        if buf is None:
            buf = self._buffers[key] = InstanceBuffer(kind)
        return buf

    def draw(self, top: bool = True) -> None:
        """`top`: the pyglet batch too (only the window something is being edited in
        shows its handles and caret)."""
        self.draw_instances()
        if top:
            self.batch.draw()

    def draw_instances(self) -> None:
        """Just the instance buffers (what the miniview draws again, at its own zoom)."""
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        now = time.monotonic() % 3600.0  # (kept small: it's a float32 in the shaders)
        for key in sorted(self._buffers):
            self._buffers[key].draw(self.offset, now)
        gl.glDisable(gl.GL_BLEND)

    def counts(self) -> dict[str, int]:
        """Instances in use per kind (for the curious / tests)."""
        out: dict[str, int] = {}
        for (_, _, name), buf in self._buffers.items():
            out[name] = out.get(name, 0) + int(buf.used[: buf.end].sum())
        return out
