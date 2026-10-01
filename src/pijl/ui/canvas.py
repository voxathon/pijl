"""World drawing without pyglet shapes: every shape is one *instance* in a numpy array.

pyglet's shapes cost a vertex list each -- allocation, several ctypes buffer writes
per property change, and a per-shape Python object tree. At tens of thousands of
parts that was most of the time spent creating, moving and recoloring things.

Here each shape kind (see sdf_shapes.py, sdf_text.py) has one InstanceBuffer per
layer: a numpy structured array with one record per shape, mirrored in one GL
buffer and drawn with a single instanced draw call. The quad's corners come from
gl_VertexID, so the only per-shape data is the record itself. Setting a property is
a numpy element write plus a dirty mark; dirty slots are uploaded once per frame.

Shapes carry both of their colors (off and on) plus a state byte, so a pin or wire
switching on/off is a one-byte write and the shader picks the color. The state bytes
live apart from the records, in their own array and GL buffer (`state`, the shaders'
`state` attribute): on a busy board most of them change every tick, and uploading
one byte per instance is far cheaper than the whole record around it.

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
        self.state = np.zeros(capacity, np.uint8)  # see the module doc; 0 = SHOW_OFF
        self.state_dirty = False  # upload `state` (all of it in use) next draw
        self.used = np.zeros(capacity, bool)
        self.dirty = np.zeros(capacity, bool)
        # What each slot shows the state of (see ui/sync.py): a pin slot or a wire
        # slot in the circuit, or -1. Freeing a slot forgets it. None until the first
        # show_pins / show_wires: most buffers (text, tags, overlays) never need them.
        self.pin_src: np.ndarray | None = None
        self.wire_src: np.ndarray | None = None
        self.f = {name: self.data[name] for name in kind.dtype.names}
        self.any_dirty = False
        self.realloc = True  # upload everything (new, or grown)
        self.free_slots: list[int] = []  # heap: lowest first
        self.end = 0  # slots below this were handed out at some point
        self.top = 0  # highest slot in use + 1: what gets drawn
        self.gen = 0  # bumped whenever pin_src / wire_src may have changed (see ViewSync)
        self._make_gl()

    def _make_gl(self) -> None:
        vbo, svbo = gl.GLuint(), gl.GLuint()
        gl.glGenBuffers(1, ctypes.byref(vbo))
        gl.glGenBuffers(1, ctypes.byref(svbo))
        self.vbo, self.svbo = vbo, svbo
        self.vao = self.make_vao(self.program)

    def make_vao(self, program: ShaderProgram) -> gl.GLuint:
        """A vertex array feeding this buffer's instances to `program` (its own, or an
        echo's: see Echo)."""
        vao = gl.GLuint()
        gl.glGenVertexArrays(1, ctypes.byref(vao))
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        stride = self.dtype.itemsize
        for name in self.dtype.names:
            loc = gl.glGetAttribLocation(program.id, name.encode())
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
        loc = gl.glGetAttribLocation(program.id, b"state")
        if loc >= 0:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.svbo)
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(
                loc, 1, gl.GL_UNSIGNED_BYTE, gl.GL_TRUE, 1, ctypes.c_void_p(0)
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
        if self.state[slot]:
            self.state[slot] = 0
            self.state_dirty = True
        self.used[slot] = False
        for src in (self.pin_src, self.wire_src):
            if src is not None and src[slot] >= 0:
                src[slot] = -1
                self.gen += 1
        self.mark(slot)
        heapq.heappush(self.free_slots, slot)
        while self.top and not self.used[self.top - 1]:
            self.top -= 1

    def free_many(self, slots: np.ndarray) -> None:
        """free() for many slots at once."""
        if not slots.size:
            return
        self.data[slots] = self._zero
        self.state[slots] = 0
        self.state_dirty = True
        self.used[slots] = False
        for src in (self.pin_src, self.wire_src):
            if src is not None:
                src[slots] = -1
        self.gen += 1
        self.mark_many(slots)
        self.free_slots.extend(slots.tolist())
        heapq.heapify(self.free_slots)
        in_use = np.flatnonzero(self.used[: self.top])
        self.top = int(in_use[-1]) + 1 if in_use.size else 0

    def _grow(self) -> None:
        n = len(self.data)
        for name, fill in (
            ("data", 0),
            ("state", 0),
            ("used", 0),
            ("dirty", 0),
            ("pin_src", -1),
            ("wire_src", -1),
        ):
            old = getattr(self, name)
            if old is None:
                continue
            new = np.full(2 * n, fill, old.dtype)
            new[:n] = old
            setattr(self, name, new)
        self.f = {name: self.data[name] for name in self.dtype.names}
        self.realloc = True

    def show_pins(self, slots, pins) -> None:
        """These slots show these pins' states (pin slots in the circuit)."""
        if self.pin_src is None:
            self.pin_src = np.full(len(self.data), -1, np.int32)
        self.pin_src[slots] = pins
        self.gen += 1

    def show_wires(self, slots, wires) -> None:
        """These slots show these wires' states (wire slots in the circuit)."""
        if self.wire_src is None:
            self.wire_src = np.full(len(self.data), -1, np.int32)
        self.wire_src[slots] = wires
        self.gen += 1

    def set_state(self, slots, values) -> None:
        """These instances' state bytes (SHOW_*, see sdf_shapes)."""
        self.state[slots] = values
        self.state_dirty = True

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
        if self.realloc or self.state_dirty:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.svbo)
            if self.realloc:
                gl.glBufferData(
                    gl.GL_ARRAY_BUFFER,
                    self.state.nbytes,
                    self.state.ctypes.data,
                    gl.GL_DYNAMIC_DRAW,
                )
            elif self.top:
                gl.glBufferSubData(
                    gl.GL_ARRAY_BUFFER, 0, self.top, self.state.ctypes.data
                )
            self.state_dirty = False
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
        gl.glBindVertexArray(self.vao)
        gl.glDrawArraysInstanced(gl.GL_TRIANGLE_STRIP, 0, 4, self.top)
        gl.glBindVertexArray(0)
        self.program.stop()


class Echo:
    """Another buffer's instances drawn again, in another layer, by another program
    (`kind`: the same instance layout, other shaders). Selection outlines and wire
    halos are echoes: the shapes carry a `sel` flag, and the echo's shaders draw only
    the selected ones -- so selecting is a flag write, not new shapes, and the
    highlight moves, lifts and goes away with its shape by itself."""

    def __init__(self, kind: Kind, source: InstanceBuffer) -> None:
        if kind.dtype != source.dtype:
            raise TypeError(f"{kind.name} can't echo {source.kind.name}: other layout")
        self.kind, self.source = kind, source
        self.program = kind.program
        self.vao = source.make_vao(self.program)

    def draw(self, offset: tuple[float, float] = (0.0, 0.0), now: float = 0.0) -> None:
        src = self.source
        if not src.top:
            return
        src._upload()  # (it may come after us in the draw order)
        self.program.use()
        self.program["lift_offset"] = offset
        if "time" in self.program.uniforms:
            self.program["time"] = now
        gl.glBindVertexArray(self.vao)
        gl.glDrawArraysInstanced(gl.GL_TRIANGLE_STRIP, 0, 4, src.top)
        gl.glBindVertexArray(0)
        self.program.stop()


class Canvas:
    """The world: instance buffers by layer, then a plain pyglet batch (`batch`) on top
    for the few things still made of pyglet shapes (wire-edit handles, the caret)."""

    def __init__(self, batch: pyglet.graphics.Batch) -> None:
        self.batch = batch
        self._buffers: dict[tuple[int, int, str], InstanceBuffer] = {}
        self._echoes: dict[tuple[int, int, str], Echo] = {}
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

    def buffers(self):
        return self._buffers.values()

    def echo(self, kind: Kind, layer, source_kind: Kind, source_layer) -> None:
        """Draw source_kind's instances in source_layer again, in `layer`, with `kind`'s
        shaders (see Echo). Once per canvas; asking again does nothing."""
        key = (getattr(layer, "order", layer), kind.rank, kind.name)
        if key not in self._echoes:
            self._echoes[key] = Echo(kind, self.buffer(source_kind, source_layer))

    def draw(self) -> None:
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        now = time.monotonic() % 3600.0  # (kept small: it's a float32 in the shaders)
        drawn = {**self._buffers, **self._echoes}
        for key in sorted(drawn):
            drawn[key].draw(self.offset, now)
        gl.glDisable(gl.GL_BLEND)
        self.batch.draw()

    def delete(self) -> None:
        """Free the GL buffers (a canvas that's going away: see inside.py)."""
        for buf in self._buffers.values():
            gl.glDeleteBuffers(2, (gl.GLuint * 2)(buf.vbo, buf.svbo))
            gl.glDeleteVertexArrays(1, ctypes.byref(buf.vao))
        for echo in self._echoes.values():
            gl.glDeleteVertexArrays(1, ctypes.byref(echo.vao))
        self._buffers.clear()
        self._echoes.clear()

    def counts(self) -> dict[str, int]:
        """Instances in use per kind (for the curious / tests)."""
        out: dict[str, int] = {}
        for (_, _, name), buf in self._buffers.items():
            out[name] = out.get(name, 0) + int(buf.used[: buf.end].sum())
        return out
