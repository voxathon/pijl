"""Let any pyglet batch draw in any of our windows.

pyglet's windows share GL objects (buffers, textures, shader programs) through its
shadow window -- except vertex arrays, which GL never shares. And a pyglet vertex domain
makes its one vertex array in whatever context is current when it's created. So a shape
or label made while another window's context happened to be current (after it drew, say)
draws garbage or crashes in its own window.

The fix: a domain keeps one vertex array per context, each set up the same way (its
attribute buffers, its index buffer) the first time it's drawn there. Growing a buffer
re-specifies it under the same name, so those stay valid; a closed window's vertex
arrays go with its context.

This reaches into pyglet's internals (2.1's graphics/vertexdomain.py); tests/test_glshare.py
fails if those change shape. canvas.py does the same for its own raw-GL buffers.
"""

from __future__ import annotations

import weakref

from pyglet import gl
from pyglet.graphics import vertexarray, vertexdomain

_patched = False


def _vao(self: vertexdomain.VertexDomain) -> vertexarray.VertexArray:
    vaos = self.__dict__["_vaos"]
    context = gl.current_context
    vao = vaos.get(context)
    if vao is None:  # first use in this window: set it up like __init__ did
        vao = vaos[context] = vertexarray.VertexArray()
        vao.bind()
        for buffer, attribute in self.buffer_attributes:
            buffer.bind()
            attribute.enable()
            attribute.set_pointer(buffer.ptr)
            if attribute.instance:
                attribute.set_divisor()
        index_buffer = getattr(self, "index_buffer", None)
        if index_buffer is not None:
            index_buffer.bind_to_index_buffer()
        vao.unbind()
    return vao


def _set_vao(self: vertexdomain.VertexDomain, vao: vertexarray.VertexArray) -> None:
    """(__init__'s `self.vao = VertexArray()`: that one is the current context's.)"""
    vaos = self.__dict__.setdefault("_vaos", weakref.WeakKeyDictionary())
    vaos[vao._context] = vao  # noqa: SLF001


def install() -> None:
    """Patch pyglet (once; call before making windows)."""
    global _patched
    if _patched:
        return
    vertexdomain.VertexDomain.vao = property(_vao, _set_vao)
    _patched = True
