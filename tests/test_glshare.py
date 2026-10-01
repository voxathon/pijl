"""glshare.py patches pyglet's vertex domains: fail loudly if what it relies on moves."""

import inspect

from pyglet.graphics import vertexdomain

from pijl.ui import glshare


def test_domain_init_is_what_the_patch_repeats():
    init = inspect.getsource(vertexdomain.VertexDomain.__init__)
    assert "self.vao = vertexarray.VertexArray()" in init
    for step in ("buffer.bind()", "attribute.enable()", "attribute.set_pointer(buffer.ptr)"):
        assert step in init
    assert "attribute.set_divisor()" in init
    indexed = inspect.getsource(vertexdomain.IndexedVertexDomain.__init__)
    assert "self.index_buffer.bind_to_index_buffer()" in indexed


def test_domains_only_reach_their_vertex_array_through_vao():
    src = inspect.getsource(vertexdomain)
    assert "vertexarray.VertexArray()" in src
    assert src.count("VertexArray()") == 1  # (made in one place: VertexDomain.__init__)


def test_install_makes_vao_a_property():
    glshare.install()
    assert isinstance(vertexdomain.VertexDomain.__dict__["vao"], property)
