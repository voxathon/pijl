from pijl.sim import Circuit


def settle(c: Circuit, steps: int = 10) -> None:
    for _ in range(steps):
        c.step()


def test_nand_truth_table():
    c = Circuit()
    a, b, g, out = c.add_chip("IN"), c.add_chip("IN"), c.add_chip("NAND"), c.add_chip("OUT")
    c.connect(a.outputs[0], g.inputs[0])
    c.connect(g.inputs[1], b.outputs[0])  # reversed order must work too
    c.connect(g.outputs[0], out.inputs[0])
    for va in (False, True):
        for vb in (False, True):
            a.outputs[0].state, b.outputs[0].state = va, vb
            settle(c)
            assert out.inputs[0].state == (not (va and vb))


def test_invalid_connections_rejected():
    c = Circuit()
    a, b = c.add_chip("IN"), c.add_chip("IN")
    g = c.add_chip("NOT")
    assert c.connect(a.outputs[0], b.outputs[0]) == (None, None)  # out -> out
    assert c.connect(g.inputs[0], g.outputs[0]) == (None, None)   # same chip
    assert c.wires == []


def test_rewiring_input_replaces_old_wire():
    c = Circuit()
    a, b, g = c.add_chip("IN"), c.add_chip("IN"), c.add_chip("NOT")
    first, _ = c.connect(a.outputs[0], g.inputs[0])
    second, replaced = c.connect(b.outputs[0], g.inputs[0])
    assert replaced is first and c.wires == [second]


def test_remove_chip_removes_its_wires():
    c = Circuit()
    a, g = c.add_chip("IN"), c.add_chip("NOT")
    c.connect(a.outputs[0], g.inputs[0])
    c.remove_chip(g)
    assert c.wires == [] and c.chips == [a]


def test_sr_latch_from_nands_holds_state():
    # Active-low SR latch: feedback loops must settle without blowing up.
    c = Circuit()
    s, r = c.add_chip("IN"), c.add_chip("IN")
    n1, n2 = c.add_chip("NAND"), c.add_chip("NAND")
    c.connect(s.outputs[0], n1.inputs[0])
    c.connect(r.outputs[0], n2.inputs[0])
    c.connect(n1.outputs[0], n2.inputs[1])
    c.connect(n2.outputs[0], n1.inputs[1])
    q = n1.outputs[0]

    s.outputs[0].state, r.outputs[0].state = False, True  # set
    settle(c)
    assert q.state is True
    s.outputs[0].state = True  # release: hold
    settle(c)
    assert q.state is True
    r.outputs[0].state = False  # reset
    settle(c)
    assert q.state is False
    r.outputs[0].state = True  # release: hold
    settle(c)
    assert q.state is False
