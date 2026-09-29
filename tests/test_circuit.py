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
    assert c.connect(a.outputs[0], b.outputs[0]) == (None, [])  # out -> out
    assert c.connect(g.inputs[0], g.outputs[0]) == (None, [])   # same chip
    assert c.wires == []


def test_pin_to_pin_wires_are_normalized():
    c = Circuit()
    a, g = c.add_chip("IN"), c.add_chip("NOT")
    w, _ = c.connect(g.inputs[0], a.outputs[0])
    assert w.src is a.outputs[0] and w.dst is g.inputs[0]


def test_rewiring_input_replaces_old_wire():
    c = Circuit()
    a, b, g = c.add_chip("IN"), c.add_chip("IN"), c.add_chip("NOT")
    first, _ = c.connect(a.outputs[0], g.inputs[0])
    second, replaced = c.connect(b.outputs[0], g.inputs[0])
    assert replaced == [first] and c.wires == [second]


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


# ---- junctions / nets ------------------------------------------------------


def test_branch_fans_out_one_driver():
    c = Circuit()
    a, n1, n2 = c.add_chip("IN"), c.add_chip("NOT"), c.add_chip("NOT")
    trunk, _ = c.connect(a.outputs[0], n1.inputs[0])
    branch, _ = c.connect(trunk, n2.inputs[0])  # wire -> input pin
    assert branch.src is trunk and branch.dst is n2.inputs[0]
    a.outputs[0].state = True
    settle(c)
    assert n1.inputs[0].state and n2.inputs[0].state
    assert c.wire_state(branch) == (True, False)


def test_wire_ending_on_wire_joins_nets():
    c = Circuit()
    a, n1, n2 = c.add_chip("IN"), c.add_chip("NOT"), c.add_chip("NOT")
    w1, _ = c.connect(a.outputs[0], n1.inputs[0])
    stub, _ = c.connect(n2.inputs[0], w1)  # input pin -> wire, either order
    assert stub.src is w1 and stub.dst is n2.inputs[0]
    a.outputs[0].state = True
    settle(c)
    assert n2.inputs[0].state


def test_two_drivers_agreeing_is_fine_disagreeing_is_a_conflict():
    c = Circuit()
    a, b, led = c.add_chip("IN"), c.add_chip("IN"), c.add_chip("OUT")
    w, _ = c.connect(a.outputs[0], led.inputs[0])
    w2, _ = c.connect(b.outputs[0], w)  # second driver onto the same net
    for va, vb, expect_value, expect_conflict in [(False, False, False, False),
                                                  (True, True, True, False),
                                                  (True, False, False, True),   # X reads as 0 (placeholder)
                                                  (False, True, False, True)]:
        a.outputs[0].state, b.outputs[0].state = va, vb
        settle(c)
        assert led.inputs[0].state == expect_value
        assert c.wire_state(w) == c.wire_state(w2) == (expect_value, expect_conflict)


def test_undriven_net_reads_zero():
    c = Circuit()
    n1, n2 = c.add_chip("NOT"), c.add_chip("NOT")
    assert not c.can_connect(n1.inputs[0], n2.inputs[0])  # in -> in pin-to-pin is still rejected...
    a = c.add_chip("IN")
    trunk, _ = c.connect(a.outputs[0], n1.inputs[0])
    c.connect(trunk, n2.inputs[0])
    c.remove_chip(a)  # ...but a net can lose its driver
    settle(c)
    assert n1.inputs[0].state is False and n2.inputs[0].state is False


def test_removing_a_wire_removes_its_branches():
    c = Circuit()
    a, n1, n2, n3 = c.add_chip("IN"), c.add_chip("NOT"), c.add_chip("NOT"), c.add_chip("NOT")
    trunk, _ = c.connect(a.outputs[0], n1.inputs[0])
    b1, _ = c.connect(trunk, n2.inputs[0])
    b2, _ = c.connect(b1, n3.inputs[0])  # branch of a branch
    assert c.remove_wire(trunk) == [trunk, b1, b2]
    assert c.wires == []


def test_replacing_input_wire_cannot_saw_off_own_branch():
    c = Circuit()
    a, n1 = c.add_chip("IN"), c.add_chip("NOT")
    trunk, _ = c.connect(a.outputs[0], n1.inputs[0])
    b = c.add_chip("NOT")
    branch, _ = c.connect(trunk, b.inputs[0])
    # wiring n1's input from `branch` would first remove trunk (and thus branch)
    assert not c.can_connect(n1.inputs[0], branch)
    assert not c.can_connect(n1.inputs[0], trunk)


def test_merge_splices_branch_onto_trunk():
    c = Circuit()
    a, n1, n2, n3 = c.add_chip("IN"), c.add_chip("NOT"), c.add_chip("NOT"), c.add_chip("NOT")
    trunk, _ = c.connect(a.outputs[0], n1.inputs[0])
    branch, _ = c.connect(trunk, n2.inputs[0])
    twig, _ = c.connect(branch, n3.inputs[0])  # hangs off the branch
    c.merge(trunk, branch)
    assert trunk.src is a.outputs[0] and trunk.dst is n2.inputs[0]
    assert twig.src is trunk and branch not in c.wires
    assert c.wires == [trunk, twig]  # parents still before children
    a.outputs[0].state = True
    settle(c)
    assert n2.inputs[0].state and n3.inputs[0].state and not n1.inputs[0].state  # n1 was cut off


def test_wire_uids_are_stable():
    c = Circuit()
    a, g = c.add_chip("IN"), c.add_chip("NOT")
    w, _ = c.connect(a.outputs[0], g.inputs[0])
    c.remove_wire(w)
    again, _ = c.connect(a.outputs[0], g.inputs[0], uid=w.uid)
    assert again.uid == w.uid
    b = c.add_chip("NOT")
    assert c.connect(a.outputs[0], b.inputs[0])[0].uid == w.uid + 1
