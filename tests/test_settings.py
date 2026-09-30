"""Part settings and actions: the declarations, the registry's checks, the engine's
set / changed / action calls, and loading saved values."""

import pytest

from pijl.logic import ONE, ZERO
from pijl.parts import (Action, Choice, Number, PartType, Registry, Text, Toggle, check_props,
                        fresh_props)
from pijl.sim import Circuit
from pijl.storage import decode, encode
from pijl.snapshot import Snapshot


class Clock(PartType):
    kind = "CLOCK"
    outs = ("out",)
    settings = {
        "period": Number(250, 1, 10_000, unit="ms", log=True),
        "running": Toggle(True),
        "mode": Choice(("square", "pulse"), "square", labels=("Square", "Pulse")),
        "note": Text("", max_len=5),
        "duty": Number(0.5, 0, 1, step=0.1, live=True),
    }
    actions = {"reset": Action("Reset phase"), "forget": Action(danger=True)}
    props = {"phase": 0}

    def __init__(self):
        self.calls = []

    def eval(self, ctx):
        return [ONE if p["running"] else ZERO for p in ctx.props]

    def changed(self, ctx, key, old):
        self.calls.append(("changed", [p.uid for p in ctx.parts], key, old))

    def action(self, ctx, name):
        self.calls.append(("action", [p.uid for p in ctx.parts], name))
        for p in ctx.parts:
            p.props["phase"] = 0


@pytest.fixture
def clock():
    reg = Registry()
    reg.add(Clock)
    return reg.get("CLOCK")


@pytest.fixture
def circuit(clock):
    reg = Registry()
    reg.types["CLOCK"] = clock
    return Circuit(reg)


# ---- the declarations ----------------------------------------------------------------


def test_number_clamps_snaps_and_keeps_ints_whole():
    n = Number(5, 0, 10)
    assert n.parse(3.4) == 3 and type(n.parse(3.4)) is int
    assert n.parse(-4) == 0 and n.parse(99) == 10
    f = Number(0.5, 0, 1, step=0.1)
    assert f.parse(0.33) == 0.3 and f.parse(0.7) == 0.7  # (not 0.7000000000000001)
    assert Number(0.5, 0, 1).parse(0.123456) == 0.123456  # continuous
    for bad in (True, "3", None, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            n.parse(bad)


def test_number_from_text():
    n = Number(250, 1, 10_000, unit="ms")
    assert n.parse_text(" 300 ms") == 300 and n.parse_text("1e9") == 10_000
    assert n.show(300) == "300 ms"
    with pytest.raises(ValueError, match="isn't a number"):
        n.parse_text("fast")


def test_choice_matches_type_exactly():
    c = Choice((0, 1, 2), 0)
    assert c.parse(1) == 1
    with pytest.raises(ValueError):
        c.parse(True)  # True == 1, but it isn't one of the values
    with pytest.raises(ValueError):
        c.parse(1.0)
    assert Choice(("a", "b"), "a", labels=("A", "B")).show("b") == "B"


def test_toggle_and_text():
    assert Toggle().parse(True) is True
    with pytest.raises(ValueError):
        Toggle().parse(1)
    assert Text(max_len=3).parse("ab\ncdef") == "abc"
    with pytest.raises(ValueError):
        Text().parse(3)


def test_titles_default_to_the_key():
    assert Toggle().title("auto_reset") == "Auto reset"
    assert Toggle(label="Go").title("auto_reset") == "Go"
    assert Action().title("clear_all") == "Clear all"


# ---- the registry --------------------------------------------------------------------


@pytest.mark.parametrize("settings, match", [
    ({"x": Choice((), 0)}, "needs values"),
    ({"x": Choice((1, 2), 3)}, "default"),
    ({"x": Choice((1, 1), 1)}, "unique"),
    ({"x": Choice((1, 2), 1, labels=("one",))}, "labels"),
    ({"x": Choice(([1], 2), 2)}, "must be str"),
    ({"x": Number(5, 10, 0)}, "min < max"),
    ({"x": Number(5, 0, 10, step=0)}, "step"),
    ({"x": Number(5, 0, 10, log=True)}, "log"),
    ({"x": Number(50, 0, 10)}, "default"),
    ({"x": Toggle(1)}, "default"),
    ({"x": Text(max_len=0)}, "max_len"),
    ({"x": Text("toolong", max_len=3)}, "default"),
    ({"not ok": Toggle()}, "identifiers"),
    ({"x": (1, 2)}, "not a Setting"),
    ({"color": Toggle()}, "reserved"),
])
def test_bad_settings_are_refused(settings, match):
    class Bad(PartType):
        kind = "BAD"
    Bad.settings = settings
    with pytest.raises((ValueError, TypeError), match=match):
        Registry().add(Bad)


def test_a_setting_isnt_also_a_prop():
    class Both(PartType):
        kind, props, settings = "BOTH", {"x": 1}, {"x": Number(1, 0, 2)}
    with pytest.raises(ValueError, match="not also in props"):
        Registry().add(Both)


def test_actions_need_the_hook():
    class NoHook(PartType):
        kind, actions = "NOHOOK", {"go": Action()}
    with pytest.raises(ValueError, match="action"):
        Registry().add(NoHook)

    class BadName(Clock):
        kind, actions = "BADNAME", {"go now": Action()}
    with pytest.raises(ValueError, match="identifiers"):
        Registry().add(BadName)


def test_fresh_props_include_the_settings(clock):
    assert fresh_props(clock) == {"phase": 0, "period": 250, "running": True, "mode": "square",
                                  "note": "", "duty": 0.5}


def test_saved_values_are_checked(clock):
    props, bad = check_props(clock, {"period": 99_999, "mode": "sine", "old_setting": 3})
    assert props["period"] == 10_000       # clamped, silently
    assert props["mode"] == "square"       # not a value: reset to the default...
    assert len(bad) == 1 and "mode" in bad[0]  # ...and said so
    assert props["old_setting"] == 3       # unknown keys are kept
    assert props["running"] is True        # missing: the default


def test_loading_a_bad_saved_value_warns(clock):
    reg = Registry()
    reg.types["CLOCK"] = clock
    data = encode(Snapshot({1: ("CLOCK", "", 0.0, 0.0, {"running": "yes"})}, {}))
    loaded = decode(data, reg)
    assert loaded.snapshot.parts[1][4]["running"] is True
    assert len(loaded.warnings) == 1 and "running" in loaded.warnings[0]


# ---- the engine ----------------------------------------------------------------------


def test_set_setting_writes_all_and_calls_changed_once(circuit, clock):
    a, b, c = (circuit.add_part("CLOCK") for _ in range(3))
    b.props["period"] = 500
    old = circuit.set_setting([a, b, c], "period", 500.4)
    assert old == [250, 500, 250]
    assert [p.props["period"] for p in (a, b, c)] == [500, 500, 500]
    # one call, and only for the parts whose value changed
    assert clock.calls == [("changed", [a.uid, c.uid], "period", [250, 250])]


def test_set_setting_refuses_bad_values_and_mixed_kinds(circuit):
    a = circuit.add_part("CLOCK")
    with pytest.raises(ValueError):
        circuit.set_setting([a], "mode", "sine")
    assert a.props["mode"] == "square"
    with pytest.raises(ValueError, match="one kind"):
        circuit.set_setting([a, circuit.add_part("IN")], "mode", "pulse")


def test_a_drag_notifies_once_at_the_end(circuit, clock):
    a = circuit.add_part("CLOCK")
    start = circuit.set_setting([a], "period", 300, notify=False)
    circuit.set_setting([a], "period", 400, notify=False)
    assert clock.calls == []  # still dragging
    circuit.settings_changed([a], "period", start)
    assert clock.calls == [("changed", [a.uid], "period", [250])]


def test_a_live_setting_notifies_while_dragging(circuit, clock):
    a = circuit.add_part("CLOCK")
    circuit.set_setting([a], "duty", 0.2, notify=False)
    assert clock.calls == [("changed", [a.uid], "duty", [0.5])]


def test_put_setting_takes_an_edit_back(circuit, clock):
    a, b = circuit.add_part("CLOCK"), circuit.add_part("CLOCK")
    b.props["period"] = 10
    old = circuit.set_setting([a, b], "period", 700, notify=False)
    circuit.put_setting([a, b], "period", old, notify=False)
    assert (a.props["period"], b.props["period"]) == (250, 10) and clock.calls == []


def test_eval_sees_a_setting_on_the_next_step(circuit):
    a = circuit.add_part("CLOCK")
    circuit.step()
    assert a.outputs[0].state is ONE
    circuit.set_setting([a], "running", False)
    circuit.step()
    assert a.outputs[0].state is ZERO


def test_actions_run_once_for_the_live_parts(circuit, clock):
    a, b = circuit.add_part("CLOCK"), circuit.add_part("CLOCK", live=False)
    circuit.run_action([a, b], "reset")
    assert clock.calls == [("action", [a.uid], "reset")]
    with pytest.raises(KeyError):
        circuit.run_action([a], "explode")


def test_a_raising_hook_disables_the_kind(circuit, clock):
    a = circuit.add_part("CLOCK")
    clock.changed = lambda ctx, key, old: 1 / 0
    circuit.set_setting([a], "mode", "pulse")
    assert "CLOCK" in circuit.faults and "changed" in circuit.faults["CLOCK"]
    assert a.props["mode"] == "pulse"  # the write itself stands


def test_slider_positions():
    lin = Number(5, 0, 10)
    assert lin.fraction(5) == 0.5 and lin.at(0.5) == 5 and lin.at(-1) == 0 and lin.at(2) == 10
    log = Number(100, 1, 10_000, log=True)
    assert log.fraction(100) == pytest.approx(0.5) and log.at(0.5) == 100 and log.at(1) == 10_000
    assert Number(0.5, 0, 1, step=0.25).at(0.4) == 0.5  # snapped
