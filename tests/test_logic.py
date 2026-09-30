"""Four-state logic, checked exhaustively against the definition: a value is the set
of bits it could be (Z reads as "either" at a gate input), and an operator's result
is every output the possible inputs could produce."""

import itertools

import numpy as np
import pytest

from pijl.logic import ONE, X, Z, ZERO, Level, Logic, codes, fights, resolve, where

LEVELS = (ZERO, ONE, X, Z)
POSSIBLE = {ZERO: {0}, ONE: {1}, X: {0, 1}, Z: {0, 1}}  # as a gate input


def level_of(bits: set[int]) -> Level:
    return {frozenset({0}): ZERO, frozenset({1}): ONE}.get(frozenset(bits), X)


def reference(fn, *ins: Level) -> Level:
    return level_of({fn(*bits) for bits in itertools.product(*(POSSIBLE[v] for v in ins))})


OPS = {
    "and": (lambda a, b: a & b, lambda a, b: a & b),
    "or": (lambda a, b: a | b, lambda a, b: a | b),
    "xor": (lambda a, b: a ^ b, lambda a, b: a ^ b),
    "nand": (lambda a, b: ~(a & b), lambda a, b: 1 - (a & b)),
}


def pairs():
    return list(itertools.product(LEVELS, repeat=2))


@pytest.mark.parametrize("name", OPS)
def test_binary_ops_match_definition_on_arrays(name):
    op, bit_fn = OPS[name]
    a = Logic([p[0] for p in pairs()])
    b = Logic([p[1] for p in pairs()])
    got = list(op(a, b))
    assert got == [reference(bit_fn, x, y) for x, y in pairs()]


@pytest.mark.parametrize("name", OPS)
def test_binary_ops_match_definition_on_levels(name):
    op, bit_fn = OPS[name]
    for x, y in pairs():
        got = op(x, y)
        assert isinstance(got, Level) and got is reference(bit_fn, x, y), (x, y)


def test_not():
    assert list(~Logic(list(LEVELS))) == [ONE, ZERO, X, X]
    assert ~X is X and ~Z is X and ~ONE is ZERO


def test_known_values_behave_like_bools():
    rng = np.random.default_rng(1)
    a, b = rng.random(100) < 0.5, rng.random(100) < 0.5
    assert np.array_equal((~(Logic(a) & Logic(b))).is1, ~(a & b))
    assert np.array_equal((Logic(a) ^ b).is1, a ^ b)  # mixing in a plain bool array
    assert np.array_equal((b | Logic(a)).is1, a | b)  # ... on either side


def test_classic_x_cases():
    assert (ZERO & X) is ZERO and (ONE | X) is ONE  # a controlling input wins
    assert (ONE & X) is X and (ZERO | X) is X
    assert (ZERO & Z) is ZERO and (ONE & Z) is X  # a floating input reads as X


def test_coercion():
    assert list(Logic([True, False, 1, 0, 5])) == [ONE, ZERO, ONE, ZERO, ONE]
    assert list(Logic([ZERO, ONE, X, Z, True])) == [ZERO, ONE, X, Z, ONE]  # Levels aren't their int codes
    assert codes(ONE) == ONE.value and codes(True) == ONE.value
    assert list(Logic.full(3, Z)) == [Z, Z, Z]


def test_truthiness():
    assert [bool(v) for v in LEVELS] == [False, True, False, False]
    with pytest.raises(TypeError):
        bool(Logic([ONE]))


def test_where_matches_definition():
    for c, a, b in itertools.product(LEVELS, repeat=3):
        expect = {ONE: a, ZERO: b}.get(c)
        if expect is None:  # unknown select: only a known value both sides agree on survives
            expect = a if a == b and a in (ZERO, ONE) else X
        assert where(c, a, b)[()] is expect, (c, a, b)
    assert list(where(np.array([True, False]), ONE, Z)) == [ONE, Z]  # a tri-state buffer


def test_resolve_and_fights():
    # (no empty groups: reduceat can't do them, and the engine only resolves nets with drivers)
    groups = [[Z], [ZERO], [ONE], [X], [ZERO, Z], [ONE, Z, Z], [ZERO, ONE], [ZERO, X], [ONE, ONE], [Z, Z]]
    flat = np.array([v.value for g in groups for v in g], np.uint8)
    starts = np.cumsum([0] + [len(g) for g in groups[:-1]])
    value = [Level(int(v)) for v in resolve(flat, starts)]
    fight = fights(flat, starts).tolist()
    for g, v, f in zip(groups, value, fight):
        known = {x for x in g if x is not Z}
        expect = Z if not known else next(iter(known)) if len(known) == 1 else X
        assert v is expect, g
        assert f == ({ZERO, ONE} <= known), g


def test_repr():
    assert repr(Logic([ZERO, ONE, X, Z])) == "Logic(01XZ)"
    assert str(X) == "X"
