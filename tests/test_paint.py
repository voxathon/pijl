from pijl.ui import theme as T
from pijl.ui.paint import mix, sample, with_hue

RED, BLUE, YELLOW = (
    T.WIRE_COLORS["red"][1],
    T.WIRE_COLORS["blue"][1],
    T.WIRE_COLORS["yellow"][1],
)


def test_red_tint_keeps_the_classic_theme():
    for color in (*T.SWITCH_ON, *T.SWITCH_OFF, *T.LED_ON, T.PIN_ON):
        assert with_hue(color, RED) == color
    assert with_hue(T.PIN_ON, None) == T.PIN_ON


def test_tint_swaps_hue_only():
    blue_switch = with_hue(T.SWITCH_ON[0], BLUE)
    assert blue_switch[2] > blue_switch[0]  # blue now
    assert max(blue_switch) == max(T.SWITCH_ON[0])  # same brightness


def test_mix_ends_and_no_grey_middle():
    assert mix(BLUE, YELLOW, 0) == BLUE and mix(BLUE, YELLOW, 1) == YELLOW
    mid = mix(BLUE, YELLOW, 0.5)
    assert max(mid) - min(mid) > 40  # still a color, not the RGB-average grey


def test_sample_gradient():
    stops = [
        (0.0, ((0, 0, 0), BLUE)),
        (0.5, ((0, 0, 0), RED)),
        (1.0, ((0, 0, 0), YELLOW)),
    ]
    assert sample(stops, 0.0)[1] == BLUE
    assert sample(stops, 0.5)[1] == RED
    assert sample(stops, 1.0)[1] == YELLOW
    assert sample([(0.0, (BLUE, BLUE))], 0.7) == (BLUE, BLUE)  # one stop: solid
