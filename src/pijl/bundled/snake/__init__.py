"""Snake: replaces the editor with a game of snake.

The editor's entry point is pijl.ui.editor.run (cli imports it only when it starts
the editor, after the mods have loaded), so swapping that one name takes over
`pijl gui` and START in the launcher. The snake is a lit wire, the food is an LED,
and a dead snake goes X.

    arrows / WASD   steer
    space / P       pause
    R               start over
    Esc             back to the launcher
"""

from __future__ import annotations

import json
import logging
import random
from collections import deque
from pathlib import Path

from pijl.mods import after_import

HERE = Path(__file__).parent
BEST = HERE / "best.json"
COLS, ROWS = 32, 22
START_HZ, MAX_HZ = 8.0, 20.0
FOOD = ("NAND", "NOR", "XOR", "AND", "OR", "NOT", "MUX", "DFF", "ADD", "LED", "CLK", "SPLIT")

log = logging.getLogger(__name__)


def read_best() -> int:
    try:
        return int(json.loads(BEST.read_text(encoding="utf-8"))["best"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def write_best(n: int) -> None:
    try:
        BEST.write_text(json.dumps({"best": n}) + "\n", encoding="utf-8")
    except OSError as e:
        log.warning("can't save the best score: %s", e)


class Game:
    """The rules, without pyglet: step() moves one cell."""

    DIRS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.reset()

    def reset(self) -> None:
        y = ROWS // 2
        self.body = deque([(5, y), (4, y), (3, y)])  # (head first)
        self.heading = (1, 0)
        self.turns: deque[tuple[int, int]] = deque()
        self.score = 0
        self.dead = False
        self.paused = False
        self.eaten = ""
        self.place_food()

    def place_food(self) -> None:
        taken = set(self.body)
        free = [(x, y) for x in range(COLS) for y in range(ROWS) if (x, y) not in taken]
        self.food = self.rng.choice(free) if free else None
        self.food_kind = self.rng.choice(FOOD)

    def steer(self, name: str) -> None:
        """Queue a turn (a few, so quick taps aren't lost); no turning back on itself."""
        d = self.DIRS[name]
        last = self.turns[-1] if self.turns else self.heading
        if d != last and d != (-last[0], -last[1]) and len(self.turns) < 3:
            self.turns.append(d)

    def step(self) -> None:
        if self.dead or self.paused:
            return
        if self.turns:
            self.heading = self.turns.popleft()
        hx, hy = self.body[0]
        head = (hx + self.heading[0], hy + self.heading[1])
        grows = head == self.food
        rest = set(self.body) if grows else set(list(self.body)[:-1])  # (the tail moves away)
        if not (0 <= head[0] < COLS and 0 <= head[1] < ROWS) or head in rest:
            self.dead = True
            return
        self.body.appendleft(head)
        if grows:
            self.score += 1
            self.eaten = self.food_kind
            self.place_food()
        else:
            self.body.pop()

    @property
    def hz(self) -> float:
        return min(MAX_HZ, START_HZ + self.score * 0.4)


def run(project: str | None = None, settle_ticks: int = 0) -> bool:
    """Stands in for the editor. True: back to the launcher (as the editor's run)."""
    import pyglet
    from pyglet import shapes
    from pyglet.window import key

    from pijl import mods
    from pijl.ui import theme as T

    cell = round(20 * T.UI_SCALE)
    top = round(34 * T.UI_SCALE)
    width, height = COLS * cell, ROWS * cell + top
    window = pyglet.window.Window(width, height, caption="pijl: snake")
    game = Game()
    best = [read_best()]
    relaunch = [False]
    clock = [0.0]

    hud = pyglet.text.Label(
        "", x=10, y=height - top // 2, anchor_y="center",
        font_name="Consolas", font_size=round(11 * T.UI_SCALE), color=T.PART_TEXT,
    )
    banner = pyglet.text.Label(
        "", x=width // 2, y=(height - top) // 2 + cell, anchor_x="center", anchor_y="center",
        font_name="Consolas", font_size=round(20 * T.UI_SCALE), weight="bold", color=T.PART_TEXT,
    )
    hint = pyglet.text.Label(
        "", x=width // 2, y=(height - top) // 2 - cell, anchor_x="center", anchor_y="center",
        font_name="Consolas", font_size=round(10 * T.UI_SCALE), color=T.HELP_TEXT,
    )

    def center(c: tuple[int, int]) -> tuple[float, float]:
        return c[0] * cell + cell / 2, c[1] * cell + cell / 2

    def tick(dt: float) -> None:
        game.step()
        if game.dead and game.score > best[0]:
            best[0] = game.score
            write_best(best[0])
        pyglet.clock.unschedule(tick)  # (faster as it grows)
        pyglet.clock.schedule_interval(tick, 1 / game.hz)

    def animate(dt: float) -> None:
        clock[0] += dt

    steer = {
        key.UP: "up", key.W: "up", key.DOWN: "down", key.S: "down",
        key.LEFT: "left", key.A: "left", key.RIGHT: "right", key.D: "right",
    }

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol in steer:
            if game.dead:
                game.reset()
            game.steer(steer[symbol])
        elif symbol in (key.SPACE, key.P) and not game.dead:
            game.paused = not game.paused
        elif symbol == key.R:
            game.reset()
        elif symbol == key.ESCAPE:
            relaunch[0] = True
            window.close()
        return pyglet.event.EVENT_HANDLED  # (or Esc closes the window by itself)

    @window.event
    def on_draw():
        window.clear()
        batch = pyglet.graphics.Batch()
        keep = []  # (a shape leaves the batch when it's collected)
        keep.append(shapes.Rectangle(0, 0, width, height, color=T.BACKGROUND[:3], batch=batch))
        keep.append(shapes.Rectangle(0, height - top, width, top, color=T.PICKER_HEADER, batch=batch))
        for x in range(1, COLS):  # (the board's grid dots)
            for y in range(1, ROWS):
                keep.append(shapes.Circle(x * cell, y * cell, 1.2, color=T.PICKER_BORDER, batch=batch))

        if game.food is not None:  # an LED, blinking
            fx, fy = center(game.food)
            fill, ring = T.LED_ON if int(clock[0] * 3) % 2 == 0 else T.LED_OFF
            keep.append(shapes.Circle(fx, fy, cell * 0.42, color=ring, batch=batch))
            keep.append(shapes.Circle(fx, fy, cell * 0.32, color=fill, batch=batch))

        points = [center(c) for c in game.body]
        if game.dead:  # the missing-texture X look, bands scrolling
            band = int(clock[0] * 6)
            colors = [T.LOGIC_X[(i + band) % 2] for i in range(len(points))]
        else:
            colors = [T.WIRE_ON] * len(points)
        for a, b, color in zip(points, points[1:], colors):
            keep.append(shapes.Line(*a, *b, thickness=cell * 0.3, color=color, batch=batch))
        for p, color in zip(points, colors):
            keep.append(shapes.Circle(*p, cell * 0.18, color=color, batch=batch))
        hx, hy = points[0]  # the head is a part
        fill, edge = T.PART_BODY
        keep.append(shapes.BorderedRectangle(
            hx - cell * 0.45, hy - cell * 0.45, cell * 0.9, cell * 0.9, border=2,
            color=fill, border_color=edge, batch=batch,
        ))
        dx, dy = game.heading
        for side in (-1, 1):  # two pins, looking ahead
            ex = hx + dx * cell * 0.2 + dy * side * cell * 0.2
            ey = hy + dy * cell * 0.2 - dx * side * cell * 0.2
            keep.append(shapes.Circle(ex, ey, cell * 0.1, color=T.PIN_ON, batch=batch))
        batch.draw()

        ate = f"   ate {game.eaten}" if game.eaten else ""
        hud.text = f"SNAKE   score {game.score}   best {best[0]}{ate}"
        hud.draw()
        if game.dead:
            banner.text, hint.text = "X", "R or an arrow: again    Esc: launcher"
        elif game.paused:
            banner.text, hint.text = "PAUSED", "space: go on"
        else:
            banner.text = hint.text = ""
        banner.draw()
        hint.draw()

    @window.event
    def on_close():
        pyglet.app.exit()

    pyglet.clock.schedule_interval(tick, 1 / game.hz)
    pyglet.clock.schedule_interval(animate, 1 / 30)
    mods.settled()  # (it's up: no crash to blame on the mods)
    try:
        pyglet.app.run()
    finally:
        pyglet.clock.unschedule(tick)
        pyglet.clock.unschedule(animate)
    return relaunch[0]


@after_import("pijl.ui.editor")
def _(m):
    m.run = run
