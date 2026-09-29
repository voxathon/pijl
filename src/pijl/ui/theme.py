"""Colors and sizes in one place so the look can be tweaked without hunting."""

BACKGROUND = (28, 28, 34, 255)

# Chip bodies are (fill, border) pairs. Bright fills get a darker border,
# dark fills a lighter one, so every block reads against the background.
CHIP_BODY = ((62, 84, 150), (34, 47, 92))
SWITCH_OFF = ((70, 40, 40), (112, 62, 62))
SWITCH_ON = ((220, 60, 60), (140, 30, 30))
LED_OFF = ((45, 45, 52), (86, 86, 98))
LED_ON = ((240, 70, 70), (155, 35, 35))
CHIP_TEXT = (235, 235, 245, 255)

PIN_OFF = (18, 18, 22)
PIN_ON = (240, 70, 70)

WIRE_OFF = (85, 85, 98)
WIRE_ON = (235, 60, 60)
WIRE_PREVIEW = (200, 200, 210, 160)
WIRE_PREVIEW_SNAP = (120, 220, 140, 220)

TOOLBAR_BG = ((40, 40, 48), (62, 62, 74))      # (fill, border), like chips
TOOLBAR_HOVER = ((64, 64, 78), (110, 110, 130))
TOOLBAR_BORDER = 1
HELP_TEXT = (150, 150, 165, 255)
GHOST_OPACITY = 150

# World-space sizes
PIN_RADIUS = 6
PIN_SPACING = 22
CHIP_WIDTH = 72
IO_WIDTH = 40
WIRE_THICKNESS = 3
CHIP_BORDER = 2

# pyglet picks circle smoothness from the radius at creation, which is far too
# coarse once zoomed in. These stay smooth up to the max zoom (8x).
PIN_SEGMENTS = 48
JOINT_SEGMENTS = 24

# Screen-space sizes
HIT_SLOP_PX = 4        # how forgiving pin/wire clicks are, in screen pixels
DRAG_THRESHOLD_PX = 4  # mouse must move this far before a press becomes a drag
