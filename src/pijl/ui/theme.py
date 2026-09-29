"""Colors and sizes in one place so the look can be tweaked without hunting."""

BACKGROUND = (28, 28, 34, 255)

# Screen-space UI (part picker, context menus) is drawn this many times its base size.
UI_SCALE = 1.2

# Part bodies are (fill, border) pairs. Bright fills get a darker border,
# dark fills a lighter one, so every block reads against the background.
PART_BODY = ((62, 84, 150), (34, 47, 92))
MACRO_SWATCH = ((128, 84, 160), (74, 46, 98))  # saved macros in the part picker
MACRO_BODY = ((92, 66, 138), (52, 36, 82))      # placed macros
SWITCH_OFF = ((70, 40, 40), (112, 62, 62))
SWITCH_ON = ((220, 60, 60), (140, 30, 30))
LED_OFF = ((45, 45, 52), (86, 86, 98))
LED_ON = ((240, 70, 70), (155, 35, 35))
PART_TEXT = (235, 235, 245, 255)
LABEL_TEXT = (200, 200, 215, 255)  # user labels next to parts

PIN_OFF = (18, 18, 22)
PIN_ON = (240, 70, 70)

WIRE_OFF = (85, 85, 98)
WIRE_ON = (235, 60, 60)
WIRE_CONFLICT = (240, 175, 40)  # net driven to different values at once (future: X)
WIRE_PREVIEW = (200, 200, 210, 160)
WIRE_PREVIEW_SNAP = (120, 220, 140, 220)

PICKER_BG = (34, 34, 41)
PICKER_HEADER = (40, 40, 48)  # the panel's title bar, and its collapsed strip's button
PICKER_SECTION = (38, 38, 46)  # collection rows
PICKER_HOVER = (58, 58, 72)
PICKER_LIFT = (70, 70, 88)  # a row being dragged
PICKER_BORDER = (62, 62, 74)
PICKER_DIM_TEXT = (120, 120, 135, 255)
HELP_TEXT = (150, 150, 165, 255)

MENU_PANEL = ((36, 36, 44), (84, 84, 100))  # (fill, border)
MENU_HOVER = (60, 60, 76)
MENU_DANGER = (240, 110, 110, 255)
CARET = (235, 235, 245)

# Selection
SELECT = (90, 170, 255)
SELECT_WIRE = (90, 170, 255, 110)  # glow under selected wires
SELECT_BOX_FILL = (90, 170, 255, 30)
SELECT_OUTSET = 4  # part outline distance from the body, world units
SELECT_THICKNESS = 2

# Wire editing
WIRE_HALO = (255, 255, 255, 40)
HANDLE_FILL = (235, 235, 245)
HANDLE_BORDER = (20, 20, 26)
HANDLE_HOVER = (120, 220, 140)  # same green as a valid wire target
ADD_HANDLE_FILL = (28, 28, 34)  # hollow look: background-colored center
ADD_HANDLE_BORDER = (200, 200, 215)
GHOST_OPACITY = 150

# Grid: (minor line, major line) colors. Brighter while Ctrl-snapping.
GRID_COLORS = ((34, 34, 42), (44, 44, 54))
GRID_SNAPPING = ((42, 42, 52), (62, 62, 78))

# World-space sizes
PIN_RADIUS = 6
# Snapping grid. Pins are PIN_SPACING apart and centered on each part side, so a
# side with an odd pin count is offset by half a spacing from an even one. With
# GRID = PIN_SPACING / 2 and part sizes in multiples of GRID, every pin of a
# snapped part lands exactly on a grid point.
GRID = 10
GRID_MAJOR_EVERY = 4
SUBGRID_DIVISIONS = 2  # Ctrl+Shift snaps to GRID / this (off the pin grid, on purpose)
PIN_SPACING = 20
PART_WIDTH = 80
IO_WIDTH = 40
WIRE_THICKNESS = 3
JUNCTION_RADIUS = 4.5  # dot where a wire attaches to another wire
PART_BORDER = 2
LABEL_SIZE = 10  # pt at zoom 1
LABEL_GAP = 6  # between a part and its label
PIN_LABEL_SIZE = 9     # pt at zoom 1: pin name tags next to macro pins
PIN_TAG_BG = (0, 0, 0, 170)  # the tag behind each name
PIN_TAG_PAD = (4, 3)   # tag padding around the text (x, y), world units
PIN_TAG_GAP = 4        # between the pin dot's edge and its tag
TITLE_PAD = 12         # a part's title keeps this much room on each side
TITLE_SIZE, IO_TITLE_SIZE = 12, 10

# pyglet picks circle smoothness from the radius at creation, which is far too
# coarse once zoomed in. These stay smooth up to the max zoom (8x).
PIN_SEGMENTS = 48
JOINT_SEGMENTS = 24

# Screen-space sizes
HIT_SLOP_PX = 4  # how forgiving pin/wire clicks are, in screen pixels
DRAG_THRESHOLD_PX = 4  # mouse must move this far before a press becomes a drag
