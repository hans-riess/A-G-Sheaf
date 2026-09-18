"""
The torpedo icon the local previews draw in place of the Robotarium's GTernal robot.

Cosmetic, and local-preview-only. The geometry lived in the robotarium submodule for a while,
edited in place; it lives here instead so that `libs/robotarium` stays a clean checkout. Nothing
about the physics reads these numbers -- ROBOT_DIAMETER, the barrier certificates and the
unicycle model are all untouched, and the hull is deliberately drawn longer and slimmer than the
0.11 m footprint the physics uses.

`install()` is the whole interface, and it has to work by rebinding a name inside rps:
ARobotarium._create_robot_patches calls the module-global `gternal_patch()` with no argument to
inject a different one through (see robotarium_abc.py, in the figure setup). So the entry points
that want the reskin call install() before they construct their GridWorld, and the ones that do
not -- notably exp/robotarium/experiment_template.py, which is what gets uploaded -- are drawn
with the stock GTernal icon. A physical run is unaffected either way: the testbed executes
against its own copy of rps, which this repository does not ship to it.

Coordinate convention (body frame):
    +y = forward (nose of the vehicle),  +x = right,  origin = centre of the hull.

Forward is +y, not +x, and the geometry has to be built that way: GTERNALRobotPatch._transform
rotates by ``theta - pi/2``, which is exactly the correction that turns a +y-forward body frame
into the world's +x-forward heading convention. (The stock GTernal body was laid out this way
too -- its arch was built along +y -- though the docstring there describes the frame as
+x-forward.) Build the hull along +x instead and every robot is drawn 90 degrees off its actual
heading.

One hard constraint on the part list: rps hardcodes GTERNALRobotPatch.LED_FACE_IDX = 3 and
_draw_robots calls set_led on every step, so the sensor light must be the *fourth* part or the
first step indexes past the end. Parts after it are unconstrained, which is where the propeller
hub goes.
"""

from __future__ import annotations

import numpy as np

from rps.patch_creation.gternal_patch import GTERNALPatchData

# ----------------------------------------------------------------------------------------------
# Hull dimensions (metres). A torpedo silhouette is carried by proportion rather than by detail:
# a long parallel midbody between a short blunt nose and a slow tapering run aft, with the fins
# at the very back. The nose is rounded, not pointed -- a sharp ogive reads as a dart or a
# rocket, while a real torpedo carries its diameter almost to the tip.
#
# The midbody carries the agent numeral (exp/run.py's place_agent_badges draws it over the robot
# at zorder 2.6) and is the numeral's only backing -- the badge is white text with no disc behind
# it, so where the hull is not, the numeral is white on white. That sets a floor on the width but
# a low one: at the figure scale a 12 x 8 grid is drawn at (~340 px/m), 0.054 m of beam is ~18 px
# against a ~6 px glyph, which is clearance enough. Everything left over is spent on length,
# because slenderness is what separates a torpedo from a bomb at this size.
# ----------------------------------------------------------------------------------------------
HALF_WIDTH = 0.027          # hull half-width, held constant down the whole midbody
STRAIGHT_HALF = 0.058       # half-length of the parallel midbody
NOSE_LENGTH = 0.030         # short rounded nose, forward of the shoulder
TAIL_LENGTH = 0.046         # run aft of the shoulder, to the stern
STERN_FRAC = 0.50           # stern half-width, as a fraction of the hull's

#: Bluntness of the nose: the half-width falls off as sqrt(1 - t**2) over the nose, an elliptical
#: cap that leaves the tip rounded rather than closing to a point. This is the single strongest
#: torpedo cue in the whole outline.
NOSE_EXPONENT = 0.5

#: Fullness of the run aft. Above 1 the tail holds its beam well past the shoulder before
#: drawing in, which is what a torpedo's afterbody does; a straight taper instead puts a visible
#: kink at the shoulder and reads as the tail fins of a bomb.
TAIL_EXPONENT = 1.8

# Cruciform tail fins, drawn as one trapezoid spanning the hull: what shows is the pair of
# triangles either side, swept back and outward the way a real fin set is. Wider than the hull on
# purpose -- fins flush with the body would be invisible at this scale.
FIN_ROOT_HALF = 0.018       # half-width where the fin leaves the hull
FIN_TIP_HALF = 0.040        # half-width at the fin tips
FIN_FRONT_INSET = 0.024     # how far forward of the stern the fin's leading edge starts
FIN_OVERHANG = 0.000        # how far aft of the stern the fin tips trail

HUB_RADIUS = 0.005          # propeller hub, tucked just inside the stern
LED_RADIUS = 0.0042         # sensor light, set into the nose cone

STERN_Y = -(STRAIGHT_HALF + TAIL_LENGTH)
LED_Y = STRAIGHT_HALF + NOSE_LENGTH * 0.35
HUB_Y = STERN_Y + HUB_RADIUS

HULL_NAVY = np.array([0.14, 0.20, 0.27])
FIN_SLATE = np.array([0.09, 0.13, 0.18])
NOSE_ORANGE = np.array([0.91, 0.39, 0.09])   # an AUV-style safety-orange nose cone
HUB_BRASS = np.array([0.62, 0.51, 0.24])
LED_WHITE = np.array([1.00, 1.00, 1.00])


def _nose_profile(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    The nose's centreline stations and half-widths, bow-ward from the shoulder to the tip.

    Shared by the hull outline and the nose cone so the cone sits exactly on the hull it caps;
    computing it twice is how the two drift apart when the dimensions above are edited.
    """
    t = np.linspace(0.0, 1.0, n)
    y = STRAIGHT_HALF + NOSE_LENGTH * t
    half = HALF_WIDTH * (1.0 - t ** 2) ** NOSE_EXPONENT
    return y, half


def _tail_profile(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    The afterbody's centreline stations and half-widths, running aft from the shoulder to the
    stern. Curved rather than straight, so the beam carries past the shoulder and closes in
    toward the stern with no kink where the midbody ends.
    """
    s = np.linspace(0.0, 1.0, n)
    y = -STRAIGHT_HALF - TAIL_LENGTH * s
    half = HALF_WIDTH * (1.0 - (1.0 - STERN_FRAC) * s ** TAIL_EXPONENT)
    return y, half


def _ring(radius: float, centre_y: float, n: int) -> np.ndarray:
    """A closed polygon approximating a disc of ``radius`` centred on the axis at ``centre_y``."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), centre_y + radius * np.sin(angles)])


def torpedo_patch() -> GTERNALPatchData:
    """
    Patch data for a torpedo body: swept tail fins, a hull of blunt nose and long parallel
    midbody over a tapering run aft, a safety-orange nose cone, a sensor light set into it, and a
    propeller hub at the stern.

    Returned in the shape rps expects -- vertices, one index array per part, one RGB row per
    part -- so that the stock GTERNALRobotPatch draws it unmodified.

    Heading is carried three ways over, which is what lets the icon stay legible at a tile's
    size: the blunt capped nose against the finned stern, the colour break at the nose cone, and
    the fins themselves being the widest thing on the body.
    """
    nose_y, nose_half = _nose_profile(16)
    tail_y, tail_half = _tail_profile(12)
    stern_half = HALF_WIDTH * STERN_FRAC

    # ------------------------------------------------------------------------------------------
    # Part 0: tail fins. Drawn before the hull so the hull covers the root and only the swept
    # triangles either side show. One trapezoid, widening aft.
    # ------------------------------------------------------------------------------------------
    fin_verts = np.array([
        [-FIN_ROOT_HALF, STERN_Y + FIN_FRONT_INSET],
        [FIN_ROOT_HALF, STERN_Y + FIN_FRONT_INSET],
        [FIN_TIP_HALF, STERN_Y - FIN_OVERHANG],
        [-FIN_TIP_HALF, STERN_Y - FIN_OVERHANG],
    ])

    # ------------------------------------------------------------------------------------------
    # Part 1: hull. One closed loop up the right side from the stern, around the nose, and back
    # down the left. The parallel midbody is the implicit straight edge between the last tail
    # station and the first nose station; the flat stern is the closing edge. Long axis on y, bow
    # at +y (see the frame note in the module docstring).
    # ------------------------------------------------------------------------------------------
    tail_right = np.column_stack([tail_half[::-1], tail_y[::-1]])   # stern forward to the shoulder
    nose_right = np.column_stack([nose_half, nose_y])               # shoulder up around to the tip
    nose_left = np.column_stack([-nose_half[-2::-1],                # back down, minus the shared tip
                                 nose_y[-2::-1]])
    tail_left = np.column_stack([-tail_half, tail_y])               # shoulder back aft to the stern

    hull_verts = np.vstack([tail_right, nose_right, nose_left, tail_left])

    # ------------------------------------------------------------------------------------------
    # Part 2: nose cone. The nose region of the hull outline, recoloured -- a warhead section
    # rather than a dot on the front, so the colour break lands on a real edge of the silhouette.
    # ------------------------------------------------------------------------------------------
    cone_verts = np.vstack([
        [HALF_WIDTH, STRAIGHT_HALF],
        nose_right,
        nose_left,
        [-HALF_WIDTH, STRAIGHT_HALF],
    ])

    # ------------------------------------------------------------------------------------------
    # Part 3: sensor light. The index rps recolours at runtime, so it has to sit exactly here.
    # Set into the cone rather than out at its rim, so that a run which leaves it black (the
    # usual case -- GridWorld never calls set_leds, and _leds defaults to zeros) reads as a
    # sensor eye rather than as a nick out of the hull's outline.
    # ------------------------------------------------------------------------------------------
    led_verts = _ring(LED_RADIUS, LED_Y, 12)

    # ------------------------------------------------------------------------------------------
    # Part 4: propeller hub, over the fin roots at the stern. Free to sit after the LED -- only
    # index 3 is spoken for.
    # ------------------------------------------------------------------------------------------
    hub_verts = _ring(HUB_RADIUS, HUB_Y, 12)

    parts = [
        (fin_verts, FIN_SLATE),
        (hull_verts, HULL_NAVY),
        (cone_verts, NOSE_ORANGE),
        (led_verts, LED_WHITE),
        (hub_verts, HUB_BRASS),
    ]

    vertices = np.vstack([verts for verts, _ in parts])

    faces = []
    offset = 0
    for verts, _ in parts:
        faces.append(np.arange(offset, offset + len(verts), dtype=int))
        offset += len(verts)

    colors = np.vstack([color for _, color in parts])

    return GTERNALPatchData(vertices=vertices, faces=faces, colors=colors)


def install() -> None:
    """
    Points the Robotarium's figure setup at the torpedo icon, for the rest of this process.

    Call it before constructing a GridWorld -- the patch data is read once, when the figure is
    built. Rebinding the name in rps.robotarium_abc is what there is to work with: the call site
    takes no argument (see the module docstring), and the alternative is editing the submodule.
    """
    import rps.robotarium_abc

    rps.robotarium_abc.gternal_patch = torpedo_patch
