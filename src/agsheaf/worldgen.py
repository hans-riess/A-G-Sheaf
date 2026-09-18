"""
Builds a demonstration world at a requested size, with the report's seeded conflicts placed by
construction rather than hoped for.

This is what `agsheaf sims --num-agents N --num-obstacles K` runs. It exists because the three
conflicts of doc/ms4_report (Subtask 5.1) do not survive random placement: hazards scattered by
`exp/benchmark.py`'s sampler make a world the flow can be measured on, but not one that
demonstrates anything, because nothing then depends on which tile of a region carries a hazard.
Every rule below was arrived at by watching a hand-placed world fail to produce the conflict it
was supposed to, so they are constraints, not decoration:

  Both walls in one region.  The true wall and the misplaced copy of it must lie INTERIOR to the
    same region, so that they push to the same region summary and the error is invisible at the
    interface. A wall straddling two regions makes the misplacement visible in the interface
    vocabulary, and the flow then corrects it like any other disagreement -- which is a different
    (and much less interesting) experiment.

  Partial walls, with staggered gaps.  Each wall leaves one row open, and the true wall's gap is
    at the far end from the false wall's. A wall sealing its region is routed around the region
    entirely; two walls sharing a gap row turn that row into a corridor clearing both at once. In
    either case the error-holding agent sails past and never meets the truth. Staggering is what
    forces the detour around the imagined wall to cross the real one.

  Four long crossings, and nobody parked.  Agents 0, 1, 2 and N-1 cross the arena; everyone else
    makes a short trip between two rooms. An exclusive region claim will not pass a transiting
    agent through a room that already holds a stationary one, so an agent that never leaves its
    starting room holds that room against all comers for the whole run. Both halves of this were
    measured: eight agents all crossing deadlocked the centre for ~200 steps against 34 balanced,
    and eight agents with the short trips degenerated to start and target in one room deadlocked
    the arena for over ten minutes against 100 seconds once they travelled.

The four long routes are the ones exp/worlds/variation.yaml and exp/worlds/scaled.yaml use, which were
measured rather than reasoned: at N=4 this reproduces variation.yaml's layout, and at N=8
scaled.yaml's. The walls differ in shape from those files -- one placement rule for every
region, where the hand-written worlds shaped each wall to its room -- so the scores will not
match theirs tile for tile.

KNOWN LIMITATION, and the reason these worlds do not yet replace the hand-authored ones in the
report. Two of the three conflicts hold here by construction: both walls sit inside one region
(so the error stays below the interface vocabulary), and room A is flagged and excluded at once.
The third does not fire. The error-holding agent is never actually driven into the true wall --
generated worlds run to ZERO hazard entries in both arms, where exp/worlds/variation.yaml's control arm
puts its agent on the wall at step 7.

The cause is understood and the fix is not: forcing the collision needs each wall to leave a
single-row gap, and a single-row gap is also the room's only through-route, which deadlocks the
arena at eight agents (measured past ten minutes against ~90 seconds). Widening the gaps clears
the deadlock and lets the agent slip past both walls diagonally. Squaring the two wants a wall
shape this module does not have -- an L, or a gap that moves along the wall's length -- rather
than another tuning of these constants.

So: use these to exercise the flow at a size, and to check that the machinery scales. Use
exp/worlds/variation.yaml and exp/worlds/scaled.yaml for the demonstration itself.

Nothing here is uploaded to the Robotarium: this module is not in build_submission.py's
AGSHEAF_MODULES, and a submission carries a configuration already baked into a module.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

#: The demonstration arena and its partition: six 4x4 rooms tiling a 12x8 grid, the `regions:`
#: block of exp/worlds/primary.yaml. The network scales; the vocabulary does not, which is the point --
#: what is being tested is whether a FIXED abstraction still absorbs a larger network's
#: disagreement.
GRID = (12, 8)
ROOMS = {
    "A": (0, 4, 3, 7), "B": (0, 0, 3, 3),
    "C": (4, 4, 7, 7), "D": (4, 0, 7, 3),
    "E": (8, 4, 11, 7), "F": (8, 0, 11, 3),
}

#: Which rooms take a wall, in order. D first: it is the room the error-holding agent crosses, so
#: the wall whose misplaced copy drives the whole diameter-length conflict has to be there. A is
#: last because the region-level conflict is seeded in it and wants free tiles.
OBSTACLE_ORDER = ("D", "E", "C", "F", "B", "A")
PRIMARY_REGION = "D"

#: Where the agents that are not crossing the arena go, as (start room, target room). They TRAVEL
#: between rooms rather than staying put, and that is a correctness constraint rather than a
#: flourish: an agent whose start and target share a room is parked there from the first tick to
#: the last, and its exclusive claim on that room then refuses entry to the long crosser arriving
#: into it. Generating eight agents that way deadlocks the arena -- measured, at over ten minutes
#: against 100 seconds for the same world with these routes. The cycle is the one
#: exp/worlds/scaled.yaml uses, which was measured before it was reasoned about.
LOCAL_ROUTES = [("B", "D"), ("F", "E"), ("A", "B"), ("E", "C"), ("D", "F"), ("C", "A")]

#: The measured long routes, as (start, target). Index 0 holds the true wall, and the last agent
#: holds it misplaced; the chain puts them at opposite ends, so the correction has to cross the
#: network's diameter. The last of these is the crossing that runs through room D.
LONG_ROUTES = [((1, 5), (10, 1)), ((0, 1), (10, 7)), ((11, 6), (1, 6))]
ERROR_ROUTE = ((11, 2), (1, 1))

MIN_AGENTS = 3

#: Above this many obstacles a generated world has been seen to deadlock. Measured at six agents,
#: seed 0: one, two and three obstacles finish in 77-84 s, four does not finish in 420. The cause
#: is the same crowding the header describes -- every wall an agent actually knows about is a
#: detour, the detours funnel onto the same open rows, and the arena jams -- and it got worse, not
#: better, when the beliefs below were fixed to cover the hazards, because agents that know about
#: a wall are the ones that route around it.
CROWDED_OBSTACLES = 3

#: How far west of the true wall the error-holding agent puts its copy. The copy has to stay
#: inside the same room, which is what keeps it invisible in the interface vocabulary.
MISPLACEMENT = 2


#: Which rows of its room each wall occupies: the true wall the top half, the false one the
#: bottom, so their gaps sit at opposite ends.
#:
#: Two tiles rather than three, and this is the unresolved compromise in this module -- see
#: "Known limitation" in the header. Three-tile walls leave each room a single gap row, which is
#: what forces the error-holding agent onto the true wall, but a single gap row is also the
#: room's only through-route, and an eight-agent world then deadlocks past ten minutes whether or
#: not anyone is placed on that row. Two-tile walls run cleanly at every size tried; the price is
#: that the two-row gaps overlap in the middle of the room, so the agent slips past the true wall
#: low and its imagined one high and never collides.
TRUE_WALL_ROWS = (2, 3)
FALSE_WALL_ROWS = (0, 1)


def wall_tiles(rect: tuple, reserved: set = frozenset()) -> list:
    """
    The true wall of a room: a vertical segment across the room's top half.

    Args:
        rect (tuple): The room as (x0, y0, x1, y1), inclusive.
        reserved (set): Tiles that must not be walled.
    """
    _, y0, _, _ = rect
    return [(wall_column(rect, reserved), y0 + row) for row in TRUE_WALL_ROWS]


def wall_column(rect: tuple, reserved: set = frozenset()) -> int:
    """
    Which column of a room its wall stands in: the third, or the second where the third would
    fall on a tile already spoken for by a start or a target.

    A room is four wide, so neither candidate is ever an outer column. That is why the corner
    tiles which starts and targets prefer stay clear of walls, and why every corner stays
    reachable from the room's open row.

    Args:
        rect (tuple): The room as (x0, y0, x1, y1), inclusive.
        reserved (set): Tiles that must not be walled.
    """
    x0, y0, _, _ = rect
    for column in (x0 + 2, x0 + 1):
        if not any((column, y0 + row) in reserved
                   for row in set(TRUE_WALL_ROWS) | set(FALSE_WALL_ROWS)):
            return column

    raise AssertionError(
        "Cannot place a wall in the room at %s: both candidate columns hold a tile already "
        "spoken for by a start or a target." % (rect,))


def false_wall_tiles(rect: tuple, reserved: set = frozenset()) -> list:
    """
    Where the error-holding agent believes the wall is: MISPLACEMENT columns west of the truth,
    inside the same room, and across the room's BOTTOM half -- the opposite end from the true
    wall, which is what makes the detour cross the truth. See the header.

    Args:
        rect (tuple): The room as (x0, y0, x1, y1), inclusive.
        reserved (set): Tiles that must not be walled, as passed to the true wall.
    """
    x0, y0, _, _ = rect
    column = wall_column(rect, reserved) - MISPLACEMENT
    assert column >= x0, \
        "The misplaced wall would fall outside the room at %s, where it would flag a different " \
        "region from the true wall and so become visible in the interface vocabulary -- which is " \
        "the one thing this construction exists to prevent." % (rect,)
    return [(column, y0 + row) for row in FALSE_WALL_ROWS]


#: How far apart two agents' squares must be, as a Chebyshev distance between tiles.
#:
#: Two is not fussiness. Corners are offered first, and the corners of ADJACENT ROOMS are
#: adjacent tiles -- room B's bottom-right (3, 0) touches room D's bottom-left (4, 0) -- so
#: preferring corners quietly packs parked robots into pairs that straddle a room boundary. On a
#: 0.2 m tile grid, two robots standing side by side with repulsion radii of 0.13-0.19 m close
#: the corridor between their rooms, and a third trying to pass simply cannot. That jams the
#: arena with the sheaf switched OFF as readily as on, so it is congestion and nothing to do with
#: the flow.
SPACING = 2


def _room_tiles(rect: tuple, blocked: set, occupied: set, rng: np.random.Generator) -> list:
    """
    Somewhere to put a start or a target in a room: corners first, and never crowding a square
    already given to another agent.

    Corners are preferred because they spread a small number of agents to the edges of the arena
    rather than clumping them, and because every corner stays reachable -- the wall occupies one
    interior column, so the columns either side of it run clear.

    Args:
        rect (tuple): The room as (x0, y0, x1, y1), inclusive.
        blocked (set): Tiles already spoken for, or walled.
        occupied (set): Squares given to agents, which the result must keep SPACING clear of.
        rng (np.random.Generator): Source of the order the non-corner tiles are offered in.
    """
    x0, y0, x1, y1 = rect
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    rest = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)
            if (x, y) not in corners]
    rng.shuffle(rest)

    free = [tile for tile in corners + rest if tile not in blocked]

    def clear_of_others(tile):
        return all(max(abs(tile[0] - x), abs(tile[1] - y)) >= SPACING for x, y in occupied)

    # Spacing is a preference rather than a guarantee: a densely populated world runs out of
    # room for it before it runs out of tiles, and a crowded world is better than none.
    return [tile for tile in free if clear_of_others(tile)] or free


def generate_world(
    num_agents: int = 4,
    num_obstacles: int = 2,
    seed: int = 0,
    experiment_name: Optional[str] = None
) -> dict:
    """
    A complete experiment configuration, in the shape exp/run.py reads and exp/worlds/variation.yaml is
    written in.

    Args:
        num_agents (int): How many agents. The chain's diameter is one less than this.
        num_obstacles (int): How many wall segments, one per room, in OBSTACLE_ORDER.
        seed (int): Seeds both the run and the choice of the tiles not fixed by construction.
        experiment_name (Optional[str]): What the run calls itself in its log.
    """
    assert num_agents >= MIN_AGENTS, \
        "Cannot generate a world for %d agents. The seeded conflicts need at least %d: one agent " \
        "holding the wall correctly, one holding it wrong at the other end of the chain, and one " \
        "dissenting on the contested room." % (num_agents, MIN_AGENTS)
    assert 1 <= num_obstacles <= len(OBSTACLE_ORDER), \
        "Cannot generate a world with %d obstacles. There is one room for each, so between 1 and " \
        "%d." % (num_obstacles, len(OBSTACLE_ORDER))

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------------------------------
    # Walls. The primary one is in D whatever else is asked for, because the diameter-length
    # conflict is built on it.
    # ------------------------------------------------------------------------------------------
    # The fixed crossings are chosen before the walls, so the walls have to keep off them: a
    # target that is also a hazard is scored as both, and an agent sent to one can never stop
    # paying for arriving.
    reserved = {tile for route in LONG_ROUTES + [ERROR_ROUTE] for tile in route}

    walled_rooms = list(OBSTACLE_ORDER[:num_obstacles])
    walls = {room: wall_tiles(ROOMS[room], reserved) for room in walled_rooms}
    primary_wall = walls[PRIMARY_REGION]
    misplaced_wall = false_wall_tiles(ROOMS[PRIMARY_REGION], reserved)

    unsafe = [tile for room in walled_rooms for tile in walls[room]]
    blocked = set(unsafe)

    # ------------------------------------------------------------------------------------------
    # Routes. Agents 0..2 and the last agent cross the arena; the rest stay local, spread over the
    # rooms that carry no through traffic.
    # ------------------------------------------------------------------------------------------
    starts: dict = {}
    assignments: dict = {}

    # The last agent takes the error route whatever the size, because it is the one holding the
    # misplaced wall; the others fill the remaining crossings from the front.
    routes = {num_agents - 1: ERROR_ROUTE}
    routes.update(zip(range(min(len(LONG_ROUTES), num_agents - 1)), LONG_ROUTES))

    long_agents = sorted(routes)
    for agent, (start, target) in routes.items():
        starts[agent] = start
        assignments[agent] = target
        blocked |= {start, target}

    # Everything an agent stands on, start or finish, kept SPACING apart so that no two parked
    # robots close a corridor between them.
    occupied = set(starts.values()) | set(assignments.values())

    local_agents = [agent for agent in range(num_agents) if agent not in long_agents]
    for index, agent in enumerate(local_agents):
        from_room, to_room = LOCAL_ROUTES[index % len(LOCAL_ROUTES)]
        for room, where in ((from_room, starts), (to_room, assignments)):
            available = _room_tiles(ROOMS[room], blocked, occupied, rng)
            assert available, \
                "Cannot generate a world for %d agents: room %s has no free tile left for agent " \
                "%d. Fewer agents, or fewer obstacles, will fit." % (num_agents, room, agent)
            where[agent] = available[0]
            blocked.add(available[0])
            occupied.add(available[0])

    # ------------------------------------------------------------------------------------------
    # The contested tile of the region-level conflict: a free tile of room A that agents 0 and 1
    # believe holds a target and agent 2 believes is hazardous, so A is flagged and excluded at
    # once. Chosen after the routes so it cannot land on a start, a target or a wall.
    # ------------------------------------------------------------------------------------------
    # No spacing here: these are tiles agents hold BELIEFS about, not squares anybody stands on,
    # so crowding costs nothing.
    contested = _room_tiles(ROOMS["A"], blocked, set(), rng)
    assert len(contested) >= 2, \
        "Cannot generate a world: room A has no free tiles left to seed the region-level " \
        "conflict on."
    contested_target, contested_hazard = contested[0], contested[1]

    # ------------------------------------------------------------------------------------------
    # Beliefs. Agent 0 alone holds the true wall; the last agent holds it two columns west and
    # believes the true column clear. The secondary walls are DEALT OUT round-robin among the
    # agents between them, one tile each in turn.
    #
    # Dealing them out rather than handing a couple of agents the same two tiles is the point:
    # what the network holds between it has to COVER the hazards, or communicating cannot help.
    # An earlier version gave each middle agent secondary[:2] or secondary[-2:], which left four
    # of eight hazard tiles known to nobody in a four-obstacle world, and a measured run walked
    # into three of them. No amount of fusion invents a fact the network does not have; the whole
    # question a demonstration asks is whether the network can assemble one it holds in pieces.
    # ------------------------------------------------------------------------------------------
    secondary = [tile for room in walled_rooms[1:] for tile in walls[room]]
    middle = list(range(1, num_agents - 1))
    dealt = {agent: secondary[index::len(middle)] for index, agent in enumerate(middle)} \
        if middle else {}

    beliefs: dict = {}
    beliefs[0] = {"unsafe": [list(tile) for tile in primary_wall],
                  "target": [list(contested_target), list(assignments[0])]}
    # Agent 0 is confidently wrong about a tile somebody else holds correctly, so the fusion has
    # a contradiction to settle and not merely gaps to fill.
    if secondary:
        beliefs[0]["safe"] = [list(secondary[-1])]

    for agent in middle:
        held: dict = {"target": [list(assignments[agent])]}
        if agent == 1:
            held["target"] = [list(contested_target), list(assignments[agent])]
        if agent == 2:
            held["unsafe"] = [list(contested_target), list(contested_hazard)]
        if dealt.get(agent):
            held["unsafe"] = held.get("unsafe", []) + [list(tile) for tile in dealt[agent]]
        beliefs[agent] = held

    beliefs[num_agents - 1] = {
        "unsafe": [list(tile) for tile in misplaced_wall],
        "safe": [list(tile) for tile in primary_wall],
        "target": [list(assignments[num_agents - 1])],
    }

    # ------------------------------------------------------------------------------------------
    # The chain of the report: diameter N-1, with the truth-holder and the error-holder at its
    # two ends.
    # ------------------------------------------------------------------------------------------
    edges = [[agent, agent + 1] for agent in range(num_agents - 1)]

    return {
        "experiment_name": experiment_name or "generated_a%d_o%d" % (num_agents, num_obstacles),
        "seed": seed,
        "gridworld": {
            "grid_width": GRID[0], "grid_height": GRID[1], "num_agents": num_agents,
            "show_figure": True, "asynchronous_moves": False, "move_schedule": None,
            "max_distance_repulsion": 0.19, "min_distance_repulsion": 0.13,
            "arrival_distance": 0.045,
        },
        "goals": {"max_steps": 100},
        "starts": {agent: list(tile) for agent, tile in sorted(starts.items())},
        "assignments": {agent: list(tile) for agent, tile in sorted(assignments.items())},
        "communication": {"edges": edges},
        "beliefs": {
            "views": ["ground_truth", [0, 1], "ground_truth", [num_agents - 2, num_agents - 1]],
            "switch_view_every": 2,
            "observe_on_arrival": True,
            "agents": beliefs,
        },
        "ground_truth": {
            "default": "safe",
            "unsafe": [list(tile) for tile in unsafe],
            "target": [list(tile) for _, tile in sorted(assignments.items())],
        },
        "regions": {room: {"rect": list(rect)} for room, rect in ROOMS.items()},
        "sheaf": {"enabled": True, "sweeps_per_step": 1},
        "rendering": {"show_sheaf_status": False},
    }
