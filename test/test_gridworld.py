"""
The planner-side notions the sheaf layer is written against.

`route_corridor` is the one the mission monitor rests on: an agent's reliance is
the set of squares its *plan* needs passable, so a contract over that plan is
only as honest as the corridor is faithful to what the planner would actually
do. These tests pin the two together -- the corridor detours where
`choose_belief_aware_step` detours, and crosses where it has no choice.
"""

import numpy as np

from agsheaf.gridworld import (DEFAULT_TILE_COSTS, choose_belief_aware_step,
                               route_corridor)


GRID = (5, 3)


def labels(**by_label):
    """Every tile of GRID "safe", then the named tiles relabelled."""
    grid_width, grid_height = GRID
    tile_labels = {(x, y): "safe" for x in range(grid_width) for y in range(grid_height)}
    for label, tiles in by_label.items():
        for tile in tiles:
            tile_labels[tile] = label
    return tile_labels


def is_a_walk(corridor, start):
    """Whether the corridor steps one tile at a time from `start`, never repeating."""
    walk = [start] + list(corridor)
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(walk, walk[1:])]
    return (all(abs(dx) + abs(dy) == 1 for dx, dy in steps)
            and len(set(walk)) == len(walk))


class TestRouteCorridor:
    def test_open_grid_takes_the_straight_run(self):
        corridor = route_corridor((0, 0), (4, 0), GRID, labels())
        assert corridor == [(1, 0), (2, 0), (3, 0), (4, 0)]

    def test_excludes_the_start_and_ends_at_the_goal(self):
        corridor = route_corridor((0, 0), (4, 2), GRID, labels())
        assert (0, 0) not in corridor
        assert corridor[-1] == (4, 2)
        assert is_a_walk(corridor, (0, 0))

    def test_standing_on_the_goal_relies_on_nothing(self):
        assert route_corridor((2, 1), (2, 1), GRID, labels()) == []

    def test_detours_around_a_believed_wall(self):
        """A gap in the wall is worth the longer way round, at 30 against 1."""
        wall = [(2, 0), (2, 1)]
        corridor = route_corridor((0, 0), (4, 0), GRID, labels(unsafe=wall))
        assert not set(corridor) & set(wall)
        assert (2, 2) in corridor
        assert is_a_walk(corridor, (0, 0))

    def test_crosses_a_sealed_wall_exactly_once(self):
        """
        Unsafe is expensive but finite, so a goal walled off is reached rather
        than given up on -- and the corridor names which square the plan spends.
        That square is exactly what the reliance contract is about.
        """
        wall = [(2, 0), (2, 1), (2, 2)]
        corridor = route_corridor((0, 0), (4, 0), GRID, labels(unsafe=wall))
        assert corridor[-1] == (4, 0)
        assert len(set(corridor) & set(wall)) == 1

    def test_prefers_a_known_route_to_an_unexplored_one(self):
        """
        unknown is priced above safe, so a longer known way round wins: two
        unexplored tiles cost 3 each against a detour of four safe ones.
        """
        unexplored = [(1, 0), (2, 0)]
        corridor = route_corridor((0, 0), (3, 0), GRID, labels(unknown=unexplored))
        assert not set(corridor) & set(unexplored)
        assert corridor[-1] == (3, 0)

    def test_first_tile_is_the_step_the_planner_commits_to(self):
        """
        The corridor is the plan, so its head must be the move
        `choose_belief_aware_step` makes from the same beliefs; a corridor that
        drifted from the planner would have the monitor guarding a route nobody
        drives.
        """
        tile_labels = labels(unsafe=[(2, 0), (2, 1)])
        start, goal = (0, 0), (4, 0)
        corridor = route_corridor(start, goal, GRID, tile_labels)

        step = choose_belief_aware_step(np.array([[start[0]], [start[1]]]),
                                        np.array([[goal[0]], [goal[1]]]),
                                        GRID, {0: tile_labels},
                                        tile_costs=DEFAULT_TILE_COSTS)
        assert (int(step[0, 0]), int(step[1, 0])) == corridor[0]
