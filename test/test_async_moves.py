"""
Planning and booking tiles when agents move asynchronously.

The synchronous planner keeps two agents off a tile by planning them in sequence against
one array of positions, every agent at rest on a tile while it does so. None of that holds
once agents move when they please: an agent in transit is on no tile in particular, and
there is no moment at which every agent is somewhere to be read off. `TileReservations`
replaces the observation with a booking, and `choose_agent_step` is the planner's body
made callable for one agent at a time.

The load-bearing claim, tested first, is that pulling that body out changed nothing: the
synchronous planner is a loop over it and must still choose exactly what it chose before.
"""

import numpy as np
import pytest

from agsheaf.gridworld import (DEFAULT_TILE_COSTS, TileReservations, choose_agent_step,
                               choose_belief_aware_step, score_agent_step, score_robot_step)


GRID = (5, 3)


def labels(**by_label):
    """Every tile of GRID "safe", then the named tiles relabelled."""
    grid_width, grid_height = GRID
    tile_labels = {(x, y): "safe" for x in range(grid_width) for y in range(grid_height)}
    for label, tiles in by_label.items():
        for tile in tiles:
            tile_labels[tile] = label
    return tile_labels


def for_agents(number_of_agents, **by_label):
    return {i: labels(**by_label) for i in range(number_of_agents)}


class TestTheExtractionChangedNothing:
    """`choose_belief_aware_step` is now a loop over `choose_agent_step`."""

    @pytest.mark.parametrize("starts, goals", [
        ([[0, 4], [0, 2]], [[4, 0], [2, 0]]),
        ([[0, 1, 2], [0, 0, 0]], [[4, 4, 4], [0, 1, 2]]),
        ([[2, 2], [0, 2]], [[2, 2], [2, 0]]),          # head on, one tile apart
        ([[0, 1], [1, 1]], [[0, 1], [1, 1]]),          # both already home
    ])
    def test_the_round_is_the_loop(self, starts, goals):
        coords, targets = np.array(starts), np.array(goals)
        number_of_agents = coords.shape[1]
        tile_labels = for_agents(number_of_agents, unsafe=[(2, 1)])

        whole_round = choose_belief_aware_step(coords, targets, GRID, tile_labels)

        # The same thing by hand: nearest to its goal first, each planned against what the
        # ones before it committed to, and the settled set fixed at what the round began with
        goal_distances = np.linalg.norm(coords - targets, axis=0)
        settled = frozenset(i for i in range(number_of_agents) if goal_distances[i] == 0)
        by_hand = coords.copy()
        for i in np.argsort(goal_distances).tolist():
            if goal_distances[i] == 0:
                continue
            by_hand[:, i] = choose_agent_step(
                i, by_hand, targets, GRID, tile_labels,
                settled_tiles=frozenset((int(by_hand[0, other]), int(by_hand[1, other]))
                                        for other in settled if other != i))

        assert np.array_equal(whole_round, by_hand)

    def test_the_agent_scorer_is_the_round_scorer(self):
        ground_truth = {(x, y): "safe" for x in range(5) for y in range(3)}
        ground_truth[(2, 1)] = "unsafe"
        ground_truth[(4, 0)] = "target"
        coords = np.array([[4, 2], [0, 1]])

        by_round = np.zeros(2, dtype=bool)
        round_scores = score_robot_step(coords, ground_truth, by_round)

        by_agent_flags = np.zeros(2, dtype=bool)
        by_agent = [score_agent_step(i, (int(coords[0, i]), int(coords[1, i])),
                                     ground_truth, by_agent_flags)
                    for i in range(2)]

        assert list(round_scores) == by_agent
        assert list(by_round) == list(by_agent_flags)


class TestReservations:
    def test_an_idle_agent_holds_its_tile(self):
        reservations = TileReservations(np.array([[0, 4], [0, 2]]))
        assert reservations.blocked_for(0) == frozenset()
        assert reservations.in_transit() == frozenset()

    def test_a_departing_agent_holds_both_tiles(self):
        reservations = TileReservations(np.array([[0, 4], [0, 2]]))
        reservations.depart(0, (1, 0))

        # The tile it left is blocked for everyone else, the tile it took is its position
        assert reservations.blocked_for(1) == {(0, 0)}
        assert tuple(reservations.destinations[:, 0]) == (1, 0)
        assert reservations.in_transit() == {0}

    def test_arriving_releases_the_tile_behind_it(self):
        reservations = TileReservations(np.array([[0, 4], [0, 2]]))
        reservations.depart(0, (1, 0))
        reservations.arrive(0)

        assert reservations.blocked_for(1) == frozenset()
        assert reservations.in_transit() == frozenset()

    def test_standing_still_books_nothing(self):
        """Departing for the tile already held is not a move."""
        reservations = TileReservations(np.array([[0, 4], [0, 2]]))
        reservations.depart(0, (0, 0))
        assert reservations.blocked_for(1) == frozenset()

    def test_an_agent_is_not_blocked_by_its_own_reservation(self):
        reservations = TileReservations(np.array([[0, 4], [0, 2]]))
        reservations.depart(0, (1, 0))
        assert reservations.blocked_for(0) == frozenset()


class TestPlanningAroundAgentsInTransit:
    def test_an_agent_will_not_enter_the_tile_another_is_leaving(self):
        """
        The rule with no synchronous counterpart. Agent 1 would step onto (1,0) but agent 0
        is driving out of it and has not arrived, so the tile is not free.
        """
        coords = np.array([[2, 1], [0, 1]])          # agent 0 heading to (2,0), agent 1 at (1,1)
        targets = np.array([[4, 1], [0, 0]])
        tile_labels = for_agents(2)

        reservations = TileReservations(np.array([[1, 1], [0, 1]]))
        reservations.depart(0, (2, 0))

        free = choose_agent_step(1, coords, targets, GRID, tile_labels)
        blocked = choose_agent_step(1, coords, targets, GRID, tile_labels,
                                    blocked_tiles=reservations.blocked_for(1))

        assert tuple(free) == (1, 0)
        assert tuple(blocked) != (1, 0)

    def test_standing_still_is_never_blocked(self):
        """
        An agent boxed in by reservations holds position. Refusing it its own tile would
        leave it no move at all, and it is already there.
        """
        coords = np.array([[1], [1]])
        targets = np.array([[4], [1]])
        tile_labels = for_agents(1)

        every_neighbour = frozenset([(0, 1), (2, 1), (1, 0), (1, 2)])
        held = choose_agent_step(0, coords, targets, GRID, tile_labels,
                                 blocked_tiles=every_neighbour)
        assert tuple(held) == (1, 1)

    def test_an_agent_waits_for_a_reserved_tile_rather_than_rounding_it(self):
        """
        A reservation is not terrain. The tile an agent is driving out of clears in a moment,
        so a reserved tile is worth waiting behind where a settled agent is worth walking
        around -- the same distinction the synchronous planner already drew between an agent
        still moving and one parked on its goal.

        This falls out of leaving reservations out of the cost-to-go rather than being coded
        for: the route through the reserved tile is still the cheapest one, so standing still
        on it beats stepping off it. Feeding reservations into the cost-to-go instead would
        send agents on detours around tiles that free up before they could finish one.
        """
        coords = np.array([[0], [1]])
        targets = np.array([[4], [1]])
        tile_labels = for_agents(1)

        step = choose_agent_step(0, coords, targets, GRID, tile_labels,
                                 blocked_tiles=frozenset([(1, 1)]))
        assert tuple(step) == (0, 1)

    def test_settled_agents_are_still_terrain(self):
        """
        The one thing that must survive from the synchronous planner: an agent parked on its
        own goal is routed around, not queued behind, since it will never clear.
        """
        coords = np.array([[0, 1], [1, 1]])
        targets = np.array([[4, 1], [1, 1]])         # agent 1 is home at (1,1)
        tile_labels = for_agents(2)

        step = choose_agent_step(0, coords, targets, GRID, tile_labels)
        assert tuple(step) in {(0, 0), (0, 2)}
