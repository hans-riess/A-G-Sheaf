"""
Firing sets at the two levels a run has them.

`GridWorld.update(firing=...)` is the lower one: who receives a velocity on a given 0.033 s
control tick. It decides how the robots are driven and not where they go, which is what most
of this file pins down -- a live schedule changes the trajectory, never the tiles the robots
end on or whether they get there at all. Nothing reads it from configuration; it is the API
for a caller modelling a controller that misses updates, and it is what `hold_projection`'s
"command zero rather than say nothing" rests on.

`parse_move_schedule` is the upper one, and the one the experiments read: whose turn it is to
claim the next tile. See `test_async_moves` for the planner underneath it.

These run the simulator headless (`show_figure=False`, `sim_in_real_time=False`), so a tick
is a physics step and nothing else.
"""

import random

import numpy as np
import pytest

from agsheaf.gridworld import GridWorld, StarvedRobotError, parse_move_schedule
from agsheaf.sheaf import random_firing, round_robin


GRID = (5, 3)

# A drive is a few hundred ticks; anything past this is a hang, not a slow robot
TICK_BUDGET = 3000


@pytest.fixture
def world():
    """Two robots a clear distance apart, headless. Fresh per test -- each builds a simulator."""
    return GridWorld(
        number_of_robots=2,
        grid_width_height=GRID,
        show_figure=False,
        sim_in_real_time=False,
        initial_coordinates=np.array([[0, 4], [0, 2]]),
    )


def drive(world, firing_of=None, budget=TICK_BUDGET):
    """
    Ticks until every robot has arrived, or gives up. Deliberately not
    `update_until_robots_done_moving`: a bounded loop reports a hang as a failed assertion
    rather than by never returning.

    `firing_of(tick)` is the firing set for that tick; None commands everyone.
    """
    for tick in range(budget):
        if world.robots_done_moving():
            return tick
        world.update(firing=None if firing_of is None else firing_of(tick))
    assert False, "robots still moving after %d ticks" % budget


def tiles(world):
    """Where the robots actually are, to the nearest tile."""
    positions = world.robot_poses[0:2, :]
    origin = world.bottom_left_tile_pos_xy
    return np.rint((positions - origin) / world.tile_width).astype(int)


class TestFiringSets:
    def test_no_firing_set_is_the_synchronous_case(self, world):
        """`firing=None` is what every existing caller passes, and must not have changed."""
        world.move_to_tile(0, np.array([1, 0]))
        world.move_to_tile(1, np.array([3, 2]))
        before = world.robot_poses.copy()
        world.update()
        assert not np.allclose(world.robot_poses, before)

    def test_a_held_robot_is_commanded_zero(self, world):
        """
        The `hold` reading of "no command". Not commanding at all would leave the simulator
        driving the robot on its last velocity, which is the one thing this must not do.
        """
        world.move_to_tile(0, np.array([1, 0]))
        world.move_to_tile(1, np.array([3, 2]))

        world.update()                      # both get a real command
        world.update(firing={0})            # robot 1 is held

        velocities = world.robotarium._velocities
        assert np.allclose(velocities[:, 1], 0.0)
        assert not np.allclose(velocities[:, 0], 0.0)

    def test_a_held_robot_does_not_move(self, world):
        world.move_to_tile(0, np.array([1, 0]))
        world.move_to_tile(1, np.array([3, 2]))
        world.update()

        held_before = world.robot_poses[:, 1].copy()
        for _ in range(20):
            world.update(firing={0})

        assert np.allclose(world.robot_poses[:, 1], held_before)

    def test_holding_everyone_moves_nobody(self, world):
        world.move_to_tile(0, np.array([1, 0]))
        world.move_to_tile(1, np.array([3, 2]))
        world.update()

        before = world.robot_poses.copy()
        for _ in range(10):
            world.update(firing=set())
        assert np.allclose(world.robot_poses, before)

    def test_a_firing_set_outside_the_robots_is_rejected(self, world):
        with pytest.raises(AssertionError, match="firing"):
            world.update(firing={7})


class TestArrivalIsIndependentOfCommanding:
    def test_round_robin_terminates(self, world):
        """
        The regression. Arrival used to be detected inside the command generators, so a
        robot the schedule skipped was never asked whether it had arrived: under any
        partial schedule `robots_done_moving` could stay False forever and the drive loop
        in the Robotarium template would spin. `drive` fails on its budget rather than
        hanging, which is what makes this runnable at all.
        """
        world.move_to_tile(0, np.array([1, 0]))
        world.move_to_tile(1, np.array([3, 2]))
        drive(world, firing_of=lambda tick: {tick % 2})
        assert world.robots_done_moving()

    def test_a_robot_arriving_while_held_is_still_noticed(self, world):
        """Arrival is a fact about where a robot is, not about it having been commanded."""
        world.move_to_tile(0, np.array([1, 0]))
        for _ in range(TICK_BUDGET):
            if world.robot_done_moving(0):
                break
            world.update(firing={0})

        # Park it, hold it forever, and it stays arrived
        for _ in range(50):
            world.update(firing=set())
        assert world.robot_done_moving(0)


class TestScheduleChangesTheTrajectoryNotTheDestination:
    @pytest.mark.parametrize("name, firing_of", [
        ("round_robin", lambda tick: {tick % 2}),
        ("one_lags", lambda tick: {0} if tick % 3 else {0, 1}),
    ])
    def test_the_robots_end_on_the_same_tiles(self, name, firing_of):
        targets = np.array([[1, 3], [0, 2]])

        def settle(firing_of):
            world = GridWorld(number_of_robots=2, grid_width_height=GRID, show_figure=False,
                              sim_in_real_time=False,
                              initial_coordinates=np.array([[0, 4], [0, 2]]))
            for i in range(2):
                world.move_to_tile(i, targets[:, i])
            ticks = drive(world, firing_of=firing_of)
            return tiles(world), ticks

        synchronous, sync_ticks = settle(None)
        asynchronous, async_ticks = settle(firing_of)

        assert np.array_equal(synchronous, targets)
        assert np.array_equal(asynchronous, targets)
        # Holding a robot can only slow it down, never speed it up
        assert async_ticks >= sync_ticks

    def test_the_sheaf_schedules_drive_robots_unchanged(self, world):
        """
        `round_robin` and `random_firing` take a sequence of nodes and know nothing about
        what the nodes are, so robot indices work as well as sheaf vertices do. One notion
        of a schedule, both layers.
        """
        world.move_to_tile(0, np.array([1, 0]))
        world.move_to_tile(1, np.array([3, 2]))

        schedule = round_robin(range(2))
        for _ in range(TICK_BUDGET):
            if world.robots_done_moving():
                break
            world.update(firing=next(schedule))
        assert world.robots_done_moving()

    def test_update_until_robots_done_moving_takes_a_schedule(self):
        world = GridWorld(number_of_robots=2, grid_width_height=GRID, show_figure=False,
                          sim_in_real_time=False,
                          initial_coordinates=np.array([[0, 4], [0, 2]]))
        world.move_to_tile(0, np.array([1, 0]))
        world.move_to_tile(1, np.array([3, 2]))
        world.update_until_robots_done_moving(
            firing_sequence=lambda: random_firing(range(2), random.Random(0)))
        assert world.robots_done_moving()


class TestTheMoveSchedule:
    """
    `gridworld.move_schedule` -- whose turn it is to claim a tile. The one schedule the
    experiments read; `GridWorld.update`'s firing set, tested above, is a level below it and
    changes only how the robots are driven.
    """

    def test_null_lets_everyone_move(self):
        assert parse_move_schedule(None, 3) is None

    def test_it_is_the_sheaf_firing_sequences(self):
        """The same generators that drive the Laplacian, over agent indices."""
        assert next(parse_move_schedule("round_robin", 3)) == next(round_robin(range(3)))

    def test_random_draws_from_its_own_stream(self):
        """
        Seeding a generator of its own rather than drawing from the module-level `random`
        keeps the number of turns taken from perturbing everything else a run samples.
        """
        random.seed(0)
        before = random.random()
        random.seed(0)
        schedule = parse_move_schedule("random", 3, seed=0)
        for _ in range(50):
            next(schedule)
        assert random.random() == before

    def test_round_robin_gives_the_turn_to_one_agent(self):
        schedule = parse_move_schedule("round_robin", 3)
        turns = [next(schedule) for _ in range(6)]
        assert all(len(turn) == 1 for turn in turns)
        assert [sorted(turn)[0] for turn in turns] == [0, 1, 2, 0, 1, 2]

    def test_random_is_reproducible_from_the_seed(self):
        assert (next(parse_move_schedule("random", 4, seed=3))
                == next(parse_move_schedule("random", 4, seed=3)))

    def test_an_unknown_schedule_is_rejected(self):
        with pytest.raises(AssertionError, match="move_schedule"):
            parse_move_schedule("whenever", 3)


class TestStarvation:
    def test_a_starved_robot_is_diagnosed(self):
        """
        A schedule that never fires robot 1 stalls the run. Spinning to the tick budget and
        blaming the physics would be the wrong diagnosis -- name the robot, as
        `sheaf.LivenessError` names the starved nodes.
        """
        world = GridWorld(number_of_robots=2, grid_width_height=GRID, show_figure=False,
                          sim_in_real_time=False, starvation_ticks=30,
                          initial_coordinates=np.array([[0, 4], [0, 2]]))
        world.move_to_tile(0, np.array([1, 0]))
        world.move_to_tile(1, np.array([3, 2]))

        with pytest.raises(StarvedRobotError) as excinfo:
            for _ in range(200):
                world.update(firing={0})
        assert "[1]" in str(excinfo.value)

    def test_an_arrived_robot_never_starves(self):
        """It wants nothing, so going uncommanded says nothing about the schedule."""
        world = GridWorld(number_of_robots=2, grid_width_height=GRID, show_figure=False,
                          sim_in_real_time=False, starvation_ticks=30,
                          initial_coordinates=np.array([[0, 4], [0, 2]]))
        for _ in range(100):
            world.update(firing=set())
        assert world.robots_done_moving()

    def test_waiting_is_not_charged_against_the_next_tile(self):
        """
        A robot that sat parked and uncommanded has not been starved by the schedule, so the
        wait must not count against the move it is next given -- otherwise the first tick
        after a long idle stretch raises.
        """
        world = GridWorld(number_of_robots=2, grid_width_height=GRID, show_figure=False,
                          sim_in_real_time=False, starvation_ticks=30,
                          initial_coordinates=np.array([[0, 4], [0, 2]]))
        for _ in range(100):
            world.update(firing=set())

        world.move_to_tile(0, np.array([1, 0]))
        world.update(firing={0})
        world.update(firing={0})

    def test_the_check_can_be_switched_off(self):
        world = GridWorld(number_of_robots=2, grid_width_height=GRID, show_figure=False,
                          sim_in_real_time=False, starvation_ticks=None,
                          initial_coordinates=np.array([[0, 4], [0, 2]]))
        world.move_to_tile(1, np.array([3, 2]))
        for _ in range(100):
            world.update(firing={0})
        assert not world.robot_done_moving(1)
