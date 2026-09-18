from __future__ import annotations
import heapq
import random
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from typing import Iterable, Literal, Optional
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import PathPatch, FancyArrowPatch

from rps.robotarium import Robotarium

from .sheaf import random_firing, round_robin


# Potential-field strengths, shared by the command generators and the reachability check.
# The distances they act over are per-GridWorld, since they have to suit the tile width.
MAX_DISTANCE_ATTRACTION = 0.3   # [m] Robots pull towards their target at full effort beyond this, linearly inside it
MAX_MAGNITUDE_ATTRACTION = 0.2  # [m/s] Maximum magnitude of an attractive force
MAX_MAGNITUDE_REPULSION = 0.4   # [m/s] Maximum magnitude of a repulsive force

#: The schedules parse_move_schedule accepts, besides None for letting every agent move
MOVE_SCHEDULES = ("round_robin", "random")

#: What is added to an agent's index when it is *drawn*. Indices stay 0-based everywhere they
#: are reasoned about -- configuration, logs, the sheaf's own vertices -- because that is what
#: they are; this is only what a human reads off the floor. Set from rendering.agent_label_offset.
AGENT_LABEL_OFFSET = [0]


def agent_label(robot_index) -> str:
    """
    How an agent is written where a person reads it.

    Args:
        robot_index (int): The agent's index, 0-based as everything else is.
    """
    return str(int(robot_index) + AGENT_LABEL_OFFSET[0])


def set_agent_label_offset(offset: int) -> None:
    """
    Sets what is added to an agent's index when it is drawn. 1 numbers the agents from one on
    the floor while leaving every index in the configuration and the logs alone.

    Args:
        offset (int): Added to the index for display only.
    """
    assert isinstance(offset, int) and not isinstance(offset, bool), "Failed to set the agent label offset. Expected an integer, received %r." % (offset,)
    AGENT_LABEL_OFFSET[0] = offset


#: Every entry the 'beliefs' configuration block may carry, checked by parse_display_views
BELIEF_KEYS = frozenset({"agents", "default", "follow_moving_agent", "mark_other_agents_unsafe",
                         "observe_on_arrival", "switch_view_every", "tour_views_every", "view",
                         "views"})


class StarvedRobotError(RuntimeError):
    """
    A firing sequence left a robot uncommanded for so long that the run is better stopped
    than continued. The motion counterpart of `agsheaf.sheaf.LivenessError`: Riess and
    Ghrist (2022), Assumption 2 asks every node to fire infinitely often, and a schedule
    that starves a robot stalls the drive loop for the same reason it robs the contract
    flow of its convergence guarantee. Raising names the robots; the alternative is a
    caller waiting on `robots_done_moving` forever with nothing on screen to explain it.
    """


class GridWorld:
    """
    Class to run a grid world simulation on the Robotarium
    """

    def __init__(
        self, 
        number_of_robots: int = -1,
        grid_width_height: tuple[int, int] = (10,5),
        show_figure: bool = True,
        show_arrows: bool = False,
        sim_in_real_time: bool = True,
        initial_coordinates: NDArray[np.integer] = np.array([]),
        collision_avoidance_type: Literal["Off", "PotentialField"] = "PotentialField",
        grid_safety_gap: float = 0.2,
        min_distance_repulsion: float = 0.2,
        max_distance_repulsion: float = 0.35,
        starvation_ticks: Optional[int] = 500,
        robot_close_to_target_distance: float = 0.05
    ):
        """
        Instantiate the grid world simulation. 
        Each tile in the grid has an (X,Y) coordinate, where X and Y are integers.
        The bottom-left tile has coordinate (0,0).

        Args:
            number_of_agents (int): The number of robots in the simulation.
            grid_width_height (tuple[int, int]): The width and height of the grid.
            show_figure (bool): Whether to display the simulation figure window.
            show_arrows (bool): Whether to display agent controls arrows.
            sim_in_real_time (bool): Whether to run the simulation in real time.
            initial_coordinates (np.ndarray[int]): A 2xN numpy array specifying the (X,Y) coordinates of the N robots.
            collision_avoidance_type (str): A string specifying the method of collision avoidance to use, either "Off" or "PotentialField".
            grid_safety_gap (float): Minimum gap in meters between the grid and the arena walls.
            min_distance_repulsion (float): Distance in meters within which repulsion is at full strength.
            max_distance_repulsion (float): Distance in meters beyond which nothing repels.
            starvation_ticks (Optional[int]): How many consecutive updates a robot that is still
                moving may go without a command before StarvedRobotError is raised. None disables
                the check. Only relevant when update() is given a firing set; commanding every
                robot every tick, the default, can never starve one.
            robot_close_to_target_distance (float): Metres from a tile centre at which a robot
                counts as having arrived. The main lever on how long a move takes; see where it
                is stored below.

        The repulsion distances are tied to the tile width, not just to the robots: a robot
        is repelled by the robot on the next tile over whenever a tile is narrower than
        max_distance_repulsion, and by the wall whenever the outermost tiles sit inside that
        band. Either one, if strong enough, cancels the attraction that brings a robot the
        last few centimeters onto its tile, and it never arrives. A finer grid therefore
        needs these scaled down with it; _assert_grid_is_reachable checks the pairing.
        """

        # Check argument validity
        assert isinstance(number_of_robots,    int),          "Failed to create GridWorld. The argument number_of_robots must have type 'int'. Received type %r."              % type(number_of_robots).__name__
        assert isinstance(grid_width_height,   tuple),        "Failed to create GridWorld. The argument grid_width_height must have type 'tuple'. Received type %r."           % type(grid_width_height).__name__
        assert isinstance(show_figure,         bool),         "Failed to create GridWorld. The argument show_figure must have type 'bool'. Received type %r."                  % type(show_figure).__name__
        assert isinstance(sim_in_real_time,    bool),         "Failed to create GridWorld. The argument sim_in_real_time must have type 'bool'. Received type %r."             % type(sim_in_real_time).__name__
        assert isinstance(initial_coordinates, np.ndarray),   "Failed to create GridWorld. The argument initial_coordinates must have type 'numpy.ndarray'. Received type %r." % type(initial_coordinates).__name__
        assert isinstance(collision_avoidance_type, str),     "Failed to create GridWorld. The argument collision_avoidance_type must have type 'str'. Received type %r."      % type(collision_avoidance_type).__name__

        assert (number_of_robots >= 0 and number_of_robots <= 50), "Failed to create GridWorld. The argument number_of_robots must be >=0 and <=50. Received %g." % number_of_robots
        if initial_coordinates.size > 0:
            assert initial_coordinates.shape == (2, number_of_robots), "Failed to create GridWorld. The argument initial_coordinates must have shape 3xN, where N is the number of robots used. Expected shape 2x%g, received shape %gx%g." % (number_of_robots, initial_coordinates.shape[0], initial_coordinates.shape[1])

        assert collision_avoidance_type in ["Off", "PotentialField"], "Failed to create GridWorld. The argument collision_avoidance_type must be either \"Off\" or \"PotentialField\". Received \"%s\"." % collision_avoidance_type

        # Save parameters
        self.number_of_robots = number_of_robots
        self.grid_width = grid_width_height[0]
        self.grid_height = grid_width_height[1]
        self.show_figure = show_figure
        self.show_arrows = show_arrows
        self.sim_in_real_time = sim_in_real_time
        self.initial_coordinates = initial_coordinates
        self.collision_avoidance_type = collision_avoidance_type
        self.grid_safety_gap = grid_safety_gap
        self.min_distance_repulsion = min_distance_repulsion
        self.max_distance_repulsion = max_distance_repulsion
        self.starvation_ticks = starvation_ticks

        assert starvation_ticks is None or starvation_ticks > 0, "Failed to create GridWorld. The argument starvation_ticks must be positive, or None to disable the check. Received %r." % starvation_ticks
        assert robot_close_to_target_distance > 0, "Failed to create GridWorld. The argument robot_close_to_target_distance must be positive, received %g." % robot_close_to_target_distance

        assert 0 <= min_distance_repulsion <= max_distance_repulsion, "Failed to create GridWorld. The repulsion distances must satisfy 0 <= min_distance_repulsion <= max_distance_repulsion, received %g and %g." % (min_distance_repulsion, max_distance_repulsion)

        # Define robotarium boundaries
        self.robotarium_width = 3.2
        self.robotarium_height = 2.0

        # Calculate grid positions
        safe_robotarium_width = self.robotarium_width - 2*grid_safety_gap
        safe_robotarium_height = self.robotarium_height - 2*grid_safety_gap
        assert safe_robotarium_width > 0 and safe_robotarium_height > 0, "Failed to create GridWorld. A grid_safety_gap of %g leaves no room inside the %gx%g arena." % (grid_safety_gap, self.robotarium_width, self.robotarium_height)
        self.tile_width = min(safe_robotarium_width / self.grid_width, safe_robotarium_height / self.grid_height)

        self.bottom_left_tile_pos_xy = np.array([
            [0 - self.grid_width /2 * self.tile_width + 0.5 * self.tile_width],
            [0 - self.grid_height/2 * self.tile_width + 0.5 * self.tile_width]])

        # Distance at which a robot is considered to have reached its tile. This is the single
        # biggest lever on how long a move takes, because the attraction pulling a robot in
        # decays linearly inside MAX_DISTANCE_ATTRACTION: the approach is exponential, so the
        # last few centimetres cost as much as the first twenty. Crossing a 0.31 m tile takes
        # ln(0.31/r) / (MAX_MAGNITUDE_ATTRACTION/MAX_DISTANCE_ATTRACTION) seconds -- 2.7 s at
        # the 0.05 m default, 1.7 s at 0.10 m. Raising it also raises the attraction a robot
        # still has as it arrives, which is what _assert_grid_is_reachable weighs against the
        # repulsion, so a larger radius makes that check easier to satisfy rather than harder.
        # What it costs is precision: the robot stops further from the tile centre, and at some
        # point stops looking like it is on the tile at all. Keep it well under tile_width/2.
        self.robot_close_to_target_distance = robot_close_to_target_distance

        assert robot_close_to_target_distance < self.tile_width/2, "Failed to create GridWorld. A robot counts as arrived within %g m of a tile centre, but tiles are only %g m across, so it would count as arrived on a tile it is not on. Keep robot_close_to_target_distance well under %g m." % (robot_close_to_target_distance, self.tile_width, self.tile_width/2)

        # Checked before the simulator is built, so an unworkable grid fails here rather
        # than by robots circling a tile they can never land on
        if self.collision_avoidance_type == "PotentialField":
            self._assert_grid_is_reachable()

        # If initial_coordinates is not given, randomly sample it
        if initial_coordinates.size == 0:
            initial_coordinates = generate_random_grid_coords(grid_width_height, self.number_of_robots)

        # Convert initial_coordinates to initial_conditions
        initial_conditions_xy = self._convert_coord_to_pos(initial_coordinates)
        initial_conditions_theta = 0 * np.ones(shape=(1,number_of_robots))
        initial_conditions = np.vstack((initial_conditions_xy, initial_conditions_theta))

        # Initialize robotarium simulator
        self.robotarium = Robotarium(
            number_of_robots=number_of_robots, 
            show_figure=show_figure,
            sim_in_real_time=sim_in_real_time,
            initial_conditions=initial_conditions)
        self.robot_wheel_radius = self.robotarium.WHEEL_RADIUS
        self.robot_base_length = self.robotarium.BASE_LENGTH
        self.robot_max_wheel_velocity = self.robotarium.MAX_WHEEL_VELOCITY
        self.robot_max_linear_velocity = self.robotarium.MAX_LINEAR_VELOCITY
        self.robot_max_angular_velocity = self.robotarium.MAX_ANGULAR_VELOCITY

        # Expose figure/axis handle. The simulator builds them only when it draws, so a
        # headless run has neither, and both stay None: a run that writes no video and opens
        # no window is the path for batch physics testing and for a Robotarium submission.
        self.figure = self.robotarium._fig if self.show_figure else None
        self.axes = self.robotarium._axes_handle if self.show_figure else None
        
        # Initialize arrays for robot control
        self.robots_target_tile_coord_xy = initial_coordinates
        self.robots_moving = [False] * self.number_of_robots

        # Consecutive updates each robot has gone without a command, for the starvation check
        self.ticks_since_commanded = [0] * self.number_of_robots

        # Render grid
        if self.show_figure:
            patch_grid_verts = []
            patch_grid_codes = []

            # Vertical lines
            for i in range(self.grid_width+1):
                patch_grid_verts.extend([(self.tile_width*(-self.grid_width/2 + i), self.tile_width*(-self.grid_height/2)), (self.tile_width*(-self.grid_width/2 + i), self.tile_width*(self.grid_height/2))])
                patch_grid_codes.extend([Path.MOVETO, Path.LINETO])

            # Horizontal lines
            for i in range(self.grid_height+1):
                patch_grid_verts.extend([(self.tile_width*(-self.grid_width/2), self.tile_width*(-self.grid_height/2+i)), (self.tile_width*(self.grid_width/2), self.tile_width*(-self.grid_height/2+i))])
                patch_grid_codes.extend([Path.MOVETO, Path.LINETO])
            
            self.patch_grid = PathPatch(Path(patch_grid_verts, patch_grid_codes), edgecolor="gray", facecolor="none", linewidth=1.0, zorder=1)
            self.axes.add_patch(self.patch_grid)
            self.axes.set_aspect("equal")
        
        # Init arrows
        if self.show_figure:
            self.patch_arrows : list[dict[str, FancyArrowPatch]] = []
            for i in range(self.number_of_robots):
                self.patch_arrows.append({})

        self.robot_poses = self.robotarium.get_poses()


    def get_tile(self, robot_index:int) -> np.ndarray:
        """
        Get the (X,Y) coordinate of a robot.

        Args:
            robot_index (int): The index of the robot.
        """
        return self.robots_target_tile_coord_xy[:,robot_index]
    

    def move_to_tile(self, robot_index:int, tile_coord_xy:np.ndarray) -> None:
        """
        Move a robot to a specified (X,Y) coordinate.

        Args:
            robot_index (int): The index of the robot to move.
            tile_coord_xy (np.ndarray): A 2x1 numpy array specifying the (X,Y) coordinate to move to.
        """

        # Store new target coordinates
        self.robots_target_tile_coord_xy[:,robot_index] = tile_coord_xy

        # Evaluate if robot is close enough already
        robot_position = self.robot_poses[0:2,robot_index]
        target_position = self._convert_coord_to_pos(tile_coord_xy)
        distance_from_target = np.linalg.norm(robot_position - target_position)
        if distance_from_target > self.robot_close_to_target_distance:
            self.robots_moving[robot_index] = True

    
    def robot_done_moving(self, robot_index:int) -> bool:
        """
        Returns True if the robot is done moving and False otherwise.

        Args:
            robot_index (int): The index of the robot to move.
        """
        return not self.robots_moving[robot_index]


    def robots_done_moving(self) -> bool:
        """
        Returns True if all robots are done moving and False otherwise.
        """
        return not np.any(self.robots_moving)


    def update_until_robots_done_moving(self, firing_sequence: Optional[Iterable] = None) -> None:
        """
        Calls update() until robots_done_moving() returns True.

        Args:
            firing_sequence (Optional[Iterable]): A schedule of firing sets, one per tick:
                an iterable of them, or a callable returning one. None -- the default --
                commands every robot every tick. The generators in `agsheaf.sheaf`,
                `round_robin` and `random_firing`, take robot indices and serve here
                unchanged, so a single notion of a schedule drives both the communication
                and the motion.
        """
        if callable(firing_sequence):
            firing_sequence = firing_sequence()
        schedule = iter(firing_sequence) if firing_sequence is not None else None

        while not self.robots_done_moving():
            self.update(firing=None if schedule is None else next(schedule))


    def update(self, firing: Optional[Iterable[int]] = None) -> None:
        """
        Plays GridWorld for one time unit (0.033 second). Call at the end of each step.

        Args:
            firing (Optional[Iterable[int]]): The robots receiving a command this tick.
                None -- the default -- commands every robot, which is the synchronous case
                and the behaviour of every caller that passes nothing. Any other value
                *holds* the robots left out: they are commanded zero velocity and stand
                still until they next fire.

        Held robots are commanded zero rather than left out of the `set_velocities` call,
        and the difference matters. The simulator keeps the last velocity it was given for
        any robot not named in a call, so omitting one leaves it driving on a stale
        command -- and a command is a (v, omega) pair, so a robot coasting on a stale omega
        does not continue straight but arcs. It also stops being avoided: the potential
        field keeps robots apart only because every robot recomputes its repulsion every
        tick, and a robot driving on an old command is not being repelled by anything it is
        about to hit. Zero is the reading of "no command" that keeps both invariants.
        """
        active = set(range(self.number_of_robots)) if firing is None else {int(i) for i in firing}
        assert all(0 <= i < self.number_of_robots for i in active), "Failed to update GridWorld. The argument firing must contain robot indices in [0, %g), received %s." % (self.number_of_robots, sorted(active))

        # Arrival is a fact about where a robot is, not about whether it was commanded, so
        # it is settled for every robot before the firing set narrows anything
        self._update_arrivals()
        self._note_commands(active)

        # Compute robot commands
        if self.collision_avoidance_type == "Off":
            commands = self._generate_robot_commands(active)
        elif self.collision_avoidance_type == "PotentialField":
            commands = self._generate_robot_commands_potential_fields(active)

        self.robotarium.set_velocities(np.arange(self.number_of_robots), commands)

        # END OF UPDATE
        self.robotarium.step()
        self.robot_poses = self.robotarium.get_poses()


    ## Private methods
    def _update_arrivals(self) -> None:
        """
        Marks every robot that has reached its target tile as done moving.

        This runs over all robots on every update, commanded or not, and that is the point
        of it being here rather than inside the command generators where it used to live.
        A robot skipped by a firing set is not being asked where it is; if arrival were
        only noticed while computing a command, a robot held over the moment it crossed
        its arrival radius would stay flagged as moving forever, and any caller waiting on
        robots_done_moving() would wait with it.
        """
        for i in range(self.number_of_robots):
            if not self.robots_moving[i]:
                continue

            robot_pos = np.reshape(self.robot_poses[0:2, i], (2,1))
            target_coord = np.reshape(self.robots_target_tile_coord_xy[:, i], (2,1))
            distance_to_target = np.linalg.norm(self._convert_coord_to_pos(target_coord) - robot_pos)

            if distance_to_target < self.robot_close_to_target_distance:
                self.robots_moving[i] = False
                self._set_arrows_visible(i, False)


    def _note_commands(self, active:set) -> None:
        """
        Records which robots were commanded this tick and stops the run if a schedule has
        left one of them waiting too long.

        Only robots that are still moving can starve. One that has arrived wants nothing and
        may sit uncommanded for the rest of the run -- and its count is held at zero while it
        waits, so that the wait is not charged against the next tile it is sent to.

        Args:
            active (set): The indices of the robots commanded this tick.
        """
        for i in range(self.number_of_robots):
            if i in active or not self.robots_moving[i]:
                self.ticks_since_commanded[i] = 0
            else:
                self.ticks_since_commanded[i] += 1

        if self.starvation_ticks is None:
            return

        starved = [i for i in range(self.number_of_robots)
                   if self.robots_moving[i] and self.ticks_since_commanded[i] >= self.starvation_ticks]
        if starved:
            raise StarvedRobotError(
                "Robots %s are still short of their tiles and have gone %g updates (%.1f s) "
                "without a command, so this run is not going to finish. Either the firing "
                "sequence starves them -- Riess and Ghrist (2022), Assumption 2 asks every "
                "node to fire infinitely often -- or its period is simply longer than "
                "starvation_ticks (%g), which is the same thing seen from inside a run. "
                "Raise starvation_ticks, or pass None to let it drive anyway."
                % (starved, max(self.ticks_since_commanded[i] for i in starved),
                   max(self.ticks_since_commanded[i] for i in starved) * 0.033,
                   self.starvation_ticks))


    def _set_arrows_visible(self, robot_index:int, visible:bool) -> None:
        """
        Shows or hides a robot's force arrows. A robot with no command this tick has no
        forces acting on it, so leaving its arrows up would draw a force that is not there.

        Args:
            robot_index (int): The index of the robot.
            visible (bool): Whether the arrows should be shown.
        """
        if not (self.show_figure and self.show_arrows):
            return
        for arrow_patch in self.patch_arrows[robot_index].values():
            arrow_patch.set_visible(visible)


    def _repulsion_magnitude(self, distance:float) -> float:
        """
        The repulsion a robot feels from something the given distance away: full strength
        close in, easing quadratically to nothing at max_distance_repulsion.

        Args:
            distance (float): Distance in meters to the robot or wall doing the repelling.
        """
        if distance < self.min_distance_repulsion:
            return MAX_MAGNITUDE_REPULSION
        if distance < self.max_distance_repulsion:
            quadratic_coeff = MAX_MAGNITUDE_REPULSION / np.power(self.max_distance_repulsion - self.min_distance_repulsion, 2)
            return float(quadratic_coeff * np.power(distance - self.max_distance_repulsion, 2))
        return 0.0


    def _assert_grid_is_reachable(self) -> None:
        """
        Checks that the grid and the repulsion distances suit each other, so that a robot
        can actually land on a tile rather than hovering beside it forever.

        A robot arrives when it is within robot_close_to_target_distance of the tile center,
        and by then the attraction pulling it in has fallen off almost to nothing. Any
        repulsion still acting on it at that point wins, the robot never registers as
        arrived, and any caller waiting on robots_done_moving() waits forever. Two things
        repel: the robot on the neighboring tile, one tile_width away, and the arena wall,
        which the outermost tile centers sit grid_safety_gap + tile_width/2 away from.
        """
        arrival_attraction = MAX_MAGNITUDE_ATTRACTION / MAX_DISTANCE_ATTRACTION * self.robot_close_to_target_distance

        neighbor_repulsion = self._repulsion_magnitude(self.tile_width)
        assert neighbor_repulsion < arrival_attraction, "Failed to create GridWorld. Tiles are %g m apart, close enough that a robot on the next tile repels with %g m/s, which beats the %g m/s of attraction left as a robot arrives, so robots would never finish a move. Widen the tiles (fewer of them) or lower max_distance_repulsion (currently %g m) below the tile width." % (self.tile_width, neighbor_repulsion, arrival_attraction, self.max_distance_repulsion)

        wall_repulsion = self._repulsion_magnitude(self.grid_safety_gap + self.tile_width/2)
        assert wall_repulsion < arrival_attraction, "Failed to create GridWorld. The outermost tile centers sit %g m from the arena wall, which repels them with %g m/s, beating the %g m/s of attraction left as a robot arrives, so robots would never finish a move onto an edge tile. Raise grid_safety_gap (currently %g m) or lower max_distance_repulsion (currently %g m)." % (self.grid_safety_gap + self.tile_width/2, wall_repulsion, arrival_attraction, self.grid_safety_gap, self.max_distance_repulsion)


    def _convert_coord_to_pos(self, coordinates:np.ndarray) -> np.ndarray:
        """
        Converts grid coordinates into world positions.

        Args:
            coordinates (np.ndarray): A 2xN numpy array specifying the (X,Y) coordinates to convert.
        """
        positions = self.bottom_left_tile_pos_xy + self.tile_width * coordinates
        return positions


    def _generate_robot_commands(self, active:set) -> np.ndarray:
        """
        Computes robot commands without collision avoidance.

        Args:
            active (set): The indices of the robots to command. Every other robot keeps the
                zero the command array is initialized with, which holds it still.
        """

        # Parameters
        max_distance_attraction = MAX_DISTANCE_ATTRACTION
        max_magnitude_attraction = MAX_MAGNITUDE_ATTRACTION

        omega_v_ratio = 3/0.2 # [rad/s / m/s] Ratio of turning rate to linear command (used for decomposing force into linear/angular rates)

        wheel_speed_safety_fraction = 0.99 # Should be less than 1, used to ensure robots operate within valid wheel speeds
        
        commands = np.zeros(shape=(2,self.number_of_robots))

        for i in range(self.number_of_robots):
            if not self.robots_moving[i]:
                # Robot has already reached goal, skip
                continue
            if i not in active:
                # Robot is held this tick: no command, so no forces to draw either
                self._set_arrows_visible(i, False)
                continue
            self._set_arrows_visible(i, True)

            robot_pos = np.reshape(self.robot_poses[0:2, i], (2,1))
            robot_theta = self.robot_poses[2, i]
            target_coord = np.reshape(self.robots_target_tile_coord_xy[:, i], (2,1))
            target_pos = self._convert_coord_to_pos(target_coord)

            # Offset to target. Arrival was settled by _update_arrivals before this ran, so
            # every robot still flagged as moving is short of its tile.
            offset_to_target = target_pos - robot_pos
            distance_to_target = np.linalg.norm(offset_to_target)

            # Calculate attractive force to target
            attraction_magnitude = np.min([max_magnitude_attraction/max_distance_attraction*distance_to_target, max_magnitude_attraction])
            attraction_force_unit = self._unit_vector(offset_to_target)
            attraction_force = attraction_magnitude * attraction_force_unit

            if self.show_figure and self.show_arrows:
                if "attract" not in self.patch_arrows[i]:
                    arrow = FancyArrowPatch((0,0),(0,0),mutation_scale=10,color="blue", zorder=1.5)
                    self.axes.add_patch(arrow)
                    self.patch_arrows[i]["attract"] = arrow
                
                self.patch_arrows[i]["attract"].set_positions(
                    (_scalar(robot_pos[0]), _scalar(robot_pos[1])),
                    (_scalar(robot_pos[0] + attraction_force[0]), _scalar(robot_pos[1] + attraction_force[1]))
                )
                if attraction_magnitude < 0.01:
                    self.patch_arrows[i]["attract"].set_visible(False)

            # Calculate total force
            total_force = attraction_force

            # Calculate command
            forward_unit = np.array([[ np.cos(robot_theta)], [np.sin(robot_theta)]])
            left_unit =    np.array([[-np.sin(robot_theta)], [np.cos(robot_theta)]])
            v =     _scalar(total_force.T @ forward_unit)
            omega = _scalar(total_force.T @ left_unit * omega_v_ratio)

            # Calculate scale limit factor to enforce max linear/angular speeds
            scale_limit_factor = 1.0
            if np.abs(v) > 0:
                scale_limit_factor = np.min([scale_limit_factor, self.robot_max_linear_velocity/np.abs(v)])
            if np.abs(omega) > 0:
                scale_limit_factor = np.min([scale_limit_factor, self.robot_max_angular_velocity/np.abs(omega)])

            # Calculate scale limit factor to enforce max wheel speed
            wheel_speeds = np.vstack((
                1/(2*self.robot_wheel_radius)*(2*v-self.robot_base_length*omega),
                1/(2*self.robot_wheel_radius)*(2*v+self.robot_base_length*omega)))
            max_wheel_speed = np.max(np.abs(wheel_speeds))
            if max_wheel_speed > 0:
                scale_limit_factor = np.min([scale_limit_factor, self.robot_max_wheel_velocity*wheel_speed_safety_fraction/max_wheel_speed])
            
            # Scale commands
            v = v * scale_limit_factor
            omega = omega * scale_limit_factor

            # Save commands
            commands[0,i] = v
            commands[1,i] = omega

        return commands


    def _generate_robot_commands_potential_fields(self, active:set) -> np.ndarray:
        """
        Computes robot commands with potential fields.

        Args:
            active (set): The indices of the robots to command. Every other robot keeps the
                zero the command array is initialized with, which holds it still.
        """

        # Parameters
        max_distance_attraction = MAX_DISTANCE_ATTRACTION
        max_magnitude_attraction = MAX_MAGNITUDE_ATTRACTION

        max_distance_repulsion = self.max_distance_repulsion # [m] Robots aren't repulsed outside of this distance (quadratic in between)
        min_distance_repulsion = self.min_distance_repulsion # [m] Robots are repulsed with maximum force within this distance (quadratic in between)
        max_magnitude_repulsion = MAX_MAGNITUDE_REPULSION

        omega_v_ratio = 3/0.2 # [rad/s / m/s] Ratio of turning rate to linear command (used for decomposing force into linear/angular rates)

        wheel_speed_safety_fraction = 0.99 # Should be less than 1, used to ensure robots operate within valid wheel speeds
        
        repulsion_quadratic_coeff = max_magnitude_repulsion / np.power(max_distance_repulsion-min_distance_repulsion,2)

        commands = np.zeros(shape=(2,self.number_of_robots))

        for i in range(self.number_of_robots):
            if not self.robots_moving[i]:
                # Robot has already reached goal, skip
                continue
            if i not in active:
                # Robot is held this tick: no command, so no forces to draw either. It still
                # repels the robots that are moving -- a stationary robot is an obstacle.
                self._set_arrows_visible(i, False)
                continue
            self._set_arrows_visible(i, True)

            robot_pos = np.reshape(self.robot_poses[0:2, i], (2,1))
            robot_theta = self.robot_poses[2, i]
            target_coord = np.reshape(self.robots_target_tile_coord_xy[:, i], (2,1))
            target_pos = self._convert_coord_to_pos(target_coord)

            # Offset to target. Arrival was settled by _update_arrivals before this ran, so
            # every robot still flagged as moving is short of its tile.
            offset_to_target = target_pos - robot_pos
            distance_to_target = np.linalg.norm(offset_to_target)

            # Calculate attractive force to target
            attraction_magnitude = np.min([max_magnitude_attraction/max_distance_attraction*distance_to_target, max_magnitude_attraction])
            attraction_force_unit = self._unit_vector(offset_to_target)
            attraction_force = attraction_magnitude * attraction_force_unit

            if self.show_figure and self.show_arrows:
                if "attract" not in self.patch_arrows[i]:
                    arrow = FancyArrowPatch((0,0),(0,0),mutation_scale=10,color="blue", zorder=1.5)
                    self.axes.add_patch(arrow)
                    self.patch_arrows[i]["attract"] = arrow
                
                self.patch_arrows[i]["attract"].set_positions(
                    (_scalar(robot_pos[0]), _scalar(robot_pos[1])),
                    (_scalar(robot_pos[0] + attraction_force[0]), _scalar(robot_pos[1] + attraction_force[1]))
                )
                self.patch_arrows[i]["attract"].set_mutation_scale(50 * attraction_magnitude)

            # Init zero repulsion force
            sum_repulsion_force = np.zeros(shape=(2,1))

            # Calculate repulsion from other agents
            for j in range(self.number_of_robots):
                if i == j:
                    continue

                robot_pos_j = np.reshape(self.robot_poses[0:2, j], (2,1))
                offset_to_j = robot_pos_j - robot_pos
                distance_to_j = np.linalg.norm(offset_to_j)
                if distance_to_j < min_distance_repulsion:
                    repulsion_magnitude = max_magnitude_repulsion
                elif distance_to_j < max_distance_repulsion:
                    repulsion_magnitude = repulsion_quadratic_coeff * np.power(distance_to_j - max_distance_repulsion, 2)
                else:
                    repulsion_magnitude = 0

                repulsion_force_unit = self._unit_vector(-offset_to_j)
                repulsion_force = repulsion_magnitude * repulsion_force_unit

                if self.show_figure and self.show_arrows:
                    arr_name = "r%g" % j
                    if arr_name not in self.patch_arrows[i]:
                        arrow = FancyArrowPatch((0,0),(0,0),mutation_scale=10,color="red", zorder=1.5)
                        self.axes.add_patch(arrow)
                        self.patch_arrows[i][arr_name] = arrow
                    
                    self.patch_arrows[i][arr_name].set_positions(
                        (_scalar(robot_pos[0]), _scalar(robot_pos[1])),
                        (_scalar(robot_pos[0] + repulsion_force[0]), _scalar(robot_pos[1] + repulsion_force[1]))
                    )
                    self.patch_arrows[i][arr_name].set_mutation_scale(50 * repulsion_magnitude)
                
                sum_repulsion_force = sum_repulsion_force + repulsion_force
            
            # Calculate repulsion force from walls
            closest_wall_points = np.array([
                [self.robotarium_width/2,    _scalar(robot_pos[0]), -self.robotarium_width/2,   _scalar(robot_pos[0])],
                [  _scalar(robot_pos[1]), self.robotarium_height/2,    _scalar(robot_pos[1]), -self.robotarium_height/2]
            ])
            for j in range(4):
                wall_pos = np.reshape(closest_wall_points[:,j], (2,1))
                offset_to_wall = wall_pos - robot_pos
                distance_to_wall = np.linalg.norm(offset_to_wall)
                if distance_to_wall < min_distance_repulsion:
                    repulsion_magnitude = max_magnitude_repulsion
                elif distance_to_wall < max_distance_repulsion:
                    repulsion_magnitude = repulsion_quadratic_coeff * np.power(distance_to_wall - max_distance_repulsion, 2)
                else:
                    repulsion_magnitude = 0

                repulsion_force_unit = self._unit_vector(-offset_to_wall)
                repulsion_force = repulsion_magnitude * repulsion_force_unit

                if self.show_figure and self.show_arrows:
                    arr_name = "w%g" % j
                    if arr_name not in self.patch_arrows[i]:
                        arrow = FancyArrowPatch((0,0),(0,0),mutation_scale=10,color="green", zorder=1.5)
                        self.axes.add_patch(arrow)
                        self.patch_arrows[i][arr_name] = arrow
                    
                    self.patch_arrows[i][arr_name].set_positions(
                        (_scalar(robot_pos[0]), _scalar(robot_pos[1])),
                        (_scalar(robot_pos[0] + repulsion_force[0]), _scalar(robot_pos[1] + repulsion_force[1]))
                    )
                    self.patch_arrows[i][arr_name].set_mutation_scale(50 * repulsion_magnitude)
                
                sum_repulsion_force = sum_repulsion_force + repulsion_force
            
            # Calculate total force
            total_force = attraction_force + sum_repulsion_force

            # Calculate command
            forward_unit = np.array([[ np.cos(robot_theta)], [np.sin(robot_theta)]])
            left_unit =    np.array([[-np.sin(robot_theta)], [np.cos(robot_theta)]])
            v =     _scalar(total_force.T @ forward_unit)
            omega = _scalar(total_force.T @ left_unit * omega_v_ratio)

            # Calculate scale limit factor to enforce max linear/angular speeds
            scale_limit_factor = 1.0
            if np.abs(v) > 0:
                scale_limit_factor = np.min([scale_limit_factor, self.robot_max_linear_velocity/np.abs(v)])
            if np.abs(omega) > 0:
                scale_limit_factor = np.min([scale_limit_factor, self.robot_max_angular_velocity/np.abs(omega)])

            # Calculate scale limit factor to enforce max wheel speed
            wheel_speeds = np.vstack((
                1/(2*self.robot_wheel_radius)*(2*v-self.robot_base_length*omega),
                1/(2*self.robot_wheel_radius)*(2*v+self.robot_base_length*omega)))
            max_wheel_speed = np.max(np.abs(wheel_speeds))
            if max_wheel_speed > 0:
                scale_limit_factor = np.min([scale_limit_factor, self.robot_max_wheel_velocity*wheel_speed_safety_fraction/max_wheel_speed])
            
            # Scale commands
            v = v * scale_limit_factor
            omega = omega * scale_limit_factor

            # Save commands
            commands[0,i] = v
            commands[1,i] = omega

        return commands
    

    def _unit_vector(self, vector:np.ndarray) -> np.ndarray:
        """
        For a given vector, returns the unit vector or zero vector.

        Args:
            vector (np.ndarray): A numpy array specifying the vector to unitize.
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        else:
            return vector / norm


def generate_random_grid_coords(
    grid_width_height: tuple[int, int],
    N:int,
    replace: bool = False,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generates an 2xN numpy array with (X,Y) coordinates sampled uniformly from the specified grid dimensions.

    Args:
        grid_width_height (tuple[int, int]): The width and height of the grid.
        N (int): The number of coordinates to sample.
        replace (bool): Whether to sample with replacement or not.
        rng (Optional[np.random.Generator]): Numpy random number generator to use. 
    """
    if rng is None:
        rng = np.random.default_rng()

    grid_width, grid_height = grid_width_height
    tile_ids = rng.choice(grid_width*grid_height, size=(1,N), replace=replace)
    coordinates = np.vstack([
        np.mod(tile_ids, grid_width),
        np.floor(tile_ids / grid_width)
    ])
    return coordinates


def choose_belief_aware_step(
    robot_coords:np.ndarray,
    goal_coords:np.ndarray,
    grid_width_height:tuple[int, int],
    tile_labels_by_agent:dict,
    tile_costs:Optional[dict[str, float]] = None,
    debug_print:bool = False
) -> np.ndarray:
    """
        Calculate robot step from what each robot believes about the tiles around it.

        One robot moves at a time, in increasing order of distance from their goal, never
        leaving the grid and never stepping onto another robot, ranking the remaining
        moves by belief. Each robot scores a move by the cheapest total cost of
        getting from that tile to its goal, where entering a tile costs according to what
        that robot believes about it (see DEFAULT_TILE_COSTS). Robots therefore round tiles
        they believe are unsafe, prefer tiles they believe are safe, and head for their goal.

        Scoring by cost-to-go rather than by the cost of the next tile alone matters: a
        robot beside a believed-unsafe region would otherwise weigh one expensive step
        against standing still and stall there, even with a cheap detour available.

        Returns a new array and does not modify robot_coords.

        Args:
            robot_coords (np.ndarray): A 2xN numpy array specifying the (X,Y) coordinates of the N robots.
            goal_coords (np.ndarray): A 2xN numpy array specifying the (X,Y) coordinates of the N goals.
            grid_width_height (tuple[int, int]): The width and height of the grid.
            tile_labels_by_agent (dict): Each robot's labels, keyed by robot index then by (X,Y) tile. See resolve_tile_labels.
            tile_costs (Optional[dict[str, float]]): Cost of entering a tile, by label. Defaults to DEFAULT_TILE_COSTS.
            debug_print (bool): Whether to print debug logs.
    """

    tile_costs = _validated_tile_costs(tile_costs)

    number_of_robots = robot_coords.shape[1]
    assert goal_coords.shape[1] == number_of_robots, "Failed to choose a step. Expected goals for %g robots, received %g." % (number_of_robots, goal_coords.shape[1])
    assert len(tile_labels_by_agent) >= number_of_robots, "Failed to choose a step. Expected tile labels for %g robots, received %g." % (number_of_robots, len(tile_labels_by_agent))

    # Calculate distances from goals
    goal_distances = np.linalg.norm(robot_coords - goal_coords, axis=0)
    robot_indices = np.argsort(goal_distances).tolist()

    # Which robots count as terrain, decided once from the positions this step began at. A
    # robot that arrives at its goal *during* this step is not yet part of the terrain for the
    # robots planned after it -- it was still moving when the step was planned.
    settled_by_index = frozenset(other for other in range(number_of_robots)
                                 if goal_distances[other] == 0)

    # Init next steps, leaving the caller's array alone
    next_robot_coords = robot_coords.copy()

    # Iterate through robots. Each is planned against the moves already committed to by the
    # robots before it, which is what keeps two of them off the same tile.
    if debug_print:
        print("EVALUATING BELIEF-AWARE MOVES:")
    for robot_index in robot_indices:
        if debug_print:
            print("\tRobot %g with goal coord (%g, %g):" % (robot_index, goal_coords[0,robot_index], goal_coords[1, robot_index]))

        # Check if robot has already reached goal
        if goal_distances[robot_index] == 0:
            if debug_print:
                print("\t\tAlready reached goal, skipping...")
            continue

        settled_tiles = frozenset(
            (int(next_robot_coords[0,other]), int(next_robot_coords[1,other]))
            for other in settled_by_index if other != robot_index)

        next_robot_coords[:, robot_index] = choose_agent_step(
            robot_index, next_robot_coords, goal_coords, grid_width_height,
            tile_labels_by_agent, tile_costs=tile_costs, settled_tiles=settled_tiles,
            debug_print=debug_print)

    return next_robot_coords


def choose_agent_step(
    robot_index:int,
    robot_coords:np.ndarray,
    goal_coords:np.ndarray,
    grid_width_height:tuple[int, int],
    tile_labels_by_agent:dict,
    tile_costs:Optional[dict[str, float]] = None,
    blocked_tiles:Iterable[tuple[int, int]] = (),
    settled_tiles:Optional[Iterable[tuple[int, int]]] = None,
    extra_costs:Optional[dict[tuple[int, int], float]] = None,
    debug_print:bool = False
) -> np.ndarray:
    """
        Calculate one robot's next tile from what it believes about the tiles around it.

        This is the body of choose_belief_aware_step, which is a loop over it -- planning one
        robot at a time was always what that function did, and pulling the body out is what
        lets a robot be planned on its own, when it arrives, rather than only as part of a
        round in which every robot is planned together.

        The rules are unchanged: never leave the grid, never enter a tile another robot holds,
        and among what is left take the move with the cheapest total cost of getting to the
        goal from it, where entering a tile costs by what this robot believes about it.

        Two tile sets say where the other robots are, and they answer different questions.
        `robot_coords` is where every robot is or is going, and rules those tiles out as
        moves. `blocked_tiles` rules out more of them without their being anybody's position
        -- the tile a robot is driving *out of*, which it still holds until it arrives. That
        distinction has no synchronous counterpart, where every robot is at rest on its tile
        when the planning happens.

        Args:
            robot_index (int): Which robot to plan for.
            robot_coords (np.ndarray): A 2xN array of the tiles the N robots hold -- where each
                is, or where it is driving to.
            goal_coords (np.ndarray): A 2xN numpy array specifying the (X,Y) coordinates of the N goals.
            grid_width_height (tuple[int, int]): The width and height of the grid.
            tile_labels_by_agent (dict): Each robot's labels, keyed by robot index then by (X,Y) tile. See resolve_tile_labels.
            tile_costs (Optional[dict[str, float]]): Cost of entering a tile, by label. Defaults to DEFAULT_TILE_COSTS.
            blocked_tiles (Iterable[tuple[int, int]]): Further tiles this robot may not enter,
                over and above the tiles in robot_coords.
            settled_tiles (Optional[Iterable[tuple[int, int]]]): The tiles of robots that have
                finished, which route planning treats as terrain -- a robot still moving is
                worth waiting behind, but one that has arrived never clears, so routing through
                it means waiting forever rather than paying for a longer way round. None reads
                them off robot_coords and goal_coords, which is what an agent planning for
                itself wants; choose_belief_aware_step passes the set its round began with.
            extra_costs (Optional[dict]): Additional per-tile entry costs for this robot --
                the mission layer's region overlay; see _compute_cost_to_go.
            debug_print (bool): Whether to print debug logs.
    """
    grid_width, grid_height = grid_width_height
    tile_costs = _validated_tile_costs(tile_costs)
    extra_costs = extra_costs or {}
    number_of_robots = robot_coords.shape[1]

    if settled_tiles is None:
        goal_distances = np.linalg.norm(robot_coords - goal_coords, axis=0)
        settled_tiles = frozenset(
            (int(robot_coords[0,other]), int(robot_coords[1,other]))
            for other in range(number_of_robots)
            if other != robot_index and goal_distances[other] == 0)
    settled_tiles = frozenset(settled_tiles)
    blocked_tiles = frozenset(tuple(tile) for tile in blocked_tiles)

    robot_coord = np.reshape(robot_coords[:,robot_index], (2,1))
    robot_tile = (int(robot_coords[0,robot_index]), int(robot_coords[1,robot_index]))
    goal_tile = (int(goal_coords[0,robot_index]), int(goal_coords[1,robot_index]))

    # Work out what every tile would cost this robot on the way to its goal
    cost_to_go = _compute_cost_to_go(goal_tile, grid_width_height, tile_labels_by_agent[robot_index], tile_costs, blocked_tiles=settled_tiles, extra_costs=extra_costs)

    # If settled robots seal the goal off entirely, plan as though they were not there,
    # which leaves the robot waiting for a way through rather than wandering
    if robot_tile not in cost_to_go:
        cost_to_go = _compute_cost_to_go(goal_tile, grid_width_height, tile_labels_by_agent[robot_index], tile_costs, extra_costs=extra_costs)
        if debug_print:
            print("\t\tGoal is sealed off by settled robots, planning as though it were not")

    # Define all possible moves
    all_moves = np.array([
        [0, 1, 0, -1, 0],
        [0, 0, 1, 0, -1]
    ])

    next_coords = robot_coord + all_moves
    valid_mask = np.full(shape=(5), fill_value=True)
    for j in range(5):
        next_coord = np.reshape(next_coords[:,j], (2,1))
        # Check if move is in bounds
        if next_coord[0] < 0 or next_coord[0] >= grid_width:
            valid_mask[j] = False
            if debug_print:
                print("\t\tRemoving coord index %g (out of x-bounds)" % j)
            continue
        if next_coord[1] < 0 or next_coord[1] >= grid_height:
            valid_mask[j] = False
            if debug_print:
                print("\t\tRemoving coord index %g (out of y-bounds)" % j)
            continue
        # Check if move conflicts with where another robot currently is
        other_robot_coords = np.delete(robot_coords, robot_index, axis=1)
        distances_to_other_robots = np.linalg.norm(next_coord - other_robot_coords, axis=0)
        if np.any(distances_to_other_robots == 0):
            valid_mask[j] = False
            if debug_print:
                print("\t\tRemoving coord index %g (overlapping with robot)" % j)
            continue
        # Check if the move is onto a tile a robot holds without standing on it -- the one it
        # is driving out of. Standing still is never blocked this way: a robot always holds
        # its own tile, and refusing it a move it is already making would leave it nothing.
        if j != 0 and (int(next_coords[0,j]), int(next_coords[1,j])) in blocked_tiles:
            valid_mask[j] = False
            if debug_print:
                print("\t\tRemoving coord index %g (reserved by a robot in transit)" % j)
            continue
        # Don't need to check for two robots swapping adjacent positions because it's impossible!
    next_valid_coords = next_coords[:, valid_mask]

    # Score the remaining moves by what the robot believes they lead to: what entering
    # that tile costs, plus the cheapest cost onward from it. Charging entry is what
    # makes a believed-unsafe tile expensive to step into rather than merely to sit on.
    # Standing still scores worse than the best move by the cost of the robot's own
    # tile, so a robot only holds position when every move is blocked.
    # A tile with no way through to the goal is left out of cost_to_go, and is no move at all
    next_costs = np.array([
        tile_costs[tile_labels_by_agent[robot_index][(int(next_valid_coords[0,k]), int(next_valid_coords[1,k]))]]
        + extra_costs.get((int(next_valid_coords[0,k]), int(next_valid_coords[1,k])), 0.0)
        + cost_to_go.get((int(next_valid_coords[0,k]), int(next_valid_coords[1,k])), np.inf)
        for k in range(next_valid_coords.shape[1])
    ])
    if debug_print:
        for k in range(len(next_costs)):
            print("\t\t Coord (%g, %g) costs %.2f" % (float(next_valid_coords[0,k]), float(next_valid_coords[1,k]), float(next_costs[k])))
    best_move_index = np.argmin(next_costs)
    best_coord = next_valid_coords[:, best_move_index]
    if debug_print:
        print("\t\tBest coord is (%g, %g)" % (float(best_coord[0]), float(best_coord[1])))

    return best_coord


class TileReservations:
    """
    Which tiles each agent holds, while the agents move one tile at a time asynchronously.

    An agent holds the tile it stands on. While it drives between two tiles it holds both:
    the one it is heading for, which is not free at all, and the one it is leaving, which is
    not free until it has actually got out of it. Holding both is what replaces the rule the
    synchronous planner could rely on -- that no two agents share a tile because each is at
    rest on one when the planning happens. With agents in transit that is no longer true of
    any particular moment, so the tiles have to be booked rather than observed.

    Holding the source until arrival is the conservative reading. An agent physically clear
    of its old tile is still charged for it, which costs some throughput on a crowded grid
    and buys the guarantee that no agent is ever driven at ground another is still on.
    """

    def __init__(self, robot_coords:np.ndarray):
        """
        Args:
            robot_coords (np.ndarray): A 2xN array of the tiles the agents start on.
        """
        self.destinations = np.array(robot_coords, dtype=int).copy()
        self.sources: dict[int, tuple[int, int]] = {}

    def depart(self, robot_index:int, tile) -> None:
        """
        Books a move: the agent takes the tile it is heading for and keeps the one it is
        leaving. Departing for the tile already held is standing still, and books nothing.

        Args:
            robot_index (int): The agent moving.
            tile: The (X,Y) tile it is driving to.
        """
        held = (int(self.destinations[0,robot_index]), int(self.destinations[1,robot_index]))
        target = (int(tile[0]), int(tile[1]))
        if target != held:
            self.sources[robot_index] = held
        self.destinations[:, robot_index] = target

    def arrive(self, robot_index:int) -> None:
        """
        Releases the tile the agent was driving out of, it having arrived.

        Args:
            robot_index (int): The agent that arrived.
        """
        self.sources.pop(robot_index, None)

    def blocked_for(self, robot_index:int) -> frozenset:
        """
        The tiles another agent holds without standing on: what to pass to
        choose_agent_step as `blocked_tiles`. The tiles agents are heading for are not
        here -- they are in `destinations`, which the planner reads as their positions.

        Args:
            robot_index (int): The agent being planned for.
        """
        return frozenset(tile for other, tile in self.sources.items() if other != robot_index)

    def in_transit(self) -> frozenset:
        """The agents currently between two tiles."""
        return frozenset(self.sources)


def _validated_tile_costs(tile_costs:Optional[dict[str, float]]) -> dict[str, float]:
    """
    The tile costs to plan with, defaulted and checked to cover every label exactly.

    Args:
        tile_costs (Optional[dict[str, float]]): Cost of entering a tile, by label.
    """
    if tile_costs is None:
        tile_costs = DEFAULT_TILE_COSTS
    for label in tile_costs:
        assert label in TILE_LABELS, "Failed to choose a step. The tile cost label \"%s\" is not one of %s." % (label, str(TILE_LABELS))
    for label in TILE_LABELS:
        assert label in tile_costs, "Failed to choose a step. No cost was given for tiles labeled \"%s\"; every one of %s needs one." % (label, str(TILE_LABELS))
    return tile_costs


def _compute_cost_to_go(
    goal_tile:tuple[int, int],
    grid_width_height:tuple[int, int],
    tile_labels:dict[tuple[int, int], str],
    tile_costs:dict[str, float],
    blocked_tiles:frozenset = frozenset(),
    extra_costs:Optional[dict[tuple[int, int], float]] = None
) -> dict[tuple[int, int], float]:
    """
    Computes, for every tile, the cheapest total cost of reaching the goal from it, given
    what one robot believes about the grid. Entering a tile costs tile_costs[label], plus
    that tile's entry in `extra_costs` if it has one; the goal tile itself costs nothing
    to arrive at.

    This is Dijkstra run outwards from the goal, which is the same thing as the shortest
    path to the goal from everywhere, since moves are undirected.

    Tiles that cannot be routed through are simply left out of the result, so a caller
    should treat a missing tile as one with no way to the goal.

    Args:
        goal_tile (tuple[int, int]): The (X,Y) coordinate the robot is heading for.
        grid_width_height (tuple[int, int]): The width and height of the grid.
        tile_labels (dict[tuple[int,int], str]): This robot's label for each (X,Y) tile.
        tile_costs (dict[str, float]): Cost of entering a tile, by label.
        blocked_tiles (frozenset): (X,Y) coordinates that cannot be routed through.
        extra_costs (Optional[dict]): Additional per-tile entry costs, over and above the
            label costs -- the mission layer's overlay for regions the network has flagged
            hazardous or a neighbor has claimed. Finite, so a warned-off region is
            expensive rather than forbidden, exactly like a believed-unsafe tile.
    """
    grid_width, grid_height = grid_width_height
    extra_costs = extra_costs or {}

    cost_to_go = {}
    frontier = [(0.0, goal_tile)]
    while frontier:
        cost, tile = heapq.heappop(frontier)
        if tile in cost_to_go:
            continue
        cost_to_go[tile] = cost

        tile_x, tile_y = tile
        for neighbor in ((tile_x+1, tile_y), (tile_x-1, tile_y), (tile_x, tile_y+1), (tile_x, tile_y-1)):
            if not (0 <= neighbor[0] < grid_width and 0 <= neighbor[1] < grid_height):
                continue
            if neighbor in cost_to_go or neighbor in blocked_tiles:
                continue
            # Stepping from the neighbor towards the goal means entering this tile
            heapq.heappush(frontier, (cost + tile_costs[tile_labels[tile]]
                                      + extra_costs.get(tile, 0.0), neighbor))

    return cost_to_go


def route_corridor(
    start_tile:tuple[int, int],
    goal_tile:tuple[int, int],
    grid_width_height:tuple[int, int],
    tile_labels:dict[tuple[int, int], str],
    tile_costs:Optional[dict[str, float]] = None,
    extra_costs:Optional[dict[tuple[int, int], float]] = None
) -> list[tuple[int, int]]:
    """
    The tiles one agent's current plan routes through: a greedy descent of the very
    cost-to-go choose_belief_aware_step ranks its moves by, from where the robot stands to
    its goal, excluding the tile it stands on and including the goal.

    This is the route as a whole rather than the one step of it the planner commits to, which
    is what a contract over the plan has to be written against: the reliance of an agent's
    mission contract is the set of squares it needs passable, not the next square alone (see
    gridsheaf.reliance_contract). The descent passes through believed-unsafe tiles exactly
    when the planner would -- their cost is large but finite -- which is precisely the case a
    reliance check has something to say about.

    The corridor stops short of the goal if the descent has nowhere left to go, which is what
    a goal walled off by tiles with no route to it looks like from the planner's side.

    Args:
        start_tile (tuple[int, int]): The (X,Y) tile the robot is on.
        goal_tile (tuple[int, int]): The (X,Y) tile it is heading for.
        grid_width_height (tuple[int, int]): The width and height of the grid.
        tile_labels (dict[tuple[int,int], str]): This robot's label for each (X,Y) tile.
        tile_costs (Optional[dict[str, float]]): Cost of entering a tile, by label. Defaults
            to DEFAULT_TILE_COSTS.
        extra_costs (Optional[dict]): Additional per-tile entry costs -- the mission layer's
            region overlay, passed so the corridor is the route the overlaid planner would
            actually drive; see _compute_cost_to_go.
    """
    if tile_costs is None:
        tile_costs = DEFAULT_TILE_COSTS
    extra_costs = extra_costs or {}

    grid_width, grid_height = grid_width_height
    cost_to_go = _compute_cost_to_go(goal_tile, grid_width_height, tile_labels, tile_costs,
                                     extra_costs=extra_costs)

    corridor: list[tuple[int, int]] = []
    tile = start_tile
    seen = {tile}
    while tile != goal_tile and len(corridor) < grid_width * grid_height:
        tile_x, tile_y = tile
        options = [
            candidate
            for candidate in ((tile_x+1, tile_y), (tile_x-1, tile_y), (tile_x, tile_y+1), (tile_x, tile_y-1))
            if 0 <= candidate[0] < grid_width and 0 <= candidate[1] < grid_height
            and candidate not in seen and candidate in cost_to_go
        ]
        if not options:
            break
        tile = min(options, key=lambda candidate: tile_costs[tile_labels[candidate]]
                   + extra_costs.get(candidate, 0.0) + cost_to_go[candidate])
        corridor.append(tile)
        seen.add(tile)
    return corridor


# Points earned for moving onto a tile, by its ground truth label (not by what the agent
# believes). An agent that has already reached a target tile earns no further points for
# entering a safe or target tile, though it can still lose points for later stepping onto
# an unsafe one -- see score_robot_step.
DEFAULT_TILE_SCORES = {
    "unsafe": -2.0,
    "safe": 1.0,
    "target": 5.0,
}


def score_robot_step(
    next_robot_coords: np.ndarray,
    ground_truth: dict[tuple[int, int], str],
    reached_target: np.ndarray,
    tile_scores: Optional[dict[str, float]] = None,
) -> np.ndarray:
    """
    Scores one step by ground truth, not by what the agents believe.

    Moving onto a ground-truth unsafe tile always costs points, whether or not the agent
    has already reached a target. Moving onto a safe or target tile earns points only up
    until the agent's first arrival at a target tile; once there, the agent is free to move
    on without needing to stay, but no longer scores for safe or target tiles, since the
    goal has already been met.

    Intended to be called once per step, with the same reached_target array passed each
    time: it is updated in place so that scoring stays correct across a whole run.

    Args:
        next_robot_coords (np.ndarray): A 2xN array of the robots' new (X,Y) coordinates.
        ground_truth (dict[tuple[int,int], str]): The world's true labeling.
        reached_target (np.ndarray): Boolean array of length N, True once an agent has
            reached a target tile. Updated in place.
        tile_scores (Optional[dict[str, float]]): Points earned by ground truth label.
            Defaults to DEFAULT_TILE_SCORES.
    """
    if tile_scores is None:
        tile_scores = DEFAULT_TILE_SCORES
    for label in tile_scores:
        assert label in GROUND_TRUTH_LABELS, "Failed to score step. The tile score label \"%s\" is not one of %s. Scoring reads the truth, which leaves no tile unknown." % (label, str(GROUND_TRUTH_LABELS))
    for label in GROUND_TRUTH_LABELS:
        assert label in tile_scores, "Failed to score step. No score was given for tiles labeled \"%s\"; every one of %s needs one." % (label, str(GROUND_TRUTH_LABELS))

    number_of_robots = next_robot_coords.shape[1]
    assert len(reached_target) == number_of_robots, "Failed to score step. Expected reached_target for %g robots, received %g." % (number_of_robots, len(reached_target))

    step_scores = np.zeros(number_of_robots)
    for i in range(number_of_robots):
        tile = (int(next_robot_coords[0, i]), int(next_robot_coords[1, i]))
        step_scores[i] = score_agent_step(i, tile, ground_truth, reached_target,
                                          tile_scores=tile_scores)

    return step_scores


def score_agent_step(
    robot_index: int,
    tile: tuple[int, int],
    ground_truth: dict[tuple[int, int], str],
    reached_target: np.ndarray,
    tile_scores: Optional[dict[str, float]] = None,
) -> float:
    """
    Scores one agent's move onto one tile, by ground truth rather than by belief.

    The body of score_robot_step, which is a loop over it. Pulled out for the same reason
    choose_agent_step was: when agents move asynchronously there is no round in which every
    agent has just moved, so a score has to be attributable to the one agent that did.

    `reached_target` is updated in place, so calling this for an agent that did not move
    would score it again for the tile it is already sitting on. Call it once per move.

    Args:
        robot_index (int): Which agent moved.
        tile (tuple[int, int]): The tile it moved onto.
        ground_truth (dict[tuple[int,int], str]): The world's true labeling.
        reached_target (np.ndarray): Boolean array of length N, True once an agent has
            reached a target tile. Updated in place.
        tile_scores (Optional[dict[str, float]]): Points earned by ground truth label.
            Defaults to DEFAULT_TILE_SCORES.
    """
    if tile_scores is None:
        tile_scores = DEFAULT_TILE_SCORES

    label = ground_truth[tile]
    if label == "unsafe":
        return tile_scores["unsafe"]
    if not reached_target[robot_index]:
        if label == "target":
            reached_target[robot_index] = True
        return tile_scores[label]
    return 0.0


def _scalar(
    array: np.ndarray
):
    """
    Converts a numpy array to a scalar
    """
    assert array.size == 1, "Failed to convert numpy array of shape %s to scalar." % str(array.shape)
    return float(np.reshape(array, ()))


# Tile labels an agent can hold a belief about.
# Note that "unknown" is a genuine label: a tile that is not believed unsafe is
# not thereby believed safe.
TILE_LABELS = ("target", "safe", "unsafe", "unknown")

# Labels the world itself carries. The truth is total, so "unknown" is not among them.
GROUND_TRUTH_LABELS = ("target", "safe", "unsafe")

# Name used in place of an agent index to ask for the world's true labeling
GROUND_TRUTH_VIEW = "ground_truth"

# How two agents' labels for the same tile relate. Note the asymmetry between
# "one_sided" and "conflict": one agent knowing more than another is not the same
# as the two of them contradicting each other.
COMPARISON_LABELS = ("agree", "one_sided", "conflict", "both_unknown")

# Default fill colors used when drawing an agent's beliefs (Georgia Tech palette)
DEFAULT_LABEL_COLORS = {
    "target": "#EAAA00",  # buzz
    "safe": "#066034",    # techlawn
    "unsafe": "#D90368",  # azalea
    "unknown": "none",
}

# Default cost of entering a tile, by what the agent believes about it. Safe is cheapest,
# so agents favor ground they believe is safe; unknown is priced above safe, so they prefer
# a known-good route to an unexplored one; unsafe is large but finite, so an agent detours
# around what it believes is unsafe whenever a route exists, and crosses only as a last resort.
DEFAULT_TILE_COSTS = {
    "target": 1.0,
    "safe": 1.0,
    "unknown": 3.0,
    "unsafe": 30.0,
}

# Default outline colors used when drawing two agents' beliefs against each other.
# The split-tile fills already show agreement, one-sidedness and emptiness on their own,
# so only a genuine contradiction is outlined by default.
DEFAULT_COMPARISON_COLORS = {
    "agree": "none",
    "one_sided": "none",
    "conflict": "#051E39",    # navy: both hold beliefs, and they differ
    "both_unknown": "none",
}


def parse_label_beliefs(
    beliefs_config: Optional[dict],
    number_of_robots: int,
    grid_width_height: tuple[int, int]
) -> list[dict[tuple[int, int], str]]:
    """
    Parses a configuration block (e.g. loaded from config.yaml) into per-agent label beliefs.

    The expected structure of beliefs_config is:

        agents:
          0:
            safe:   [[0, 0], [1, 0], [2, 0]]
            unsafe: [[5, 2], [5, 3]]
            target: [[6, 4]]
          1:
            unsafe: [[3, 0], [3, 1], [3, 2]]

    Each label maps to a list of (X,Y) tile pairs such as [5, 2], every tile written out.
    Within one agent, labels are applied in the order they appear, so a later label wins on
    overlapping tiles.

    Tiles not mentioned for an agent are left unlabeled here; the caller decides their
    default label (see initialize_safety_color_grid).

    Args:
        beliefs_config (Optional[dict]): The parsed configuration block, or None for no beliefs.
        number_of_robots (int): The number of robots in the simulation.
        grid_width_height (tuple[int, int]): The width and height of the grid.
    """
    beliefs: list[dict[tuple[int, int], str]] = [{} for _ in range(number_of_robots)]

    if not beliefs_config:
        return beliefs

    assert isinstance(beliefs_config, dict), "Failed to parse label beliefs. Expected a mapping, received type %r." % type(beliefs_config).__name__

    agents_config = beliefs_config.get("agents") or {}
    assert isinstance(agents_config, dict), "Failed to parse label beliefs. The 'agents' entry must be a mapping from agent index to labels, received type %r." % type(agents_config).__name__

    for agent_key, agent_config in agents_config.items():
        agent_index = int(agent_key)
        assert 0 <= agent_index < number_of_robots, "Failed to parse label beliefs. Agent index %g is out of range for %g robots." % (agent_index, number_of_robots)

        if not agent_config:
            continue
        assert isinstance(agent_config, dict), "Failed to parse label beliefs for agent %g. Expected a mapping from label to tiles, received type %r." % (agent_index, type(agent_config).__name__)

        for label, tile_spec in agent_config.items():
            assert label in TILE_LABELS, "Failed to parse label beliefs for agent %g. The label \"%s\" is not one of %s." % (agent_index, label, str(TILE_LABELS))
            for tile in _parse_tile_spec(tile_spec, grid_width_height):
                beliefs[agent_index][tile] = label

    return beliefs


def resolve_tile_labels(
    robot_coords: np.ndarray,
    grid_width_height: tuple[int, int],
    assigned_targets: np.ndarray,
    label_beliefs: Optional[list[dict[tuple[int, int], str]]] = None,
    default_label: str = "safe",
    mark_other_agents_unsafe: bool = True
) -> dict[int, dict[tuple[int, int], str]]:
    """
    Resolves what every robot believes about every tile, assembled in layers:

        1. Every tile starts at default_label.
        2. The robot's static beliefs from label_beliefs are applied.
        3. If mark_other_agents_unsafe, the other robots' current tiles are labeled "unsafe".
        4. The square this robot has been assigned is labeled "target".

    Layers 2 and 4 are the two different things a robot can know about a target. Layer 4 is
    the square it has been told to reach, which it holds with certainty; layer 2 carries,
    among the rest of its beliefs, its guesses about where the other robots have been sent,
    which may be right or wrong. Both come out as "target", so a robot routes towards any of
    them equally; the renderer numbers the assigned one so the two stay legible apart.

    This is what both the renderer and the planner read, so robots plan against exactly the
    labeling drawn for them.

    Args:
        robot_coords (np.ndarray): A 2xN numpy array specifying the (X,Y) coordinates of the N robots.
        grid_width_height (tuple[int, int]): The width and height of the grid.
        assigned_targets (np.ndarray): A 2xN numpy array of the (X,Y) square assigned to each robot. See parse_assignments.
        label_beliefs (Optional[list[dict[tuple[int,int], str]]]): Per-robot static beliefs. See parse_label_beliefs.
        default_label (str): The label given to tiles no belief is held about.
        mark_other_agents_unsafe (bool): Whether the other robots' tiles are labeled unsafe.
    """
    grid_position_list = _create_grid_position_list(grid_width_height)
    robot_tile_positions = _determine_robot_tile_position(robot_coords, grid_position_list)
    assigned_tiles = _determine_assigned_tile_indices(assigned_targets, grid_position_list)

    return _determine_robot_safety(
        robot_coords, grid_position_list, robot_tile_positions, assigned_tiles,
        label_beliefs=label_beliefs,
        default_label=default_label,
        mark_other_agents_unsafe=mark_other_agents_unsafe)


def parse_assignments(
    assignments_config: Optional[dict],
    number_of_robots: int,
    grid_width_height: tuple[int, int]
) -> np.ndarray:
    """
    Parses the square each robot has been assigned to reach, returned as a 2xN array of
    (X,Y) coordinates in robot order.

    The expected structure of assignments_config is one tile per robot index:

        0: [6, 4]
        1: [5, 4]
        2: [6, 3]
        3: [5, 3]

    An assignment is something the robot is told, not something it works out, so every robot
    needs exactly one and no two may share a square: two robots cannot stand on one tile, so
    a shared assignment leaves a run that can never finish.

    Args:
        assignments_config (Optional[dict]): The parsed configuration block.
        number_of_robots (int): The number of robots in the simulation.
        grid_width_height (tuple[int, int]): The width and height of the grid.
    """
    assert assignments_config, "Failed to parse assignments. Every robot needs a square to head for, but no assignments were given."
    return _parse_square_per_robot(
        assignments_config, number_of_robots, grid_width_height,
        what="assignments",
        verb="assigned",
        sharing_reason="two robots cannot occupy one tile, so the run could never finish")


def parse_start_tiles(
    starts_config: Optional[dict],
    number_of_robots: int,
    grid_width_height: tuple[int, int],
    rng=None
) -> np.ndarray:
    """
    Parses the square each robot starts the run on, returned as a 2xN array of (X,Y)
    coordinates in robot order. Same structure as parse_assignments, one tile per robot:

        0: [3, 1]
        1: [4, 2]
        2: [8, 3]
        3: [0, 3]

    Given no configuration the squares are sampled at random from rng, which is how a run
    reads its starting layout off the seed alone. Naming them is what a demo wants instead:
    a seed can only be shopped for, and it hands out all four squares together, so there is
    no way to ask for one agent starting far from its target and another already close.

    Nothing else in the experiments draws from rng, so a run that names its starts and one
    that samples them differ in nothing but the starts.

    A start may be any square, including one the ground truth calls unsafe and including a
    robot's own assigned square -- a robot that begins where it was sent has simply arrived.

    Args:
        starts_config (Optional[dict]): The parsed configuration block, or None to sample.
        number_of_robots (int): The number of robots in the simulation.
        grid_width_height (tuple[int, int]): The width and height of the grid.
        rng (Optional[np.random.Generator]): Source of the random layout, when sampling.
    """
    if not starts_config:
        return generate_random_grid_coords(grid_width_height, number_of_robots, rng=rng)

    return _parse_square_per_robot(
        starts_config, number_of_robots, grid_width_height,
        what="starting squares",
        verb="started on",
        sharing_reason="two robots cannot occupy one tile, and the Robotarium will not place them there")


def _parse_square_per_robot(
    squares_config,
    number_of_robots: int,
    grid_width_height: tuple[int, int],
    what: str,
    verb: str,
    sharing_reason: str
) -> np.ndarray:
    """
    One [x, y] square per robot index, as a 2xN array in robot order. Every robot needs
    exactly one and no two may share, which is what both the assignments and the starting
    squares are; the wording of what a violation means is the caller's, since the same
    mistake is a different mistake in each.

    Args:
        squares_config: The parsed configuration block, a mapping from robot index to a tile.
        number_of_robots (int): The number of robots in the simulation.
        grid_width_height (tuple[int, int]): The width and height of the grid.
        what (str): What is being parsed, naming the failure.
        verb (str): What the robot is having done to it, for the per-robot messages.
        sharing_reason (str): Why two robots sharing a square cannot be allowed.
    """
    assert isinstance(squares_config, dict), "Failed to parse %s. Expected a mapping from robot index to an [x, y] square, received type %r." % (what, type(squares_config).__name__)

    squares: dict[int, tuple[int, int]] = {}
    for robot_key, tile_spec in squares_config.items():
        robot_index = int(robot_key)
        assert 0 <= robot_index < number_of_robots, "Failed to parse %s. Robot index %g is out of range for %g robots." % (what, robot_index, number_of_robots)

        tiles = _parse_tile_spec(tile_spec, grid_width_height)
        assert len(tiles) == 1, "Failed to parse %s. Robot %g must be %s exactly one [x, y] square, received %g." % (what, robot_index, verb, len(tiles))
        squares[robot_index] = tiles[0]

    missing = [robot_index for robot_index in range(number_of_robots) if robot_index not in squares]
    assert not missing, "Failed to parse %s. No square was given for robot(s) %s; every one of %g robots needs one." % (what, str(missing), number_of_robots)

    for robot_index, tile in squares.items():
        sharing = [other for other, other_tile in squares.items() if other != robot_index and other_tile == tile]
        assert not sharing, "Failed to parse %s. Robots %s are all %s (%g, %g), but %s." % (what, str(sorted([robot_index] + sharing)), verb, tile[0], tile[1], sharing_reason)

    return np.array([[squares[robot_index][0] for robot_index in range(number_of_robots)],
                     [squares[robot_index][1] for robot_index in range(number_of_robots)]])


def parse_communication_topology(
    communication_config: Optional[dict],
    number_of_robots: int
) -> Optional[nx.Graph]:
    """
    Parses which robots can talk to each other, returned as an undirected graph.

    The expected structure of communication_config is an edge list:

        edges: [[0, 1], [1, 2], [2, 3]]

    Every robot becomes a node whether or not it has an edge, so a robot that can talk to
    nobody is an isolated node rather than absent from the graph. The graph being undirected,
    [0, 1] and [1, 0] are the same edge and a repeat of either is not an error.

    Nothing reads the topology to decide how robots move: it is declared, drawn and logged,
    ready for robots to share what they believe along it.

    Args:
        communication_config (Optional[dict]): The parsed configuration block, or None for no topology.
        number_of_robots (int): The number of robots in the simulation.
    """
    if not communication_config:
        return None

    assert isinstance(communication_config, dict), "Failed to parse communication topology. Expected a mapping, received type %r." % type(communication_config).__name__

    edge_specs = communication_config.get("edges") or []
    assert isinstance(edge_specs, list), "Failed to parse communication topology. The 'edges' entry must be a list of [a, b] pairs, received type %r." % type(edge_specs).__name__

    topology = nx.Graph()
    topology.add_nodes_from(range(number_of_robots))

    for edge_spec in edge_specs:
        assert isinstance(edge_spec, (list, tuple)) and len(edge_spec) == 2, "Failed to parse communication topology. Expected an [a, b] pair of agent indices, received %r." % (edge_spec,)

        first, second = edge_spec
        for agent_index in (first, second):
            assert isinstance(agent_index, int), "Failed to parse communication topology. Expected an agent index, received type %r." % type(agent_index).__name__
            assert 0 <= agent_index < number_of_robots, "Failed to parse communication topology. Agent index %g is out of range for %g robots." % (agent_index, number_of_robots)
        assert first != second, "Failed to parse communication topology. Agent %g is given an edge to itself, but a topology says who an agent can talk to besides itself." % first

        topology.add_edge(first, second)

    return topology


def parse_move_schedule(
    move_schedule: Optional[str],
    number_of_robots: int,
    seed: int = 0
) -> Optional[Iterable]:
    """
    Parses whose turn it is to take a tile, returned as a schedule of agent sets -- or None,
    which lets every agent move as soon as it is able to.

    These are `agsheaf.sheaf`'s firing sequences over agent indices rather than sheaf
    vertices -- the same `round_robin` and `random_firing` that drive the asynchronous Tarski
    Laplacian (Riess and Ghrist 2022, Def. 5), applied to who moves rather than to who
    broadcasts. Liveness, their Assumption 2, is what both layers need of a schedule.

    `GridWorld.update` takes a firing set of its own, one level down: who receives a velocity
    on a given control tick, which decides how the robots are driven and not where they go.
    Nothing reads that from configuration -- it is there for callers modelling a controller
    that misses updates.

    A turn is over when every agent in it has had its chance -- taken a tile, been found with
    nowhere to go, or been found already home. An agent whose turn comes up while it is still
    driving is waited for, which is what serialises the run and what the schedule is for; an
    agent that is merely blocked is not, or a boxed-in agent would hold up the rotation
    indefinitely.

      * "round_robin" gives the turn to one agent at a time, cycling. Agents claim tiles in
        strict rotation. They may still be in transit together -- the rotation orders who
        departs next, not who is moving.
      * "random" gives it to each agent independently with probability 1/2, which is live
        almost surely. Drawn from its own generator seeded from `seed`.

    Args:
        move_schedule (Optional[str]): The configured schedule, or None to let everyone move.
        number_of_robots (int): How many agents the schedule cycles over.
        seed (int): Seed for "random". Ignored by the others.
    """
    if move_schedule is None:
        return None

    assert move_schedule in MOVE_SCHEDULES, "Failed to parse the gridworld configuration. The move_schedule entry must be one of %s, or null to let every agent move as soon as it can. Received %r." % (str(MOVE_SCHEDULES), move_schedule)
    return _schedule_over_agents(move_schedule, number_of_robots, seed)


def _schedule_over_agents(schedule: str, number_of_robots: int, seed: int) -> Iterable:
    """
    The named firing sequence over agent indices. `agsheaf.sheaf`'s generators take a sequence
    of nodes and never ask what a node is, so they serve here unchanged.

    Args:
        schedule (str): One of MOVE_SCHEDULES.
        number_of_robots (int): How many agents to cycle over.
        seed (int): Seed for "random".
    """
    if schedule == "round_robin":
        return round_robin(range(number_of_robots))
    return random_firing(range(number_of_robots), random.Random(seed))


def assert_assignments_are_targets(
    assigned_targets: np.ndarray,
    ground_truth: dict[tuple[int, int], str]
) -> None:
    """
    Checks that every assigned square really is a target in the world, rather than somewhere
    a robot has been sent by mistake. Kept apart from parse_assignments so that parsing an
    assignment does not require a ground truth to have been configured.

    Args:
        assigned_targets (np.ndarray): A 2xN numpy array of assigned (X,Y) squares.
        ground_truth (dict[tuple[int,int], str]): The world's true labeling.
    """
    for robot_index in range(assigned_targets.shape[1]):
        tile = (int(assigned_targets[0, robot_index]), int(assigned_targets[1, robot_index]))
        assert ground_truth[tile] == "target", "Failed to check assignments. Robot %g is assigned (%g, %g), which is truly \"%s\", not a target." % (robot_index, tile[0], tile[1], ground_truth[tile])


def parse_ground_truth(
    ground_truth_config: Optional[dict],
    grid_width_height: tuple[int, int]
) -> Optional[dict[tuple[int, int], str]]:
    """
    Parses the world's true labeling, the thing the agents hold beliefs about.

    The expected structure of ground_truth_config is:

        default: safe                   # label for tiles not mentioned below
        unsafe: [[3, 0], [3, 1], [5, 2]]
        target: [[5, 3], [6, 3], [5, 4], [6, 4]]

    Tiles are given the same way as in parse_label_beliefs: every one written out as an
    [x, y] pair. Unlike a belief, the truth is total: every tile is safe or unsafe apart
    from the targets, so "unknown" is rejected here.

    The targets need not sit together or form any particular shape. What each robot has to
    reach is settled by the assignments, and assert_assignments_are_targets checks each of
    those is truly a target, so nothing downstream cares how the targets are arranged.

    Args:
        ground_truth_config (Optional[dict]): The parsed configuration block, or None for no truth.
        grid_width_height (tuple[int, int]): The width and height of the grid.
    """
    if not ground_truth_config:
        return None

    assert isinstance(ground_truth_config, dict), "Failed to parse ground truth. Expected a mapping, received type %r." % type(ground_truth_config).__name__

    default_label = ground_truth_config.get("default", "safe")
    assert default_label in ("safe", "unsafe"), "Failed to parse ground truth. The default label must be \"safe\" or \"unsafe\", received \"%s\"." % default_label

    grid_position_list = _create_grid_position_list(grid_width_height)
    ground_truth = {tile: default_label for tile in grid_position_list}

    for label, tile_spec in ground_truth_config.items():
        if label == "default":
            continue
        assert label in GROUND_TRUTH_LABELS, "Failed to parse ground truth. The label \"%s\" is not one of %s. The truth leaves no tile unknown." % (label, str(GROUND_TRUTH_LABELS))
        for tile in _parse_tile_spec(tile_spec, grid_width_height):
            ground_truth[tile] = label

    assert any(label == "target" for label in ground_truth.values()), "Failed to parse ground truth. No tile is labeled \"target\", so no robot has anywhere to be sent."

    return ground_truth


def parse_display_views(
    beliefs_config: Optional[dict],
    number_of_robots: int
) -> list[int | tuple[int, int]]:
    """
    Parses the view, or sequence of views, to draw while a demo runs.

    One view for the whole run, which is what reading a video for debugging wants, is
    given as the 'view' entry:

        view: 0                   # agent 0's own labels
        view: ground_truth        # the world as it really is
        view: [0, 1]              # how agents 0 and 1 agree or conflict
        view: [ground_truth, 0]   # where agent 0's beliefs depart from the truth

    Several views cycled through as the run proceeds, which is what a demo video wants,
    are given as the 'views' entry instead, paired with a 'switch_view_every' of 'never'
    or a number of steps:

        views:
          - ground_truth
          - 0
          - [0, 1]
        switch_view_every: 20

    A run showing one view at a time can only ever draw one of them at any given moment of
    the run. 'tour_views_every' instead holds the figure on every configured view in turn,
    robots stationary, every so many steps, so a single video shows the whole set of them
    at the same moment -- which is what a demo filmed in one take needs:

        tour_views_every: 1

    Only one of 'view' and 'views' may be given. A view is either one side, drawn as that
    side's tile labels, or a pair of sides, drawn as the agreement between them. A side is
    an agent index or "ground_truth". Both forms are accepted by the function returned from
    initialize_safety_color_grid.

    Args:
        beliefs_config (Optional[dict]): The parsed configuration block, or None for the default view.
        number_of_robots (int): The number of robots in the simulation.
    """
    if not beliefs_config:
        return [0]

    # Every key this block may carry. A misspelling is otherwise silent -- the reader falls
    # back to its default and the run proceeds looking exactly as though the key had not been
    # given, which is the hardest kind of configuration bug to see. 'switch_view_every' and
    # 'tour_views_every' differ in the plural, so each is the other's most likely misspelling.
    unknown = sorted(set(beliefs_config) - BELIEF_KEYS)
    assert not unknown, "Failed to parse display views. The beliefs configuration has no %s entry. Expected one of %s." % (
        ", ".join(repr(key) for key in unknown), ", ".join(sorted(BELIEF_KEYS)))

    # A run drives every agent to its assigned square once, so there are no rounds left for a
    # view to follow; it advances by step count or not at all
    switch_view_every = beliefs_config.get("switch_view_every", "never")
    assert switch_view_every == "never" or isinstance(switch_view_every, int), "Failed to parse display views. The switch_view_every entry must be \"never\" or a number of steps, received %r." % (switch_view_every,)

    # Checked here rather than where it is read, so a run naming it wrongly fails at parse
    # time alongside everything else about the views
    tour_views_every = beliefs_config.get("tour_views_every", "never")
    assert tour_views_every == "never" or (isinstance(tour_views_every, int) and tour_views_every > 0), "Failed to parse display views. The tour_views_every entry must be \"never\" or a positive number of steps, received %r." % (tour_views_every,)

    single_view_spec = beliefs_config.get("view")
    view_specs = beliefs_config.get("views")
    assert single_view_spec is None or view_specs is None, "Failed to parse display views. Give either 'view' for one view or 'views' for a sequence of them, not both."

    if view_specs is None:
        return [_parse_display_view(0 if single_view_spec is None else single_view_spec, number_of_robots)]

    assert isinstance(view_specs, list) and len(view_specs) > 0, "Failed to parse display views. Expected a non-empty list of views, received %r." % (view_specs,)

    # Following the moving agent picks the view by that agent's index, so there has to be one
    # view per agent for it to pick from. Checked here rather than by an index error mid-run.
    follow_moving_agent = beliefs_config.get("follow_moving_agent", False)
    assert isinstance(follow_moving_agent, bool), "Failed to parse display views. The follow_moving_agent entry must be true or false, received %r." % (follow_moving_agent,)
    assert not follow_moving_agent or len(view_specs) == number_of_robots, "Failed to parse display views. With follow_moving_agent the view is chosen by the index of the agent that moved, so 'views' needs one entry per agent -- %g of them, received %g." % (number_of_robots, len(view_specs))

    return [_parse_display_view(view_spec, number_of_robots) for view_spec in view_specs]


def _parse_display_view(view_spec, number_of_robots: int) -> int | str | tuple:
    """
    Parses a single view, either one side or a pair of sides to compare, where a side is
    an agent index or "ground_truth".

    Args:
        view_spec: The view specification to parse.
        number_of_robots (int): The number of robots in the simulation.
    """
    if isinstance(view_spec, (int, str)):
        return _validate_view_side(view_spec, number_of_robots)

    assert isinstance(view_spec, (list, tuple)) and len(view_spec) == 2, "Failed to parse display view. Expected a side or a pair of sides, received %r." % (view_spec,)
    return (_validate_view_side(view_spec[0], number_of_robots), _validate_view_side(view_spec[1], number_of_robots))


def _validate_view_side(view_side, number_of_robots: int) -> int | str:
    """
    Checks that a view side names an existing robot or the ground truth, and returns it.

    Args:
        view_side: The agent index or "ground_truth" to check.
        number_of_robots (int): The number of robots in the simulation.
    """
    if isinstance(view_side, str):
        assert view_side == GROUND_TRUTH_VIEW, "Failed to parse display view. Expected an agent index or \"%s\", received \"%s\"." % (GROUND_TRUTH_VIEW, view_side)
        return view_side

    assert isinstance(view_side, int), "Failed to parse display view. Expected an agent index, received type %r." % type(view_side).__name__
    assert 0 <= view_side < number_of_robots, "Failed to parse display view. Agent index %g is out of range for %g robots." % (view_side, number_of_robots)
    return view_side


def describe_view(view, halves: bool = False) -> str:
    """
    Names a view in prose, e.g. "agent 0", "ground truth", "agent 0 vs agent 1".

    With halves, a comparison view carries the glyph saying which half of a split tile is
    whose -- "agent 0 ◣ vs agent 1 ◥" -- which is the same convention the legend title
    uses. That is what a caption drawn over the tiles wants; a log record or a startup
    print, which has no tiles beside it to relate the glyphs to, wants the plain form.

    Args:
        view: One side, or a pair of sides, as returned by parse_display_views.
        halves (bool): Whether to mark which half of a split tile each side is drawn in.
    """
    if isinstance(view, (list, tuple)):
        template = "%s ◣ vs %s ◥" if halves else "%s vs %s"
        return template % (_describe_view_side(view[0]), _describe_view_side(view[1]))
    return _describe_view_side(view)


def _parse_tile_spec(tile_spec, grid_width_height: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Expands a tile specification into a list of (X,Y) tile coordinates.

    A tile is an [x, y] pair, and a specification is either one tile or a list of them,
    so every tile named in a configuration is written out in full.

    Args:
        tile_spec: The tile specification to expand.
        grid_width_height (tuple[int, int]): The width and height of the grid.
    """
    if tile_spec is None:
        return []

    assert not isinstance(tile_spec, dict), "Failed to parse tile specification. Tiles are given as [x, y] pairs; the {x: ..., y: ...} rectangle form is no longer accepted, so list the tiles out instead."
    assert isinstance(tile_spec, (list, tuple)), "Failed to parse tile specification. Expected an [x, y] pair or a list of them, received type %r." % type(tile_spec).__name__

    # A bare [x, y] pair describes a single tile
    if len(tile_spec) == 2 and all(isinstance(item, int) for item in tile_spec):
        return [_validate_tile((int(tile_spec[0]), int(tile_spec[1])), grid_width_height)]

    tiles = []
    for item in tile_spec:
        tiles.extend(_parse_tile_spec(item, grid_width_height))
    return tiles


def _validate_tile(tile: tuple[int, int], grid_width_height: tuple[int, int]) -> tuple[int, int]:
    """
    Checks that a tile coordinate lies inside the grid and returns it.

    Args:
        tile (tuple[int, int]): The (X,Y) tile coordinate to check.
        grid_width_height (tuple[int, int]): The width and height of the grid.
    """
    grid_width, grid_height = grid_width_height
    assert 0 <= tile[0] < grid_width and 0 <= tile[1] < grid_height, "Failed to parse tile (%g, %g). It lies outside the %gx%g grid." % (tile[0], tile[1], grid_width, grid_height)
    return tile


def initialize_communication_lines(
    grid_world,
    topology: Optional[nx.Graph],
    line_color: str = "#051E39",
    line_alpha: float = 0.25,
    line_width: float = 0.8,
    line_style: str = "--",
    section_color: Optional[str] = None,
    section_alpha: Optional[float] = None,
    section_width: Optional[float] = None,
    section_style: str = "-",
    message_colors: Optional[tuple] = None,
    message_alpha: float = 1.0,
    message_width_scale: float = 2.5,
    show_when_idle: bool = True
):
    """
    Builds a function that draws the communication topology onto the grid world figure, as a
    faint line between each pair of robots that can talk to each other.

    Lines run between where the robots actually are rather than between tile centers, so
    calling the returned function every simulation frame keeps them attached to the robots as
    they drive. They are drawn under the robots and over the tiles.

    An edge of the topology is an interface of the belief sheaf over it, and an interface
    either agrees -- both endpoints restricting to the same thing on their shared edge stalk,
    which is the sheaf condition holding locally -- or does not yet. Given the section_*
    styling, the returned function takes a mapping saying which interfaces have closed and
    draws those differently, so a video shows the sheaf condition being met one interface at a
    time rather than only at the end.

    Returns a function taking an optional {(u, v): bool} mapping, keyed by sorted pair. Given
    no topology, or a headless grid world with nothing to draw onto, that function does
    nothing, so a caller need not check whether a topology was configured or a figure exists.

    Args:
        grid_world (GridWorld): The grid world whose axes are drawn onto.
        topology (Optional[nx.Graph]): Who can talk to whom. See parse_communication_topology.
        line_color (str): Color of the lines.
        line_alpha (float): Opacity of the lines, kept faint so they read as context.
        line_width (float): Width of the lines.
        line_style (str): Matplotlib line style, dashed by default.
        section_color (Optional[str]): Color for an interface that agrees. None leaves every
            interface drawn the same way, whatever the mapping says.
        section_alpha (Optional[float]): Opacity for an interface that agrees; defaults to
            line_alpha.
        section_width (Optional[float]): Width for an interface that agrees; defaults to
            line_width.
        section_style (str): Line style for an interface that agrees, solid by default, so a
            closed interface reads as the settled thing it is against the dashed rest.
        message_colors (Optional[tuple]): (agreeing, disagreeing) colors for an interface with
            a message on it. None never lights one up, leaving the resting styling throughout.
            Deliberately separate from section_color: an interface can be lit by a message
            whether or not the run draws the settled state of its interfaces at all.
        message_alpha (float): Opacity of an interface while a message is on it. Full by
            default: a message is the one thing on the figure that is happening rather than
            holding, so it is the one thing drawn at full strength.
        message_width_scale (float): How much thicker an interface is drawn while a message is
            on it, as a multiple of the width it would otherwise have.
        show_when_idle (bool): Whether an interface is drawn at all when nothing is crossing it.
            False shows the topology only as it is used, so a line on the floor means a message
            is passing rather than that two agents could in principle talk. The topology is then
            no longer legible from a still frame, which is the trade.

    An edge carries a message on the iterations just after the agent at one end broadcasts, and
    is drawn bold and opaque for them -- the first of `message_colors` where the interface now
    agrees, the second where it does not, in the line style of that state where the run draws
    the states apart by style as well. A run then shows knowledge moving along the topology hop
    by hop rather than only the state it leaves behind, which is what the flow actually is.
    """
    if topology is None or grid_world.axes is None:
        def draw_no_communication_lines(sections: Optional[dict] = None,
                                        messaging: Optional[Iterable] = None) -> None:
            pass
        return draw_no_communication_lines

    for agent_index in topology.nodes:
        assert 0 <= agent_index < grid_world.number_of_robots, "Failed to initialize communication lines. Agent index %g is out of range for %g robots." % (agent_index, grid_world.number_of_robots)

    # One line per edge, created once and repositioned from then on. zorder sits above the
    # tile outlines at 1.2 and below the Robotarium's robot handles at 2.
    communication_lines = []
    for first, second in topology.edges:
        line = Line2D([], [], color=line_color, alpha=line_alpha, linewidth=line_width,
                      linestyle=line_style, zorder=1.4)
        grid_world.axes.add_line(line)
        communication_lines.append((first, second, line))

    def draw_communication_lines(sections: Optional[dict] = None,
                                 messaging: Optional[Iterable] = None) -> None:
        carrying = frozenset(tuple(sorted(edge)) for edge in (messaging or ()))

        for first, second, line in communication_lines:
            first_x, first_y = grid_world.robot_poses[0:2, first]
            second_x, second_y = grid_world.robot_poses[0:2, second]
            line.set_data([first_x, second_x], [first_y, second_y])

            edge = tuple(sorted((first, second)))
            agrees = bool(sections.get(edge, False)) if sections else False

            # What the interface looks like when nothing is happening on it: its settled state
            # where the run draws that, and the plain line where it does not.
            if section_color is not None and sections is not None and agrees:
                settled_color = section_color
                settled_alpha = section_alpha if section_alpha is not None else line_alpha
                settled_width = section_width if section_width is not None else line_width
                settled_style = section_style
            else:
                settled_color, settled_alpha = line_color, line_alpha
                settled_width, settled_style = line_width, line_style

            line.set_visible(show_when_idle or edge in carrying)

            if message_colors is not None and edge in carrying:
                # A message is on this interface right now. Bold and opaque, and coloured by
                # what the interface makes of what it just heard.
                line.set_color(message_colors[0] if agrees else message_colors[1])
                line.set_alpha(message_alpha)
                line.set_linewidth(settled_width * message_width_scale)
                # It keeps the line style of its settled state wherever the run draws that
                # state, so an interface lit while it has not closed still says which it is on
                # a surface where the two colors arrive the same -- the projected floor, where
                # every interface is black. Solid otherwise: the dashes of a plain topology are
                # its resting styling rather than a state worth carrying into the blink.
                line.set_linestyle(settled_style if section_color is not None and sections is not None else "-")
            else:
                line.set_color(settled_color)
                line.set_alpha(settled_alpha)
                line.set_linewidth(settled_width)
                line.set_linestyle(settled_style)

    return draw_communication_lines


def initialize_region_overlay(
    grid_world,
    regions,
    line_color: str = "#051E39",
    line_alpha: float = 0.5,
    line_width: float = 1.8,
    show_names: bool = True,
    name_color: str = "#051E39",
    name_alpha: float = 0.45,
    name_fontsize: float = 6.0
) -> None:
    """
    Draws the region partition onto the grid world figure: a heavier line along every
    boundary between two tiles of different regions, and optionally each region's name at
    its centroid.

    The partition is the abstraction the mission sheaf communicates through
    (`agsheaf.regions`), so drawing it is what lets a viewer read the two levels off one
    frame: tile colors are what an agent believes, region boundaries are the resolution at
    which the interfaces measure agreement. Static for a run, so this draws once at
    initialization rather than returning a per-frame function. Does nothing for a headless
    grid world, an absent partition, or the identity partition -- one boundary per tile
    would just repaint the grid.

    Args:
        grid_world (GridWorld): The grid world whose axes are drawn onto.
        regions: The `agsheaf.regions.Regions` partition, or None.
        line_color (str): Color of the boundary lines.
        line_alpha (float): Opacity of the boundary lines.
        line_width (float): Width of the boundary lines, heavier than the tile grid.
        show_names (bool): Whether to write each region's name at its centroid.
        name_color (str): Color of the region names.
        name_alpha (float): Opacity of the region names.
        name_fontsize (float): Size of the region names.
    """
    if regions is None or grid_world.axes is None or regions.is_identity():
        return

    grid_width, grid_height = regions.grid_width_height
    assert (grid_width, grid_height) == (grid_world.grid_width, grid_world.grid_height), \
        "Failed to draw the region overlay. The partition is over a %gx%g grid but the world is %gx%g." % (grid_width, grid_height, grid_world.grid_width, grid_world.grid_height)

    tile_width = grid_world.tile_width
    anchor = grid_world.bottom_left_tile_pos_xy

    def center(tile):
        return (float(anchor[0, 0] + tile_width * tile[0]),
                float(anchor[1, 0] + tile_width * tile[1]))

    # A segment along each internal edge whose two tiles lie in different regions. zorder
    # sits above the tile patches and outlines, below the robots at 2.
    for x in range(grid_width):
        for y in range(grid_height):
            cx, cy = center((x, y))
            here = regions.region_of((x, y))
            if x + 1 < grid_width and regions.region_of((x + 1, y)) != here:
                grid_world.axes.add_line(Line2D(
                    [cx + tile_width / 2, cx + tile_width / 2],
                    [cy - tile_width / 2, cy + tile_width / 2],
                    color=line_color, alpha=line_alpha, linewidth=line_width,
                    solid_capstyle="projecting", zorder=1.3))
            if y + 1 < grid_height and regions.region_of((x, y + 1)) != here:
                grid_world.axes.add_line(Line2D(
                    [cx - tile_width / 2, cx + tile_width / 2],
                    [cy + tile_width / 2, cy + tile_width / 2],
                    color=line_color, alpha=line_alpha, linewidth=line_width,
                    solid_capstyle="projecting", zorder=1.3))

    if show_names:
        for name in regions.names:
            tiles = regions.tiles(name)
            xs, ys = zip(*(center(t) for t in tiles))
            grid_world.axes.text(
                sum(xs) / len(xs), sum(ys) / len(ys), name,
                color=name_color, alpha=name_alpha, fontsize=name_fontsize,
                ha="center", va="center", zorder=1.35)


def initialize_status_caption(
    grid_world,
    enabled: bool = True,
    text_color: str = "#051E39",
    fontsize: float = 7.0,
    position: tuple[float, float] = (0.025, 0.93),
    horizontal_alignment: str = "left",
    vertical_alignment: str = "top",
    rotation: float = 0.0,
    weight: str = "normal",
    background: bool = True
):
    """
    Builds a function that writes a one-line caption onto the figure.

    Two things about a frame cannot be read off the tiles themselves. The first is whose
    beliefs they are: the same grid drawn for agent 0, for agent 2, or for the ground truth
    is three different pictures of the same moment, and nothing in the colors says which one
    is on screen. The second is where the flow has got to -- a sweep of the Laplacian happens
    between frames, so a video shows its effect but not the thing itself, the tiles simply
    being different from one step to the next. The caption is where both go, which is what
    turns a run into a legible record rather than a sequence of grids.

    Returns a function taking the caption to show, or "" for none. With enabled False, or a
    headless grid world with nothing to write onto, that function does nothing, so a caller
    need not check whether the caption was configured or a figure exists.

    Args:
        grid_world (GridWorld): The grid world whose axes are drawn onto.
        enabled (bool): Whether to draw the caption at all.
        text_color (str): Color of the caption.
        fontsize (float): Size of the caption.
        position (tuple[float, float]): Where it sits, in axes coordinates, where (0, 0) is the
            bottom left of the arena and (1, 1) the top right. The grid never fills the arena --
            it is square-tiled inside a 3.2x2.0 m rectangle -- so the margin the leftover strip
            leaves is where a caption can be drawn large without covering ground the robots
            drive over. Projected onto the floor, that is the difference between a caption the
            overhead footage can read and one it cannot.
        horizontal_alignment (str): Which side of the text `position` names, or "center".
        vertical_alignment (str): Which end of the text `position` names, or "center".
        rotation (float): Degrees counterclockwise. 90 runs the caption up the side of the
            arena, which is what fits a large one into the strip beside a wide grid.
        weight (str): Matplotlib font weight, "bold" being the other one worth having.
        background (bool): Whether to back the text with a white box. It keeps the caption
            readable over the tiles; drawn clear of them, on a floor that is already white,
            the box is a rectangle of glare and nothing else.
    """
    if not enabled or grid_world.axes is None:
        def set_no_status_caption(caption: str) -> None:
            pass
        return set_no_status_caption

    # Upper left by default, in axes coordinates, opposite the legend so the two never overlap,
    # and inset far enough to clear the arena's own border.
    status_text = grid_world.axes.text(
        position[0], position[1], "", transform=grid_world.axes.transAxes, color=text_color,
        fontsize=fontsize, fontweight=weight, rotation=rotation,
        ha=horizontal_alignment, va=vertical_alignment, zorder=3.0,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5) if background else None)

    def set_status_caption(caption: str) -> None:
        status_text.set_text(caption)

    return set_status_caption


# Creating closure safety_color_grid function

def initialize_safety_color_grid(
    grid_world,
    assigned_targets: np.ndarray,
    label_beliefs: Optional[list[dict[tuple[int, int], str]]] = None,
    ground_truth: Optional[dict[tuple[int, int], str]] = None,
    default_label: str = "safe",
    mark_other_agents_unsafe: bool = True,
    label_colors: Optional[dict[str, str]] = None,
    comparison_colors: Optional[dict[str, str]] = None,
    fill_ground_truth_safe: bool = False,
    show_legend: bool = False,
    legend_fontsize: str | float = "xx-small",
    fill_alpha: float = 0.45,
    assigned_number_fontsize: float = 7.0,
    assigned_number_color: str = "#051E39",
    static_assignment_numbers: bool = False,
    show_assignment_numbers: bool = True,
    contested_marker: Optional[str] = None,
    contested_color: str = "#051E39",
    contested_fontsize: float = 8.0,
    section_colors: Optional[tuple[str, str]] = None
):
    """
    Builds a function that draws tile-label beliefs onto the grid world figure, either
    one agent's own beliefs or two agents' beliefs held against each other.

    Each agent's labeling is assembled in layers, later layers overriding earlier ones:
        1. Every tile starts at default_label.
        2. The agent's static beliefs from label_beliefs are applied.
        3. If mark_other_agents_unsafe, the other agents' next steps are labeled "unsafe".
        4. The square the agent has been assigned is labeled "target".

    An agent's assigned square and its guesses about where the others were sent are both
    drawn as targets, in the one color, since the agent routes towards them all the same
    way. The assigned square carries the agent's number in its corner, so which is which
    stays readable: a numbered square is one an agent was told to reach, an unnumbered one
    is a guess it holds about somebody else. A view numbers only the squares it is entitled
    to know about -- an agent's own view numbers its own square, the ground truth view
    numbers every one.

    A view is either one side or a pair of sides, where a side is an agent index or
    "ground_truth". The same machinery therefore shows one agent's beliefs, two agents held
    against each other, the world as it really is, or an agent held against the truth.

    Every tile is drawn as two triangular halves split along the anti-diagonal. Given a
    single side, both halves carry that side's label, so each tile reads as one flat color.
    Given a pair, the lower-left half carries the first side's label and the upper-right
    half the second's, which makes the four ways two sides can relate directly legible:

        both_unknown  empty tile          neither side has a belief
        one_sided     half-filled tile    exactly one side has a belief
        agree         uniform tile        same label on both sides
        conflict      split tile          both have beliefs, and they differ

    Conflicts are additionally outlined, since a contradiction is the thing worth
    catching the eye. Tiles labeled "unknown" are left unfilled, and so, by default, are
    the safe tiles of a view of the ground truth alone -- see `fill_ground_truth_safe`.

    A labeling is all a view can draw, but an agent communicating over a belief sheaf holds
    more than one: a tile whose possibility set has emptied under disagreement keeps the
    agent's own label, so nothing in the fills says it is contested. Given contested_marker,
    the returned function takes those tiles per agent and marks them, which puts the sheaf's
    ∅ read-out on the figure instead of only in the log.

    Given a headless grid world there is nothing to draw onto, and the returned function does
    nothing, so a caller need not check whether a figure exists. What the configuration says
    is still checked here, since a headless run is a run of the same experiment.

    Args:
        grid_world (GridWorld): The grid world whose axes are drawn onto.
        assigned_targets (np.ndarray): A 2xN numpy array of the (X,Y) square assigned to each
            agent. See parse_assignments.
        label_beliefs (Optional[list[dict[tuple[int,int], str]]]): Per-agent static beliefs,
            mapping (X,Y) tile coordinates to labels. See parse_label_beliefs.
        ground_truth (Optional[dict[tuple[int,int], str]]): The world's true labeling, required
            for any view naming "ground_truth". See parse_ground_truth.
        default_label (str): The label given to tiles no belief is held about.
        mark_other_agents_unsafe (bool): Whether other agents' next steps are labeled unsafe.
        label_colors (Optional[dict[str, str]]): Fill color to draw each label with.
        comparison_colors (Optional[dict[str, str]]): Outline color for each way two agents
            can relate. Defaults outline conflicts only.
        fill_ground_truth_safe (bool): Whether a view of the ground truth alone fills the tiles
            it labels safe. The truth labels every tile, and away from the hazards and the
            targets that label is "safe", so filling them washes the whole arena in one color
            that says only that nothing is wrong there -- while the hazards and the targets,
            which are what the view is looked at for, have to be picked out of it. A view
            holding the truth against an agent fills them regardless: there "safe" is a claim
            the agent is making, which the truth can be short of or contradict.
        show_legend (bool): Whether to draw a legend for whichever view is being rendered. It
            sits inside the arena, so on a full grid it covers a corner tile; a run being read
            for where the robots went rather than for what the colors mean wants it off.
        legend_fontsize (str | float): Matplotlib size for the legend, which is what its
            footprint over the arena mostly comes down to.
        fill_alpha (float): Opacity of the tile fills, kept low enough to see the robots.
        assigned_number_fontsize (float): Size of the numeral drawn on an assigned square.
        assigned_number_color (str): Color of the numeral drawn on an assigned square.
        static_assignment_numbers (bool): Whether every assigned square stays numbered whatever
            the view. False numbers only the squares the view is entitled to know about, which
            says something true -- an agent knows where it was sent and only guesses at the
            rest -- at the cost of the numerals appearing and vanishing as the view moves. On a
            run whose view changes every move that reads as flicker rather than as epistemics,
            and the assignments are the one thing on the floor that never moves, so True draws
            them as the fixed reference they are.
        show_assignment_numbers (bool): Whether an assigned square carries a numeral at all.
            False leaves the target fills unlabelled, which is what a projected run wants when
            the numeral is too small on the floor to be read: a numeral nobody can resolve is
            a smudge over a tile rather than an answer to which target is whose.
        contested_marker (Optional[str]): The glyph marking a tile an agent's belief sheaf
            holds no possibility for. None draws no marks, whatever the caller passes.
        contested_color (str): Color of that glyph.
        contested_fontsize (float): Size of that glyph.
        section_colors (Optional[tuple[str, str]]): The colors initialize_communication_lines
            draws an interface in when it does and does not agree. The lines are not this
            function's to draw, but the legend is, and a mark nothing names is a mark nobody
            can read.
    """

    # Defining important values
    bottom_left_xy_point = grid_world.bottom_left_tile_pos_xy
    grid_width_height = (grid_world.grid_width, grid_world.grid_height)

    assert isinstance(grid_width_height, tuple)
    assert all(isinstance(item, int) for item in grid_width_height)
    assert isinstance(bottom_left_xy_point, np.ndarray)
    assert len(bottom_left_xy_point) == 2
    assert default_label in TILE_LABELS, "Failed to initialize safety color grid. The default label \"%s\" is not one of %s." % (default_label, str(TILE_LABELS))
    assert assigned_targets.shape == (2, grid_world.number_of_robots), "Failed to initialize safety color grid. Expected assigned targets of shape 2x%g, received shape %gx%g." % (grid_world.number_of_robots, assigned_targets.shape[0], assigned_targets.shape[1])
    if label_beliefs is not None:
        assert len(label_beliefs) == grid_world.number_of_robots, "Failed to initialize safety color grid. Expected label beliefs for %g robots, received %g." % (grid_world.number_of_robots, len(label_beliefs))

    # Resolving colors used to draw each label
    tile_label_colors = dict(DEFAULT_LABEL_COLORS)
    if label_colors is not None:
        for label in label_colors:
            assert label in TILE_LABELS, "Failed to initialize safety color grid. The label \"%s\" is not one of %s." % (label, str(TILE_LABELS))
        tile_label_colors.update(label_colors)

    # Resolving colors used to draw each pairwise comparison
    tile_comparison_colors = dict(DEFAULT_COMPARISON_COLORS)
    if comparison_colors is not None:
        for label in comparison_colors:
            assert label in COMPARISON_LABELS, "Failed to initialize safety color grid. The comparison \"%s\" is not one of %s." % (label, str(COMPARISON_LABELS))
        tile_comparison_colors.update(comparison_colors)

    # Nothing to draw onto, so nothing is drawn. The colors above were still checked, so a
    # headless run rejects the same configurations an animated one does.
    if grid_world.axes is None:
        def create_no_safety_color_grid(robot_coords: np.ndarray,
                                        robot_id: int | str | tuple,
                                        contested: Optional[dict] = None):
            pass
        return create_no_safety_color_grid

    # Creating constant grid position list
    grid_position_list = _create_grid_position_list(grid_width_height)

    # Initializing patches info
    tile_width = grid_world.tile_width
    patch_width = tile_width * 0.95
    linewidth = 2.0

    # Initializing patches for each tile. Each tile gets a whole-tile fill used when both
    # sides carry the same label, two triangular halves used when they differ, and an outline
    # drawn over them. All sit below the robots (zorder 1.5).
    tile_fill_patches = {}
    tile_half_patches = {}
    tile_outline_patches = {}
    for tile_index in grid_position_list:
        # Locating the bottom-left corner of this tile's patches
        tile_index_np = np.reshape(np.array(tile_index), (2, 1))
        corner_x, corner_y = tuple(((tile_index_np * tile_width + bottom_left_xy_point) - patch_width/2).flatten().tolist())

        # Splitting the tile along its anti-diagonal: lower-left half, then upper-right half
        lower_left_half = plt.Polygon(
            [(corner_x, corner_y), (corner_x + patch_width, corner_y), (corner_x, corner_y + patch_width)],
            closed=True, edgecolor='none', facecolor='none', alpha=fill_alpha, zorder=1.1)
        upper_right_half = plt.Polygon(
            [(corner_x + patch_width, corner_y), (corner_x + patch_width, corner_y + patch_width), (corner_x, corner_y + patch_width)],
            closed=True, edgecolor='none', facecolor='none', alpha=fill_alpha, zorder=1.1)

        # A whole-tile fill, so that a tile both sides agree on is drawn as one flat square
        # rather than as two halves meeting along a seam
        whole_fill = plt.Rectangle((corner_x, corner_y), width=patch_width, height=patch_width,
                                   edgecolor='none', facecolor='none', alpha=fill_alpha, zorder=1.1)

        outline = plt.Rectangle((corner_x, corner_y), width=patch_width, height=patch_width,
                                linewidth=linewidth, edgecolor='none', facecolor='none', zorder=1.2)

        tile_fill_patches[tile_index] = whole_fill
        tile_half_patches[tile_index] = (lower_left_half, upper_right_half)
        tile_outline_patches[tile_index] = outline
        for patch in (whole_fill, lower_left_half, upper_right_half, outline):
            grid_world.axes.add_patch(patch)

    # Numbering each robot's assigned square. Every robot's target beliefs are drawn in the
    # one color, so the numeral is what tells the square a robot was told to reach from the
    # squares it merely guesses others were sent to. Positions never change, so they are
    # placed once here and only shown or hidden as the view changes. The numeral sits in the
    # tile's lower-left corner, clear of the badge the robot carries once it arrives. Left
    # undrawn altogether where the numerals are turned off, so the loop that shows and hides
    # them below runs over nothing rather than needing a guard of its own.
    assignment_number_labels = {}
    for robot_index in range(assigned_targets.shape[1] if show_assignment_numbers else 0):
        assigned_tile = np.reshape(assigned_targets[:, robot_index], (2, 1))
        tile_center_x, tile_center_y = tuple((assigned_tile * tile_width + bottom_left_xy_point).flatten().tolist())
        assignment_number_labels[robot_index] = grid_world.axes.text(
            tile_center_x - patch_width/2 * 0.78, tile_center_y - patch_width/2 * 0.78,
            agent_label(robot_index), color=assigned_number_color, fontsize=assigned_number_fontsize,
            ha="center", va="center", zorder=1.3, visible=False)

    # A glyph per tile marking it contested, placed once and only shown or hidden, the same
    # way the assignment numerals are. It sits in the tile's center, above the fills.
    contested_marks = {}
    if contested_marker is not None:
        for tile_index in grid_position_list:
            tile_index_np = np.reshape(np.array(tile_index), (2, 1))
            center_x, center_y = tuple((tile_index_np * tile_width + bottom_left_xy_point).flatten().tolist())
            contested_marks[tile_index] = grid_world.axes.text(
                center_x, center_y, contested_marker, color=contested_color,
                fontsize=contested_fontsize, ha="center", va="center", zorder=1.3, visible=False)

    # A legend per view, built the first time that view is drawn and then only shown or
    # hidden. Rendering several views of one run alternates between them every frame, and
    # rebuilding a legend that often is far too slow.
    legend_artists = {}

    # The resolved labeling depends only on where the robots are, so drawing several views of
    # one frame resolves it once rather than once per view
    label_cache = {}

    def create_safety_color_grid(robot_coords: np.ndarray, # Like [[0, 4], [1, 3]], where top row is x and bottom is y
                             robot_id: int | str | tuple, # Side whose labels are drawn, or a pair of sides to compare
                             contested: Optional[dict] = None # Per agent, the tiles its belief sheaf holds no possibility for
                            ):

        # Variable assertions
        assert robot_coords.shape[0] == 2, "Robot coords are not as expected."
        assert robot_coords.shape[1] == assigned_targets.shape[1], "Robot and assigned target dimensions do not line up."

        number_of_robots = robot_coords.shape[1]
        comparing = isinstance(robot_id, (tuple, list))
        if comparing:
            assert len(robot_id) == 2, "Robot ID is not valid. Expected a single side or a pair of sides to compare, received %g." % len(robot_id)
            view_sides = list(robot_id)
        else:
            view_sides = [robot_id]
        for side in view_sides:
            if side == GROUND_TRUTH_VIEW:
                assert ground_truth is not None, "Cannot draw the ground truth view: no ground truth was given to initialize_safety_color_grid."
                continue
            assert isinstance(side, (int, np.integer)), "Robot ID is not valid."
            assert 0 <= int(side) < number_of_robots, "Robot ID %g is not valid for %g robots." % (side, number_of_robots)
        view_sides = [side if side == GROUND_TRUTH_VIEW else int(side) for side in view_sides]

        # The same resolution the planner reads, so robots move by what is drawn for them.
        # The resolution depends on the beliefs as well as on where the robots are, and a
        # communicating run rewrites the belief dicts in place between steps, so both go in
        # the cache key: keying on the coordinates alone would redraw a settled frame from
        # before the sweep whenever two consecutive steps happened to move nobody.
        coords_key = (robot_coords.tobytes(),
                      None if label_beliefs is None else tuple(sorted(beliefs.items()) for beliefs in label_beliefs))
        if label_cache.get("key") != coords_key:
            label_cache["key"] = coords_key
            label_cache["labels"] = resolve_tile_labels(
                robot_coords, grid_width_height, assigned_targets,
                label_beliefs=label_beliefs,
                default_label=default_label,
                mark_other_agents_unsafe=mark_other_agents_unsafe)
        robot_safety = label_cache["labels"]

        # Result assertions
        assert len(robot_safety) == robot_coords.shape[1], "Robot safety wrapper dictionary has wrong size."

        # Deciding what each half of a tile shows. A single side fills both halves, so its
        # tiles read as one flat color; a pair fills one half each, so they split where the
        # two disagree.
        labels_by_side = {side: (ground_truth if side == GROUND_TRUTH_VIEW else robot_safety[side]) for side in view_sides}
        if comparing:
            labels_lower_left = labels_by_side[view_sides[0]]
            labels_upper_right = labels_by_side[view_sides[1]]
            comparisons = _compare_tile_labels(labels_lower_left, labels_upper_right, grid_position_list)
            legend_key = ("comparison", view_sides[0], view_sides[1])
        else:
            labels_lower_left = labels_by_side[view_sides[0]]
            labels_upper_right = labels_lower_left
            comparisons = None
            legend_key = ("labels", view_sides[0])

        # The world's own account of the ground carries a label on every tile, and everywhere
        # away from the hazards and the targets that label is "safe": filled, it is a wash of
        # one color over the whole arena saying only that the tiles it covers are the ones
        # nothing is wrong with. So the truth drawn on its own leaves them empty and shows what
        # is actually worth looking at. Held against an agent it is a different statement --
        # "safe" is then a claim that agent is making, which the other side can be short of or
        # contradict -- so a comparison fills it like any other label.
        view_colors = tile_label_colors
        if not comparing and view_sides[0] == GROUND_TRUTH_VIEW and not fill_ground_truth_safe:
            view_colors = {**tile_label_colors, "safe": "none"}

        # Assigning patch properties based on input values
        for tile_index in grid_position_list:
            label_lower_left = labels_lower_left[tile_index]
            label_upper_right = labels_upper_right[tile_index]
            if label_lower_left not in view_colors or label_upper_right not in view_colors:
                raise ValueError("Tile in robot safety dictionary does not have valid tile status.")

            lower_left_half, upper_right_half = tile_half_patches[tile_index]
            if label_lower_left == label_upper_right:
                # Both sides say the same thing: draw one flat square
                tile_fill_patches[tile_index].set_facecolor(view_colors[label_lower_left])
                lower_left_half.set_facecolor('none')
                upper_right_half.set_facecolor('none')
            else:
                # The sides differ: split the tile so both beliefs stay readable
                tile_fill_patches[tile_index].set_facecolor('none')
                lower_left_half.set_facecolor(view_colors[label_lower_left])
                upper_right_half.set_facecolor(view_colors[label_upper_right])

            # Outlining tiles the two agents relate on in a noteworthy way, conflicts by default
            outline_color = 'none' if comparisons is None else tile_comparison_colors[comparisons[tile_index]]
            tile_outline_patches[tile_index].set_edgecolor(outline_color)

        # Numbering the assigned squares. Statically, that is all of them throughout: the
        # assignments are fixed for a run and are the one fixed reference on the floor, so a
        # view changing every move should not take them with it. Otherwise only the squares the
        # view is entitled to know about -- a robot knows where it was sent and only guesses at
        # the others', so its own view numbers one square, while the ground truth view numbers
        # them all, being the world's own account of who was sent where.
        if static_assignment_numbers:
            numbered_robots = set(range(number_of_robots))
        else:
            numbered_robots = set()
            for side in view_sides:
                if side == GROUND_TRUTH_VIEW:
                    numbered_robots.update(range(number_of_robots))
                else:
                    numbered_robots.add(side)
        for robot_index, number_label in assignment_number_labels.items():
            number_label.set_visible(robot_index in numbered_robots)

        # Marking the tiles contested for whichever agents this view shows. The ground truth
        # is not an agent and holds no sheaf stalk, so a view of it alone marks nothing.
        if contested_marks:
            marked = set()
            for side in view_sides:
                if side != GROUND_TRUTH_VIEW:
                    marked.update(tuple(tile) for tile in (contested or {}).get(side, ()))
            for tile_index, mark in contested_marks.items():
                mark.set_visible(tile_index in marked)

        # Showing the legend belonging to whichever view is being rendered, building it the
        # first time that view comes up
        if show_legend:
            if legend_key not in legend_artists:
                legend_artists[legend_key] = _draw_label_legend(
                    grid_world.axes, view_colors, tile_comparison_colors, view_sides, fill_alpha,
                    contested_marker=contested_marker, contested_color=contested_color,
                    section_colors=section_colors, fontsize=legend_fontsize)
            for key, legend in legend_artists.items():
                if legend is not None:
                    legend.set_visible(key == legend_key)

    return create_safety_color_grid


def _compare_tile_labels(labels_a: dict, labels_b: dict, grid_position_list: list[tuple]) -> dict[tuple, str]:
    """
    Compares two agents' tile labels, tile by tile.

    A tile is "agree" when both agents give it the same label, "conflict" when both label it
    but differently, "one_sided" when exactly one of them labels it, and "both_unknown" when
    neither does. Note that "one_sided" is not disagreement: one agent simply knows more.

    Args:
        labels_a (dict): The first agent's labels, keyed by (X,Y) tile coordinate.
        labels_b (dict): The second agent's labels, keyed by (X,Y) tile coordinate.
        grid_position_list (list[tuple]): The (X,Y) coordinates of every tile in the grid.
    """
    comparison = {}
    for tile in grid_position_list:
        label_a = labels_a[tile]
        label_b = labels_b[tile]
        if label_a == label_b:
            comparison[tile] = "both_unknown" if label_a == "unknown" else "agree"
        elif label_a == "unknown" or label_b == "unknown":
            comparison[tile] = "one_sided"
        else:
            comparison[tile] = "conflict"
    return comparison


def _draw_label_legend(axes, tile_label_colors: dict[str, str], tile_comparison_colors: dict[str, str],
                       view_sides: list, fill_alpha: float,
                       contested_marker: Optional[str] = None, contested_color: str = "#D90368",
                       section_colors: Optional[tuple[str, str]] = None,
                       fontsize: str | float = "xx-small"):
    """
    Draws a legend for one view: the label fills, plus the outlines and the half-tile
    convention when two sides are being compared. Returns the legend so that a caller drawing
    several views can keep one per view and show whichever it needs, or None when the view
    has nothing to put in a legend.

    The legend is re-added as a plain artist, since axes.legend() keeps only the newest and
    would drop the legends belonging to the other views.

    Args:
        axes: The matplotlib axes to draw the legend onto.
        tile_label_colors (dict[str, str]): Fill color used for each label.
        tile_comparison_colors (dict[str, str]): Outline color used for each comparison.
        view_sides (list): The side being drawn, or the pair being compared.
        fill_alpha (float): Opacity the tile fills are drawn with.
        contested_marker (Optional[str]): The glyph marking a contested tile, if any are marked.
        contested_color (str): Color of that glyph.
        section_colors (Optional[tuple[str, str]]): The colors an interface is drawn in when it
            does and does not agree, if the lines are styled that way.
        fontsize (str | float): Matplotlib size for the entries and the title.
    """
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=tile_label_colors[label], edgecolor="none", alpha=fill_alpha, label=label)
        for label in TILE_LABELS if tile_label_colors[label] != "none"
    ]

    if len(view_sides) == 2:
        # Each side on its own line, with the glyph saying which half of a split tile is whose
        title = "%s ◣\n%s ◥" % (_describe_view_side(view_sides[0]), _describe_view_side(view_sides[1]))
        handles.extend([
            plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=tile_comparison_colors[comparison], linewidth=2.0, label=comparison)
            for comparison in COMPARISON_LABELS if tile_comparison_colors[comparison] != "none"
        ])
    else:
        title = _describe_view_side(view_sides[0])

    # The sheaf's own marks. Naming them here is what keeps a video self-describing: a
    # contested tile and a "conflict" outline are different things -- the outline is two views
    # of the same tile disagreeing, the mark is the network having no label left for it at all.
    if contested_marker:
        handles.append(plt.Line2D([], [], linestyle="none", marker="$%s$" % contested_marker,
                                  color=contested_color, markersize=6, label="contested: no label left"))
    if section_colors is not None:
        agreeing_color, disagreeing_color = section_colors
        handles.extend([
            plt.Line2D([], [], color=disagreeing_color, linewidth=1.6, label="interface: not yet agreed"),
            plt.Line2D([], [], color=agreeing_color, linewidth=1.6, label="interface: agrees"),
        ])

    if not handles:
        return None

    # The legend sits inside the arena, so every row of it is arena a tile could have been
    # read in. The sheaf marks made it seven rows deep, enough to cover a corner tile, hence
    # the tight geometry: short handles, little padding, and a size the caller sets.
    legend = axes.legend(handles=handles, title=title, loc="upper right", bbox_to_anchor=(0.99, 0.98),
                         fontsize=fontsize, title_fontsize=fontsize, framealpha=0.8,
                         handlelength=1.1, handleheight=0.7, handletextpad=0.5,
                         labelspacing=0.3, borderpad=0.4, borderaxespad=0.0)
    axes.add_artist(legend)
    return legend


def _describe_view_side(view_side) -> str:
    """
    Names a view side for the legend.

    Args:
        view_side: The agent index or "ground_truth" being drawn.
    """
    return "ground truth" if view_side == GROUND_TRUTH_VIEW else "agent %s" % agent_label(view_side)

# Making helper functions for create_safety_color_grid

def _create_grid_position_list(grid_width_height) -> list[tuple]:
    # Creating width/height index (Check this)
    grid_width = range(grid_width_height[0])
    grid_height = range(grid_width_height[1])
    grid_position_list = [(col, row) for row in grid_height for col in grid_width]

    assert len(grid_position_list) == len(grid_width) * len(grid_height), "Grid position list size is wrong."
    
    return grid_position_list

def _determine_robot_tile_position(robot_coords, grid_position_list) -> dict[int, tuple]:
    # Making helper function to determine what robot is on what tile based on distance from tile
    number_robots = np.arange(robot_coords.shape[1])
    robot_tile_positions = {}

    for robot in number_robots:
        robot_coordinates = robot_coords[:2, robot]

        assert tuple(robot_coordinates) in grid_position_list
        
        robot_tile_positions[robot] = tuple(robot_coordinates)
    
    return robot_tile_positions
    
def _determine_assigned_tile_indices(assigned_targets, grid_position_list) -> dict[int, tuple]:
    # Determining tile tuple from each robot's assigned square
    number_robots = np.arange(assigned_targets.shape[1])
    assigned_tile_indices = {}

    for robot in number_robots:
        assigned_coordinates = assigned_targets[:2, robot]

        assert tuple(assigned_coordinates) in grid_position_list

        assigned_tile_indices[int(robot)] = tuple(assigned_coordinates)

    return assigned_tile_indices

def _determine_robot_safety(robot_coords, grid_position_list, robot_tile_positions, assigned_tile_indices,
                            label_beliefs=None,
                            default_label="safe",
                            mark_other_agents_unsafe=True):
    # Making robot tile dictionaries (0 --> N - 1)
    number_robots = np.arange(robot_coords.shape[1])
    robot_safety = {}
    for robot in number_robots:
        robot = int(robot)

        # Layer 1: tiles the robot holds no belief about
        tile_labels = {tile: default_label for tile in grid_position_list}

        # Layer 2: static beliefs programmed for this robot
        if label_beliefs is not None:
            for tile, label in label_beliefs[robot].items():
                if tile in tile_labels:
                    tile_labels[tile] = label

        # Layer 3: the other robots' next steps are unsafe
        if mark_other_agents_unsafe:
            robot_other_tiles = {bot: tile_index for bot, tile_index in robot_tile_positions.items() if bot != robot} # Tile indices besides self one
            for tile in set(robot_other_tiles.values()):
                tile_labels[tile] = "unsafe"

        # Layer 4: the square this robot has been assigned is a target it holds with
        # certainty, so it goes on last and overrides any belief underneath it
        tile_labels[assigned_tile_indices[robot]] = "target"

        robot_safety[robot] = tile_labels

    return robot_safety
