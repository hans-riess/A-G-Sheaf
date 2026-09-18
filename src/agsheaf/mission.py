"""
The drive loop every experiment runs, in one place.

The runners -- exp/run.py in its several arms, and the Robotarium bundle --
differ in what they draw and what they record, and not at all in how a run proceeds: agents
plan a tile, drive to it, communicate, and do it again until everyone is home. That loop used
to be written out three times, which is how `asynchronous_moves` came to work in one of them
and be silently ignored by the other two.

So it lives here, and the runners supply the parts that genuinely differ, as callbacks: what to
draw on a control iteration, what to do with a Laplacian iterate, and what "hold still and look
at this" means -- stopping the robots on a testbed floor, or writing more frames to a video.

One loop covers both ways of moving, because they differ in a single predicate: who is allowed
to plan. Synchronously an agent may plan only when every agent is idle, which is the barrier --
a grid step is not over until the last robot is on its tile. Asynchronously an agent plans the
moment it arrives, whoever else is still driving.
"""

from __future__ import annotations

import numpy as np

from .gridsheaf import communicate, forced_empty, observe_tiles, parse_sweeps_per_step, reliance_contract
from .gridworld import (TileReservations, choose_agent_step, describe_view, parse_move_schedule,
                        resolve_tile_labels, route_corridor, score_agent_step, score_robot_step)
# agsheaf.regionsheaf is imported lazily inside _read_the_region_monitor: only a run whose
# sheaf carries a region partition needs it, and the Robotarium submission bundle -- which
# ships this module -- does not carry the region machinery.

#: Seconds of testbed time one control iteration takes
TICK_SECONDS = 0.033


class Mission:
    """
    Drives every agent to its assigned square, once.

    The assignment is fixed for a run, so this is a single pass rather than a round per goal:
    agents step until all have arrived or until one of the budgets stops them.
    """

    def __init__(self, config, grid_world, assigned_targets, label_beliefs, *,
                 belief_sheaf=None, ground_truth=None, experiment_log=None,
                 display_views=(0,), on_tick=None, on_sweep=None, on_hold=None,
                 announce=print):
        """
        Args:
            config (dict): The merged experiment configuration. Every key the loop obeys is read
                from here, which is what keeps the runners in step with each other.
            grid_world (GridWorld): The world to drive.
            assigned_targets (np.ndarray): A 2xN array of the square each agent must reach.
            label_beliefs (dict): Each agent's tile beliefs, as parse_label_beliefs returns them.
                Mutated in place by observation and by the flow.
            belief_sheaf (Optional[ContractSheaf]): The sheaf the agents communicate over, or
                None for a run where nobody talks.
            ground_truth (Optional[dict]): The world's true labeling, or None not to score.
            experiment_log (Optional[ExperimentLog]): Where steps and poses are recorded.
            display_views (Sequence): The views the projection cycles through.
            on_tick (Optional[Callable]): `on_tick(coords, view)`, once per control iteration,
                before the world is stepped. Where a runner draws and records poses.
            on_sweep (Optional[Callable]): `on_sweep(index, conflicts, changed)`, after each
                Laplacian sweep, with the assignment as that sweep left it.
            on_hold (Optional[Callable]): `on_hold(coords, view)`, to hold the run on one view
                with the agents stationary. Called only for a synchronous view tour, there being
                no moment in an asynchronous run at which everybody is standing still.
            announce (Callable): Where the loop's running commentary goes.
        """
        self.config = config
        self.grid_world = grid_world
        self.assigned_targets = assigned_targets
        self.label_beliefs = label_beliefs
        self.belief_sheaf = belief_sheaf
        self.ground_truth = ground_truth
        self.experiment_log = experiment_log
        self.display_views = list(display_views)
        self.announce = announce

        self._on_tick = on_tick
        self._on_sweep = on_sweep
        self._on_hold = on_hold

        gridworld_cfg = config["gridworld"]
        goals_cfg = config.get("goals") or {}
        beliefs_cfg = config.get("beliefs") or {}
        planning_cfg = config.get("planning") or {}
        scoring_cfg = config.get("scoring") or {}
        sheaf_cfg = config.get("sheaf") or {}

        self.number_of_agents = gridworld_cfg["num_agents"]
        self.grid_width_height = (gridworld_cfg["grid_width"], gridworld_cfg["grid_height"])

        self.asynchronous_moves = bool(gridworld_cfg.get("asynchronous_moves", False))
        self.move_schedule = parse_move_schedule(gridworld_cfg.get("move_schedule"),
                                                 self.number_of_agents,
                                                 seed=config.get("seed") or 0)

        self.max_steps = goals_cfg.get("max_steps", 100)
        self.max_ticks = goals_cfg.get("max_ticks", 18000)
        self.stall_ticks = goals_cfg.get("stall_ticks", 600)

        self.tile_costs = planning_cfg.get("tile_costs")
        self.tile_scores = scoring_cfg.get("tile_scores")
        self.route_around_after_ticks = planning_cfg.get("route_around_after_ticks", 150)
        assert self.route_around_after_ticks is None or (isinstance(self.route_around_after_ticks, int) and self.route_around_after_ticks > 0), \
            "Failed to read the planning configuration. The route_around_after_ticks entry must be a positive number of control iterations, or null to let agents wait indefinitely. Received %r." % (self.route_around_after_ticks,)

        self.belief_options = dict(
            label_beliefs=label_beliefs,
            default_label=beliefs_cfg.get("default", "safe"),
            mark_other_agents_unsafe=beliefs_cfg.get("mark_other_agents_unsafe", True),
        )
        self.observe_on_arrival = bool(beliefs_cfg.get("observe_on_arrival", False)) and ground_truth is not None
        self.tour_views_every = beliefs_cfg.get("tour_views_every", "never")
        self.switch_view_every = beliefs_cfg.get("switch_view_every", "never")
        self.follow_moving_agent = bool(beliefs_cfg.get("follow_moving_agent", False))

        self.sweeps_per_step = parse_sweeps_per_step(sheaf_cfg.get("sweeps_per_step", 1),
                                                     sheaf_cfg.get("max_sweeps", 100))
        self.sweep_every_ticks = sheaf_cfg.get("sweep_every_ticks", 15)
        assert self.sweep_every_ticks is None or (isinstance(self.sweep_every_ticks, int) and self.sweep_every_ticks > 0), \
            "Failed to read the sheaf configuration. The sweep_every_ticks entry must be a positive number of control iterations, or null to sweep only when agents move. Received %r." % (self.sweep_every_ticks,)
        self.mission_monitor = bool(sheaf_cfg.get("mission_monitor", False)) and belief_sheaf is not None

        # The region-abstracted sheaf carries mission contracts the monitor reads back as
        # assumption violations and turns into a planner cost overlay; see
        # _read_the_mission_monitor. The legacy bijective sheaf keeps the old reliance
        # read-out. region_penalty is the extra cost of entering one tile of a region the
        # network has flagged hazardous or a higher-priority neighbor has claimed -- finite,
        # so a warned-off region is expensive rather than forbidden.
        contracts_cfg = config.get("contracts") or {}
        self.region_sheaf = (belief_sheaf is not None
                             and belief_sheaf.graph.get('mode') == 'regions')
        self.region_penalty = float(contracts_cfg.get("region_penalty", 12.0))
        self._extra_costs: dict = {}

        # How long an interface stays lit after a message crosses it. A sweep is instantaneous
        # and would otherwise be invisible between two frames; this is how many control
        # iterations it is held on screen afterwards.
        self.message_blink_ticks = (config.get("rendering") or {}).get("message_blink_ticks", 10)

        # --- run state -----------------------------------------------------
        agents = self.number_of_agents
        self.view_index = 0
        self.step_count = 0
        self.tick_count = 0
        self.rounds_moved = 0
        self.ticks_since_a_move = 0
        self.agent_scores = np.zeros(agents)
        self.reached_target = np.zeros(agents, dtype=bool)
        self.moves_made = np.zeros(agents, dtype=int)

        # Control iterations each agent has spent wanting to move and finding every move
        # blocked. Counts only that: an agent driving, or parked on its goal, is not being
        # kept from anything.
        self.blocked_ticks = np.zeros(agents, dtype=int)
        self._reported_stalled = frozenset()

        # Which tiles each agent holds. Synchronously this is just their positions -- every
        # agent is on its tile whenever anything is decided. Asynchronously an agent in transit
        # holds the tile it is driving out of as well, since nobody else may be sent onto
        # ground it has not left.
        self.reservations = TileReservations(grid_world.robots_target_tile_coord_xy)

        # Everyone has to have broadcast since the last thing the flow changed before it can be
        # called settled. Under a full broadcast that is the first quiet sweep, which is what a
        # synchronous run sees; under the partial broadcasts of an asynchronous run it is not,
        # and a sweep that changed nothing proves only that the agents who spoke had nothing to
        # say (see test_async).
        self._everyone = set(range(agents)) if belief_sheaf is not None else set()
        self._broadcast_since_change = set()
        self.communication_settled = belief_sheaf is None
        self._last_sweep_changed = False
        self.sweeps_run = 0

        # Whose turn it is to claim a tile, and which of them have not yet taken it
        self._turn_pending = set()

        # What the flow has turned up since the last step record. Synchronously a step is a
        # round and these are filled and emptied inside it; asynchronously the sweeps happen
        # between the steps, so they accumulate and are carried by whichever record comes next.
        self._step_conflicts = []
        self._step_sweeps = []

        # The interfaces the last sweep sent something along, and when. An edge carries a
        # message when the agent at one end of it broadcast, which is what the firing set says.
        self._message_edges = frozenset()
        self._message_tick = None

    # -- what the runner needs while a run is in progress --------------------

    @property
    def robot_coords(self):
        """The tile each agent holds: where it is, or where it is driving to."""
        return self.reservations.destinations

    @property
    def view(self):
        """The view the projection is on."""
        return self.display_views[self.view_index % len(self.display_views)]

    @property
    def messaging_edges(self):
        """
        The interfaces with a message on them right now, as sorted (u, v) pairs.

        An edge carries a message when the agent at one end of it broadcast, so this is the
        firing set of the last sweep read as edges. It empties again `message_blink_ticks`
        control iterations later: a sweep takes no time, and something that happens between one
        frame and the next cannot be filmed unless it is held.
        """
        if self._message_tick is None or not self.message_blink_ticks:
            return frozenset()
        if self.tick_count - self._message_tick >= self.message_blink_ticks:
            return frozenset()
        return self._message_edges

    def record_sweep(self, entry):
        """
        Adds an entry to the sweeps the next step record will carry. For a runner that
        snapshots the sheaf inside its `on_sweep`.

        Args:
            entry (dict): Whatever the runner wants kept about that iterate.
        """
        self._step_sweeps.append(entry)

    # -- the run -------------------------------------------------------------

    def agents_at_goal(self):
        """
        The agents whose held tile is the square they were assigned -- including one still
        driving onto it, which is what makes this the right notion for planning: nobody else
        may take that tile, and the agent itself has nowhere further to go.
        """
        return frozenset(i for i in range(self.number_of_agents)
                         if tuple(self.reservations.destinations[:, i]) == tuple(self.assigned_targets[:, i]))

    def finished(self):
        """
        Whether every agent is on its assigned square and standing on it.

        Holding a tile is not being on it. An agent that has just departed for its goal already
        counts as at its goal -- it is going nowhere else -- but the run is not over until it
        has physically arrived, or the last move would be cut off partway by whatever runs next.
        """
        return (len(self.agents_at_goal()) == self.number_of_agents
                and self.grid_world.robots_done_moving())

    def run(self):
        """
        Drives the run to its end and returns (all_arrived, stop_reason).

        `stop_reason` is None when everyone arrived, and otherwise names which budget ran out.
        """
        agents = self.number_of_agents
        self.announce("Driving %d agents to their assigned squares on a %dx%d grid"
                      % (agents, self.grid_width_height[0], self.grid_width_height[1]))
        self.announce("Motion: %s" % ("asynchronous -- each agent departs as soon as it arrives"
                                      if self.asynchronous_moves else
                                      "synchronous -- every agent steps together"))

        while (not self.finished() and self.tick_count < self.max_ticks
               and self.ticks_since_a_move < self.stall_ticks
               and (self.asynchronous_moves or self.rounds_moved < self.max_steps)):
            # An agent that has stopped moving is on its tile, so the tile behind it is free
            idle = [i for i in range(agents) if self.grid_world.robot_done_moving(i)]
            for i in idle:
                self.reservations.arrive(i)

            # The agents keep talking while they drive. A sweep of the Laplacian takes no time
            # on the testbed, so there is nothing to be gained by only running it when somebody
            # is about to move: an agent crossing a tile is not busy, and a run is seconds of
            # driving for every decision. On this clock the flow advances a hop twice a second
            # rather than once per arrival, and an agent that arrives has been listening the
            # whole way there. Everyone broadcasts, which makes it a synchronous flow on its own
            # clock; the partial firing set earns its keep in the sweep below.
            if (self.asynchronous_moves and self.sweep_every_ticks
                    and self.tick_count % self.sweep_every_ticks == 0):
                self._run_sweeps(None)

            planning = self._agents_allowed_to_plan(idle)
            if planning:
                self._take_a_step(planning)

            # One control iteration, everybody driving at once -- those just sent somewhere and
            # those still on their way to where they were sent earlier.
            if self._on_tick is not None:
                self._on_tick(self.robot_coords, self.view)
            if self.experiment_log is not None:
                self.experiment_log.record_poses(self.step_count, self.grid_world.robot_poses)

            self.grid_world.update()
            self.tick_count += 1
            self.ticks_since_a_move += 1

        all_arrived = len(self.agents_at_goal()) == agents
        stop_reason = None
        if not all_arrived:
            if self.ticks_since_a_move >= self.stall_ticks:
                stop_reason = ("no agent has completed a move in %d control iterations (%.0f s). "
                               "The agents still short of their squares cannot get to them."
                               % (self.stall_ticks, self.stall_ticks * TICK_SECONDS))
            elif self.tick_count >= self.max_ticks:
                stop_reason = ("the run hit its budget of %d control iterations (%.0f s)."
                               % (self.max_ticks, self.max_ticks * TICK_SECONDS))
            else:
                stop_reason = "no agent has any move left within max_steps."
            self.announce("Stopped: %s" % stop_reason)

        return bool(all_arrived), stop_reason

    # -- the pieces of a step ------------------------------------------------

    def _agents_allowed_to_plan(self, idle):
        """
        Which agents may claim a tile this iteration.

        Args:
            idle (list): The agents standing still.
        """
        agents = self.number_of_agents
        at_goal = self.agents_at_goal()
        ready = [i for i in idle if i not in at_goal and self.moves_made[i] < self.max_steps]

        if self.asynchronous_moves:
            allowed = ready
        else:
            # The barrier: nobody plans until everybody is idle
            allowed = ready if len(idle) == agents else []

        if self.move_schedule is None:
            return allowed

        # Whose turn it is. The turn is held until every agent in it has had its chance, so an
        # agent whose turn comes up mid-drive is waited for, and the rotation is over moves
        # rather than over control iterations.
        eligible = {i for i in range(agents)
                    if i not in at_goal and self.moves_made[i] < self.max_steps}
        # Draw turns until one of them can still be taken. Bounded, so a schedule that keeps
        # naming agents with nowhere to go cannot spin here.
        for _ in range(agents + 1):
            self._turn_pending &= eligible
            if self._turn_pending or not eligible:
                break
            self._turn_pending = set(next(self.move_schedule))
        return [i for i in allowed if i in self._turn_pending]

    def _take_a_step(self, planning):
        """
        Communicates, plans a tile for each agent in `planning`, sends them, and records it.

        Args:
            planning (list): The agents claiming a tile this iteration.
        """
        agents = self.number_of_agents
        step_index = self.step_count
        robot_coords = self.reservations.destinations

        # A communication round before these agents move, so everything downstream --
        # resolution, planning, projection -- reads the fused beliefs. Synchronously this is the
        # only place the flow runs and everyone broadcasts, which is Riess and Ghrist (2022)
        # Eq. (4). Asynchronously the agents about to plan are the ones needing fresh beliefs,
        # so it is their own neighbours that broadcast -- the firing set tau_t of their Def. 5,
        # licensed by their Theorem 1, the sections being the fixed points of the flow under any
        # schedule that starves nobody.
        if not self.asynchronous_moves:
            self._run_sweeps(None)
        elif self.belief_sheaf is not None:
            self._run_sweeps({neighbour for i in planning
                              for neighbour in self.belief_sheaf.neighbors(i)})

        # Hold on every configured view in turn, agents stationary, before anyone moves: one
        # take then carries the whole set of views of this moment rather than only the one the
        # run is on. There is no such moment in an asynchronous run -- somebody is always
        # driving -- so the tour belongs to the synchronous one alone.
        if (not self.asynchronous_moves and self._on_hold is not None
                and self.tour_views_every != "never" and step_index % self.tour_views_every == 0):
            for tour_view in self.display_views:
                self._on_hold(robot_coords, tour_view)

        # Step by what each agent believes about the tiles around it
        tile_labels = resolve_tile_labels(robot_coords, self.grid_width_height,
                                          self.assigned_targets, **self.belief_options)
        goal_distances = np.linalg.norm(robot_coords - self.assigned_targets, axis=0)
        at_goal = self.agents_at_goal()

        # What counts as terrain to route around rather than as somebody to queue behind: the
        # agents parked on their own goal, and the ones that have been trying and failing to
        # move for long enough that waiting on them is no longer a plan.
        stalled = frozenset(i for i in range(agents)
                            if self.route_around_after_ticks
                            and self.blocked_ticks[i] >= self.route_around_after_ticks)
        for i in sorted(stalled - self._reported_stalled):
            self.announce("Step %d: agent %d has been blocked for %d iterations; the others will "
                          "plan around it" % (step_index, i, self.blocked_ticks[i]))
        self._reported_stalled = stalled
        settled_tiles = frozenset(tuple(self.reservations.destinations[:, other])
                                  for other in (at_goal | stalled))

        mission_records = self._read_the_mission_monitor(robot_coords, tile_labels, step_index)

        # Nearest to its goal first, each planned against the tiles the ones before it just took
        moved = []
        for i in sorted(planning, key=lambda i: goal_distances[i]):
            held = tuple(self.reservations.destinations[:, i])
            next_tile = choose_agent_step(
                i, self.reservations.destinations, self.assigned_targets, self.grid_width_height,
                tile_labels, tile_costs=self.tile_costs,
                blocked_tiles=self.reservations.blocked_for(i),
                settled_tiles=settled_tiles - {held},
                extra_costs=self._extra_costs.get(i))

            if tuple(next_tile) == held:
                # Every move is blocked, so the agent holds position and tries again next
                # iteration. It costs the agent nothing from its budget: waiting is not a step
                # taken. What it does cost is patience -- long enough here and the others stop
                # waiting on it.
                self.blocked_ticks[i] += 1
                continue

            self.reservations.depart(i, next_tile)
            self.grid_world.move_to_tile(i, next_tile)
            self.moves_made[i] += 1
            self.blocked_ticks[i] = 0
            moved.append(i)

        # Everyone who planned has had their turn, whether they took a tile or found nowhere to
        # go. Only being blocked does not hold the rotation up; still driving does.
        self._turn_pending -= set(planning)
        robot_coords = self.reservations.destinations

        step_scores = self._score(robot_coords, moved)
        step_observations = self._observe(robot_coords, moved)

        self.step_count += 1
        if moved:
            self.ticks_since_a_move = 0
            self.rounds_moved += 1

        self._advance_the_view(moved)

        if self.experiment_log is not None:
            self.experiment_log.record_step(
                step_index, robot_coords, step_scores=step_scores,
                cumulative_scores=self.agent_scores, reached_target=self.reached_target,
                tile_labels_by_agent=tile_labels,
                conflicts=[{**conflict._asdict(), "tile": list(conflict.tile)}
                           for conflict in self._step_conflicts] or None,
                mission=mission_records,
                sweeps=list(self._step_sweeps) or None,
                observations=step_observations,
                settled=self.communication_settled if self.belief_sheaf is not None else None,
                moved=moved if self.asynchronous_moves else None,
                tick=self.tick_count if self.asynchronous_moves else None,
                view=describe_view(self.view))

        # Emptied rather than rebound, so the accumulator a runner appends to through
        # record_sweep stays the one this record just carried away
        self._step_conflicts.clear()
        self._step_sweeps.clear()

    def _run_sweeps(self, firing):
        """
        One communication round: `sweeps_per_step` hops of the Laplacian flow, broadcast by the
        agents in `firing`, or by everyone if it is None.

        Settling is judged by Riess and Ghrist (2022) Theorem 1's hypothesis rather than by a
        quiet sweep. Under a full broadcast the first quiet sweep is a fixed point; under a
        partial one it proves only that the agents who spoke had nothing to say, so the flow is
        settled only once every agent has broadcast since the last thing that changed.

        Args:
            firing: The agents that broadcast, the set tau_t of their Def. 5.
        """
        if (self.belief_sheaf is None or self.communication_settled
                or (firing is not None and not firing)):
            return

        def watch(index, conflicts, changed):
            self._last_sweep_changed = bool(changed)
            self.sweeps_run += 1
            if self._on_sweep is not None:
                self._on_sweep(index, conflicts, changed)

        conflicts, _ = communicate(self.belief_sheaf, sweeps=self.sweeps_per_step,
                                   on_sweep=watch, firing=firing)
        self._step_conflicts.extend(conflicts)

        # Light the interfaces that just carried something: every edge with a broadcasting
        # agent at one end of it. Firing None is everyone talking, so that is every edge.
        broadcasters = set(range(self.number_of_agents)) if firing is None else set(firing)
        self._message_edges = frozenset(
            tuple(sorted((i, neighbour)))
            for i in broadcasters for neighbour in self.belief_sheaf.neighbors(i))
        self._message_tick = self.tick_count
        self._broadcast_since_change = (
            set() if self._last_sweep_changed
            else self._broadcast_since_change | (self._everyone if firing is None else set(firing)))
        self.communication_settled = self._broadcast_since_change >= self._everyone
        if self.communication_settled:
            self.announce("Communication settled after step %d: %d sweeps"
                          % (self.step_count, self.sweeps_run))

    def _read_the_mission_monitor(self, robot_coords, tile_labels, step_index):
        """
        Each agent's mission assumption checked against its fused stalk.

        Over the legacy bijective sheaf this is the original reliance read-out: the
        corridor's tiles met against the stalk, emptied possibility sets split into
        contested (the network disagrees) and broken (the network agrees it is unsafe and
        the plan crosses anyway).

        Over the region sheaf it is the contract layer proper (regionsheaf): the monitor
        reads each agent's `assumption_status` -- corridor tiles the network has flagged
        hazardous, route regions a neighbor is known to claim -- and turns what it finds
        into a per-agent cost overlay on the offending regions' tiles. The agent then
        replans against the overlaid costs, and if the corridor that comes out differs
        from the one it committed to, it *recommits*: the sheaf's mission contract is
        rewritten (new claims, new reliance), stale claims are retracted network-wide,
        and the settled latch is cleared so the flow re-closes the sections the new
        commitment opened. Claim conflicts yield by index: an agent routes around a
        region only when a lower-indexed neighbor claims it, which breaks the symmetric
        mutual-yield oscillation two equal agents would otherwise fall into.

        Read before anybody moves, from where the agents are rather than where they are
        about to be: the corridor is what the plan relies on, and the plan is made from
        here.

        Args:
            robot_coords (np.ndarray): Where the agents are.
            tile_labels (dict): Each agent's labels.
            step_index (int): Which step this is, for the notice.
        """
        if not self.mission_monitor:
            return None
        if self.region_sheaf:
            return self._read_the_region_monitor(robot_coords, tile_labels, step_index)

        records = {}
        for i in range(self.number_of_agents):
            current_tile = (int(robot_coords[0, i]), int(robot_coords[1, i]))
            goal_tile = (int(self.assigned_targets[0, i]), int(self.assigned_targets[1, i]))
            if current_tile == goal_tile:
                continue
            corridor = route_corridor(current_tile, goal_tile, self.grid_width_height,
                                      tile_labels[i], tile_costs=self.tile_costs)
            stalk_vars = self.belief_sheaf.nodes[i]['v']
            viability = self.belief_sheaf.contract(i).meet(reliance_contract(stalk_vars, corridor))
            contested = forced_empty(self.belief_sheaf.contract(i), stalk_vars, corridor)
            broken = [tile for tile in forced_empty(viability, stalk_vars, corridor)
                      if tile not in contested]
            records[str(i)] = {
                "corridor": [list(tile) for tile in corridor],
                "contested": [list(tile) for tile in contested],
                "broken": [list(tile) for tile in broken],
                "refines_initial": bool(self.belief_sheaf.contract(i).refines(self.belief_sheaf.initial(i))),
            }
            if broken:
                self.announce("Step %d: agent %d's corridor is broken at %s" % (step_index, i, broken))
        return records

    def _read_the_region_monitor(self, robot_coords, tile_labels, step_index):
        """
        The mission monitor over the region sheaf; see _read_the_mission_monitor.
        Also maintains self._extra_costs, the per-agent overlay the planning loop
        passes to choose_agent_step.

        Args:
            robot_coords (np.ndarray): Where the agents are.
            tile_labels (dict): Each agent's labels.
            step_index (int): Which step this is, for the notice.
        """
        from .regionsheaf import assumption_status, expected_claims, hazard_regions, recommit

        sheaf = self.belief_sheaf
        regions = sheaf.graph['regions']
        records = {}
        for i in range(self.number_of_agents):
            current_tile = (int(robot_coords[0, i]), int(robot_coords[1, i]))
            goal_tile = (int(self.assigned_targets[0, i]), int(self.assigned_targets[1, i]))

            if current_tile == goal_tile:
                # An arrived agent has nothing left to rely on: withdraw its claims, so
                # the regions it held stop costing its neighbors detours.
                self._extra_costs.pop(i, None)
                if recommit(sheaf, i, []):
                    self._clear_settled_latch()
                continue

            status = assumption_status(sheaf, i)
            flagged = hazard_regions(sheaf, i)
            avoid = set(flagged)
            yielded = []
            for r, claimants in expected_claims(sheaf, i).items():
                if any(j < i for j in claimants):
                    avoid.add(r)
                    yielded.append(r)
            overlay = {tile: self.region_penalty
                       for r in avoid for tile in regions.tiles(r)}
            self._extra_costs[i] = overlay

            corridor = route_corridor(current_tile, goal_tile, self.grid_width_height,
                                      tile_labels[i], tile_costs=self.tile_costs,
                                      extra_costs=overlay)
            recommitted = recommit(sheaf, i, corridor)
            if recommitted:
                self._clear_settled_latch()
                self.announce("Step %d: agent %d recommitted to regions %s"
                              % (step_index, i, sheaf.nodes[i]['route_regions']))
            if status.violated:
                self.announce("Step %d: agent %d's assumption is violated in %s "
                              "(hazards %s, claimed %s)"
                              % (step_index, i, status.violated,
                                 status.hazards, status.claimed))

            records[str(i)] = {
                "corridor": [list(tile) for tile in corridor],
                "route_regions": list(sheaf.nodes[i]['route_regions']),
                "violated": list(status.violated),
                "hazards": {r: [list(t) for t in tiles]
                            for r, tiles in status.hazards.items()},
                "claimed": {r: list(js) for r, js in status.claimed.items()},
                "hazard_regions": list(flagged),
                "yielded": yielded,
                "recommitted": bool(recommitted),
                "refines_initial": bool(sheaf.contract(i).refines(sheaf.initial(i))),
            }
        return records

    def _clear_settled_latch(self):
        """
        An out-of-flow update (observation, recommitment) makes a fixed point no longer
        final: the flow must run again and everyone must broadcast again before the
        communication can be called settled.
        """
        self.communication_settled = self.belief_sheaf is None
        self._broadcast_since_change = set()

    def _score(self, robot_coords, moved):
        """
        Scores the step by ground truth.

        Synchronously that is every agent, every round, scored for the tile it now sits on
        whether or not it moved onto it -- an agent short of a target and unable to move keeps
        earning for where it stands. Asynchronously there is no round to score, only the agents
        that just arrived somewhere, so a blocked agent earns nothing for waiting. The two
        therefore do not score a stalled run alike.

        Args:
            robot_coords (np.ndarray): Where the agents now are.
            moved (list): The agents that just claimed a tile.
        """
        step_scores = np.zeros(self.number_of_agents)
        if self.ground_truth is None:
            return step_scores

        if self.asynchronous_moves:
            for i in moved:
                step_scores[i] = score_agent_step(
                    i, (int(robot_coords[0, i]), int(robot_coords[1, i])),
                    self.ground_truth, self.reached_target, tile_scores=self.tile_scores)
        else:
            step_scores = score_robot_step(robot_coords, self.ground_truth, self.reached_target,
                                           tile_scores=self.tile_scores)
        self.agent_scores += step_scores
        return step_scores

    def _observe(self, robot_coords, moved):
        """
        Each agent sees the square it has just stepped onto, and what it sees for itself
        overrides what it was told.

        This is the one thing in a run that enters an agent's knowledge from outside the flow,
        so it is also the one thing that can reopen a settled flow: a fixed point reached before
        somebody went and looked is no longer one.

        Args:
            robot_coords (np.ndarray): Where the agents now are.
            moved (list): The agents that just claimed a tile.
        """
        if not self.observe_on_arrival or not (moved or not self.asynchronous_moves):
            return None

        observations = observe_tiles(
            self.label_beliefs,
            [(int(robot_coords[0, i]), int(robot_coords[1, i]))
             for i in range(self.number_of_agents)],
            self.ground_truth, sheaf=self.belief_sheaf)
        if any(record["changed"] for record in observations):
            self._clear_settled_latch()
        return observations

    def _advance_the_view(self, moved):
        """
        Moves the projection on, if it is configured to move.

        Following the moving agent points it at whoever just set off, so that while an agent
        drives into a square the floor is showing that agent's beliefs -- including about the
        square it is driving into. One view per agent, so the agent's index is the view's; with
        several departing at once the first of them has it, which makes it worth configuring
        only for a run where they depart one at a time.

        Args:
            moved (list): The agents that just claimed a tile.
        """
        if self.follow_moving_agent and moved:
            self.view_index = moved[0]
        elif (isinstance(self.switch_view_every, int)
                and self.step_count % self.switch_view_every == 0):
            self.view_index += 1
