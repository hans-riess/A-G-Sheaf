"""
The gridworld experiment as submitted to the Robotarium.

Agents share what they know along the communication topology by running the mission sheaf's
Tarski Laplacian and plan over the fused knowledge -- the same run exp/run.py performs, with
the same planner calls in the same order, so the two produce the same tiles step for step.
Which arm runs is the configuration's decision, exactly as it is locally: a `regions:`
partition selects the abstraction-level contract sheaf (per-tile agent alphabets, per-region
interface vocabulary, mission contracts with monitored assumptions and route claims), its
absence the legacy f = id belief sheaf, and `sheaf.enabled: false` the no-communication
control arm.

Unlike a local run this draws onto the testbed itself. The Robotarium scales the figure window
to the arena and projects it onto the surface the robots drive on (Robotarium Python Guide,
2.7.8: "anything that you plot within the boundaries shown in the Python simulator will be
projected onto the testbed"), the 3.2m x 2m arena being the projector's coverage area. So the
figure is not a local convenience to be switched off here -- it is the display, and what goes
on it is what makes the run legible: each agent's beliefs as tile fills under the robots, the
communication topology as lines between them, the tiles the network has emptied, and the
caption naming which Laplacian iterate is on the floor.

What is dropped is only the video. The Robotarium films the arena from overhead and hands that
back, so recording the figure as well would cost a canvas draw and an encode on every control
step for a second copy of what the testbed already records -- and a control loop that stalls
longer than half a second leaves the robots pausing themselves.

This file is the main file of the submission. It is uploaded alongside the flattened agsheaf
modules, all as siblings in one flat directory; z3 comes from the executors. See
exp/robotarium/build_submission.py, which assembles the bundle; do not edit the copy inside
submission/, since building overwrites it.

Results come back three ways: the projection is filmed, everything printed here lands in the
run's log.txt, and the JSON this writes is returned as a data file.
"""
import os
import platform
import random
import sys
import numpy

# A preflight, before anything that could fail obscurely.
#
# The contract layer is z3 all the way down, and z3 lives on the executors rather than in this
# bundle -- it was installed there in response to robotarium_python_simulator#43, which at the
# time of writing is confirmed but unmerged, so the record of it living there is a comment on a
# pull request. If it is ever absent, everything after this collapses into a cascade that says
# nothing about the cause, and each diagnosis costs a queue round-trip. So the run states
# plainly what it is standing on before it leans on it.
print("=== preflight ===")
print("python      : %s (%s)" % (sys.version.split()[0], platform.machine()))
print("numpy     : %s"  % (numpy.__version__))
print("working dir : %s" % os.getcwd())
print("files here  : %s" % ", ".join(sorted(os.listdir("."))))

# Which of the two submissions this is. Printed before anything can fail, so a returned log.txt
# says which variant it belongs to rather than leaving it to be inferred from the results.
from experiment_config import CONFIG

print("variant     : %s" % CONFIG.get("experiment_name", "unnamed"))

try:
    import z3
    print("z3          : %s, from %s" % (z3.get_version_string(), z3.__file__))
except Exception as error:
    print("z3          : FAILED TO IMPORT -- %s: %s" % (type(error).__name__, error))
    print("              The executors are expected to provide z3. If they no longer do, the")
    print("              hmr/flatten_z3 branch builds a bundle that carries its own.")
    raise

# numpy and matplotlib come from the executors as well, and are older there than on the machine
# a bundle is built on -- old enough that np.reshape's `shape=` keyword, which numpy 2.1 added,
# is absent, which is what the first submission died of. A build's dry runs cannot see that:
# they run against the builder's own numpy, where every such spelling works. So the versions are
# stated here instead, where a returned log.txt carries them back and the next build can be
# written to the floor they name rather than to a guess at it.
#
# Importing matplotlib is not choosing its backend -- that happens on the first pyplot import,
# left to the Robotarium below -- so this settles nothing the testbed is entitled to settle. The
# path is printed beside the version because the executors have carried two matplotlib installs
# at once, which the returned log announces as a warning about Axes3D and nothing more.
import matplotlib
import numpy

print("numpy       : %s, from %s" % (numpy.__version__, numpy.__file__))
print("matplotlib  : %s, from %s" % (matplotlib.__version__, matplotlib.__file__))
print("=== preflight ok ===\n")

# No matplotlib backend is chosen here on purpose. The Robotarium projects the figure, so it
# owns that decision; forcing a non-interactive backend would leave the arena floor blank.
# A local check of this bundle can set MPLBACKEND=Agg in the environment instead.
import traceback

import numpy as np

# CONFIG is already imported, by the preflight above
from agsheaf_gridworld import (GridWorld, agent_label, set_agent_label_offset,
                               assert_assignments_are_targets, describe_view,
                               initialize_communication_lines, initialize_region_overlay,
                               initialize_safety_color_grid, initialize_status_caption,
                               parse_assignments, parse_communication_topology,
                               parse_display_views, parse_ground_truth, parse_label_beliefs,
                               parse_move_schedule, parse_start_tiles, resolve_tile_labels,
                               route_corridor)
from agsheaf_gridsheaf import build_belief_sheaf, render_contracts, sheaf_state
from agsheaf_mission import Mission
from agsheaf_log import ExperimentLog, encode_sheaf_state
from agsheaf_regions import parse_regions
from matplotlib.patches import Circle

# Where the run is written. The Robotarium hands back the files an experiment leaves behind, so
# a fixed name in the working directory is what makes the record retrievable; nothing here
# timestamps its output the way a local run does, there being one run per submission.
RESULTS_FILENAME = "results.json"

gridworld_cfg = CONFIG["gridworld"]

# Agents are numbered from one where a person reads them -- the badges, the assigned squares, the
# captions -- and from zero everywhere they are reasoned about, which is what they are indexed by
# in this file, in the configuration and in the log. Set before anything is drawn or printed.
set_agent_label_offset((CONFIG.get("rendering") or {}).get("agent_label_offset", 0))
goals_cfg = CONFIG["goals"]
rendering_cfg = CONFIG["rendering"]
robotarium_cfg = CONFIG.get("robotarium") or {}
beliefs_cfg = CONFIG.get("beliefs") or {}
planning_cfg = CONFIG.get("planning") or {}
scoring_cfg = CONFIG.get("scoring") or {}
logging_cfg = CONFIG.get("logging") or {}
sheaf_cfg = CONFIG.get("sheaf") or {}

# All three generators, as exp/sheaf_grid.py seeds them: this experiment draws from its own,
# while the simulator draws from numpy's global legacy state and from the standard library's
# random for the poses it scatters the robots to and for its sensor noise. On the testbed the
# hardware places the robots and the extra seeding changes nothing, but a local dry run of this
# bundle is then the same run every time, which is what the build's duration is measured off.
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
rng = np.random.default_rng(seed=CONFIG["seed"])

N = gridworld_cfg["num_agents"]

# The one planning setting this file reads for itself, for the initial corridors below. The
# rest of the run's behavior -- budgets, schedules, the asynchronous barrier, the deadlock
# tie-break -- is the Mission's to read and validate from CONFIG, the same loop exp/run.py
# drives, so nothing of it is duplicated here.
tile_costs = planning_cfg.get("tile_costs")

grid_width = gridworld_cfg["grid_width"]
grid_height = gridworld_cfg["grid_height"]
grid_width_height = (grid_width, grid_height)

# The squares the agents start on, resolved exactly as exp/run.py resolves them: named by
# the experiment file, or sampled from the configured seed when it does not name them. Same
# config, same seed, same rng, same call, so the submission starts the world the local run
# starts, which is what lets the two be compared step for step.
robot_coords = parse_start_tiles(CONFIG.get("starts"), N, grid_width_height, rng=rng)


## the projection is not the experiment


# Everything drawn on the arena floor is a report of the run rather than part of it. The sheaf
# fuses, the planner plans, the robots drive and the log fills whether or not a single tile is
# ever coloured in: nothing downstream reads anything back out of the drawing.
#
# The first submission did not treat it that way. It died in the setup of the belief tiles,
# before a robot had moved, and a testbed slot that had waited days in a queue came back with a
# traceback and no data -- no results.json, no sheaf iterates, no scores. The cause was a numpy
# spelling and is fixed, but the shape of that loss is not specific to it: a submission gets one
# run, and any drawing call that raises during one is a whole run thrown away for the sake of
# something that was only ever illustration.
#
# So the projection is held at arm's length. Every piece of it is built and called through
# projection_part below, and a piece that cannot be built is simply not drawn while a piece that
# throws mid-run stops being drawn and the run keeps going. The experiment reaches its end and
# writes its results either way.
#
# Not quietly, though. A projection that failed invisibly would be its own way of wasting the
# slot: the returned footage would show a floor that is subtly wrong rather than obviously
# broken, and the numbers would be read as if they had been watched. So each distinct failure is
# announced in full the first time it happens, counted after that -- drawing happens on every
# one of some thousands of control iterations, and a traceback per frame would bury the
# experiment's own output in the log they share -- gathered into a summary at the end, and
# written into results.json, where it travels with the data it qualifies.
projection_failures = {}


def note_projection_failure(what, error):
    """
    Records that one piece of the projection failed: loudly the first time, by count after.

    Args:
        what (str): The piece that failed, named the way a summary should name it.
        error (Exception): What it failed with. Its traceback is taken from the exception
            being handled, so this is only meaningful from inside an except block.
    """
    failure = projection_failures.get(what)
    if failure is None:
        failure = projection_failures[what] = {
            "error": "%s: %s" % (type(error).__name__, error),
            "traceback": traceback.format_exc(),
            "count": 0,
        }
        print("\n" + "!" * 79)
        print("THE PROJECTION FAILED: %s" % what)
        print("")
        print("The experiment itself is unaffected -- the beliefs, the planning and the robots")
        print("do not depend on anything being drawn -- so it is carrying on and will write its")
        print("results as usual. What the testbed films from here is missing this piece, so read")
        print("the footage against this notice rather than on its own.")
        print("!" * 79)
        print(failure["traceback"].rstrip())
        print("!" * 79 + "\n")
    failure["count"] += 1


def projection_part(what, build):
    """
    Builds one piece of the projection, and returns a way of drawing it that cannot end the run.

    A piece that fails to build at all comes back as a function that draws nothing, so that
    every call site downstream stays a plain call rather than a call guarded by whether the
    thing it draws exists.

    Args:
        what (str): The piece being built, as note_projection_failure should name it.
        build (callable): Builds the piece, returning the function that draws it.
    """
    try:
        draw = build()
    except Exception as error:
        note_projection_failure("%s, while being set up" % what, error)
        return lambda *arguments, **keywords: None

    def draw_guarded(*arguments, **keywords):
        try:
            return draw(*arguments, **keywords)
        except Exception as error:
            note_projection_failure(what, error)
        return None

    return draw_guarded


def build_grid_world(show_figure):
    """
    Builds the world, drawing or not.

    Args:
        show_figure (bool): Whether to build the figure the testbed projects.
    """
    return GridWorld(
        number_of_robots=N,
        grid_width_height=grid_width_height,
        show_figure=show_figure,
        show_arrows=gridworld_cfg["show_arrows"],
        sim_in_real_time=gridworld_cfg["sim_in_real_time"],
        initial_coordinates=robot_coords,
        grid_safety_gap=gridworld_cfg["grid_safety_gap"],
        min_distance_repulsion=gridworld_cfg["min_distance_repulsion"],
        max_distance_repulsion=gridworld_cfg["max_distance_repulsion"],
        robot_close_to_target_distance=gridworld_cfg.get("arrival_distance", 0.05)
    )


# The figure is made with the world rather than after it, so it is the one piece of the
# projection that cannot be held at arm's length by projection_part -- failing to make it takes
# the world with it. What it gets instead is a second attempt without it: the run happens
# blind, the robots drive the experiment they were sent to drive and the results come back,
# which is a poor outcome only next to a projected run and an excellent one next to nothing.
#
# The retry builds a second simulator, the first having thrown partway through building one. On
# a testbed that is not free of consequence -- but it is only ever reached when the alternative
# is a slot that returns nothing at all, and if it fails too, that exception stands.
try:
    grid_world = build_grid_world(gridworld_cfg["show_figure"])
except Exception as error:
    if not gridworld_cfg["show_figure"]:
        raise
    note_projection_failure("the figure itself, so nothing can be projected at all", error)
    print("Building the world again with no figure, so that the run still happens and still")
    print("writes its results. Nothing will appear on the arena floor.\n")
    grid_world = build_grid_world(False)

# Whose turn it is to claim a tile. This is the schedule at the granularity the grid is played
# on -- who moves next, rather than who is being driven this 0.033 s -- and None lets every
# agent move as soon as it can, which is the fastest the run goes.
move_schedule = parse_move_schedule(gridworld_cfg.get("move_schedule"), N, seed=CONFIG["seed"])
if move_schedule is not None:
    print("Moving in turn: %s" % gridworld_cfg["move_schedule"])

label_beliefs = parse_label_beliefs(beliefs_cfg, N, grid_width_height)
ground_truth = parse_ground_truth(CONFIG.get("ground_truth"), grid_width_height)
display_views = parse_display_views(beliefs_cfg, N)
print("Projecting %s" % " then ".join(describe_view(view) for view in display_views))

# How the projection moves between views, whether agents learn the ground they drive over, and
# the rest of the run's behavior are the Mission's to read from CONFIG -- the same loop
# exp/run.py drives -- so nothing of it is duplicated here.

assigned_targets = parse_assignments(CONFIG.get("assignments"), N, grid_width_height)
if ground_truth is not None:
    assert_assignments_are_targets(assigned_targets, ground_truth)

communication_topology = parse_communication_topology(CONFIG.get("communication"), N)
if communication_topology is not None:
    print("Communicating over %d edges: %s" % (communication_topology.number_of_edges(),
                                               sorted(sorted(edge) for edge in communication_topology.edges)))

# The region partition: the fibers of the abstraction the interfaces speak. Configured, it
# selects the abstraction-level sheaf; absent, the legacy f = id construction. Kept as its
# own object because the projection draws it and the sheaf communicates through it.
regions = None
if CONFIG.get("regions"):
    regions = parse_regions(CONFIG["regions"], grid_width_height)
    print("Regions: %s" % regions)

# Each agent holds its assigned square with certainty, so it is seeded into the shareable
# beliefs: the network can learn where everyone was sent, and a wrong guess can be contradicted
if sheaf_cfg.get("share_assignments", True):
    for i in range(N):
        label_beliefs[i][(int(assigned_targets[0, i]), int(assigned_targets[1, i]))] = "target"

# The corridors the agents' first plans rely on: the reliance of their initial mission
# contracts, and the route regions of their initial claims. Computed exactly the way the
# mission monitor will recompute them, so the first monitor read finds the commitments it
# would itself have made. Only the region sheaf takes contracts; the legacy arm ignores it.
initial_corridors = None
if regions is not None:
    resolved = resolve_tile_labels(robot_coords, grid_width_height, assigned_targets,
                                   label_beliefs=label_beliefs,
                                   default_label=beliefs_cfg.get("default", "safe"),
                                   mark_other_agents_unsafe=beliefs_cfg.get("mark_other_agents_unsafe", True))
    initial_corridors = {
        i: route_corridor((int(robot_coords[0, i]), int(robot_coords[1, i])),
                          (int(assigned_targets[0, i]), int(assigned_targets[1, i])),
                          grid_width_height, resolved[i], tile_costs=tile_costs)
        for i in range(N)}

belief_sheaf = None
if (sheaf_cfg.get("enabled", True) and communication_topology is not None
        and communication_topology.number_of_edges() > 0):
    belief_sheaf = build_belief_sheaf(label_beliefs, grid_width_height, communication_topology,
                                      regions=regions, corridors=initial_corridors,
                                      contracts_cfg=(CONFIG.get("contracts") or {}) if regions is not None else None)
region_sheaf = belief_sheaf is not None and regions is not None

# How far the flow runs before the agents move, kept as it was written -- a count of hops or
# the word 'converge' -- since that is what the results file should say ran. The Mission
# resolves and validates it for the run itself.
configured_sweeps_per_step = sheaf_cfg.get("sweeps_per_step", 1)

sheaf_legs = sheaf_cfg.get("legs", "kan")
record_iterates = belief_sheaf is not None and sheaf_cfg.get("record_iterates", True)

# Whether each iterate also records every local section as the contract it is, rather than only
# as the possibility grid that contract means. Costs a z3 simplify and a couple of KB per agent
# per sweep, which is the bulk of a communicating run's results file once it is on.
record_contracts = belief_sheaf is not None and sheaf_cfg.get("record_contracts", True)

# How long the projection dwells on each Laplacian iterate, on each view of a tour, and on the
# settled state at the end, in control iterations of 0.033 s
hold_steps_per_sweep = robotarium_cfg.get("hold_steps_per_sweep", 0)
hold_steps_per_view = robotarium_cfg.get("hold_steps_per_view", 45)
hold_steps_at_end = robotarium_cfg.get("hold_steps_at_end", 0)

# What the projection reads about the sheaf, refreshed once per sweep and held between them:
# which interfaces agree, which tiles each agent's stalk has no possibility left for, and the
# caption naming the iterate
sheaf_sections = None
sheaf_contested = None
sheaf_caption = ""

draw_section_styling = belief_sheaf is not None and rendering_cfg.get("show_sections", True)
section_colors = (rendering_cfg.get("section_color", "#066034"),
                  rendering_cfg.get("no_section_color", "#D90368")) if draw_section_styling else None

belief_options = dict(
    label_beliefs=label_beliefs,
    default_label=beliefs_cfg.get("default", "safe"),
    mark_other_agents_unsafe=beliefs_cfg.get("mark_other_agents_unsafe", True),
)

# The tile fills projected under the robots: what each agent believes about the ground it is
# driving over, which is the whole point of putting this on a testbed with a projector
create_color_grid = projection_part("the belief tiles", lambda: initialize_safety_color_grid(
    grid_world,
    assigned_targets,
    ground_truth=ground_truth,
    **belief_options,
    label_colors=rendering_cfg.get("label_colors"),
    comparison_colors=rendering_cfg.get("comparison_colors"),
    fill_ground_truth_safe=rendering_cfg.get("fill_ground_truth_safe", False),
    show_legend=rendering_cfg.get("show_legend", False),
    legend_fontsize=rendering_cfg.get("legend_fontsize", "xx-small"),
    fill_alpha=rendering_cfg["belief_alpha"],
    assigned_number_fontsize=rendering_cfg.get("assigned_number_fontsize", 7.0),
    assigned_number_color=rendering_cfg.get("badge_edge_color", "#051E39"),
    static_assignment_numbers=rendering_cfg.get("static_assignment_numbers", True),
    show_assignment_numbers=rendering_cfg.get("show_assignment_numbers", True),
    contested_marker=(rendering_cfg.get("contested_marker") or None) if belief_sheaf is not None else None,
    contested_color=rendering_cfg.get("contested_color", "#D90368"),
    contested_fontsize=rendering_cfg.get("contested_fontsize", 8.0),
    section_colors=section_colors
))

# The partition drawn on the floor: a heavier boundary along every edge between two regions,
# names at their centroids. Tile colors are what an agent believes; the boundaries are the
# resolution at which the interfaces measure agreement. Static, so only the setup needs
# guarding -- the returned draw function is a no-op and is never called.


def build_region_overlay():
    """Draws the region partition once and returns a draw function with nothing left to do."""
    if rendering_cfg.get("show_regions", True):
        initialize_region_overlay(
            grid_world, regions,
            line_color=rendering_cfg.get("region_line_color", "#051E39"),
            line_alpha=rendering_cfg.get("region_line_alpha", 0.5),
            line_width=rendering_cfg.get("region_line_width", 1.8),
            show_names=rendering_cfg.get("show_region_names", True),
            name_color=rendering_cfg.get("region_name_color", "#051E39"),
            name_alpha=rendering_cfg.get("region_name_alpha", 0.45),
            name_fontsize=rendering_cfg.get("region_name_fontsize", 6.0))
    return lambda *arguments, **keywords: None


projection_part("the region partition", build_region_overlay)

# Lines drawn on the floor between robots that can talk to each other, colored by whether that
# interface of the sheaf agrees yet
if draw_section_styling:
    line_color, agreeing_color = section_colors[1], section_colors[0]
    line_alpha = rendering_cfg.get("section_alpha", 0.9)
    line_width = rendering_cfg.get("section_linewidth", 1.4)
    section_style = rendering_cfg.get("section_linestyle", "-")
    # An interface that has not closed yet is drawn in its own line style as well as its own
    # color. On a screen the color is enough; projected onto the arena floor it is not -- under
    # the projector the two are one washed-out grey, which is why the testbed layer draws every
    # interface black -- and the style is then the only thing left saying which have closed.
    line_style = rendering_cfg.get("no_section_linestyle") or section_style
else:
    line_color, agreeing_color = rendering_cfg.get("communication_color", "#051E39"), None
    line_alpha = rendering_cfg.get("communication_alpha", 0.25)
    line_width = rendering_cfg.get("communication_linewidth", 0.8)
    line_style = rendering_cfg.get("communication_linestyle", "--")
    section_style = line_style

# What an interface is lit up with while a message is crossing it: green where the two ends now
# agree, red where they still do not. Read whether or not the settled state of an interface is
# drawn, since a message is an event rather than a state and is worth showing either way.
message_colors = ((rendering_cfg.get("section_color", "#066034"),
                   rendering_cfg.get("no_section_color", "#D90368"))
                  if belief_sheaf is not None and rendering_cfg.get("show_messages", True) else None)

draw_communication_lines = projection_part("the communication lines", lambda: initialize_communication_lines(
    grid_world,
    communication_topology if rendering_cfg.get("show_communication", True) else None,
    line_color=line_color,
    line_alpha=line_alpha,
    line_width=line_width,
    line_style=line_style,
    section_color=agreeing_color,
    section_alpha=line_alpha,
    section_width=line_width,
    section_style=section_style,
    message_colors=message_colors,
    message_alpha=rendering_cfg.get("message_alpha", 1.0),
    message_width_scale=rendering_cfg.get("message_width_scale", 2.5),
    show_when_idle=rendering_cfg.get("show_idle_communication", True)
))

# The caption projected above the grid, carrying the two things about a frame the tiles cannot
# say themselves: whose beliefs they are, and which Laplacian iterate they are. It is what
# lets the overhead footage be read without a legend, which on the floor covers tiles the
# robots drive over.
show_view_in_status = rendering_cfg.get("show_view_in_status", True)
show_sheaf_status = belief_sheaf is not None and rendering_cfg.get("show_sheaf_status", True)
set_status_caption = projection_part("the caption", lambda: initialize_status_caption(
    grid_world,
    enabled=show_view_in_status or show_sheaf_status,
    text_color=rendering_cfg.get("sheaf_status_color") or rendering_cfg.get("badge_edge_color", "#051E39"),
    fontsize=rendering_cfg.get("sheaf_status_fontsize", 7.0),
    position=tuple(rendering_cfg.get("sheaf_status_position", (0.025, 0.93))),
    horizontal_alignment=rendering_cfg.get("sheaf_status_ha", "left"),
    vertical_alignment=rendering_cfg.get("sheaf_status_va", "top"),
    rotation=rendering_cfg.get("sheaf_status_rotation", 0.0),
    weight=rendering_cfg.get("sheaf_status_weight", "normal"),
    background=rendering_cfg.get("sheaf_status_background", True)
))


def status_caption(view):
    """
    The caption for a frame projected in one particular view.

    Args:
        view: The side, or pair of sides, whose labels this frame draws.
    """
    parts = []
    if show_view_in_status:
        parts.append(describe_view(view, halves=True))
    if show_sheaf_status and sheaf_caption:
        parts.append(sheaf_caption)
    return "  ·  ".join(parts)

experiment_log = None
if logging_cfg.get("enabled", True):
    experiment_log = ExperimentLog(RESULTS_FILENAME, CONFIG, grid_width_height, N,
                                   record_poses=logging_cfg.get("record_poses", True))
    experiment_log.record_setup(assigned_targets, ground_truth=ground_truth,
                                topology=communication_topology, videos=[])

    if belief_sheaf is not None:
        experiment_log.record_sheaf(
            interfaces=[sorted(interface) for interface in belief_sheaf.interfaces()],
            stalk_alphabet=("per tile a possibility set over target/safe/unsafe, per region "
                            "a route claim; interfaces speak per-region flagged/excluded "
                            "summaries and per-endpoint claims" if region_sheaf else
                            "one possibility set per tile, over the labels target/safe/unsafe"),
            legs=sheaf_legs,
            sweeps_per_step=configured_sweeps_per_step,
            initial_state=encode_sheaf_state(
                sheaf_state(belief_sheaf, legs=sheaf_legs), grid_width_height,
                contracts=render_contracts(belief_sheaf) if record_contracts else None),
            regions={name: regions.tiles(name) for name in regions.names} if region_sheaf else None)

# Robots are drawn as neutral badges carrying their agent index. On the testbed these land on
# the floor beneath the robots rather than standing in for them, so the numeral is what tells
# the overhead footage which robot is which; the belief labels stay the only other color.
# Set rendering.show_agent_badges false to leave the physical robots unmarked.
agent_badge_radius = rendering_cfg["agent_badge_radius"]
robot_length = rendering_cfg["robot_length"]
badge_edge_color = rendering_cfg["badge_edge_color"]


def build_agent_badges():
    """
    Places a badge per robot and returns the function that keeps them under the robots.

    Built inside a function rather than at module level so that it goes through
    projection_part like the rest of the projection, and so that a badge that cannot be drawn
    takes only the badges with it.
    """
    patches_badges = []
    patches_badge_labels = []
    if grid_world.axes is not None and rendering_cfg.get("show_agent_badges", True):
        for i in range(N):
            # zorder sits above the Robotarium's own robot handles, which are at 2
            patch_badge = Circle((0, 0), radius=agent_badge_radius, facecolor="white",
                                 edgecolor=badge_edge_color, linewidth=1.0, zorder=2.5)
            patches_badges.append(patch_badge)
            grid_world.axes.add_patch(patch_badge)

            badge_label = grid_world.axes.text(0, 0, agent_label(i), color=badge_edge_color, fontsize=7,
                                               ha="center", va="center", zorder=2.6)
            patches_badge_labels.append(badge_label)

    def place_agent_badges():
        """
        Reattaches each agent's badge to where its robot now is. With badges turned off, or with
        no figure to draw onto, this runs over an empty list rather than needing a guard at each
        call.
        """
        for i in range(len(patches_badges)):
            robot_x, robot_y, robot_theta = tuple(grid_world.robot_poses[:, i].tolist())
            robot_center_pos = np.array([
                robot_x + robot_length / 2 * np.cos(robot_theta),
                robot_y + robot_length / 2 * np.sin(robot_theta)
            ])
            patches_badges[i].set_center(robot_center_pos)
            patches_badge_labels[i].set_position(robot_center_pos)

    return place_agent_badges


place_agent_badges = projection_part("the agent badges", build_agent_badges)


def refresh_projection(coords, view):
    """
    Redraws everything the testbed projects, without stepping the simulation.

    Every piece it calls came through projection_part, so this cannot raise on account of the
    drawing -- which matters most here, at the one call the driving loops make on every control
    iteration. A throw from this would leave the robots uncommanded, and a robot that hears
    nothing for half a second stops itself and is no longer the experiment's to move.

    Args:
        coords (np.ndarray): The 2xN grid coordinates the tiles are resolved against, which
            while an iterate is held is where the robots still are rather than where they go.
        view: The side, or pair of sides, whose labels are drawn.
    """
    place_agent_badges()
    draw_communication_lines(sheaf_sections, messaging=mission.messaging_edges)
    set_status_caption(status_caption(view))
    create_color_grid(coords, view, sheaf_contested)


def hold_projection(steps, coords, view):
    """
    Leaves the current state on the floor for a while, robots stationary.

    A sweep of the Laplacian happens between grid steps and takes no time on the testbed, so
    without this the floor jumps from one iterate to the next while the robots keep driving and
    the flow -- the thing the run is about -- is never actually seen. Holding is what turns it
    into something a camera can record.

    The robots are commanded zero velocity rather than left alone: a robot that receives no
    command for half a second pauses itself, which would end the experiment's control of it.

    Args:
        steps (int): Control iterations to hold, each 0.033 s.
        coords (np.ndarray): The grid coordinates the tiles are resolved against.
        view: The side, or pair of sides, whose labels are drawn.
    """
    stationary = np.zeros((2, N))
    for _ in range(steps):
        refresh_projection(coords, view)
        grid_world.robotarium.set_velocities(np.arange(N), stationary)
        grid_world.robotarium.step()
        grid_world.robot_poses = grid_world.robotarium.get_poses()


# One entry per Laplacian sweep of the whole run, so a sweep can name which iterate it is
sweeps_made = []


def observe_sweep(sweep_in_step, conflicts, changed):
    """
    Called after each sweep of the flow, with the assignment as that sweep left it.

    A sweep is one application of the Tarski Laplacian, and it is the thing the run is about;
    what the agents end up believing is only the part of it a labeling can hold. So the whole
    state is snapshotted -- possibility sets, which interfaces have closed, whether the
    assignment is a global section, whether knowledge still refines what it started as -- for
    the log, for the caption on the floor, and for the tile marks and line colors the
    projection draws.

    Args:
        sweep_in_step (int): Which sweep this is within the current grid step, from 0.
        conflicts (list): The conflicts this sweep surfaced.
        changed (bool): Whether this sweep changed anything. False means the flow had already
            reached its fixed point, which the meet's monotonicity makes permanent.
    """
    global sheaf_sections, sheaf_contested, sheaf_caption

    state = sheaf_state(belief_sheaf, legs=sheaf_legs)
    sweeps_made.append(state)

    sheaf_sections = state.sections
    sheaf_contested = state.contested
    agreeing = sum(1 for agrees in state.sections.values() if agrees)
    # The caption and the printed line say the same thing at two lengths, because they are read
    # in two places. The caption is read off a frame of a moving picture, where the words a
    # reader already knows are what there is no time for; the print goes into the log the
    # Robotarium hands back, which is read afterwards and at leisure, and which is where the
    # run says what it did in full. The tick is the assignment being a global section, which is
    # all of the interfaces having closed and so is the fraction beside it reaching 1 -- said
    # twice because at a glance the fraction is two numbers to compare and the tick is not.
    section_mark = " ✓" if state.is_section else ""
    if region_sheaf:
        # The two levels side by side is the point of the construction: the interfaces agree
        # in the shared vocabulary while the tile-level pictures still differ, and the
        # disagreement on the floor is exactly the kernel of the abstraction the agents
        # communicate through.
        differing = sum(state.concrete_disagreement.values())
        sheaf_caption = "sweep %d  ·  %d/%d interfaces%s  ·  %s" % (
            len(sweeps_made), agreeing, len(state.sections), section_mark,
            "%d tiles differ" % differing if differing else "tiles agree")
        reported = "sweep %d  ·  abstract: %s (%d/%d interfaces)  ·  concrete: %s" % (
            len(sweeps_made),
            "global section" if state.is_section else "not a section",
            agreeing, len(state.sections),
            "agents differ on %d tiles" % differing if differing else "no disagreement")
    else:
        sheaf_caption = "sweep %d  ·  %d/%d interfaces%s" % (
            len(sweeps_made), agreeing, len(state.sections), section_mark)
        reported = "sweep %d  ·  %d/%d interfaces agree  ·  %s" % (
            len(sweeps_made), agreeing, len(state.sections),
            "global section" if state.is_section else "not a section")
    print("  %s" % reported)

    # Sweeps are numbered from 1 as ordinals rather than indices, so a record names the same
    # iterate the caption on the floor does
    if record_iterates:
        mission.record_sweep({
            "sweep": len(sweeps_made),
            "sweep_in_step": sweep_in_step + 1,
            "changed": bool(changed),
            "conflicts": [{**conflict._asdict(), "tile": list(conflict.tile)} for conflict in conflicts],
            **encode_sheaf_state(state, grid_width_height,
                                 contracts=render_contracts(belief_sheaf) if record_contracts else None),
        })

    # Hold this iterate on the floor, robots stationary, so the sweeps are watchable rather than
    # only their effect. Not asynchronously: holding stops every agent, including the ones this
    # sweep has nothing to do with, and a run whose whole point is that agents no longer wait for
    # each other cannot freeze them all twice a second. The iterate still reaches the floor -- the
    # projection is redrawn every tick and the caption names the sweep -- it is just not held.
    if not mission.asynchronous_moves:
        hold_projection(hold_steps_per_sweep, mission.robot_coords, mission.view)


# The drive loop itself lives in agsheaf.mission, shared with exp/run.py, so a change to how a
# run proceeds -- including the mission monitor's overlay and recommitments -- lands in both at
# once rather than in whichever one was being worked on. What stays here is what only a
# projected run has: the tiles and lines thrown onto the arena floor, and the holds that stop
# the robots so a camera can see an iterate of the flow.


def project_the_frame(coords, view):
    """
    One control iteration's worth of projection.

    Args:
        coords (np.ndarray): The tiles the agents hold.
        view: The view the run is on.
    """
    refresh_projection(coords, view)


def hold_on_the_view(coords, view):
    """
    Stops the robots and leaves one view on the floor, so that a single take carries the whole
    set of views of a moment rather than only the one the run is on.

    Args:
        coords (np.ndarray): The tiles the agents hold.
        view: The view to hold on.
    """
    hold_projection(hold_steps_per_view, coords, view)


mission = Mission(CONFIG, grid_world, assigned_targets, label_beliefs,
                  belief_sheaf=belief_sheaf, ground_truth=ground_truth,
                  experiment_log=experiment_log, display_views=display_views,
                  on_tick=project_the_frame, on_sweep=observe_sweep, on_hold=hold_on_the_view)
all_arrived, _ = mission.run()

robot_coords = mission.robot_coords
agent_scores = mission.agent_scores
step_count = mission.step_count

# End on the settled state rather than cutting at the last motion, so the footage closes on
# whatever the flow reached
hold_projection(hold_steps_at_end, robot_coords, mission.view)

# The Robotarium's own account of the run: collisions, boundary violations, actuator limits
grid_world.robotarium.debug()

if experiment_log is not None:
    experiment_log.close(final_scores=agent_scores, steps_taken=step_count, all_arrived=all_arrived,
                         projection_failures=projection_failures or None)

# How long this took in testbed time, said once in a fixed form. The builder reads this back off
# a dry run to work out what to put in the portal's duration field, which is otherwise a guess:
# the tile-level run is deterministic but its length is not, the testbed spending a variable
# while driving the robots to their initial conditions before the experiment proper begins.
# `_iteration` is an attribute of the local simulator fork only; the physical testbed's
# Robotarium does not have it (its absence crashed the 08-12 runs at this very line, after
# the experiment but before the log close -- which is why this print now comes last, guarded).
iteration = getattr(grid_world.robotarium, "_iteration", None)
if iteration is not None:
    print("DURATION_SECONDS %.1f" % (iteration * grid_world.robotarium.TIME_STEP))
    print("Wrote %s" % RESULTS_FILENAME)

print("All agents arrived: %s, after %d steps" % (all_arrived, step_count))
if ground_truth is not None:
    print("Final scores:")
    for i in range(N):
        print("  Agent %s: %+.1f" % (agent_label(i), agent_scores[i]))
    print("  Total: %+.1f" % agent_scores.sum())

# Said last as well as when it happened, because a log.txt is read from the end and the notice
# at the point of failure is thousands of lines up by now. The results above stand -- they were
# never a function of what was drawn -- but the footage does not, so the two are separated here
# rather than left to be reconciled from the video.
if projection_failures:
    print("\n" + "!" * 79)
    print("THE PROJECTION FAILED IN %d PLACE(S). The results above are unaffected; the footage"
          % len(projection_failures))
    print("is not, and shows less than the run actually did.")
    for what, failure in sorted(projection_failures.items()):
        print("  %s" % what)
        print("      %s" % failure["error"])
        print("      %d time(s); full traceback at its first occurrence above" % failure["count"])
    print("Recorded under \"projection_failures\" in %s." % RESULTS_FILENAME)
    print("!" * 79)
