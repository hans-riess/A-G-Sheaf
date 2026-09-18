# The experiment: agents drive to assigned squares through a world they hold partial,
# possibly wrong beliefs about, communicating over the mission sheaf's Tarski Laplacian.
#
# One runner covers every arm, the configuration deciding which runs:
#
#   * `regions:` configured        -- the abstraction-level contract sheaf (regionsheaf):
#     per-tile agent alphabets, per-region interface alphabets, two-slot mission
#     contracts, agreement measured in the interface vocabulary. The demonstration.
#   * no `regions:` block          -- the legacy f = id belief sheaf (bijective
#     restrictions, mask flatten), kept as the degenerate control arm.
#   * `sheaf.enabled: false`       -- nobody talks; the greedy baseline the old
#     exp/greedy_grid.py ran, on identical world/planner/scoring/rendering.
#
# This file replaces exp/sheaf_grid.py and exp/greedy_grid.py.
from agsheaf.gridworld import GridWorld, agent_label, set_agent_label_offset
from agsheaf.gridworld import parse_start_tiles, initialize_safety_color_grid, initialize_communication_lines, parse_label_beliefs, parse_ground_truth
from agsheaf.gridworld import parse_display_views, parse_assignments, parse_communication_topology, assert_assignments_are_targets, resolve_tile_labels, describe_view
from agsheaf.gridworld import initialize_region_overlay, initialize_status_caption, route_corridor
from agsheaf.gridsheaf import build_belief_sheaf, sheaf_state, parse_sweeps_per_step, render_contracts
from agsheaf.log import ExperimentLog, encode_sheaf_state
from agsheaf.mission import Mission
from agsheaf.regions import parse_regions
from agsheaf.utils import VideoSaver, load_experiment_config, run_output_path
import robot_icon
import numpy as np
import datetime
import random
import sys
from pathlib import Path

# Load this experiment on top of the general defaults. An alternative experiment file may be
# named on the command line -- `python exp/run.py my_world.yaml` -- which is how the control
# arms (the f = id partition, the no-communication baseline) run without editing the main
# configuration; the log records whichever file ran.
experiments_dir = Path(__file__).parent
config_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else experiments_dir / "worlds" / "primary.yaml"
config = load_experiment_config(config_path, experiments_dir / "defaults.yaml")

# The same config file drives every arm, so the name of the script that ran it is what
# tells one run's log from another's. An experiment file naming itself still wins.
config.setdefault("experiment_name", Path(__file__).stem)

gridworld_cfg = config["gridworld"]

# Agents are numbered from one where a person reads them -- the badges, the assigned squares, the
# captions -- and from zero everywhere they are reasoned about, which is what they are indexed by
# in this file, in the configuration and in the log. Set before anything is drawn or printed.
set_agent_label_offset((config.get("rendering") or {}).get("agent_label_offset", 0))
goals_cfg = config["goals"]
video_cfg = config["video"]
rendering_cfg = config["rendering"]
beliefs_cfg = config.get("beliefs") or {}
planning_cfg = config.get("planning") or {}
scoring_cfg = config.get("scoring") or {}
logging_cfg = config.get("logging") or {}
sheaf_cfg = config.get("sheaf") or {}
contracts_cfg = config.get("contracts") or {}

# Init the RNGs for repeatability. All three of them, because three separate streams decide what
# a run looks like: this experiment draws from its own generator, while the Robotarium simulator
# draws from numpy's global legacy state (its deadlock waypoints and sensor noise) and from the
# standard library's random (generate_initial_poses, the charger squares the robots are scattered
# onto before being driven to the initial conditions).
#
# Leaving the last two unseeded is invisible in the tiles, the beliefs and the scores, which come
# out identical either way, and plainly visible in the video: the robots reach their starting
# squares a couple of centimetres and a few degrees apart from one run to the next, so the first
# step takes a different number of control iterations to drive and every frame after it is
# offset. Seeding all three is what makes two runs of one configuration the same video, which is
# what writing a seed down is for.
random.seed(config["seed"])
np.random.seed(config["seed"])
rng = np.random.default_rng(seed=config["seed"])

# Define number of agents
N = gridworld_cfg["num_agents"]

# How agents weigh tiles when routing, by what they believe about them, and what entering a
# tile is worth when scoring the run, by what that tile truly is
tile_costs = planning_cfg.get("tile_costs")
tile_scores = scoring_cfg.get("tile_scores")

# Define grid size
grid_width = gridworld_cfg["grid_width"]
grid_height = gridworld_cfg["grid_height"]
grid_width_height = (grid_width, grid_height)

# Init grid world. The squares the agents start on are named by the experiment file, or
# sampled from the seed when it does not name them.
robot_coords = parse_start_tiles(config.get("starts"), N, grid_width_height, rng=rng)
robot_icon.install()   # draw the robots as UUVs; cosmetic, and before the figure is built
grid_world = GridWorld(
    number_of_robots=N,
    grid_width_height=(grid_width, grid_height),
    show_arrows=gridworld_cfg["show_arrows"],
    show_figure=gridworld_cfg.get("show_figure", True),
    sim_in_real_time=gridworld_cfg["sim_in_real_time"],
    initial_coordinates=robot_coords,
    grid_safety_gap=gridworld_cfg["grid_safety_gap"],
    min_distance_repulsion=gridworld_cfg["min_distance_repulsion"],
    max_distance_repulsion=gridworld_cfg["max_distance_repulsion"],
    robot_close_to_target_distance=gridworld_cfg.get("arrival_distance", 0.05)
)

# Parse the programmed per-agent label beliefs and set up the belief renderer.
# Each view is either one side -- an agent index or 'ground_truth' -- or a pair of sides
# whose labels are compared. Configuring 'view' draws one view for the whole run, which is
# what reading a video for debugging wants; 'views' cycles through several.
label_beliefs = parse_label_beliefs(beliefs_cfg, N, grid_width_height)
ground_truth = parse_ground_truth(config.get("ground_truth"), grid_width_height)
display_views = parse_display_views(beliefs_cfg, N)
print("Drawing %s" % " then ".join(describe_view(view) for view in display_views))

# Showing one view at a time means only one of them is ever on screen at a given moment of
# the run. A tour instead holds the video on every configured view in turn, robots
# stationary, so one video carries the whole set of them at that moment -- the ground truth
# and then what each agent makes of it -- which is what a demo filmed in one take needs.
frames_per_view = video_cfg.get("frames_per_view", 30)

# The square each agent has been told to reach. This is what drives the movement, and is
# separate from what an agent believes: its own square it knows, the others' it only guesses
# at, through the target tiles programmed for it under beliefs.agents.
assigned_targets = parse_assignments(config.get("assignments"), N, grid_width_height)
if ground_truth is not None:
    assert_assignments_are_targets(assigned_targets, ground_truth)

# Who can talk to whom: the base graph of the mission sheaf. Each communication round runs
# sweeps of the sheaf's Laplacian flow along it, which is the agents sharing what they
# believe -- and, over the region sheaf, what they claim.
communication_topology = parse_communication_topology(config.get("communication"), N)
if communication_topology is not None:
    print("Communicating over %d edges: %s" % (communication_topology.number_of_edges(),
                                               sorted(sorted(edge) for edge in communication_topology.edges)))

# The region partition: the fibers of the abstraction the interfaces speak. Configured, it
# selects the abstraction-level sheaf; absent, the legacy f = id construction. Kept as its
# own object because the renderer draws it and the sheaf communicates through it.
regions = None
if config.get("regions"):
    regions = parse_regions(config["regions"], grid_width_height)
    print("Regions: %s" % regions)

# Each agent holds its own assigned square with certainty, so with share_assignments it is
# seeded into the agent's shareable beliefs: the network can learn where everyone was sent,
# and a wrong guess about another agent's target can be contradicted rather than believed.
if sheaf_cfg.get("share_assignments", True):
    for i in range(N):
        label_beliefs[i][(int(assigned_targets[0, i]), int(assigned_targets[1, i]))] = "target"

# How beliefs are resolved into tile labels. Shared by the renderer and the planner, so
# agents move by exactly the labeling drawn for them.
belief_options = dict(
    label_beliefs=label_beliefs,
    default_label=beliefs_cfg.get("default", "safe"),
    mark_other_agents_unsafe=beliefs_cfg.get("mark_other_agents_unsafe", True),
)

# The corridors the agents' first plans rely on: the reliance of their initial mission
# contracts, and the route regions of their initial claims. Computed exactly the way the
# mission monitor will recompute them, so the first monitor read finds the commitments it
# would itself have made. Only the region sheaf takes contracts; the legacy arm ignores it.
initial_corridors = None
if regions is not None:
    resolved = resolve_tile_labels(robot_coords, grid_width_height, assigned_targets,
                                   **belief_options)
    initial_corridors = {
        i: route_corridor((int(robot_coords[0, i]), int(robot_coords[1, i])),
                          (int(assigned_targets[0, i]), int(assigned_targets[1, i])),
                          grid_width_height, resolved[i], tile_costs=tile_costs)
        for i in range(N)}

# The mission sheaf itself. Its stalks project into the very belief dicts the renderer and
# planner read, so each sweep's fused knowledge shows up in the videos and the routing
# without either knowing communication exists. Once a sweep changes nothing the flow is at
# its fixed point for good (it is monotone), so sweeping stops -- until an observation or a
# recommitment reopens it.
belief_sheaf = None
if (sheaf_cfg.get("enabled", True) and communication_topology is not None
        and communication_topology.number_of_edges() > 0):
    belief_sheaf = build_belief_sheaf(label_beliefs, grid_width_height, communication_topology,
                                      regions=regions, corridors=initial_corridors,
                                      contracts_cfg=contracts_cfg if regions is not None else None)
region_sheaf = belief_sheaf is not None and regions is not None

# How far the flow runs before the agents move. A number is that many communication hops
# per step; 'converge' runs it to its fixed point first, so agents step on everything the
# network between them knows rather than on what has reached them so far. The configured
# value is kept as it was written, since that is what the log should say ran.
configured_sweeps_per_step = sheaf_cfg.get("sweeps_per_step", 1)
sweeps_per_step = parse_sweeps_per_step(configured_sweeps_per_step,
                                        sheaf_cfg.get("max_sweeps", 100))
sheaf_legs = sheaf_cfg.get("legs", "kan")
record_iterates = belief_sheaf is not None and sheaf_cfg.get("record_iterates", True)

# Whether each iterate also records every mission contract as the assume/guarantee pair it
# is, rather than only as the possibility grid it means. Costs a z3 simplify and a couple of
# KB per agent per sweep, which is the bulk of a communicating run's log once it is on.
record_contracts = belief_sheaf is not None and sheaf_cfg.get("record_contracts", True)

# What the renderers read about the sheaf, refreshed once per sweep and held between them:
# which interfaces agree, which tiles each agent's stalk has no possibility left for, and the
# caption naming the iterate. None until a sheaf exists, so a run without one draws as before.
sheaf_sections = None
sheaf_contested = None
sheaf_caption = ""

# The two states an interface can be in, in the order the legend and the line drawing both
# take them: agreeing first. None when there is no sheaf and an edge is just an edge.
draw_section_styling = belief_sheaf is not None and rendering_cfg.get("show_sections", True)
section_colors = (rendering_cfg.get("section_color", "#066034"),
                  rendering_cfg.get("no_section_color", "#D90368")) if draw_section_styling else None

create_color_grid = initialize_safety_color_grid(
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
)

# The partition drawn on the floor: tile colors are what an agent believes, region
# boundaries are the resolution at which the interfaces measure agreement. Static, so it is
# drawn once here rather than per frame.
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

# Lines between agents that can talk to each other, drawn from where the robots actually are,
# so this is refreshed every frame alongside the badges. An edge of the topology is an
# interface of the sheaf, and it either agrees -- both endpoints pushing the same contract
# onto their shared edge stalk, the sheaf condition holding locally, measured in the
# interface alphabet -- or does not yet, so the line is colored by which. The faint dashed
# styling is for a run with no sheaf, where an edge has no such state to be in.
if draw_section_styling:
    line_color, agreeing_color = section_colors[1], section_colors[0]
    line_alpha = rendering_cfg.get("section_alpha", 0.9)
    line_width = rendering_cfg.get("section_linewidth", 1.4)
    section_style = rendering_cfg.get("section_linestyle", "-")
    # An interface that has not closed yet can be drawn in its own line style as well as its
    # own color, for a surface where the color does not survive -- the arena floor under the
    # projector, where the testbed layer draws every interface black.
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

draw_communication_lines = initialize_communication_lines(
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
)

# The caption, carrying the things about a frame the tiles cannot say themselves: whose
# beliefs they are, which Laplacian iterate they are -- and, over the region sheaf, the
# two levels at once: whether the interfaces agree in the abstract vocabulary, and how many
# tiles the agents still concretely differ on. Their coexistence is the abstraction working.
show_view_in_status = rendering_cfg.get("show_view_in_status", True)
show_sheaf_status = belief_sheaf is not None and rendering_cfg.get("show_sheaf_status", True)
set_status_caption = initialize_status_caption(
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
)


def status_caption(view):
    """
    The caption for a frame drawn in one particular view.

    Takes the view rather than reading the current one, because with video.one_per_view
    several videos are written from the one figure with a different view drawn in each, and
    a caption naming a view has to be set per video rather than per frame.

    Args:
        view: The side, or pair of sides, whose labels this frame draws.
    """
    parts = []
    if show_view_in_status:
        parts.append(describe_view(view, halves=True))
    if show_sheaf_status and sheaf_caption:
        parts.append(sheaf_caption)
    return "  ·  ".join(parts)


# The timestamp names the run, and every file it writes shares that one name. Nothing about
# what was configured goes in a filename: the log carries the whole configuration, the views
# included, so the run describes itself rather than relying on a name to do it.
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# With one_per_view, every configured view gets its own video of the same run: the simulation
# is stepped once and each frame is drawn once per view, so the videos come out frame for
# frame identical apart from what is drawn, ready to sit side by side. Otherwise a single
# video is written, cycling the views as switch_view_every says.
# A video is a recording of the figure, so a headless run writes none and the loops below,
# which all run over these two lists, come out empty rather than needing a guard each.
one_video_per_view = video_cfg.get("one_per_view", False)
video_views = (display_views if one_video_per_view else [None]) if grid_world.figure is not None else []

video_savers = []
if video_views:
    # output_dir may be relative to this file, or an absolute/~ path (e.g. to keep videos
    # outside a hidden directory, which confined snap players such as VLC cannot read)
    video_dir = (Path(__file__).parent / video_cfg["output_dir"]).expanduser()
    video_dir.mkdir(parents=True, exist_ok=True)
    for video_index, view in enumerate(video_views):
        # A run writing one video does not number it; one writing several numbers them from 1
        video_path = run_output_path(video_dir, timestamp, "mp4",
                                     number=None if len(video_views) == 1 else video_index + 1)
        video_savers.append(VideoSaver(grid_world.figure, str(video_path), fps=video_cfg.get("fps", 30)))

video_records = [
    {"filename": str(saver.filename),
     "view": describe_view(view) if view is not None else " then ".join(describe_view(v) for v in display_views)}
    for view, saver in zip(video_views, video_savers)
]
if video_records:
    print("Writing %d video(s): %s" % (len(video_records), ", ".join(
        "%s (%s)" % (Path(record["filename"]).name, record["view"]) for record in video_records)))
else:
    print("Running headless: no figure, so no video")

experiment_log = None
if logging_cfg.get("enabled", True):
    log_dir = (Path(__file__).parent / logging_cfg.get("output_dir", "logs")).expanduser()
    log_filename = run_output_path(log_dir, timestamp, "json")
    experiment_log = ExperimentLog(log_filename, config, grid_width_height, N,
                                   record_poses=logging_cfg.get("record_poses", True))
    experiment_log.record_setup(
        assigned_targets,
        ground_truth=ground_truth,
        topology=communication_topology,
        videos=video_records)

    # The sheaf itself, and the 0-cochain it starts from: the first term the iterates recorded
    # per step are read against. A run without a sheaf records neither.
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

# Init patches. Each robot carries its agent index as a numeral laid straight onto the drawn
# hull, so the only colors on the figure are the belief labels. There is no disc behind the
# numeral: the hull is already a solid dark shape to read against, and a disc wide enough to
# hold the glyph covered the vehicle it was meant to label. The floor arm keeps its disc --
# see exp/robotarium/experiment_template.py, where the badge is projected onto the ground and
# has no drawn hull under it. Made here rather than on the first frame the robots drive, since
# a frame held on a Laplacian iterate is written before anyone has moved and would otherwise
# come out with no robots on it. A headless run makes none.
badge_text_color = rendering_cfg.get("badge_text_color", "#FFFFFF")
patches_badge_labels = []
if grid_world.axes is not None and rendering_cfg.get("show_agent_badges", True):
    for i in range(N):
        # zorder sits above the Robotarium's own robot handles, which are at 2
        badge_label = grid_world.axes.text(0, 0, agent_label(i), color=badge_text_color,
                                           fontsize=7, weight="bold",
                                           ha="center", va="center", zorder=2.6)
        patches_badge_labels.append(badge_label)


def place_agent_badges():
    """
    Reattaches each agent's numeral to where its robot now is. A headless run has no numerals
    to reattach, so this runs over an empty list rather than needing a guard at each call.
    """
    for i in range(len(patches_badge_labels)):
        robot_x, robot_y, _ = tuple(grid_world.robot_poses[:,i].tolist())
        patches_badge_labels[i].set_position((robot_x, robot_y))


def write_frame_to_every_video(coords, view=None):
    """
    Draws the figure as it stands and writes it once to each video, without stepping the
    simulation. The frame loop below writes the frames the robots' motion produces; neither a
    sweep of the Laplacian nor a change of view produces any of its own, so holding a video
    on either needs this.

    Args:
        coords (np.ndarray): The 2xN grid coordinates the tiles are resolved against, which
            during a held frame is where the robots still are rather than where they go.
        view: A view to draw instead of whichever one the run is currently on, for holding a
            video on a view the run is only passing through. Videos pinned to one view by
            video.one_per_view keep drawing theirs regardless, so a tour costs them held
            frames of their own view and the set stays frame for frame aligned.
    """
    place_agent_badges()
    draw_communication_lines(sheaf_sections, messaging=mission.messaging_edges)
    cycling_view = mission.view if view is None else view
    for video_view, video_saver in zip(video_views, video_savers):
        frame_view = cycling_view if video_view is None else video_view
        set_status_caption(status_caption(frame_view))
        create_color_grid(coords, frame_view, sheaf_contested)
        video_saver.writeFrame()


# One entry per Laplacian sweep of the whole run, so a sweep can name which iterate it is
sweeps_made = []
frames_per_sweep = video_cfg.get("frames_per_sweep", 0)


def observe_sweep(sweep_in_step, conflicts, changed):
    """
    Called after each sweep of the flow, with the assignment as that sweep left it.

    A sweep is one application of the Tarski Laplacian, and it is the thing the run is
    actually about; what the agents end up believing is only the part of it a labeling can
    hold. So the whole state is snapshotted here -- possibility sets, which interfaces (and,
    over the region sheaf, which regions of each) have closed, whether the assignment is a
    global section, how many tiles the endpoints of each interface still concretely differ
    on -- for the log, for the caption, and for the tile marks and interface styling the
    videos draw.

    Over the region sheaf the caption carries the two levels side by side, and short enough
    to be read off a moving frame: `3/3 interfaces ✓` next to `70 tiles differ` is the point
    of the construction, the interfaces agreeing in the shared vocabulary while the
    tile-level pictures still differ, and the disagreement on screen being exactly the
    kernel of the abstraction the agents communicate through. The tick is the assignment
    being a global section, which is all of the interfaces having closed and so is the
    fraction reaching 1 -- said twice because at a glance the fraction is two numbers to
    compare and the tick is not.

    Args:
        sweep_in_step (int): Which sweep this is within the current grid step, from 0.
        conflicts (list): The conflicts this sweep surfaced.
        changed (bool): Whether this sweep changed anything. False means the flow had already
            reached its fixed point, which stays reached until an observation or a
            recommitment reopens it.
    """
    global sheaf_sections, sheaf_contested, sheaf_caption

    state = sheaf_state(belief_sheaf, legs=sheaf_legs)
    sweeps_made.append(state)

    sheaf_sections = state.sections
    sheaf_contested = state.contested
    agreeing = sum(1 for agrees in state.sections.values() if agrees)
    section_mark = " ✓" if state.is_section else ""
    if region_sheaf:
        differing = sum(state.concrete_disagreement.values())
        sheaf_caption = "sweep %d  ·  %d/%d interfaces%s  ·  %s" % (
            len(sweeps_made), agreeing, len(state.sections), section_mark,
            "%d tiles differ" % differing if differing else "tiles agree")
    else:
        sheaf_caption = "sweep %d  ·  %d/%d interfaces%s" % (
            len(sweeps_made), agreeing, len(state.sections), section_mark)

    # Sweeps are numbered from 1 as ordinals rather than indices, so that a record names the
    # same iterate the caption on the video does
    if record_iterates:
        mission.record_sweep({
            "sweep": len(sweeps_made),
            "sweep_in_step": sweep_in_step + 1,
            "changed": bool(changed),
            "conflicts": [{**conflict._asdict(), "tile": list(conflict.tile)} for conflict in conflicts],
            **encode_sheaf_state(state, grid_width_height,
                                 contracts=render_contracts(belief_sheaf) if record_contracts else None),
        })

    # Holding the video on the iterate, robots stationary, so the sweeps are watchable rather
    # than only their effect. Not when the agents move asynchronously: holding stops every
    # agent, including the ones this sweep has nothing to do with, and a run whose point is
    # that agents no longer wait for each other cannot freeze them all twice a second.
    if not mission.asynchronous_moves:
        for _ in range(frames_per_sweep):
            write_frame_to_every_video(mission.robot_coords)


# The drive loop itself lives in agsheaf.mission, so that this run and the Robotarium bundle
# obey the same configuration -- asynchronous moves, move schedules, the communication clock,
# the mission monitor and its recommitments -- rather than copies of a loop drifting apart.
# What stays here is what is genuinely this run's own: what gets drawn, and into how many
# videos.


def draw_the_frame(coords, view):
    """
    One control iteration's worth of drawing: the badges and interface lines where the robots
    now are, then the frame written once per video.

    The simulation has already been stepped, so the videos differ only in what is drawn over
    the same robots in the same places, which is what lets them be played side by side.

    Args:
        coords (np.ndarray): The tiles the agents hold.
        view: The view the run is on.
    """
    place_agent_badges()
    draw_communication_lines(sheaf_sections, messaging=mission.messaging_edges)
    for video_view, video_saver in zip(video_views, video_savers):
        frame_view = view if video_view is None else video_view
        set_status_caption(status_caption(frame_view))
        create_color_grid(coords, frame_view, sheaf_contested)
        video_saver.writeFrame()


def hold_on_the_view(coords, view):
    """
    Holds every video on one view, robots stationary, so that a single video carries the whole
    set of views of one moment rather than only the one the run is on.

    Args:
        coords (np.ndarray): The tiles the agents hold.
        view: The view to hold on.
    """
    for _ in range(frames_per_view):
        write_frame_to_every_video(coords, view)


mission = Mission(config, grid_world, assigned_targets, label_beliefs,
                  belief_sheaf=belief_sheaf, ground_truth=ground_truth,
                  experiment_log=experiment_log, display_views=display_views,
                  on_tick=draw_the_frame, on_sweep=observe_sweep, on_hold=hold_on_the_view)
all_arrived, _ = mission.run()

robot_coords = mission.robot_coords
agent_scores = mission.agent_scores
step_count = mission.step_count

# Close videos and print errors
for video_saver in video_savers:
    video_saver.close()
grid_world.robotarium.debug()

if experiment_log is not None:
    experiment_log.close(final_scores=agent_scores, steps_taken=step_count,
                         all_arrived=all_arrived)
    print("Logged run to %s" % experiment_log.path)

if ground_truth is not None:
    print("Final scores:")
    for i in range(N):
        print("  Agent %s: %+.1f" % (agent_label(i), agent_scores[i]))
    print("  Total: %+.1f" % agent_scores.sum())
