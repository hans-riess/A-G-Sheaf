"""
One frame of the figure the experiment draws, saved as a PNG, with no run behind it.

    python exp/preview_frame.py                        # the screen figure, as exp/run.py draws it
    python exp/preview_frame.py --floor                # what the Robotarium projects onto the arena
    python exp/preview_frame.py --floor --view 2 --sweeps 4 --out /tmp/floor.png

One configuration is drawn onto two surfaces -- a monitor, and white foam under the arena's
projector -- and they do not want the same graphics. A 7-point numeral in the corner of a tile
is fine on the first and gone on the second; the green of a closed interface and the magenta of
an open one separate cleanly on the first and arrive as one washed-out grey on the second.
exp/robotarium/config.yaml is where that divergence is written down, and the only way to see
what it does used to be to build a submission, upload it, and read the returned overhead
footage. This draws the same frame locally instead, from the same layered configuration:
exp/defaults.yaml, then exp/worlds/primary.yaml, then whatever testbed layers are named.

What it is not is a run. Nothing drives, nothing is scored, nothing is logged. The robots stand
on the squares the configuration starts them on, and the sheaf is the real one -- built from the
configured beliefs and swept as many times as asked -- so the tiles, the contested marks and the
interface states are a moment the run actually passes through rather than a mock-up of one.
Every interface is drawn lit, as it is on the iteration just after a sweep everybody broadcasts
on, that being the frame with the most on it. With --sweeps 0 nothing is lit, and where the
configuration hides idle interfaces (show_idle_communication: false, the default) the frame then
carries no lines at all -- which is also what the floor looks like between broadcasts.

The rendering block below is the one exp/run.py and exp/robotarium/experiment_template.py both
build their figures from, reading the same keys from the same configuration. It is copied rather
than shared because each of those is a script that runs a whole experiment when imported, so
there is nothing to import from either: a rendering key added to one belongs in all three.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

from agsheaf.gridsheaf import build_belief_sheaf, communicate, sheaf_state
from agsheaf.gridworld import (GridWorld, agent_label, set_agent_label_offset,
                               describe_view, initialize_communication_lines,
                               initialize_region_overlay, initialize_safety_color_grid,
                               initialize_status_caption, parse_assignments,
                               parse_communication_topology, parse_display_views,
                               parse_ground_truth, parse_label_beliefs, parse_start_tiles,
                               resolve_tile_labels, route_corridor)
from agsheaf.regions import parse_regions
from agsheaf.utils import compose_config

import robot_icon

EXPERIMENTS_DIR = Path(__file__).resolve().parent
ROBOTARIUM_CONFIG = EXPERIMENTS_DIR / "robotarium" / "config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENTS_DIR / "worlds" / "primary.yaml",
                        help="the world to draw, over exp/defaults.yaml")
    parser.add_argument("--floor", action="store_true",
                        help="draw it as the testbed projects it, i.e. over "
                             "exp/robotarium/config.yaml")
    parser.add_argument("--config", type=Path, nargs="*", default=[],
                        help="further layers, applied in order over the ones above -- a variant "
                             "such as exp/robotarium/no_communication.yaml, or a scratch file "
                             "holding one setting being tried out")
    parser.add_argument("--view", type=str, default=None,
                        help="whose beliefs to draw: an agent index, 'ground_truth', or a pair "
                             "of either separated by a comma, which draws them against each "
                             "other. Defaults to the first view the configuration names.")
    parser.add_argument("--sweeps", type=int, default=1,
                        help="Laplacian sweeps to run before drawing. 0 is the opening frame, "
                             "before anybody has said anything.")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the PNG. Defaults to exp/previews/floor.png or "
                             "exp/previews/screen.png.")
    parser.add_argument("--dpi", type=int, default=140)
    arguments = parser.parse_args()

    layers = ([ROBOTARIUM_CONFIG] if arguments.floor else []) + list(arguments.config)
    config = layered_config(arguments.experiment, layers)

    out_path = arguments.out or (EXPERIMENTS_DIR / "previews" /
                                 ("floor.png" if arguments.floor else "screen.png"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    draw_frame(config, view_argument=arguments.view, sweeps=arguments.sweeps,
               out_path=out_path, dpi=arguments.dpi)


def layered_config(experiment: Path, layers: list) -> dict:
    """
    The configuration the frame is drawn from: exp/defaults.yaml, then the experiment file, then
    each layer in turn, merged section by section the way build_submission.py merges them, so
    the frame is drawn from the configuration a run would be run from rather than from a second
    copy of it that could drift.

    The figure is forced on whatever the configuration says, since a frame is the entire point
    here, where in a run it is optional and off for batch physics.

    Args:
        experiment (Path): The experiment file, applied over exp/defaults.yaml.
        layers (list): Further layers, in order, each applied over the ones before it.
    """
    config = compose_config(experiment, layers, EXPERIMENTS_DIR / "defaults.yaml", announce=True)

    config["gridworld"]["show_figure"] = True
    return config


def draw_frame(config: dict, view_argument, sweeps: int, out_path: Path, dpi: int) -> None:
    """
    Builds the world, sweeps the sheaf, draws everything the run draws, and writes the PNG.

    Args:
        config (dict): The layered configuration.
        view_argument: The view named on the command line, or None for the configured one.
        sweeps (int): Laplacian sweeps to run before drawing.
        out_path (Path): Where the PNG goes.
        dpi (int): Its resolution. Only the pixel count -- the layout is set by the figure, and
            the figure is the arena.
    """
    gridworld_cfg = config["gridworld"]
    rendering_cfg = config["rendering"]
    beliefs_cfg = config.get("beliefs") or {}
    planning_cfg = config.get("planning") or {}
    sheaf_cfg = config.get("sheaf") or {}
    contracts_cfg = config.get("contracts") or {}

    set_agent_label_offset(rendering_cfg.get("agent_label_offset", 0))

    # Seeded the way a run seeds itself, so the robots stand where that run's first frame stands
    # them. The Robotarium simulator draws its scattering poses from two of these three streams.
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    rng = np.random.default_rng(seed=config["seed"])

    N = gridworld_cfg["num_agents"]
    grid_width_height = (gridworld_cfg["grid_width"], gridworld_cfg["grid_height"])
    tile_costs = planning_cfg.get("tile_costs")

    robot_coords = parse_start_tiles(config.get("starts"), N, grid_width_height, rng=rng)
    robot_icon.install()   # draw the robots as UUVs; cosmetic, and before the figure is built
    grid_world = GridWorld(
        number_of_robots=N,
        grid_width_height=grid_width_height,
        show_arrows=gridworld_cfg["show_arrows"],
        show_figure=True,
        sim_in_real_time=False,
        initial_coordinates=robot_coords,
        grid_safety_gap=gridworld_cfg["grid_safety_gap"],
        min_distance_repulsion=gridworld_cfg["min_distance_repulsion"],
        max_distance_repulsion=gridworld_cfg["max_distance_repulsion"],
        robot_close_to_target_distance=gridworld_cfg.get("arrival_distance", 0.05))

    label_beliefs = parse_label_beliefs(beliefs_cfg, N, grid_width_height)
    ground_truth = parse_ground_truth(config.get("ground_truth"), grid_width_height)
    assigned_targets = parse_assignments(config.get("assignments"), N, grid_width_height)
    communication_topology = parse_communication_topology(config.get("communication"), N)
    regions = parse_regions(config["regions"], grid_width_height) if config.get("regions") else None

    if view_argument is None:
        view = parse_display_views(beliefs_cfg, N)[0]
    else:
        view = parse_display_views({"view": parse_view_argument(view_argument)}, N)[0]

    if sheaf_cfg.get("share_assignments", True):
        for i in range(N):
            label_beliefs[i][(int(assigned_targets[0, i]), int(assigned_targets[1, i]))] = "target"

    belief_options = dict(
        label_beliefs=label_beliefs,
        default_label=beliefs_cfg.get("default", "safe"),
        mark_other_agents_unsafe=beliefs_cfg.get("mark_other_agents_unsafe", True))

    # The sheaf, and the sweeps of it whose iterate this frame is. Built exactly as a run builds
    # it -- the corridors its contracts rely on computed the way the mission monitor computes
    # them -- so the possibility sets the frame draws are ones the run passes through.
    belief_sheaf = None
    if (sheaf_cfg.get("enabled", True) and communication_topology is not None
            and communication_topology.number_of_edges() > 0):
        initial_corridors = None
        if regions is not None:
            resolved = resolve_tile_labels(robot_coords, grid_width_height, assigned_targets,
                                           **belief_options)
            initial_corridors = {
                i: route_corridor((int(robot_coords[0, i]), int(robot_coords[1, i])),
                                  (int(assigned_targets[0, i]), int(assigned_targets[1, i])),
                                  grid_width_height, resolved[i], tile_costs=tile_costs)
                for i in range(N)}
        belief_sheaf = build_belief_sheaf(label_beliefs, grid_width_height, communication_topology,
                                          regions=regions, corridors=initial_corridors,
                                          contracts_cfg=contracts_cfg if regions is not None else None)

    sheaf_sections, sheaf_contested, sheaf_caption = None, None, ""
    if belief_sheaf is not None and sweeps > 0:
        communicate(belief_sheaf, sweeps=sweeps)
        state = sheaf_state(belief_sheaf, legs=sheaf_cfg.get("legs", "kan"))
        sheaf_sections, sheaf_contested = state.sections, state.contested

        # Worded as the runners word it, since how long the caption is is one of the things
        # this is drawn to check
        agreeing = sum(1 for agrees in state.sections.values() if agrees)
        section_mark = " ✓" if state.is_section else ""
        if regions is not None:
            differing = sum(state.concrete_disagreement.values())
            sheaf_caption = "sweep %d  ·  %d/%d interfaces%s  ·  %s" % (
                sweeps, agreeing, len(state.sections), section_mark,
                "%d tiles differ" % differing if differing else "tiles agree")
        else:
            sheaf_caption = "sweep %d  ·  %d/%d interfaces%s" % (
                sweeps, agreeing, len(state.sections), section_mark)

    # --- the rendering block, as exp/run.py and the submission template build it ---

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
        section_colors=section_colors)

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

    if draw_section_styling:
        line_color, agreeing_color = section_colors[1], section_colors[0]
        line_alpha = rendering_cfg.get("section_alpha", 0.9)
        line_width = rendering_cfg.get("section_linewidth", 1.4)
        section_style = rendering_cfg.get("section_linestyle", "-")
        line_style = rendering_cfg.get("no_section_linestyle") or section_style
    else:
        line_color, agreeing_color = rendering_cfg.get("communication_color", "#051E39"), None
        line_alpha = rendering_cfg.get("communication_alpha", 0.25)
        line_width = rendering_cfg.get("communication_linewidth", 0.8)
        line_style = rendering_cfg.get("communication_linestyle", "--")
        section_style = line_style

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
        show_when_idle=rendering_cfg.get("show_idle_communication", True))

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
        background=rendering_cfg.get("sheaf_status_background", True))

    caption_parts = []
    if show_view_in_status:
        caption_parts.append(describe_view(view, halves=True))
    if show_sheaf_status and sheaf_caption:
        caption_parts.append(sheaf_caption)

    # Every interface carries a message on the iteration after a sweep everybody broadcast on,
    # which is the frame worth drawing: it is the only one where the lines are on the floor at
    # all, where the configuration hides them when idle.
    messaging = sorted(tuple(sorted(edge)) for edge in communication_topology.edges) \
        if communication_topology is not None and sweeps > 0 else []

    caption = "  ·  ".join(caption_parts)
    create_color_grid(robot_coords, view, sheaf_contested)
    draw_communication_lines(sheaf_sections, messaging=messaging)
    set_status_caption(caption)
    draw_agent_badges(grid_world, rendering_cfg, N)

    grid_world.figure.savefig(out_path, dpi=dpi)

    print("\nDrew %s%s" % (describe_view(view),
                           "" if not sheaf_caption else ", %s" % sheaf_caption))
    report_caption_placement(grid_world, caption)
    print("Wrote %s" % out_path)


def report_caption_placement(grid_world, caption: str) -> None:
    """
    Says where the caption landed, in metres of arena, and complains if that is somewhere it
    should not be.

    Two ways a caption goes wrong, and neither shows up anywhere but the floor. It can run off
    the arena, in which case the projector simply loses the end of it. Or it can sit over the
    grid, in which case it is projected onto ground the robots drive on, which is what the
    caption was made large and moved into the margin to avoid. The margin is only 0.4 m wide on
    a 12x8 grid, and how much of it a caption needs depends on the words in it -- a comparison
    view is three times the length of a single one -- so this is worth measuring rather than
    reasoning about.

    Args:
        grid_world (GridWorld): The world the caption was drawn onto.
        caption (str): The caption as it was set, or "" where the run draws none.
    """
    if not caption:
        return

    figure, axes = grid_world.figure, grid_world.axes
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    # The artist itself is inside a closure, so it is found by what it says. A caption drawn
    # over the tiles is not the only text on the figure, but it is the only text with this in it.
    drawn = [text for text in axes.texts if text.get_text() == caption]
    if not drawn:
        return

    # Rotated text measures as the upright box around the turned glyphs, which is what matters
    # here: it is the footprint on the floor rather than the reading order that is in question.
    (left, bottom), (right, top) = axes.transData.inverted().transform(
        drawn[0].get_window_extent(renderer).get_points())

    arena_x, arena_y = grid_world.robotarium_width / 2, grid_world.robotarium_height / 2
    grid_x = grid_world.grid_width * grid_world.tile_width / 2
    grid_y = grid_world.grid_height * grid_world.tile_width / 2

    outside = max(-arena_x - left, right - arena_x, -arena_y - bottom, top - arena_y)
    if outside > 0:
        print("  the caption runs %.2f m off the arena -- lower rendering.sheaf_status_fontsize, "
              "or move rendering.sheaf_status_position" % outside)
        return

    # How far the caption is from the grid, on the axis it is clear on. Both negative means it
    # overlaps: there is no side of the grid it sits beside.
    clearance = max(-grid_x - right, left - grid_x, -grid_y - top, bottom - grid_y)
    if clearance < 0:
        print("  the caption sits over the grid, so it is projected onto ground the robots "
              "drive on")
    else:
        print("  the caption clears the grid by %.2f m, inside the arena" % clearance)


def draw_agent_badges(grid_world, rendering_cfg: dict, number_of_robots: int) -> None:
    """
    Puts each robot's agent index onto it, the numeral a run keeps attached as they drive.

    Copied from the entry points for the same reason the rest of the rendering is: each of them
    builds its badges inline, around a placement function a still frame has no use for. Laid
    straight onto the drawn hull, with no disc behind it -- see exp/run.py, which this matches.

    Args:
        grid_world (GridWorld): The world whose axes are drawn onto.
        rendering_cfg (dict): The rendering configuration.
        number_of_robots (int): How many numerals to place.
    """
    if not rendering_cfg.get("show_agent_badges", True):
        return

    badge_text_color = rendering_cfg.get("badge_text_color", "#FFFFFF")
    for i in range(number_of_robots):
        robot_x, robot_y, _ = tuple(grid_world.robot_poses[:, i].tolist())
        grid_world.axes.text(robot_x, robot_y, agent_label(i), color=badge_text_color,
                             fontsize=7, weight="bold",
                             ha="center", va="center", zorder=2.6)


def parse_view_argument(text: str):
    """
    Turns a --view argument into the view specification the configuration would have written.

    An agent index, "ground_truth", or a comma-separated pair of either. Agents are named on the
    command line the way the configuration names them -- from zero -- rather than the way the
    figure labels them, since this is a configuration value and not a caption.

    Args:
        text (str): The argument as it was typed.
    """
    sides = [side.strip() for side in text.split(",")]
    assert 1 <= len(sides) <= 2, \
        "Failed to read --view. Expected a side or a comma-separated pair of them, received %r." % text

    parsed = [int(side) if side.lstrip("-").isdigit() else side for side in sides]
    return parsed[0] if len(parsed) == 1 else parsed


if __name__ == "__main__":
    main()
