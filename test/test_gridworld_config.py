"""Tests for parsing and encoding the gridworld experiment configuration."""
import numpy as np
import pytest

from agsheaf.gridworld import (parse_assignments, parse_communication_topology,
                               parse_ground_truth, assert_assignments_are_targets,
                               parse_display_views, describe_view, parse_start_tiles)
from agsheaf.log import encode_label_grid, decode_label_grid, LABEL_CHARACTERS
from agsheaf.utils import run_output_path


GRID = (9, 5)
GOOD_ASSIGNMENTS = {0: [6, 4], 1: [5, 4], 2: [6, 3], 3: [5, 3]}


# --- communication topology -------------------------------------------------

def test_topology_parses_an_edge_list():
    topology = parse_communication_topology({"edges": [[0, 1], [1, 2], [2, 3]]}, 4)
    assert topology.number_of_edges() == 3
    assert sorted(sorted(edge) for edge in topology.edges) == [[0, 1], [1, 2], [2, 3]]


def test_topology_is_undirected_and_dedupes():
    """[0,1] and [1,0] name one edge, and repeating it is not an error."""
    topology = parse_communication_topology({"edges": [[0, 1], [1, 0], [0, 1]]}, 4)
    assert topology.number_of_edges() == 1
    assert topology.has_edge(1, 0) and topology.has_edge(0, 1)


def test_topology_keeps_agents_with_no_edges_as_nodes():
    """An agent that can talk to nobody is isolated, not missing."""
    topology = parse_communication_topology({"edges": [[0, 1]]}, 4)
    assert sorted(topology.nodes) == [0, 1, 2, 3]
    assert topology.degree(3) == 0


def test_topology_absent_gives_none():
    assert parse_communication_topology(None, 4) is None
    assert parse_communication_topology({}, 4) is None


def test_topology_without_edges_is_all_isolated():
    topology = parse_communication_topology({"edges": []}, 3)
    assert sorted(topology.nodes) == [0, 1, 2]
    assert topology.number_of_edges() == 0


@pytest.mark.parametrize("edges", [
    [[0, 0]],           # self-loop
    [[0, 9]],           # index out of range
    [[0, -1]],          # negative index
    [[0, 1, 2]],        # not a pair
    [[0]],              # not a pair
    [["a", 1]],         # not an index
])
def test_topology_rejects_bad_edges(edges):
    with pytest.raises(AssertionError):
        parse_communication_topology({"edges": edges}, 4)


# --- assignments ------------------------------------------------------------

def test_assignments_parse_in_robot_order():
    assigned = parse_assignments(GOOD_ASSIGNMENTS, 4, GRID)
    assert assigned.shape == (2, 4)
    assert [[int(assigned[0, i]), int(assigned[1, i])] for i in range(4)] == [[6, 4], [5, 4], [6, 3], [5, 3]]


@pytest.mark.parametrize("assignments, reason", [
    ({0: [6, 4], 1: [6, 4], 2: [6, 3], 3: [5, 3]}, "two agents share a square"),
    ({0: [6, 4], 1: [5, 4], 2: [6, 3]}, "an agent has none"),
    ({0: [6, 4], 1: [5, 4], 2: [6, 3], 3: [99, 9]}, "off the grid"),
    ({0: [6, 4], 1: [5, 4], 2: [6, 3], 3: [[5, 3], [4, 4]]}, "two squares for one agent"),
    (None, "no assignments at all"),
])
def test_assignments_reject_bad_configs(assignments, reason):
    with pytest.raises(AssertionError):
        parse_assignments(assignments, 4, GRID)


# --- starting squares -------------------------------------------------------

def test_starts_parse_in_robot_order():
    starts = parse_start_tiles({0: [3, 1], 1: [4, 2], 2: [8, 3], 3: [0, 3]}, 4, GRID)
    assert starts.shape == (2, 4)
    assert [[int(starts[0, i]), int(starts[1, i])] for i in range(4)] == [[3, 1], [4, 2], [8, 3], [0, 3]]


@pytest.mark.parametrize("starts, reason", [
    ({0: [3, 1], 1: [3, 1], 2: [8, 3], 3: [0, 3]}, "two agents on one tile"),
    ({0: [3, 1], 1: [4, 2], 2: [8, 3]}, "an agent has none"),
    ({0: [3, 1], 1: [4, 2], 2: [8, 3], 3: [99, 9]}, "off the grid"),
    ({0: [3, 1], 1: [4, 2], 2: [8, 3], 4: [0, 3]}, "an index out of range"),
])
def test_starts_reject_bad_configs(starts, reason):
    with pytest.raises(AssertionError):
        parse_start_tiles(starts, 4, GRID)


def test_starts_absent_samples_from_the_rng():
    """No block is the old behaviour: the layout comes off the seed alone."""
    sampled = parse_start_tiles(None, 4, GRID, rng=np.random.default_rng(seed=0))
    expected = parse_start_tiles({}, 4, GRID, rng=np.random.default_rng(seed=0))
    assert sampled.shape == (2, 4)
    assert np.array_equal(sampled, expected)


def test_named_starts_ignore_the_rng():
    """Naming the squares takes the run off the seed, rather than perturbing it."""
    named = {0: [3, 1], 1: [4, 2], 2: [8, 3], 3: [0, 3]}
    first = parse_start_tiles(named, 4, GRID, rng=np.random.default_rng(seed=0))
    second = parse_start_tiles(named, 4, GRID, rng=np.random.default_rng(seed=7))
    assert np.array_equal(first, second)


def test_a_start_may_be_the_agents_own_assigned_square():
    """An agent that begins where it was sent has simply arrived."""
    starts = parse_start_tiles(GOOD_ASSIGNMENTS, 4, GRID)
    assert np.array_equal(starts, parse_assignments(GOOD_ASSIGNMENTS, 4, GRID))


def test_targets_need_not_form_a_block():
    """Scattered targets are fine: only the assignments say where a robot must go."""
    scattered = {"default": "safe", "target": [[0, 0], [8, 4], [4, 2], [1, 3]]}
    ground_truth = parse_ground_truth(scattered, GRID)
    assert sorted(t for t, l in ground_truth.items() if l == "target") == [(0, 0), (1, 3), (4, 2), (8, 4)]

    assigned = parse_assignments({0: [0, 0], 1: [8, 4], 2: [4, 2], 3: [1, 3]}, 4, GRID)
    assert_assignments_are_targets(assigned, ground_truth)


def test_ground_truth_needs_at_least_one_target():
    with pytest.raises(AssertionError):
        parse_ground_truth({"default": "safe", "unsafe": [[0, 0]]}, GRID)


def test_assignments_must_be_true_targets():
    ground_truth = parse_ground_truth(
        {"default": "safe", "target": [[5, 3], [6, 3], [5, 4], [6, 4]]}, GRID)
    assert_assignments_are_targets(parse_assignments(GOOD_ASSIGNMENTS, 4, GRID), ground_truth)

    off_target = parse_assignments({0: [6, 4], 1: [5, 4], 2: [6, 3], 3: [0, 0]}, 4, GRID)
    with pytest.raises(AssertionError):
        assert_assignments_are_targets(off_target, ground_truth)


# --- display views ----------------------------------------------------------

def test_views_parse_sides_and_pairs():
    assert parse_display_views({"views": ["ground_truth", 0, [0, 1]]}, 2) == [
        "ground_truth", 0, (0, 1)]


def test_view_and_views_are_mutually_exclusive():
    with pytest.raises(AssertionError):
        parse_display_views({"view": 0, "views": [1]}, 2)


@pytest.mark.parametrize("tour_views_every", ["sometimes", 0, -1, 1.5])
def test_tour_views_every_rejects_anything_but_never_or_a_step_count(tour_views_every):
    with pytest.raises(AssertionError):
        parse_display_views({"views": [0], "tour_views_every": tour_views_every}, 2)


def test_tour_views_every_accepts_never_and_a_step_count():
    assert parse_display_views({"views": [0], "tour_views_every": "never"}, 2) == [0]
    assert parse_display_views({"views": [0], "tour_views_every": 3}, 2) == [0]


def test_describe_view_names_sides_and_pairs():
    assert describe_view("ground_truth") == "ground truth"
    assert describe_view(0) == "agent 0"
    assert describe_view((0, 1)) == "agent 0 vs agent 1"


def test_describe_view_marks_the_halves_of_a_split_tile():
    """
    The caption is drawn over the tiles it names, so it carries the same glyphs
    the split tiles are drawn with. A single side has no halves to tell apart.
    """
    assert describe_view((0, "ground_truth"), halves=True) == "agent 0 ◣ vs ground truth ◥"
    assert describe_view(0, halves=True) == "agent 0"


# --- label grid encoding ----------------------------------------------------

def test_label_characters_are_distinct():
    assert len(set(LABEL_CHARACTERS.values())) == len(LABEL_CHARACTERS)


def test_encode_writes_highest_row_first():
    """Rows read down the page the way the grid is drawn, so y=1 comes before y=0."""
    labels = {(x, y): "safe" for x in range(3) for y in range(2)}
    labels[(0, 1)] = "target"
    labels[(2, 0)] = "unsafe"

    assert encode_label_grid(labels, (3, 2)) == ["tss", "ssu"]


def test_encode_decode_round_trip():
    labels = {}
    for x in range(GRID[0]):
        for y in range(GRID[1]):
            labels[(x, y)] = ("target", "safe", "unsafe", "unknown")[(x + y) % 4]

    encoded = encode_label_grid(labels, GRID)
    assert len(encoded) == GRID[1]
    assert all(len(row) == GRID[0] for row in encoded)
    assert decode_label_grid(encoded) == labels


def test_decode_rejects_an_unknown_character():
    with pytest.raises(AssertionError):
        decode_label_grid(["tsx"])


# --- output naming ----------------------------------------------------------

def test_one_output_of_a_kind_is_unnumbered():
    """The timestamp alone names the run; nothing configured goes in the filename."""
    assert run_output_path("videos", "20260802_172038", "mp4").name == "20260802_172038.mp4"
    assert run_output_path("logs", "20260802_172038", "json").name == "20260802_172038.json"


def test_several_videos_are_numbered_from_one():
    names = [run_output_path("videos", "20260802_172038", "mp4", number=n).name for n in (1, 2, 3)]
    assert names == ["20260802_172038-1.mp4", "20260802_172038-2.mp4", "20260802_172038-3.mp4"]


def test_output_path_keeps_its_directory():
    assert run_output_path("/tmp/videos", "20260802_172038", "mp4").parent.as_posix() == "/tmp/videos"


# --- display views ----------------------------------------------------------

def test_a_misspelled_beliefs_key_is_rejected():
    """
    The regression. `switch_view_every` and `tour_views_every` differ in the plural, and
    `switch_views_every` sat in exp/defaults.yaml being silently ignored: the reader fell back
    to "never" and the run looked exactly as though the key had not been given.
    """
    with pytest.raises(AssertionError) as excinfo:
        parse_display_views({"views": [0, 1], "switch_views_every": 1}, 2)
    assert "switch_views_every" in str(excinfo.value)
    assert "switch_view_every" in str(excinfo.value)


def test_the_spelled_key_is_accepted():
    assert parse_display_views({"views": [0, 1], "switch_view_every": 1}, 2) == [0, 1]


def test_an_agent_can_be_compared_with_ground_truth():
    """`[ground_truth, i]` is where agent i's beliefs depart from the truth."""
    views = parse_display_views(
        {"views": [["ground_truth", 0], ["ground_truth", 1]], "switch_view_every": 1}, 2)
    assert views == [("ground_truth", 0), ("ground_truth", 1)]


def test_following_the_moving_agent_needs_a_view_each():
    """The view is picked by the mover's index, so there has to be one to pick."""
    with pytest.raises(AssertionError, match="one entry per agent"):
        parse_display_views(
            {"views": [["ground_truth", 0]], "follow_moving_agent": True}, 3)


def test_following_the_moving_agent_accepts_a_view_each():
    views = parse_display_views(
        {"views": [["ground_truth", 0], ["ground_truth", 1]], "follow_moving_agent": True}, 2)
    assert views == [("ground_truth", 0), ("ground_truth", 1)]


def test_following_is_off_by_default_so_any_number_of_views_is_fine():
    assert len(parse_display_views({"views": [0, 1, "ground_truth"]}, 2)) == 3


# --- communication lines ----------------------------------------------------

def _lines_world():
    """A drawing grid world and its two-edge topology. Agg backend, so no window opens."""
    import networkx as nx
    from agsheaf.gridworld import GridWorld, initialize_communication_lines
    world = GridWorld(number_of_robots=3, grid_width_height=(5, 3), show_figure=True,
                      sim_in_real_time=False, initial_coordinates=np.array([[0, 2, 4], [0, 1, 2]]))
    topology = nx.Graph([(0, 1), (1, 2)])
    return world, topology, initialize_communication_lines


def test_a_message_lights_its_interface_up():
    """
    A sweep is instantaneous, so the run holds the interfaces it crossed lit for a few frames.
    Green where the two ends now agree, red where they do not -- and that is the whole of what
    the colour means, so it reads as an event rather than a status.
    """
    world, topology, initialize = _lines_world()
    draw = initialize(world, topology, line_color="#051E39", line_alpha=0.25, line_width=0.8,
                      message_colors=("#066034", "#D90368"), message_width_scale=2.5)
    lines = [line for line in world.axes.lines if line.get_zorder() == 1.4]

    draw({(0, 1): True, (1, 2): False}, messaging={(0, 1), (1, 2)})
    assert lines[0].get_color() == "#066034"      # agreed, so green
    assert lines[1].get_color() == "#D90368"      # still disagrees, so red
    assert all(line.get_alpha() == 1.0 for line in lines)
    assert all(line.get_linewidth() == 0.8 * 2.5 for line in lines)


def test_an_interface_goes_back_to_resting_when_the_message_passes():
    world, topology, initialize = _lines_world()
    draw = initialize(world, topology, line_color="#051E39", line_alpha=0.25, line_width=0.8,
                      message_colors=("#066034", "#D90368"))
    lines = [line for line in world.axes.lines if line.get_zorder() == 1.4]

    draw({(0, 1): True, (1, 2): False}, messaging={(0, 1)})
    draw({(0, 1): True, (1, 2): False}, messaging=set())
    assert all(line.get_alpha() == 0.25 for line in lines)
    assert all(line.get_linewidth() == 0.8 for line in lines)


def test_messages_light_up_without_section_styling_too():
    """A message is an event, so it shows whether or not the run draws settled interfaces."""
    world, topology, initialize = _lines_world()
    draw = initialize(world, topology, line_color="#051E39", line_alpha=0.25, line_width=0.8,
                      message_colors=("#066034", "#D90368"))
    lines = [line for line in world.axes.lines if line.get_zorder() == 1.4]

    draw(None, messaging={(0, 1)})
    assert lines[0].get_color() == "#D90368"      # nothing says it agrees, so red
    assert lines[1].get_color() == "#051E39"      # no message, so resting


def test_no_message_colors_means_no_blink():
    world, topology, initialize = _lines_world()
    draw = initialize(world, topology, line_color="#051E39", line_alpha=0.25, line_width=0.8)
    lines = [line for line in world.axes.lines if line.get_zorder() == 1.4]

    draw({(0, 1): True}, messaging={(0, 1), (1, 2)})
    assert all(line.get_alpha() == 0.25 for line in lines)


def test_assignment_numerals_can_be_held_static_across_views():
    """
    The numerals mark fixed squares, so on a run whose view changes every move they should not
    come and go with it. Static shows every assignment whatever the view is of.
    """
    from agsheaf.gridworld import GridWorld, initialize_safety_color_grid
    world = GridWorld(number_of_robots=2, grid_width_height=(5, 3), show_figure=True,
                      sim_in_real_time=False, initial_coordinates=np.array([[0, 4], [0, 2]]))
    targets = np.array([[4, 0], [2, 0]])
    labels = [{(x, y): "safe" for x in range(5) for y in range(3)} for _ in range(2)]

    def numerals(static):
        draw = initialize_safety_color_grid(world, targets, label_beliefs=labels,
                                            static_assignment_numbers=static)
        draw(np.array([[0, 4], [0, 2]]), 0)     # a view of agent 0 alone
        return sorted(t.get_text() for t in world.axes.texts
                      if t.get_text() in {"0", "1"} and t.get_visible())

    assert numerals(True) == ["0", "1"]
    for text in list(world.axes.texts):
        text.set_visible(False)
    assert numerals(False) == ["0"]


def test_agents_are_labelled_from_one_without_reindexing_them():
    """
    The offset is display only: an agent is still agent 0 in the configuration, the log and the
    sheaf, and reads as 1 on the floor.
    """
    from agsheaf.gridworld import agent_label, set_agent_label_offset, describe_view
    try:
        set_agent_label_offset(1)
        assert agent_label(0) == "1" and agent_label(2) == "3"
        assert describe_view(0) == "agent 1"
        assert describe_view(("ground_truth", 1)) == "ground truth vs agent 2"
    finally:
        set_agent_label_offset(0)
    assert agent_label(0) == "0"


def test_an_idle_interface_can_be_hidden_until_it_carries_something():
    """A line on the floor then means traffic rather than the possibility of it."""
    world, topology, initialize = _lines_world()
    draw = initialize(world, topology, line_color="#051E39", line_alpha=0.25, line_width=0.8,
                      message_colors=("#066034", "#D90368"), show_when_idle=False)
    lines = [line for line in world.axes.lines if line.get_zorder() == 1.4]

    draw({(0, 1): True, (1, 2): False}, messaging={(0, 1)})
    assert lines[0].get_visible() and not lines[1].get_visible()

    draw({(0, 1): True, (1, 2): False}, messaging=set())
    assert not any(line.get_visible() for line in lines)
