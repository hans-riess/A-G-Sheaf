"""
Tests for the `agsheaf` command line's configuration composition.

Everything here exercises `cli.composed`, which is where the layering the whole command line
exists to provide actually happens: defaults, then the world, then `--layer`s, then the control
arms, then `--set`s. It is worth testing on its own because it is pure -- a namespace in, a
configuration dict out -- while everything downstream of it drives robots for a minute or more.
The two regressions below were both found by hand, running the CLI, precisely because nothing
covered this function.

`arguments()` builds the namespace `main()` would have parsed, rather than going through
argparse, so a test names only the flags it is about.
"""
import argparse

import pytest

from agsheaf import cli


def arguments(**overrides):
    """
    The namespace a subcommand's parser produces, with every flag at its default.

    Args:
        **overrides: The flags this test is about, by their argparse destination.
    """
    namespace = dict(experiment=None, world=None, num_agents=None, num_obstacles=2, seed=0,
                     layer=[], no_communication=False, no_abstraction=False,
                     views=None, video=None, tour=None, set=[])
    namespace.update(overrides)
    return argparse.Namespace(**namespace)


def beliefs_of(config):
    """The beliefs block, whether or not the world wrote one."""
    return config.get("beliefs") or {}


# --- naming the world -------------------------------------------------------

def test_world_names_a_file_in_the_worlds_directory():
    assert cli.named_world("primary") == cli.WORLDS_DIR / "primary.yaml"


def test_unknown_world_says_what_there_is():
    with pytest.raises(AssertionError) as raised:
        cli.named_world("nonexistent")
    assert "primary" in str(raised.value)


def test_available_worlds_lists_the_shipped_ones():
    assert {"primary", "variation", "scaled"} <= set(cli.available_worlds())


def test_no_world_named_is_the_primary_one():
    assert cli.composed(arguments())["experiment_name"] \
        == cli.composed(arguments(world="primary"))["experiment_name"]


@pytest.mark.parametrize("named", [
    dict(world="primary", num_agents=4),
    dict(experiment=cli.WORLDS_DIR / "primary.yaml", world="primary"),
    dict(experiment=cli.WORLDS_DIR / "primary.yaml", num_agents=4),
])
def test_naming_the_world_twice_is_refused(named):
    """Which world to run would otherwise be decided by the order the code happens to check."""
    with pytest.raises(AssertionError, match="Name the world once"):
        cli.composed(arguments(**named))


def test_one_namespace_composes_twice():
    """
    Regression. `preview` composes the SAME namespace once per surface, and resolving --world by
    writing the path back onto `arguments.experiment` left the second pass seeing a world named
    both ways -- the "name the world once" assertion firing on a command line that named it once.
    """
    reused = arguments(world="scaled")
    first = cli.composed(reused)
    second = cli.composed(reused)
    assert first["gridworld"]["num_agents"] == second["gridworld"]["num_agents"] == 8


# --- generated worlds -------------------------------------------------------

def test_generating_replaces_the_world_keys():
    generated = cli.composed(arguments(num_agents=6))
    primary = cli.composed(arguments(world="primary"))
    assert generated["gridworld"]["num_agents"] == 6
    assert len(generated["assignments"]) == len(generated["beliefs"]["agents"]) == 6
    assert len(primary["assignments"]) == 4
    assert "starts" in generated                     # primary samples its starts from the seed
    assert generated["ground_truth"] != primary["ground_truth"]


def test_generating_keeps_the_interface_vocabulary_fixed():
    """
    The six rooms are the same rooms. That is the point of scaling by the network rather than by
    the grid: a FIXED abstraction absorbing the disagreement of a larger network is only readable
    off a world whose regions did not grow with it.
    """
    assert cli.composed(arguments(num_agents=8))["regions"] \
        == cli.composed(arguments(world="primary"))["regions"]


def test_generating_inherits_how_the_run_is_conducted():
    """
    A generated world replaces what a world IS and inherits the rest from primary.yaml. This is
    the check that WORLD_KEYS stays a list of world keys: `video.one_per_view` set there went
    unnoticed by every generated run before the CLI composed them this way.
    """
    generated = cli.composed(arguments(num_agents=6))
    primary = cli.composed(arguments(world="primary"))
    assert generated["video"] == primary["video"]
    assert generated["scoring"] == primary["scoring"]
    assert generated["planning"] == primary["planning"]


def test_generating_is_seeded():
    assert cli.composed(arguments(num_agents=6, seed=1))["starts"] \
        == cli.composed(arguments(num_agents=6, seed=1))["starts"]
    assert cli.composed(arguments(num_agents=6, seed=1))["seed"] == 1


# --- control arms -----------------------------------------------------------

def test_no_communication_switches_the_flow_off():
    config = cli.composed(arguments(world="primary", no_communication=True))
    assert config["sheaf"]["enabled"] is False
    assert config["rendering"]["show_sheaf_status"] is False
    assert config["rendering"]["show_sections"] is False


def test_no_abstraction_blanks_the_partition():
    config = cli.composed(arguments(world="primary", no_abstraction=True))
    assert config["regions"] is None


def test_an_arm_renames_the_run():
    """Without this both arms log under the world's name, told apart only by the config block."""
    plain = cli.composed(arguments(world="primary"))["experiment_name"]
    armed = cli.composed(arguments(world="primary", no_communication=True))["experiment_name"]
    assert armed == "%s_no_communication" % plain


def test_arms_apply_to_a_generated_world():
    config = cli.composed(arguments(num_agents=6, no_communication=True))
    assert config["sheaf"]["enabled"] is False


def test_set_beats_an_arm():
    """Contradictory, but it resolves by which instruction is more specific, not by typing order."""
    config = cli.composed(arguments(world="primary", no_communication=True,
                                    set=["sheaf.enabled=true"]))
    assert config["sheaf"]["enabled"] is True


# --- overrides --------------------------------------------------------------

def test_set_reads_values_as_yaml():
    config = cli.composed(arguments(world="primary", set=["seed=7", "sheaf.enabled=false"]))
    assert config["seed"] == 7 and config["sheaf"]["enabled"] is False


def test_set_refuses_a_key_that_does_not_exist():
    """A run silently shaped by a typo is the failure this rules out."""
    with pytest.raises(AssertionError):
        cli.composed(arguments(world="primary", set=["sheaf.enable=false"]))


# --- views ------------------------------------------------------------------

def test_views_all_is_ground_truth_then_every_agent():
    config = cli.composed(arguments(world="primary", views="all"))
    assert beliefs_of(config)["views"] == ["ground_truth", 0, 1, 2, 3]


def test_views_truth_is_just_the_one():
    assert beliefs_of(cli.composed(arguments(world="primary", views="truth")))["views"] \
        == ["ground_truth"]


def test_views_takes_a_comma_list():
    config = cli.composed(arguments(world="scaled", views="truth,0,2"))
    assert beliefs_of(config)["views"] == ["ground_truth", 0, 2]


def test_views_all_on_a_generated_world():
    """
    Regression. `beliefs` is a WORLD_KEY, so a generated world carries no beliefs block from
    primary.yaml -- `follow_moving_agent` among the keys dropped with it. Emitting the override
    that switches it off then asserted in apply_overrides, which refuses a key the configuration
    does not hold: `--num-agents 8 --views all` failed on an override that would have done
    nothing, since the key defaults to False where it is read.
    """
    config = cli.composed(arguments(num_agents=8, views="all"))
    assert beliefs_of(config)["views"] == ["ground_truth"] + list(range(8))
    assert beliefs_of(config).get("follow_moving_agent") in (None, False)


def test_naming_fewer_views_than_agents_switches_following_off():
    """follow_moving_agent picks the view by the index of the agent that moved, so it needs one
    view per agent; naming the views is the more specific instruction and wins."""
    config = cli.composed(arguments(world="primary", views="truth"))
    assert beliefs_of(config)["follow_moving_agent"] is False
    assert len(beliefs_of(config)["views"]) == 1


def test_video_and_tour_set_their_keys():
    per_view = cli.composed(arguments(world="primary", video="per-view"))
    single = cli.composed(arguments(world="primary", video="single"))
    assert per_view["video"]["one_per_view"] is True
    assert single["video"]["one_per_view"] is False
    assert beliefs_of(cli.composed(arguments(world="primary", tour=5)))["tour_views_every"] == 5


@pytest.mark.parametrize("spec", ["sideways", "truth,sideways", "1.5", ","])
def test_unreadable_views_is_refused(spec):
    """A word where an index belongs used to come out as a raw ValueError from int()."""
    with pytest.raises(AssertionError, match="Failed to read"):
        cli.composed(arguments(world="primary", views=spec))


# --- layers -----------------------------------------------------------------

def test_a_layer_lands_over_the_world():
    config = cli.composed(arguments(world="primary", layer=[cli.ROBOTARIUM_CONFIG]))
    plain = cli.composed(arguments(world="primary"))
    assert config["rendering"] != plain["rendering"]


def test_an_arm_lands_over_a_layer():
    """
    The testbed layer is applied as a --layer so it lands BEFORE the arm. Applied after, its
    rendering block puts back the section lines --no-com had just turned off, and a bundle films
    a no-communication run as though its interfaces were agreeing.
    """
    config = cli.composed(arguments(world="primary", layer=[cli.ROBOTARIUM_CONFIG],
                                    no_communication=True))
    assert config["rendering"]["show_sections"] is False
    assert config["sheaf"]["enabled"] is False
