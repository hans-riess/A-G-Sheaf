"""
The `agsheaf` command line: one entry point over the runners in exp/.

    agsheaf sims exp/worlds/variation.yaml --set sheaf.enabled=false
    agsheaf sims --num-agents 8
    agsheaf preview --experiment exp/worlds/scaled.yaml --view ground_truth
    agsheaf preview --num-agents 8 --num-obstacles 2 --seed 3
    agsheaf preview --floor --no-com --out /tmp/frame.png
    agsheaf robots --config exp/robotarium/config.yaml exp/robotarium/no_communication.yaml
    agsheaf converge --studies topology
    agsheaf scale --agents 32

Every subcommand is a thin front for a script that already exists and can still be run directly;
this adds a way to reach them by name, and a way to vary a configuration without writing a file.
The layering it composes -- defaults, then the experiment, then `--layer`s, then `--set`s -- is
the one exp/robotarium/build_submission.py already applies to build a testbed configuration, so a
run launched from here and a bundle built for the testbed are shaped the same way.

On `--set`: it exists because exp/run.py takes exactly one configuration file, so the only way to
vary one arm of an experiment used to be to copy the whole world file and edit a line -- which is
how the no-communication controls were produced, and why exp/worlds/scaled.yaml carries a comment
telling the reader to do it by hand. `--set sheaf.enabled=false` replaces that copy.

This module is deliberately NOT in build_submission.py's AGSHEAF_MODULES, so nothing here is
uploaded to the Robotarium: the testbed runs exp/robotarium/experiment_template.py against a
configuration baked into a module, and never parses YAML or a command line. Keep it that way --
argparse is stdlib and would survive the trip, but the exp/ scripts this imports would not.

None of the above reaches `agsheaf --help`, which shows DESCRIPTION and EXAMPLES below instead.
This docstring is for whoever opens the file; a terminal wants the shortest thing that gets the
command right.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from .utils import _merge_settings, apply_overrides, compose_config, merge_layers
from .worldgen import CROWDED_OBSTACLES, generate_world

#: The repository, from this file: src/agsheaf/cli.py -> src/agsheaf -> src -> here. This holds
#: for the editable install the project is set up as (see bootstrap.py), which is how the exp/
#: scripts are on hand to be run at all.
#: What `agsheaf --help` shows. Deliberately short: the reasoning lives in the module docstring.
DESCRIPTION = "Run the agsheaf experiments. Each command fronts a script in exp/."
EXAMPLES = """\
examples:
  agsheaf sims exp/worlds/scaled.yaml --no-com     an arm of a measured world
  agsheaf sims --num-agents 8 --num-obstacles 2 a generated world
  agsheaf preview --num-agents 8                screen + floor frames
  agsheaf robots                                build the submission bundle

`agsheaf COMMAND --help` for a command's own options."""

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "exp"
DEFAULTS = EXPERIMENTS_DIR / "defaults.yaml"
ROBOTARIUM_CONFIG = EXPERIMENTS_DIR / "robotarium" / "config.yaml"

WORLDS_DIR = EXPERIMENTS_DIR / "worlds"
DEFAULT_EXPERIMENT = WORLDS_DIR / "primary.yaml"

#: The keys that DEFINE a world, as opposed to how a run is conducted. A generated world replaces
#: these outright and inherits everything else from exp/worlds/primary.yaml -- video, logging, rendering,
#: planning, scoring, contracts, goals -- because that is where those are kept and asking for a
#: different world is not asking to forget them. `video.one_per_view: true` set there went
#: unnoticed by every generated run until this existed.
#:
#: They have to be replaced rather than merged: the merge joins nested mappings, so a four-agent
#: `beliefs.agents` or `starts` left underneath a six-agent world's would survive it entry by
#: entry, and the run would be given a world that is neither.
WORLD_KEYS = ("starts", "assignments", "beliefs", "ground_truth", "regions", "communication")

#: The control arms, as the keys each one changes. These are the ablations of the report's
#: three-arm design (doc/ms4_report, Subtask 5.1): every arm runs the same world, the same starts,
#: assignments, beliefs, topology, planner and scoring, and differs in exactly one respect.
#:
#: They are written here rather than as layer files because the arm and the surface are different
#: things and exp/robotarium/no_communication.yaml is already the product of both -- it carries
#: `robotarium.hold_steps_per_sweep` and names itself `_robotarium`, neither of which belongs in a
#: run on a screen. Keeping the ablation itself in one place is what stops the two drifting; a
#: testbed bundle still gets it from the robotarium layer, which build_submission.py composes the
#: same way.
ARMS = {
    "no_communication": {
        "sheaf.enabled": False,          # no belief sheaf, so no sweeps and no fusing
        "rendering.show_sheaf_status": False,   # there is no iterate to name
        "rendering.show_sections": False,       # an edge has no state to be in
        "rendering.contested_marker": "",       # nothing empties a possibility set without a flow
    },
    "no_abstraction": {
        "regions": None,                 # blanks the partition: the legacy f = id belief sheaf,
    },                                   # tile-by-tile fusion over bijective restrictions
}


def available_worlds() -> list:
    """The worlds in exp/worlds/, by the name --world takes."""
    return sorted(path.stem for path in WORLDS_DIR.glob("*.yaml")) if WORLDS_DIR.is_dir() else []


def named_world(name: str) -> Path:
    """
    Resolves a --world name to its file in exp/worlds/.

    Args:
        name (str): The world's file name without the .yaml.
    """
    path = WORLDS_DIR / ("%s.yaml" % name)
    assert path.is_file(), \
        "No world called %r in %s. There is %s." % (
            name, WORLDS_DIR, ", ".join(available_worlds()) or "nothing there")
    return path


def base_config(arguments: argparse.Namespace) -> dict:
    """
    The world a run starts from, before layers, arms and overrides: either generated at the
    requested size, or read from a file.

    `--num-agents` generates. The prepared worlds stay reachable by name --
    `agsheaf sims exp/worlds/scaled.yaml` -- so that the measured instances in the report can still
    be run exactly, and the flag means one thing rather than two.

    Args:
        arguments (argparse.Namespace): The parsed command line.
    """
    named = [source for source in (arguments.experiment, arguments.world,
                                   arguments.num_agents) if source is not None]
    assert len(named) <= 1, \
        "Name the world once: --experiment takes a path, --world takes a name in exp/worlds/, " \
        "and --num-agents generates one. Giving two leaves it ambiguous which should be run."

    # Resolved into a local rather than back onto `arguments`: `preview` composes the SAME
    # namespace twice, once per surface, and writing the resolved path into .experiment while
    # .world stayed set made the second pass look like a world named twice -- the assertion above
    # firing on a command line that named it once.
    named_experiment = arguments.experiment
    if arguments.world is not None:
        named_experiment = named_world(arguments.world)

    if arguments.num_agents is not None:
        print("Generating a world: %d agents, %d obstacles, seed %d"
              % (arguments.num_agents, arguments.num_obstacles, arguments.seed))
        if arguments.num_obstacles > CROWDED_OBSTACLES:
            print("Note: more than %d obstacles has been measured to deadlock the arena -- the "
                  "detours around them funnel onto the same rows. If this run does not finish, "
                  "that is why; %d or fewer completes in about 80 s."
                  % (CROWDED_OBSTACLES, CROWDED_OBSTACLES))
        generated = generate_world(num_agents=arguments.num_agents,
                                   num_obstacles=arguments.num_obstacles,
                                   seed=arguments.seed)

        # The generated world replaces the world; how the run is conducted still comes from
        # exp/worlds/primary.yaml. See WORLD_KEYS.
        if DEFAULT_EXPERIMENT.is_file():
            policy = compose_config(DEFAULT_EXPERIMENT, [], DEFAULTS)
            print("  world generated; run settings from %s" % DEFAULT_EXPERIMENT.name)
        else:
            policy = yaml.safe_load(DEFAULTS.read_text()) or {}
            print("  world generated; run settings from %s" % DEFAULTS.name)
        policy = {key: value for key, value in policy.items() if key not in WORLD_KEYS}

        return _merge_settings(policy, generated)

    experiment = named_experiment or DEFAULT_EXPERIMENT
    assert Path(experiment).is_file(), "No experiment file at %s." % experiment
    return compose_config(experiment, [], DEFAULTS)


def _as_yaml(value) -> str:
    """One configuration value, written the way --set would have to write it on a command line."""
    return yaml.safe_dump(value, default_flow_style=True).strip().rstrip("...").strip()


def view_overrides(arguments: argparse.Namespace, number_of_robots: int) -> dict:
    """
    What --views, --video and --tour mean as configuration keys.

    They are sugar: every one of them sets a key that could be set with --set, and they exist
    because the two cases that actually come up -- one film per view on a screen, one film
    carrying every view on the floor -- were reachable only by remembering three key names.

    Args:
        arguments (argparse.Namespace): The parsed command line.
        number_of_robots (int): How many agents, which is what "all" means.
    """
    overrides = {}

    if arguments.views is not None:
        text = arguments.views.strip().lower()
        if text == "all":
            views = ["ground_truth"] + list(range(number_of_robots))
        elif text in ("truth", "ground_truth"):
            views = ["ground_truth"]
        else:
            # Parsed a side at a time rather than as a comprehension over int(), so that a word
            # where an index belongs is reported by the message below rather than as a raw
            # ValueError traceback out of int() -- `--views sideways` used to give one.
            views = []
            for side in (side.strip() for side in text.split(",")):
                if not side:
                    continue
                if side in ("truth", "ground_truth"):
                    views.append("ground_truth")
                else:
                    assert side.isdigit(), \
                        "Failed to read --views %r: %r is not an agent index. Give \"all\", " \
                        "\"truth\", or a comma list such as truth,1,2." % (arguments.views, side)
                    views.append(int(side))
            assert views, "Failed to read --views %r. Give \"all\", \"truth\", or a comma " \
                          "list such as truth,1,2." % arguments.views
        overrides["beliefs.views"] = views

        # follow_moving_agent picks the view by the index of the agent that moved, so it demands
        # exactly one view per agent. Naming the views is a more specific instruction than that,
        # and leaving it set would only assert; so it yields, loudly.
        if len(views) != number_of_robots:
            overrides["beliefs.follow_moving_agent"] = False

    if arguments.video is not None:
        overrides["video.one_per_view"] = arguments.video == "per-view"

    if arguments.tour is not None:
        overrides["beliefs.tour_views_every"] = arguments.tour

    return overrides


def selected_arms(arguments: argparse.Namespace) -> list:
    """
    The control arms asked for on the command line, in the order ARMS declares them.

    Args:
        arguments (argparse.Namespace): The parsed command line.
    """
    return [arm for arm in ARMS if getattr(arguments, arm)]


def composed(arguments: argparse.Namespace) -> dict:
    """
    The configuration one of the world-running subcommands was asked for: the defaults, then the
    world, then any `--layer`s, then the control arms, then any `--set`s.

    The arms and the view flags come before `--set` so that an explicit override still wins over
    either -- `--no-com --set sheaf.enabled=true` is contradictory, but it resolves the way the
    more specific instruction should rather than by which was typed first.

    Args:
        arguments (argparse.Namespace): The parsed command line.
    """
    config = merge_layers(base_config(arguments), arguments.layer, announce=True)

    arms = selected_arms(arguments)
    for arm in arms:
        print("Arm: %s" % arm)
        config = apply_overrides(config, ["%s=%s" % (key, _as_yaml(value))
                                          for key, value in ARMS[arm].items()])

    # An arm changes what ran, so it has to change what the run calls itself; without this both
    # arms log themselves under the world's name and are told apart only by the config block
    # their log records.
    if arms and "experiment_name" in config:
        config["experiment_name"] = "_".join([config["experiment_name"]] + arms)

    views = view_overrides(arguments, config["gridworld"]["num_agents"])
    if views:
        # Only worth switching off if it is actually on. It defaults to False in
        # gridworld.parse_display_views, and a GENERATED world carries no beliefs block from
        # primary.yaml or defaults.yaml at all -- beliefs is a WORLD_KEY, so the whole block is
        # replaced by worldgen's, which does not write this key. Emitting the override anyway
        # asserted in apply_overrides, which refuses a key the configuration does not already
        # hold: `--num-agents 8 --views all` failed on an override that would have done nothing.
        if views.get("beliefs.follow_moving_agent") is False:
            if (config.get("beliefs") or {}).get("follow_moving_agent"):
                print("Note: follow_moving_agent switched off -- it needs one view per agent, "
                      "and --views named %d for %d agents."
                      % (len(views["beliefs.views"]), config["gridworld"]["num_agents"]))
            else:
                del views["beliefs.follow_moving_agent"]

        config = apply_overrides(config, ["%s=%s" % (key, _as_yaml(value))
                                          for key, value in views.items()])

    return apply_overrides(config, arguments.set)


def add_world_arguments(parser: argparse.ArgumentParser, positional: bool = True) -> None:
    """
    The options shared by the subcommands that put a world together -- running one and drawing
    one are the same question of which world, and differ only in what is done with it.

    Args:
        parser (argparse.ArgumentParser): The subcommand's parser.
        positional (bool): Whether the world is named by a positional argument or by
            --experiment.
    """
    world = "a world file, by path (default exp/worlds/primary.yaml)"
    if positional:
        parser.add_argument("experiment", type=Path, nargs="?", default=None, help=world)
    else:
        # preview_frame.py already names its world with --experiment, and this subcommand is a
        # front for it, so it keeps that spelling rather than inventing a positional.
        parser.add_argument("--experiment", type=Path, default=None, help=world)
    parser.add_argument("--world", default=None, metavar="NAME",
                        help="a world in exp/worlds/ by name, without the .yaml: %s"
                             % ", ".join(available_worlds()))

    parser.add_argument("--num-agents", type=int, default=None, metavar="N",
                        help="generate a world for N agents instead; chain diameter N-1. Note "
                             "the KNOWN LIMITATION in agsheaf/worldgen.py")
    parser.add_argument("--num-obstacles", type=int, default=2, metavar="K",
                        help="wall segments in a generated world, 1-6 (default 2)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds a generated world and the run (default 0)")
    parser.add_argument("--layer", type=Path, action="append", default=[], metavar="FILE",
                        help="extra config layer over the world; repeatable, applied in order")
    parser.add_argument("--no-com", "--no-communication", dest="no_communication",
                        action="store_true", help="control arm: the flow switched off")
    parser.add_argument("--no-abstraction", dest="no_abstraction", action="store_true",
                        help="control arm: partition blanked, agreement measured tile by tile")
    parser.add_argument("--views", default=None, metavar="SPEC",
                        help="whose beliefs to draw: \"all\" (ground truth then every agent), "
                             "\"truth\", or a comma list of truth and/or agent indices numbered "
                             "FROM ZERO as the configuration numbers them (the figure labels the "
                             "same agents from one), e.g. truth,0,1. A pair of sides to compare "
                             "against each other still needs --set.")
    parser.add_argument("--video", choices=("per-view", "single"), default=None,
                        help="\"per-view\" films the one run once per view, which is what "
                             "comparing views wants; \"single\" writes one film, which is all a "
                             "physical run on the floor can give.")
    parser.add_argument("--tour", type=int, default=None, metavar="STEPS",
                        help="hold the run every STEPS steps and pass through every view, so one "
                             "film carries them all. For the floor, where --video per-view is "
                             "not available.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override one existing key, read as YAML; repeatable, beats an arm")


def run_simulation(arguments: argparse.Namespace) -> int:
    """
    Runs exp/run.py on the composed configuration.

    The composed configuration is written to a file and passed as the single path run.py already
    takes, and run.py is run as a subprocess rather than imported, because it is not an importable
    module: it is top-level script code that runs a whole experiment on import, held that way
    deliberately so that it, exp/preview_frame.py and the testbed's experiment_template.py stay
    three readable copies of one drive loop. Handing it a path leaves all three untouched.

    Merging the defaults under a configuration that already carries them is what run.py will do
    with this file, and that is idempotent, so the run is the one the file describes.

    Args:
        arguments (argparse.Namespace): The parsed command line.
    """
    config = composed(arguments)

    with tempfile.TemporaryDirectory() as scratch:
        # Named for the experiment so that a traceback, and the runner's own echo of its
        # argument, name something recognisable rather than a random temporary
        composed_path = Path(scratch) / ("%s.composed.yaml" % config.get("experiment_name", "run"))
        composed_path.write_text(yaml.safe_dump(config, sort_keys=False))

        completed = subprocess.run([sys.executable, str(EXPERIMENTS_DIR / "run.py"),
                                    str(composed_path)],
                                   cwd=REPO_ROOT)

    return completed.returncode


def run_preview(arguments: argparse.Namespace, forwarded: list) -> int:
    """
    Draws the composed world on both surfaces, forwarding everything about HOW to draw it.

    Both by default, because one configuration is drawn onto two surfaces -- a monitor and the
    white foam under the arena's projector -- and the whole reason to look at a frame before
    running it is to see that what the testbed will be sent still matches what the simulator
    shows. Drawing one of them silently leaves the other stale on disk, which reads as the two
    having diverged when nothing has changed at all. --floor or --screen asks for just that one.

    preview_frame.py names its world with --experiment and takes its own layers with --config, so
    the composed configuration goes in as a file the same way the runner takes one. Composing it
    here rather than letting preview_frame.py read a world file directly is what lets a GENERATED
    world be looked at before it is run.

    Args:
        arguments (argparse.Namespace): The parsed command line.
        forwarded (list): The arguments preview_frame.py should receive untouched.
    """
    assert not any(argument.startswith("--experiment") for argument in forwarded), \
        "Name the world once: `agsheaf preview --experiment ...` is this command's own option, " \
        "and it is what gets handed to preview_frame.py."

    asked = [surface for surface, wanted in (("screen", arguments.screen),
                                             ("floor", arguments.floor)) if wanted]
    surfaces = asked or ["screen", "floor"]

    given_out = any(argument.startswith("--out") for argument in forwarded)
    assert not (given_out and len(surfaces) > 1), \
        "--out names one file, but two frames are being drawn. Add --screen or --floor to say " \
        "which one it is for, or drop --out and take exp/previews/screen.png and floor.png."

    base_layers = list(arguments.layer)

    for surface in surfaces:
        # The testbed's rendering layer is applied HERE rather than forwarded as preview_frame's
        # own --floor, so that it lands BEFORE the control arms and --set instead of after them.
        # Forwarded it lands last and quietly undoes them: a --floor --no-com frame came out
        # drawing "this interface agrees" lines for a run with no sheaf at all, which is exactly
        # what exp/robotarium/no_communication.yaml turns off for the testbed.
        arguments.layer = ([ROBOTARIUM_CONFIG] if surface == "floor" else []) + base_layers

        print("Drawing the %s frame" % surface)
        config = composed(arguments)

        # preview_frame.py picks its own output name off the --floor it is no longer being told
        # about, so the same two names are chosen here.
        out = [] if given_out else ["--out", str(EXPERIMENTS_DIR / "previews" /
                                                 ("%s.png" % surface))]

        with tempfile.TemporaryDirectory() as scratch:
            composed_path = Path(scratch) / ("%s.%s.composed.yaml"
                                             % (config.get("experiment_name", "preview"), surface))
            composed_path.write_text(yaml.safe_dump(config, sort_keys=False))

            failed = run_script(EXPERIMENTS_DIR / "preview_frame.py",
                                ["--experiment", str(composed_path)] + out + forwarded)
        if failed:
            return failed

    return 0


def run_build(arguments: argparse.Namespace, forwarded: list) -> int:
    """
    Builds the Robotarium bundle from the composed world.

    The world goes in as build_submission.py's --experiment rather than as another --config
    layer, because the layers are merged as nested mappings: a world passed as a layer would have
    exp/worlds/primary.yaml's starts, assignments and per-agent beliefs merged underneath its own, and
    the bundle would ship a world that is neither of them.

    Args:
        arguments (argparse.Namespace): The parsed command line.
        forwarded (list): The arguments build_submission.py should receive untouched.
    """
    assert not any(argument.startswith("--experiment") for argument in forwarded), \
        "Name the world once: `agsheaf robots --experiment ...` is this command's own option, " \
        "and it is what gets handed to build_submission.py."

    if arguments.num_agents is not None:
        print("Note: a generated world does not carry the hazard collision (see the KNOWN "
              "LIMITATION in agsheaf/worldgen.py). It will exercise the flow on the testbed, but "
              "the demonstration itself wants exp/worlds/variation.yaml or exp/worlds/scaled.yaml.")

    # The testbed layers are applied here rather than left to build_submission.py's own --config,
    # so that they land BEFORE the control arms instead of after them. Applied after, the
    # testbed's rendering block puts back the section lines and the contested marker that
    # --no-com had just turned off, and the bundle films a no-communication run as though its
    # interfaces were agreeing. build_submission.py is then told --config with no paths, having
    # already been handed the merged result as its world.
    testbed = [ROBOTARIUM_CONFIG] if arguments.config is None else list(arguments.config)
    arguments.layer = testbed + list(arguments.layer)
    forwarded = forwarded + ["--config"]

    config = composed(arguments)

    with tempfile.TemporaryDirectory() as scratch:
        composed_path = Path(scratch) / ("%s.composed.yaml"
                                         % config.get("experiment_name", "submission"))
        composed_path.write_text(yaml.safe_dump(config, sort_keys=False))

        return run_script(EXPERIMENTS_DIR / "robotarium" / "build_submission.py",
                          ["--experiment", str(composed_path)] + forwarded)


def run_script(path: Path, argv: list) -> int:
    """
    Runs one of the exp/ scripts exactly as `python exp/<script>.py ...` would.

    A subprocess rather than an in-process import, for the same reason `sims` uses one: these are
    scripts, not libraries. Running one in this process would have to reproduce by hand what the
    interpreter does for a script -- putting its own directory first on sys.path, which is how
    exp/preview_frame.py finds robot_icon -- and would then leave that script's import-time
    effects, matplotlib's backend among them, set for whatever ran next. Handing the whole thing
    to a fresh interpreter keeps `agsheaf preview` and `python exp/preview_frame.py` the same run.

    Args:
        path (Path): The script to run.
        argv (list): The arguments to pass it, without the program name.
    """
    completed = subprocess.run([sys.executable, str(path)] + [str(argument) for argument in argv],
                               cwd=REPO_ROOT)
    return completed.returncode


def main() -> None:
    """The `agsheaf` entry point, as declared in pyproject.toml."""
    parser = argparse.ArgumentParser(
        prog="agsheaf", usage="agsheaf COMMAND [options]",
        description=DESCRIPTION, epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # An empty metavar drops the {sims,preview,...} line, which otherwise repeats the whole list
    # directly above the one that says what each command does. Spelling the usage out by hand is
    # what lets it go: argparse would otherwise render the metavar there too.
    # prog is spelled out because a hand-written usage above would otherwise be inherited as the
    # prefix of every subcommand's, giving "agsheaf COMMAND [options] sims [-h] ...".
    subcommands = parser.add_subparsers(dest="subcommand", required=True, prog="agsheaf",
                                        title="commands", metavar="")

    simulate = subcommands.add_parser("sims", help="run an experiment (exp/run.py)")
    add_world_arguments(simulate)

    # These four already parse their own command lines, and those are the documented interfaces
    # for them, so nothing is declared here and whatever follows is collected by
    # parse_known_args below and handed over untouched. That keeps one set of flags rather than
    # a second set here that could fall behind, and it is why they are built with add_help=False:
    # `agsheaf preview --help` should reach preview_frame.py and show ITS options rather than an
    # empty wrapper's. (argparse.REMAINDER looks like the tool for this and is not -- it only
    # starts collecting at the first argument that does not look like an option, so a subcommand
    # whose arguments begin with --experiment loses them.)
    # preview composes a world the same way sims does -- looking at a generated world before
    # running it is most of what a frame preview is for -- and forwards everything about how to
    # draw it to preview_frame.py untouched.
    preview = subcommands.add_parser(
        "preview",
        help="draw one frame, no run behind it (exp/preview_frame.py)",
        epilog="Options describing HOW to draw the frame are forwarded to exp/preview_frame.py: "
               "--view, --sweeps, --out, --dpi, --config. Run "
               "`python exp/preview_frame.py --help` for those.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_world_arguments(preview, positional=False)
    preview.add_argument("--floor", action="store_true",
                         help="draw only the projected frame, over exp/robotarium/config.yaml")
    preview.add_argument("--screen", action="store_true",
                         help="draw only the screen frame, as exp/run.py draws it")

    robots = subcommands.add_parser(
        "robots",
        help="build the submission bundle (exp/robotarium/build_submission.py)",
        epilog="Options describing HOW to build are forwarded to build_submission.py: "
               "--dry-runs, --history. Run "
               "`python exp/robotarium/build_submission.py --help` for those.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_world_arguments(robots, positional=False)
    robots.add_argument("--config", type=Path, nargs="*", default=None, metavar="FILE",
                        help="testbed layers over the world, applied BEFORE any control arm. "
                             "Defaults to exp/robotarium/config.yaml; pass with no paths to "
                             "build without one.")
    subcommands.add_parser(
        "converge", add_help=False,
        help="the convergence campaign (exp/convergence.py)")
    scripts = {
        "preview": EXPERIMENTS_DIR / "preview_frame.py",
        "robots": EXPERIMENTS_DIR / "robotarium" / "build_submission.py",
        "converge": EXPERIMENTS_DIR / "convergence.py",
        "scale": EXPERIMENTS_DIR / "scale_example.py",
    }
    subcommands.add_parser(
        "scale", add_help=False,
        help="the demonstration sheaf at network scale (exp/scale_example.py)")

    arguments, passed_through = parser.parse_known_args()

    if arguments.subcommand == "sims":
        # `sims` owns its whole command line, so an argument it does not recognise is a mistake
        # rather than something meant for someone else -- unlike the passthrough subcommands,
        # where leftovers are the entire point.
        if passed_through:
            parser.error("unrecognized arguments for sims: %s" % " ".join(passed_through))
        raise SystemExit(run_simulation(arguments))

    if arguments.subcommand == "preview":
        raise SystemExit(run_preview(arguments, passed_through))

    if arguments.subcommand == "robots":
        raise SystemExit(run_build(arguments, passed_through))

    raise SystemExit(run_script(scripts[arguments.subcommand], passed_through))


if __name__ == "__main__":
    main()
