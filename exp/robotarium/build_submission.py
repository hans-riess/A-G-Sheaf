"""
Builds the flat file bundle uploaded to the Robotarium.

The runners carry z3 themselves -- it was installed on the executors in response to
robotarium/robotarium_python_simulator#43 -- so a submission imports it like any other library
and ships only this project's own code. (Should it ever go missing, the hmr/flatten_z3 branch
carries a version of this pipeline that brings its own solver: the bindings flattened out of
their package and libz3.so renamed to an extension the upload form permits.)

What remains is shaped by what the submission form accepts:

    Allowed extensions: .m .jpg .jpeg .png .gif .tiff .bmp .py .mat .npy .mex .mexw64 .mexa64
                        .mexmaci64 .jl

    Directory upload is refused outright, and one uploaded file is marked as the main one.

Two consequences, which are most of what this script does:

    No subdirectories.  Every module is written flat, and every import that named a package is
                        rewritten to name a sibling module instead. The agsheaf modules are
                        prefixed rather than kept under their own names, so that nothing on the
                        runner's path is shadowed by a file called `log.py` or `utils.py`.

    No .yaml.           The configuration cannot be uploaded, so it is baked into a module as a
                        dict literal, merged exactly as an ordinary run would merge it.

Run it from anywhere:

    python exp/robotarium/build_submission.py

Builds land under exp/robotarium/builds/, labeled by which arm of the experiment they are:

    builds/sheaf/                    what to upload -- one directory per arm, so building the
    builds/no_comm/                  no-comm baseline no longer destroys the sheaf bundle the
    builds/no_abstraction/           way a single `submission/` did

    builds/sheaf-20260814T1018.zip   a timestamped snapshot of each, kept forever, so a result
    builds/no_comm-20260814T1019.zip that comes back from the testbed days later still has the
                                     files that produced it

Upload everything in the arm's directory, marking experiment.py as the main file. (The zips
are for keeping and for sending on; the form takes neither a directory nor an archive.)

To build the same world with the belief flow switched off, for the other half of a side-by-side:

    python exp/robotarium/build_submission.py \
        --config exp/robotarium/config.yaml exp/robotarium/no_communication.yaml
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pprint
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).resolve().parent / "experiment_template.py"

# Builds live in here, two ways at once:
#
#   builds/sheaf/                    the current bundle of that arm, which is what gets
#                                    uploaded -- one directory per arm, so rebuilding one arm
#                                    no longer destroys the other's files the way a single
#                                    `submission/` did
#   builds/sheaf-20260814T1018.zip   a snapshot of it, kept forever
#
# Results come back from the testbed hours or days after the upload, by which time the arm's
# directory has moved on; the timestamped zips are what lets a returned video still be opened
# against the exact files that produced it. builds.jsonl records the parameters alongside.
BUILDS_DIR = Path(__file__).resolve().parent / "builds"

# Where a build is assembled before its arm is known: the arm is read out of the merged
# configuration, and the configuration is not merged until it has been written into the bundle.
STAGING_DIR = BUILDS_DIR / "_staging"

# Set by main() once staging begins, and read by the writers and the dry runs. A module global
# for the same reason the fixed bundle directory was one: every writer puts its file in the
# same place, and threading a directory through all of them buys nothing.
BUNDLE_DIR: Path = STAGING_DIR

# What a submission runs, layered: the general defaults, then the experiment, then whatever the
# testbed wants differently from a run on a screen. Keeping the third layer separate is what
# lets the local demo be tuned for debugging without changing what gets shipped.
ROBOTARIUM_CONFIG = Path(__file__).resolve().parent / "config.yaml"

# One line per build, appended forever: the parameters each build was made with, which is what
# --history reads to show what changed between them. The files themselves live beside it in
# BUILDS_DIR, under the same label.
BUILD_LOG = Path(__file__).resolve().parent / "builds.jsonl"

# What --history shows for the first build, which has nothing to be compared against: the
# parameters a run is usually tuned by. Every later build is shown as what changed instead, so
# this list only has to be a reasonable opening summary, not exhaustive.
HEADLINE_PARAMETERS = (
    "seed",
    "gridworld.num_agents", "gridworld.grid_width", "gridworld.grid_height",
    "goals.max_steps",
    "sheaf.enabled", "sheaf.sweeps_per_step", "sheaf.legs",
    "contracts.reliance", "contracts.exclusive_claims", "contracts.region_penalty",
    "beliefs.default", "beliefs.view", "beliefs.observe_on_arrival",
    "beliefs.tour_views_every",
    "robotarium.hold_steps_per_sweep", "robotarium.hold_steps_per_view",
)

#: What sheaf.sweeps_per_step says instead of a count to run the flow to its fixed point.
#: Restates gridsheaf.CONVERGE rather than importing it: this script assembles the bundle and
#: does not otherwise need z3 or networkx on the path to do it.
CONVERGE = "converge"

sys.path.insert(0, str(REPO_ROOT / "src"))

# The agsheaf modules the experiment reaches, flattened under a prefix. utils.py is absent on
# purpose: it exists for VideoSaver and for loading YAML, and the bundle needs neither, so
# leaving it out also drops the cv2 dependency. Order is irrelevant -- imports are rewritten.
# regions, product and regionsheaf are the abstraction-level sheaf (exp/run.py's default arm);
# a bundle built from a configuration with no `regions:` block still ships them, unused --
# three small pure-python files against the risk of a build that varies by configuration.
AGSHEAF_MODULES = ("gridworld", "contracts", "sheaf", "product", "regions", "regionsheaf",
                   "gridsheaf", "log", "mission")
AGSHEAF_PREFIX = "agsheaf_"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", type=Path, default=None,
                        help="the world the testbed layers apply over. Defaults to exp/worlds/primary.yaml. "
                             "This is how a different world is filmed -- exp/worlds/scaled.yaml, say, or "
                             "a composed one written out by `agsheaf robots`; a world passed to "
                             "--config instead would be merged UNDERNEATH exp/worlds/primary.yaml's own "
                             "starts, assignments and beliefs rather than replacing them.")
    parser.add_argument("--config", type=Path, nargs="*", default=[ROBOTARIUM_CONFIG],
                        help="the testbed layers, applied in order over exp/worlds/primary.yaml. Defaults to "
                             "exp/robotarium/config.yaml. Add another to build a variant of the same "
                             "world -- e.g. no_communication.yaml, which films it with the flow "
                             "switched off. Pass --config with no paths to ship the experiment "
                             "unmodified.")
    parser.add_argument("--dry-runs", type=int, default=0, metavar="N",
                        help="run the finished bundle N times to measure how long it takes, which is "
                             "what the portal's duration field is filled in from, and to check that it "
                             "runs at all. Defaults to 0, which skips the measurement and falls back to "
                             "declaring the 600 s maximum; pass 1 or more before a submission you mean "
                             "to upload, so the form gets a measured duration and the bundle is known "
                             "to start.")
    parser.add_argument("--history", action="store_true",
                        help="print what every past build was made with, each shown as the parameters "
                             "that changed since the one before it, and exit without building.")
    args = parser.parse_args()

    if args.history:
        show_history()
        return

    global BUNDLE_DIR

    # To milliseconds, because it is what distinguishes one entry in the log from the next, and
    # two builds a few seconds apart is the normal way of working here
    built_at = datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")

    BUNDLE_DIR = STAGING_DIR
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True)

    written = []
    written += write_agsheaf_modules()
    config_path, config = write_config(args.config, args.experiment)
    written.append(config_path)
    written.append(write_entry_point())

    written = adopt_arm_directory(config, written)

    assert_runs_on_an_older_numpy(written)

    if args.dry_runs:
        print("Timing the bundle, to answer the form's duration with a measurement:")
    fields = portal_fields(config, measure_duration(args.dry_runs), built_at)

    archive = archive_bundle(config, built_at)
    record_build(config, fields, built_at)
    report(written, built_at, fields, archive)


def arm_label(config: dict) -> str:
    """
    Which arm of the experiment this configuration is: the same three-way split the portal
    title is written from, named for the shelf rather than for the form.

    Args:
        config (dict): The merged configuration the bundle was built from.
    """
    communicating = bool((config.get("sheaf") or {}).get("enabled", True))
    if not communicating:
        return "no_comm"
    if not (config.get("regions") or {}):
        return "no_abstraction"
    return "sheaf"


def _stamp(built_at: str) -> str:
    """
    The timestamp part of an archive's name, to the minute: enough to tell two builds of one
    arm apart, short enough to read.

    Args:
        built_at (str): The build's ISO timestamp, as the build log records it.
    """
    return re.sub(r"[^0-9T]", "", built_at.split("+")[0].split(".")[0])[:13]


def adopt_arm_directory(config: dict, written: list) -> list:
    """
    Moves the staged bundle into the directory of its arm -- builds/sheaf, builds/no_comm --
    replacing whatever that arm held before.

    One directory per arm rather than one per build: this is the copy that gets uploaded, so
    it wants a stable path, and rebuilding one arm should leave the others alone. The history
    is kept as archives instead, see archive_bundle.

    Returns the written paths rebased onto that directory. They are only used for their names
    and sizes afterwards, but a stale path is a trap left for whatever is added next.

    Args:
        config (dict): The merged configuration the bundle was built from.
        written (list): The files written into the bundle, under the staging directory.
    """
    global BUNDLE_DIR

    arm = BUILDS_DIR / arm_label(config)
    if arm.exists():
        shutil.rmtree(arm)
    BUNDLE_DIR.rename(arm)
    BUNDLE_DIR = arm

    return [BUNDLE_DIR / path.name for path in written]


def archive_bundle(config: dict, built_at: str) -> Path:
    """
    Zips the finished bundle as a timestamped snapshot of its arm, and returns where it landed.

    The arm's directory is overwritten by its next build; these are not, so a result that comes
    back from the testbed days later still has the files that produced it. The upload form
    takes neither a directory nor a .zip, which is why the loose files stay too.

    Args:
        config (dict): The merged configuration the bundle was built from.
        built_at (str): The build's ISO timestamp.
    """
    label = "%s-%s" % (arm_label(config), _stamp(built_at))
    archive = BUILDS_DIR / ("%s.zip" % label)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(BUNDLE_DIR.iterdir()):
            bundle.write(path, arcname="%s/%s" % (label, path.name))
    return archive


## what the executors carry

# Spellings that work on the machine a bundle is built on and fail on the one that runs it.
#
# The executors' numpy is older than any recent developer install, and the dry runs below cannot
# tell: they exercise the bundle against the builder's own numpy, where all of this is fine. So
# a submission carrying one of these looks healthy right up until it is a traceback at the far
# end of a queue, which is what happened to the first one -- np.reshape's `shape=` keyword,
# added in numpy 2.1 as the replacement for `newshape`, absent on the testbed.
#
# This list is what has actually cost a round-trip plus its immediate neighbours, not a model of
# numpy's release history; there is no being exhaustive about an API surface without knowing
# which version the executors hold. The preflight in experiment_template.py now prints that
# version, so a returned log.txt says what the real floor is, and this list can shrink to a
# check against it once one comes back.
NEWER_THAN_THE_EXECUTORS = (
    (r"np\.reshape\([^\n]*\bshape\s*=",
     "np.reshape(a, shape=...) needs numpy 2.1. Pass the shape positionally: np.reshape(a, (2, 1))."),
    (r"np\.reshape\([^\n]*\bnewshape\s*=",
     "np.reshape(a, newshape=...) was removed in numpy 2.3. Pass the shape positionally instead."),
    (r"\bnp\.trapezoid\b",
     "np.trapezoid needs numpy 2.0. np.trapz is the spelling both versions accept."),
    (r"\bnp\.astype\(",
     "np.astype as a free function needs numpy 2.0. The .astype method is on every version."),
    (r"\bnp\.(unique_all|unique_counts|unique_inverse|unique_values)\b",
     "the np.unique_* family needs numpy 2.0. np.unique carries the same results on both."),
)


def assert_runs_on_an_older_numpy(written: list[Path]) -> None:
    """
    Reads the finished bundle for anything the executors' numpy is too old to have.

    This is a check the dry runs cannot make, and the only one here that looks at the source
    rather than at what running it does. Everything else a build verifies, it verifies by
    running; this one thing is invisible to running, because the version that would object to it
    is not the version present.

    Args:
        written (list[Path]): Every file written into the bundle.
    """
    complaints = []
    for path in written:
        lines = path.read_text().splitlines()
        for pattern, explanation in NEWER_THAN_THE_EXECUTORS:
            for number, line in enumerate(lines, start=1):
                if re.search(pattern, line):
                    complaints.append("  %s:%d  %s\n      %s" % (path.name, number, line.strip(), explanation))

    assert not complaints, (
        "Failed to build the submission. The bundle uses numpy the executors are too old for, and "
        "a dry run here would not notice, running as it does against this machine's numpy:\n\n%s\n\n"
        "Fix these in src/agsheaf/ rather than in submission/, which building overwrites."
        % "\n".join(complaints))


## the submission form


# The portal's own limits, which a build should not let a submission exceed
MAX_DURATION_SECONDS = 600
MAX_ROBOTS = 18


def measure_duration(runs: int) -> list:
    """
    Runs the built bundle a few times and reports how long each took in testbed seconds.

    The portal asks for an estimated duration, and the honest way to answer is to run the thing.
    The tile-level experiment is deterministic, but its length is not: the testbed places the
    robots at arbitrary poses and drives them to the initial conditions before the run proper,
    and that phase counts, which is why several samples are taken rather than one.

    Doing this also checks the bundle: a submission that cannot start is worth discovering here
    rather than at the far end of a queue.

    Args:
        runs (int): How many times to run it. Zero skips measuring.
    """
    samples = []
    for run_index in range(runs):
        with tempfile.TemporaryDirectory() as sandbox:
            # Run it as the runner would: from inside a flat directory of its own, so nothing is
            # resolved out of this repository, and headless, there being no projector here
            for path in BUNDLE_DIR.iterdir():
                shutil.copy2(path, Path(sandbox) / path.name)
            completed = subprocess.run([sys.executable, "experiment.py"], cwd=sandbox,
                                       capture_output=True, text=True,
                                       env={**os.environ, "MPLBACKEND": "Agg"})

        if completed.returncode != 0:
            print("\nA dry run of the bundle failed, so it would fail on the testbed too:\n")
            print(completed.stdout[-2000:])
            print(completed.stderr[-2000:])
            raise SystemExit("Failed to build the submission. The bundle does not run.")

        match = re.search(r"^DURATION_SECONDS ([\d.]+)$", completed.stdout, flags=re.MULTILINE)
        assert match, \
            "Failed to build the submission. A dry run finished without reporting its duration, so " \
            "experiment.py and this builder disagree about how that is printed."
        samples.append(float(match.group(1)))
        print("  dry run %d of %d: %.1f s" % (run_index + 1, runs, samples[-1]))

    return samples


def estimated_duration(samples: list) -> int:
    """
    Turns measured run lengths into the number to put on the form.

    Twice the longest sample, rounded up to the minute. The margin is not timidity: the samples
    only cover the initialization variance seen on this machine, the hardware has its own, and
    the cost of overestimating is nothing while the cost of underestimating is a run cut short.

    Args:
        samples (list): Measured durations in seconds, or empty if none were taken.
    """
    if not samples:
        return MAX_DURATION_SECONDS

    estimate = int(math.ceil(max(samples) * 2 / 60.0) * 60)
    return min(estimate, MAX_DURATION_SECONDS)


def portal_fields(config: dict, samples: list, built_at: str) -> dict:
    """
    Fills in what the submission form asks for, from what the bundle was built with.

    The form wants a title, a duration, a description and a robot count, and every one of them
    is already settled by the configuration -- so they are derived here rather than retyped, an
    experiment whose description does not match what it runs being worse than none.

    The title is written to be found again: submissions accumulate in a list, and the same world
    gets built repeatedly, so it carries the variant, the shape of the run and when it was built
    rather than describing the work. The description is terse on purpose -- notes, not prose --
    since it is read to tell one submission from another.

    Args:
        config (dict): The fully resolved configuration the bundle was built with.
        samples (list): Measured run lengths in seconds, or empty if none were taken.
        built_at (str): When this build was made, which stamps the title.
    """
    gridworld = config["gridworld"]
    robots = gridworld["num_agents"]
    width, height = gridworld["grid_width"], gridworld["grid_height"]
    communicating = bool((config.get("sheaf") or {}).get("enabled", True))
    edges = len(((config.get("communication") or {}).get("edges")) or [])
    sweeps = (config.get("sheaf") or {}).get("sweeps_per_step", 1)
    max_steps = (config.get("goals") or {}).get("max_steps", 100)
    regions = config.get("regions") or {}
    abstraction = communicating and bool(regions)

    assert 1 <= robots <= MAX_ROBOTS, \
        "Failed to build the submission. The Robotarium takes 1 to %d robots, and this is configured " \
        "for %d." % (MAX_ROBOTS, robots)

    # Scannable in a list: which variant, how big, and which build. The stamp is what tells two
    # submissions of the same world apart when the parameters between them have moved.
    stamp = built_at[5:16].replace("T", " ")
    shape = "%d robots, %dx%d" % (robots, width, height)
    if abstraction:
        title = "Contract sheaf - %s, %d regions - %s" % (shape, len(regions), stamp)
    elif communicating:
        title = "Sheaf fusion - %s, %d-edge comm - %s" % (shape, edges, stamp)
    else:
        title = "No-comm baseline - %s - %s" % (shape, stamp)

    if communicating:
        # sweeps_per_step is a count or the word "converge", so it is described rather than
        # formatted as a number
        if sweeps == CONVERGE:
            how_far = "the Tarski Laplacian run to its fixed point before each grid step"
        elif sweeps == 1:
            how_far = "one sweep of the Tarski Laplacian per grid step"
        else:
            how_far = "%d sweeps of the Tarski Laplacian per grid step" % sweeps
        if abstraction:
            method = (
                "Agents start out disagreeing about which tiles are unsafe and where the targets "
                "are, and hold assume-guarantee mission contracts over a %d-region partition of the "
                "grid. Knowledge, hazard summaries and route claims fused by %s over a %d-edge "
                "topology; agreement is measured in the region-level interface vocabulary, so the "
                "sections close while tile-level disagreement remains. A violated assumption -- a "
                "hazard flagged on a corridor, a neighbour claiming a region -- triggers replanning "
                "and recommitment. Projected: per-agent tile beliefs, region boundaries, comm links "
                "coloured by whether that interface agrees, and the two-level caption (abstract "
                "section / concrete disagreement)."
                % (len(regions), how_far, edges))
        else:
            method = (
                "Agents start out disagreeing about which tiles are unsafe and where the targets are. "
                "Beliefs fused by %s over a %d-edge topology, on a "
                "sheaf of assume-guarantee contracts; agents plan on the fused beliefs. "
                "Projected: per-agent tile beliefs, comm links coloured by whether that interface agrees, "
                "contested tiles, sweep caption. Robots hold still ~%.0fs per sweep so the flow is visible."
                % (how_far, edges,
                   (config.get("robotarium") or {}).get("hold_steps_per_sweep", 0) * 0.033))
    else:
        method = (
            "Control run for the contract-sheaf experiment: same world, no communication. Agents keep "
            "the beliefs they start with, so their disagreements about safety and targets are never "
            "resolved. Projected: per-agent tile beliefs, and the comm topology carrying nothing.")

    description = (
        "%d robots on a %dx%d tile grid, driving to assigned target squares by what they believe about "
        "each tile (safe / unsafe / target). %s Potential-field control with collision avoidance; one "
        "tile per step, max %d steps."
        % (robots, width, height, method, max_steps))

    return {
        "title": title,
        "duration_seconds": estimated_duration(samples),
        "number_of_robots": robots,
        "description": description,
        "measured_seconds": samples,
    }


## the build log


def record_build(config: dict, fields: dict, built_at: str) -> None:
    """
    Appends the parameters this build shipped to the build log.

    The configuration is recorded resolved rather than as the names of the files it came from.
    Naming the files would say nothing about what they contained at the time -- the layers get
    edited between builds, which is the whole reason for keeping a history -- so the log holds
    the parameters the submission actually ran with, every default included.

    The portal fields go in beside them, so the log says what was typed into the form as well as
    what the bundle was built with -- given a submission sitting in the portal, its title and
    duration are what identify the build it came from.

    Args:
        config (dict): The fully resolved configuration the bundle was built with.
        fields (dict): What to put on the submission form. See portal_fields.
        built_at (str): When this build was made, the same stamp its title carries.
    """
    entry = {
        "built_at": built_at,
        "experiment_name": config.get("experiment_name"),
        "portal": fields,
        "config": config,
    }

    with open(BUILD_LOG, "a") as build_log:
        build_log.write(json.dumps(entry, sort_keys=True) + "\n")


def show_history() -> None:
    """
    Prints the build log as a history of what changed, which is the question it exists to answer.

    Every build is listed with the parameters that differ from the build before it, since a run
    of the same world with one number moved is the common case and a full configuration printed
    each time would bury the one line that matters. The first build has nothing to differ from,
    so it is summarised by the parameters most likely to be tuned.
    """
    if not BUILD_LOG.exists():
        print("No builds recorded yet: %s does not exist." % BUILD_LOG)
        return

    entries = [json.loads(line) for line in BUILD_LOG.read_text().splitlines() if line.strip()]
    print("%d build(s) in %s\n" % (len(entries), BUILD_LOG))

    previous = None
    for entry in entries:
        config = entry.get("config") or {}
        print("%s  %s" % (entry.get("built_at", "?"), entry.get("experiment_name") or "unnamed"))

        current = _flatten(config)
        if previous is None:
            for key in HEADLINE_PARAMETERS:
                if key in current:
                    print("    %s = %r" % (key, current[key]))
        else:
            changed = [key for key in sorted(set(previous) | set(current))
                       if previous.get(key) != current.get(key)]
            if not changed:
                print("    (identical parameters to the build above)")
            for key in changed:
                print("    %s: %r -> %r" % (key, previous.get(key), current.get(key)))
        previous = current
        print()


def _flatten(config: dict, prefix: str = "") -> dict:
    """
    Flattens a nested configuration into dotted keys, so two builds can be compared entry by
    entry rather than section by section.

    Args:
        config (dict): The configuration to flatten.
        prefix (str): The dotted path accumulated so far.
    """
    flat = {}
    for key, value in config.items():
        path = "%s%s" % (prefix, key)
        if isinstance(value, dict):
            flat.update(_flatten(value, path + "."))
        else:
            flat[path] = value
    return flat


## agsheaf


def write_agsheaf_modules() -> list[Path]:
    """
    Copies the agsheaf modules out flat under a prefix, rewriting the imports between them.
    """
    written = []

    for module in AGSHEAF_MODULES:
        source = (REPO_ROOT / "src" / "agsheaf" / ("%s.py" % module)).read_text()
        destination = BUNDLE_DIR / ("%s%s.py" % (AGSHEAF_PREFIX, module))
        destination.write_text(rewrite_agsheaf_imports(source))
        written.append(destination)

    return written


def rewrite_agsheaf_imports(source: str) -> str:
    """
    Points every import of one agsheaf module at its flattened sibling.

    Four forms occur. Most are `from agsheaf.gridworld import ...`, which becomes
    `from agsheaf_gridworld import ...`. Within the package a sibling is reached relatively,
    as sheaf.py's `from .contracts import ...`, which becomes the same thing. gridsheaf.py's
    lazy dispatches use `from . import regionsheaf`, which becomes an aliased import of the
    flattened sibling. The last is gridsheaf.py's `from agsheaf import Contract,...`, which
    reads the package's own re-exports; there is no package here, so it is resolved against
    src/agsheaf/__init__.py into imports of the modules that actually define those names.

    Args:
        source (str): The module source to rewrite.
    """
    for module in AGSHEAF_MODULES:
        source = re.sub(r"^(\s*)from agsheaf\.%s import" % module,
                        r"\1from %s%s import" % (AGSHEAF_PREFIX, module), source, flags=re.MULTILINE)
        source = re.sub(r"^(\s*)from \.%s import" % module,
                        r"\1from %s%s import" % (AGSHEAF_PREFIX, module), source, flags=re.MULTILINE)
        source = re.sub(r"^(\s*)from \. import %s$" % module,
                        r"\1import %s%s as %s" % (AGSHEAF_PREFIX, module, module),
                        source, flags=re.MULTILINE)

    def resolve_package_import(match: re.Match) -> str:
        names = [name.strip() for name in match.group(2).split(",")]
        by_module: dict[str, list[str]] = {}
        for name in names:
            by_module.setdefault(defining_module(name), []).append(name)
        return "\n".join(
            "%sfrom %s%s import %s" % (match.group(1), AGSHEAF_PREFIX, module, ", ".join(sorted(imported)))
            for module, imported in sorted(by_module.items()))

    source = re.sub(r"^(\s*)from agsheaf import ([\w, ]+)$", resolve_package_import, source, flags=re.MULTILINE)

    # Anchored on a boundary, since every rewritten import now begins "from agsheaf_..."
    assert not re.search(r"^\s*(from|import) agsheaf(\.|\s|$)", source, flags=re.MULTILINE), \
        "Failed to build the submission. An import of the agsheaf package survived the rewrite, and the " \
        "bundle has no package for it to resolve against."
    assert not re.search(r"^\s*from \.", source, flags=re.MULTILINE), \
        "Failed to build the submission. A relative import survived the rewrite, and a flat bundle has no " \
        "parent package for one to resolve against. A new module may have been added to AGSHEAF_MODULES."

    return source


def defining_module(name: str) -> str:
    """
    Says which agsheaf module defines a name re-exported by the package, read from the package
    itself rather than hardcoded, so a change to what __init__.py re-exports is picked up here.

    Args:
        name (str): The re-exported name, e.g. "ContractSheaf".
    """
    init_source = (REPO_ROOT / "src" / "agsheaf" / "__init__.py").read_text()
    for line in init_source.splitlines():
        match = re.match(r"from \.(\w+) import ([\w, ]+)", line.strip())
        if match and name in [imported.strip() for imported in match.group(2).split(",")]:
            return match.group(1)
    raise AssertionError(
        "Failed to build the submission. The name %r is imported from the agsheaf package but "
        "src/agsheaf/__init__.py does not say which module defines it." % name)


## configuration and entry point


def write_config(testbed_configs: list, experiment: Path = None) -> tuple:
    """
    Bakes the merged configuration into a module, since YAML cannot be uploaded. Returns the
    file written and the configuration itself, the latter for the build log, which records the
    parameters resolved rather than the names of the files they came from.

    Layered, each overriding the one before it section by section: exp/defaults.yaml, then
    exp/worlds/primary.yaml, then each testbed layer in turn. The first two are merged by the same
    function an ordinary run uses, so the bundle runs the world the repo describes rather than a
    second copy of it that could drift; the rest are what the testbed, and the particular
    variant being filmed, want differently from a screen.

    Args:
        testbed_configs (list): The testbed layers, in order. Empty ships the experiment as it
            stands. A path that does not exist is an error rather than a silent skip -- a
            submission built without the layer it was meant to have is the kind of mistake that
            is only noticed in the returned video.
        experiment (Optional[Path]): The world the layers apply over, defaulting to
            exp/worlds/primary.yaml. A world has to come in HERE rather than as another entry in
            testbed_configs, because the merge below joins nested mappings: a second world passed
            as a layer would have exp/worlds/primary.yaml's starts, assignments and per-agent beliefs
            merged underneath its own, and a bundle would ship a world that is neither.
    """
    from agsheaf.utils import _merge_settings, load_experiment_config

    experiment = experiment or (REPO_ROOT / "exp" / "worlds" / "primary.yaml")
    assert Path(experiment).is_file(), \
        "Failed to build the submission. No experiment at %s." % experiment
    print("Experiment: %s" % experiment)

    config = load_experiment_config(experiment, REPO_ROOT / "exp" / "defaults.yaml")

    for testbed_config in testbed_configs:
        assert testbed_config.is_file(), \
            "Failed to build the submission. No testbed configuration at %s. Pass --config with no " \
            "paths to build without one." % testbed_config
        with open(testbed_config) as testbed_file:
            overrides = yaml.safe_load(testbed_file) or {}
        assert isinstance(overrides, dict), \
            "Failed to build the submission. Expected a mapping in %s, received type %r." % (
                testbed_config, type(overrides).__name__)
        config = _merge_settings(config, overrides)
        print("Testbed layer: %s" % testbed_config)

    # The figure is not up for configuration here, but the other way about from a local run:
    # the Robotarium scales it to the arena and projects it onto the floor the robots drive on,
    # so switching it off would leave the testbed blank and the beliefs invisible. A submission
    # always draws.
    config["gridworld"]["show_figure"] = True

    # Named so the record says which variant it came from, and so benchmark.py's --replay, which
    # picks logs by this field, cannot mistake a testbed run for a local one. A layer may name
    # itself; only the fallback is fixed here.
    config.setdefault("experiment_name", "run_robotarium")

    destination = BUNDLE_DIR / "experiment_config.py"
    destination.write_text(
        '"""\n'
        "The configuration this submission runs, merged from exp/worlds/primary.yaml over exp/defaults.yaml,\n"
        "with the testbed layers applied over both.\n\n"
        "Generated by exp/robotarium/build_submission.py -- edit the YAML and rebuild, not this file.\n"
        'The Robotarium accepts no .yaml upload, which is why the configuration arrives as a literal.\n'
        '"""\n\n'
        "CONFIG = %s\n" % pprint.pformat(config, indent=4, width=100, sort_dicts=True))
    return destination, config


def write_entry_point() -> Path:
    """
    Copies the authored entry point in as the main file.

    Unlike the library modules this is not a mechanical translation of an existing script: the
    experiments are written around video and a figure window, and a submission has neither --
    what it has instead is a projector pointed at the floor.
    """
    destination = BUNDLE_DIR / "experiment.py"
    shutil.copy2(TEMPLATE, destination)
    return destination


def report(written: list[Path], built_at: str, fields: dict, archive: Path) -> None:
    """
    Prints what to upload and what to type, in the order the form asks for it.

    Args:
        written (list[Path]): Every file written into the bundle.
        built_at (str): When this build was recorded, as the build log names it.
        fields (dict): What to put on the submission form. See portal_fields.
        archive (Path): The zip of the bundle, kept alongside it.
    """
    total = sum(path.stat().st_size for path in written)
    print("\nWrote %d files to %s" % (len(written), BUNDLE_DIR))
    for path in sorted(written, key=lambda p: -p.stat().st_size):
        marker = "  <-- MAIN FILE" if path.name == "experiment.py" else ""
        print("  %8.1f KB  %s%s" % (path.stat().st_size / 1e3, path.name, marker))
    print("  %8.1f KB  total\n" % (total / 1e3))

    print("Zipped to %s (%.1f KB)" % (archive.name, archive.stat().st_size / 1e3))
    print("Built %s. Parameters logged to %s -- see --history for what has changed between builds."
          % (built_at, BUILD_LOG.name))

    measured = fields["measured_seconds"]
    print("\n" + "-" * 78)
    print("SUBMISSION FORM")
    print("-" * 78)
    print("Title                        %s" % fields["title"])
    print("Estimated Duration (max 600) %d" % fields["duration_seconds"])
    if measured:
        print("                             (measured %s s over %d dry run(s); doubled and rounded "
              "up to the minute)" % ("/".join("%.0f" % sample for sample in measured), len(measured)))
    else:
        print("                             (not measured -- rebuild with --dry-runs 1 for a "
              "figure based on actually running it)")
    print("Number of Robots (1-18)      %d" % fields["number_of_robots"])
    print("Experiment Description       %s" % _wrapped(fields["description"], 29))
    print("-" * 78)

    print("\nUpload every file in %s, and tick \"Main File\" next to experiment.py."
          % BUNDLE_DIR)


def _wrapped(text: str, indent: int) -> str:
    """
    Wraps a paragraph to the width of the form block, so a description meant to be copied out
    can be read before it is.

    Args:
        text (str): The paragraph to wrap.
        indent (int): Columns the block is already indented by.
    """
    return ("\n" + " " * indent).join(textwrap.wrap(text, width=78 - indent))


if __name__ == "__main__":
    main()
