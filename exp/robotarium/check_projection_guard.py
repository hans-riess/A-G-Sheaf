"""
Breaks the projection on purpose and checks that the experiment survives it.

experiment_template.py holds everything drawn on the arena floor at arm's length, on the
grounds that it is a report of the run rather than part of one: a piece that cannot be built is
not drawn, a piece that throws mid-run stops being drawn, and either way the robots finish their
run and the results get written. That claim is worth exactly as much as the evidence for it, and
the evidence cannot come from an ordinary test -- the bundle is a script that runs a whole
experiment when imported, and the failures being guarded against are ones that only happen while
it is running.

So this damages the built bundle three ways, once per shape the failure can take, and runs each:

    the belief tiles cannot be built at all      -- what the first submission actually died of
    the belief tiles throw on every single frame -- thousands of times, once per control step
    the figure itself cannot be made             -- which takes the whole world down with it

Each case has to end with the run finished, results.json written, the failure announced at the
point it happened and again in a summary at the end, and the failure recorded in results.json.
The scores are compared against an undamaged run as well: a projection that broke must leave the
numbers exactly where an unbroken one leaves them, that being the entire claim.

Build the bundle first, then run it from anywhere:

    python exp/robotarium/build_submission.py --dry-runs 0
    python exp/robotarium/check_projection_guard.py

It takes a few minutes -- every case is a full run of the experiment, four of them in all.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parent / "submission"
RESULTS_FILENAME = "results.json"

# What the run prints when a piece of the projection fails, and what it prints at the end to
# repeat it. A log.txt is read from the end, and by then the notice at the point of failure is
# thousands of lines up, so both have to be there.
ANNOUNCEMENT = "THE PROJECTION FAILED:"
SUMMARY = "THE PROJECTION FAILED IN"

# What a run concludes, and so what breaking its projection must leave untouched.
#
# "frames" is deliberately not among them. The simulator places the robots at arbitrary poses
# and drives them to their initial conditions before the experiment begins, sampling those poses
# from numpy's global generator, which nothing here seeds -- so two identical runs record a
# slightly different number of frames and take a slightly different number of seconds. That is
# the same variance build_submission.py takes several duration samples to see around. The tile
# level above it is deterministic, and it is the tile level that carries the results.
CONCLUSIONS = ("steps_taken", "all_arrived", "final_scores", "total_score")


def main() -> None:
    if not BUNDLE_DIR.is_dir():
        raise SystemExit(
            "No bundle at %s. Build one first:\n"
            "    python exp/robotarium/build_submission.py --dry-runs 0" % BUNDLE_DIR)

    print("An undamaged run first, for the results the damaged ones have to match:")
    intact = run_bundle(lambda sandbox: None)
    assert intact["completed"].returncode == 0, \
        "Failed to check the projection guard. The bundle does not run undamaged, so nothing " \
        "can be concluded from running it damaged:\n%s" % intact["completed"].stdout[-2000:]
    assert "projection_failures" not in intact["record"], \
        "Failed to check the projection guard. An undamaged run recorded a projection failure, " \
        "so the record cannot be read as evidence of one."

    baseline = {field: intact["record"]["result"].get(field) for field in CONCLUSIONS}
    print("  %d steps, all arrived: %s, total score: %s\n"
          % (baseline["steps_taken"], baseline["all_arrived"], baseline["total_score"]))

    survived = [check(name, damage, baseline) for name, damage in CASES]

    print("=" * 79)
    if all(survived):
        print("The projection can fail in any of %d ways and the experiment still finishes, "
              "still\nscores the same, and still says what broke." % len(CASES))
    else:
        raise SystemExit(
            "The projection guard does not hold: %d of %d failures took the experiment with "
            "them, or went unreported." % (len(survived) - sum(survived), len(survived)))


def check(name: str, damage, baseline: dict) -> bool:
    """
    Runs the bundle with one piece of its projection broken, and says whether that was survived.

    Args:
        name (str): The case being checked, as the report should name it.
        damage (callable): Damages a copy of the bundle, given its directory.
        baseline (dict): The "result" block of an undamaged run, which this has to match.
    """
    outcome = run_bundle(damage)
    completed, record = outcome["completed"], outcome["record"]
    output = completed.stdout

    ran = completed.returncode == 0
    wrote = record is not None
    result = (record or {}).get("result") or {}
    concluded = {field: result.get(field) for field in CONCLUSIONS}
    scored_the_same = wrote and concluded == baseline
    reported = ANNOUNCEMENT in output and SUMMARY in output
    recorded = bool((record or {}).get("projection_failures"))

    print("=" * 79)
    print("CASE: %s" % name)
    print("  the run finished          : %s" % _yes_no(ran))
    print("  %s written      : %s" % (RESULTS_FILENAME, _yes_no(wrote)))
    print("  same results as undamaged : %s%s" % (
        _yes_no(scored_the_same),
        "" if scored_the_same or not wrote else "   %s against %s" % (concluded, baseline)))
    print("  said so, at the time      : %s" % _yes_no(ANNOUNCEMENT in output))
    print("  said so again, at the end : %s" % _yes_no(SUMMARY in output))
    print("  recorded in %s  : %s" % (RESULTS_FILENAME, _yes_no(recorded)))
    for what, failure in ((record or {}).get("projection_failures") or {}).items():
        print("      %s -- %s, %d time(s)" % (what, failure["error"], failure["count"]))

    survived = ran and wrote and scored_the_same and reported and recorded
    if not survived:
        print("\n  The last 2000 characters of the run:\n")
        print(output[-2000:])
        print(completed.stderr[-2000:])
    return survived


def run_bundle(damage) -> dict:
    """
    Copies the bundle into a sandbox, damages it, and runs it the way the testbed would.

    Headless, since there is no projector here, and from a flat directory of its own so that
    nothing resolves out of this repository -- the same conditions build_submission.py times a
    dry run under.

    Args:
        damage (callable): Damages the copied bundle, given its directory.
    """
    sandbox = Path(tempfile.mkdtemp())
    try:
        for path in BUNDLE_DIR.iterdir():
            if path.is_file():
                shutil.copy2(path, sandbox / path.name)
        damage(sandbox)

        completed = subprocess.run([sys.executable, "experiment.py"], cwd=sandbox,
                                   capture_output=True, text=True,
                                   env={**os.environ, "MPLBACKEND": "Agg"})
        results = sandbox / RESULTS_FILENAME
        return {
            "completed": completed,
            "record": json.loads(results.read_text()) if results.is_file() else None,
        }
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


## the damage


def _insert_before(sandbox: Path, filename: str, anchor: str, inserted: str) -> None:
    """
    Puts a line into the bundle just before an anchor, so that a case breaks the piece it means
    to break rather than whatever now sits at a line number.

    Args:
        sandbox (Path): The copied bundle to damage.
        filename (str): Which module to damage.
        anchor (str): The text to insert before, which must appear exactly once.
        inserted (str): The line to insert, indentation included.
    """
    path = sandbox / filename
    source = path.read_text()
    assert source.count(anchor) == 1, \
        "Failed to check the projection guard. Expected to find %r exactly once in %s, found it " \
        "%d times -- this script damages the bundle by anchoring on its source, so an edit there " \
        "needs the anchor updated here." % (anchor, filename, source.count(anchor))
    path.write_text(source.replace(anchor, inserted + "\n" + anchor, 1))


def break_the_tiles_being_built(sandbox: Path) -> None:
    """The belief tiles cannot be built. This is the shape of the first submission's failure."""
    _insert_before(
        sandbox, "agsheaf_gridworld.py",
        "    # Resolving colors used to draw each pairwise comparison",
        "    raise TypeError(\"reshape() got an unexpected keyword argument 'shape'\")")


def break_the_tiles_being_drawn(sandbox: Path) -> None:
    """The belief tiles build, then throw on every frame -- thousands of times over a run."""
    _insert_before(
        sandbox, "agsheaf_gridworld.py",
        "        # Variable assertions\n        assert robot_coords.shape[0] == 2",
        "        raise RuntimeError(\"the projector fell over\")")


def break_the_figure(sandbox: Path) -> None:
    """The figure cannot be made at all, which is the one failure that takes the world with it."""
    _insert_before(
        sandbox, "agsheaf_gridworld.py",
        "        # Render grid\n        if self.show_figure:",
        "        if self.show_figure:\n"
        "            raise RuntimeError(\"no display, no projector, nothing to draw onto\")")


CASES = (
    ("the belief tiles cannot be built", break_the_tiles_being_built),
    ("the belief tiles throw on every frame", break_the_tiles_being_drawn),
    ("the figure itself cannot be made", break_the_figure),
)


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "NO"


if __name__ == "__main__":
    main()
