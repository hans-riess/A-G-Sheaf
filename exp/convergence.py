"""
Convergence of the Tarski Laplacian and its dual, measured rather than asserted.

`test_theorems.py` proves each theorem on hand-built sheaves and
`validate_theory.py` samples initial assignments for the gridworld sheaf. Both
answer with booleans. This answers with curves: over a randomized ensemble of
contract sheaves (`agsheaf.ensembles`), it runs the primal and dual harmonic
flows under several firing schedules and records, at every step, how far the
assignment is from the fixed point and how much disagreement is left across the
interfaces (`agsheaf.diagnostics`).

Five studies, each varying one axis around a baseline, because a ten-dimensional
cartesian product produces neither a readable figure nor a finished run:

    convergence   every flow under every schedule -- the descent profiles, the
                  Dirichlet energy falling to zero, and the check that all
                  schedules reach the same limit (RG Thm. 1)
    scale         topology and agent count -- steps to a fixed point against
                  graph diameter
    abstraction   coarseness of the restriction, from bijection upward, with
                  both bisheaves -- abstract agreement against concrete
                  disagreement, and where `legs` starts to matter
    collapse      encoding against error rate -- how often disagreement takes a
                  stalk to a lattice extreme rather than staying in the lattice
    cost          canonicalisation on and off -- solver time and formula size
                  per sweep, which is what the demonstration would have to pay

Every run carries its own invariant checks, and a violated invariant is recorded
in the output rather than raised: a study that stops at the first surprise
produces no figure at all, and the surprise is the finding. `all_invariants` in
the summary is the one line to read first.

Usage:

    python exp/convergence.py --quick
    python exp/convergence.py --studies convergence,abstraction --seeds 12
    python exp/convergence.py --out benchmarks
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agsheaf.contracts import Undecided
from agsheaf.diagnostics import (Alphabets, concrete_disagreement, distance_profile,
                                 divergence_profile, edge_energies,
                                 laplacian_residual)
from agsheaf.ensembles import DegenerateEnsemble, random_sheaf
from agsheaf.measure import AlphabetTooLarge, canonical
from agsheaf.sheaf import CO, KAN, LivenessError, random_firing, round_robin
from agsheaf.simplify import ast_size

#: The four flows. `dual` and `legs` are orthogonal -- the first chooses the
#: aggregation, the second the bisheaf transported along -- so all four
#: combinations exist and three of them mean something different:
#:
#:    primal      meet along `lan -| pullback`      Milestone 3 case (1)
#:    primal_co   meet along `dual_pullback -| ran` Milestone 3 case (2)
#:    dual        join along `dual_pullback -| ran` the dual with content
#:    dual_kan    join along `lan -| pullback`      Def. 7.3 Eq. (13), literally
#:
#: The last carries no content over the Boolean base (`doc/MATH.md`, "section 7
#: degenerates"); it is run as a labelled control so that the claim can be seen
#: rather than taken on trust.
FLOWS = {
    "primal": {"dual": False, "legs": KAN},
    "primal_co": {"dual": False, "legs": CO},
    "dual": {"dual": True, "legs": CO},
    "dual_kan": {"dual": True, "legs": KAN},
}

#: Firing schedules, in the sense of RG Def. 5. `sync` is their Eq. (6) read
#: literally -- every agent reads the same snapshot -- and is Algorithm 1 of the
#: Milestone 3 report; the rest are live schedules that Thm. 1 says must reach
#: the same limit.
SCHEDULES = {
    "sync": {"in_place": False, "firing": None},
    "inplace": {"in_place": True, "firing": None},
    "round_robin": {"in_place": True, "firing": "round_robin"},
    "random": {"in_place": True, "firing": "random"},
}

#: The ensemble every study varies around. Possibility-valued facts by default,
#: because that is what the demonstration uses and what keeps a contested fact
#: from taking its whole stalk to `bot`; `collapse` is the study that varies it.
BASELINE = {
    "agents": 5,
    "topology": "path",
    "facts": 3,
    "coarseness": 2,
    "aggregator": "or",
    "encoding": "possibility",
    "labels": 3,
    "knowledge": 0.55,
    "error": 0.25,
    "assumption_density": 0.3,
    "density": 0.3,
}


class Budget(RuntimeError):
    """The run exceeded its wall-clock allowance and was abandoned."""


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------

def _canonicaliser(sheaf, alphabets):
    """Rewrite every stalk into its canonical form; semantics-preserving."""
    def rewrite():
        for i in sheaf.nodes():
            sheaf.nodes[i]['c'] = canonical(sheaf.contract(i), alphabets.node(i))
    return rewrite


def run_flow(sheaf, flow: str, schedule: str, rng: random.Random,
             max_sweeps: int = 40, canonicalise: bool = True, budget: float = 60.0,
             max_states: int = 1 << 16) -> Tuple[Dict[str, Any], Optional[Dict]]:
    """
    One flow on one sheaf: its record, and the limit it reached.

    The limit comes back alongside rather than inside the record because it is
    contracts rather than JSON -- `run_cell` needs it to compare one flow's limit
    against another's, and nothing else does.

    The iterates are snapshotted from inside `converge`'s `on_step`, which is
    the only moment they exist -- the flow overwrites each stalk in place. Every
    metric is computed afterwards against the last snapshot, so the measurement
    cannot perturb what it measures and the recorded timings are of the flow
    alone.
    """
    settings = FLOWS[flow]
    nodes = sorted(sheaf.nodes())
    firing = SCHEDULES[schedule]["firing"]
    sequence = None
    if firing == "round_robin":
        sequence = lambda: round_robin(nodes)
    elif firing == "random":
        sequence = lambda: random_firing(nodes, rng)

    alphabets = Alphabets(sheaf, max_states=max_states)
    rewrite = _canonicaliser(sheaf, alphabets) if canonicalise else None

    snapshots = [sheaf.assignment()]
    seconds: List[float] = []
    sizes: List[int] = []
    started = time.perf_counter()
    last = started

    def on_step(step, unchanged):
        nonlocal last
        if rewrite is not None:
            rewrite()
        snapshots.append(sheaf.assignment())
        now = time.perf_counter()
        seconds.append(now - last)
        last = now
        sizes.append(max(ast_size(sheaf.contract(i).a)
                         + ast_size(sheaf.contract(i).sat_g) for i in nodes))
        if now - started > budget:
            raise Budget(f"exceeded {budget}s after {step + 1} steps")

    outcome, steps = "converged", None
    try:
        steps = sheaf.converge(dual=settings["dual"], legs=settings["legs"],
                               in_place=SCHEDULES[schedule]["in_place"],
                               firing_sequence=sequence, max_sweeps=max_sweeps,
                               on_step=on_step)
    except Budget:
        outcome = "timeout"
    except Undecided:
        outcome = "undecided"
    except LivenessError:
        outcome = "starved"
    except AlphabetTooLarge:
        outcome = "unmeasurable"
    except RuntimeError:
        outcome = "max_sweeps"

    record = {
        "flow": flow,
        "schedule": schedule,
        "canonicalise": canonicalise,
        "outcome": outcome,
        "steps": steps,
        "seconds_total": round(time.perf_counter() - started, 4),
        "seconds_per_step": [round(s, 5) for s in seconds],
        "ast_size": sizes,
    }
    if outcome not in ("converged", "max_sweeps"):
        return record, None

    record.update(_measure(sheaf, alphabets, snapshots, settings))
    return record, snapshots[-1]


def _measure(sheaf, alphabets, snapshots, settings) -> Dict[str, Any]:
    """Every curve, computed against the last snapshot as the limit."""
    legs = settings["legs"]
    dual = settings["dual"]
    limit = snapshots[-1]
    interfaces = sheaf.interfaces()

    distance_to_limit, dirichlet, residual, closed = [], [], [], []
    concrete, per_node = [], []
    section_flags, energy_agrees = [], True

    for x in snapshots:
        profile = distance_profile(sheaf, alphabets, x, limit)
        per_node.append([round(profile[i], 6) for i in sorted(sheaf.nodes())])
        distance_to_limit.append(round(sum(profile.values()), 6))

        energies = edge_energies(sheaf, alphabets, legs=legs, x=x)
        total = sum(energies.values())
        dirichlet.append(round(total, 6))
        closed.append(sum(1 for e in energies.values() if e == 0.0))

        residual.append(round(laplacian_residual(sheaf, alphabets, dual=dual,
                                                 legs=legs, x=x), 6))
        concrete.append(round(sum(concrete_disagreement(sheaf, u, v, alphabets, x=x)
                                  for u, v in interfaces), 6))

        # Decided by contract equality rather than by the distance being zero,
        # so it stays an independent check on the measure rather than a
        # restatement of it.
        section = sheaf.is_section(legs=legs, x=x)
        section_flags.append(section)
        energy_agrees &= (total == 0.0) == section

    # Monotonicity: the primal flow only descends and the dual only ascends, so
    # one of the two directed distances between consecutive iterates is
    # identically zero. A violation names the step, since "it went up somewhere"
    # is not actionable.
    monotone, monotone_step = True, None
    for t in range(len(snapshots) - 1):
        before, after = snapshots[t], snapshots[t + 1]
        gap = divergence_profile(sheaf, alphabets,
                                 before if dual else after,
                                 after if dual else before)
        if any(value != 0.0 for value in gap.values()):
            monotone, monotone_step = False, t
            break

    # The measure against the solver, on the pairs most likely to expose a
    # disagreement: the last two iterates, which are equal as contracts.
    agrees_with_z3 = all(
        (distance_profile(sheaf, alphabets, snapshots[-1], snapshots[-2])[i] == 0.0)
        == (snapshots[-1][i] == snapshots[-2][i])
        for i in sheaf.nodes()) if len(snapshots) > 1 else True

    # The curve the figures draw. Because the iterates nest, the distance to the
    # fixed point cannot rise; checking it separately from `monotone` is worth
    # the two lines, since this is the claim a reader takes from the plot.
    approaches = all(distance_to_limit[t] >= distance_to_limit[t + 1]
                     for t in range(len(distance_to_limit) - 1))

    return {
        "trace": {
            "distance_to_limit": distance_to_limit,
            "dirichlet": dirichlet,
            "residual": residual,
            "closed_interfaces": closed,
            "concrete_disagreement": concrete,
            "is_section": section_flags,
        },
        "per_node_distance": per_node,
        "interfaces": len(interfaces),
        "limit": {
            "is_section": section_flags[-1],
            "collapsed": {str(k): v for k, v in sheaf.collapsed().items()},
            "dirichlet": dirichlet[-1],
            "concrete_disagreement": concrete[-1],
        },
        "invariants": {
            "monotone": monotone,
            "monotone_failed_at": monotone_step,
            "energy_zero_iff_section": bool(energy_agrees),
            "distance_zero_iff_equal": bool(agrees_with_z3),
            "approaches_the_limit": bool(approaches),
        },
    }


def build(params: Dict[str, Any], seed: int):
    """One sheaf from the ensemble, seeded so a surprising run can be replayed."""
    return random_sheaf(rng=random.Random(seed), **params)


def run_cell(params: Dict[str, Any], seed: int, flows, schedules,
             canonicalise: bool = True, **kwargs) -> List[Dict[str, Any]]:
    """
    Every (flow, schedule) on the *same* sheaf, plus the comparisons that only
    make sense between runs of one sheaf.

    A fresh instance per combination rather than `reset`, so that no run inherits
    another's memoised measurements. The instances are interchangeable: z3
    interns a constant by name and sort, so the limits are contracts over
    literally the same variables and can be compared directly.

    The two comparisons are the ones no single run can make:

      * across schedules, whether the limits coincide -- RG Thm. 1, as a
        distance rather than as a claim;
      * across flows, how far apart the limits of the two bisheaves are --
        zero exactly when the restriction is a bijection, which is the
        degeneracy `doc/DUALITY.md` is about.
    """
    rows, limits = [], {}
    reference = build(params, seed)
    alphabets = Alphabets(reference)

    for flow in flows:
        for schedule in schedules:
            sheaf = build(params, seed)
            rng = random.Random(seed * 977 + 13)
            row, limit = run_flow(sheaf, flow, schedule, rng,
                                  canonicalise=canonicalise, **kwargs)
            row.update(params=dict(params), seed=seed,
                       diameter=sheaf.graph["diameter"],
                       draws=sheaf.graph["attempts"])
            rows.append(row)
            if limit is not None:
                limits[(flow, schedule)] = limit

    comparison = _compare_limits(reference, alphabets, limits)
    if comparison:
        comparison.update(params=dict(params), seed=seed, kind="comparison",
                          diameter=reference.graph["diameter"])
        rows.append(comparison)
    return rows


def _compare_limits(sheaf, alphabets, limits) -> Dict[str, Any]:
    """Pairwise distances between the limits reached in one cell."""
    if not limits:
        return {}

    def between(a, b):
        return round(sum(distance_profile(sheaf, alphabets, limits[a], limits[b]).values()), 6)

    across_schedules = {}
    for flow in {f for f, _ in limits}:
        runs = sorted(s for f, s in limits if f == flow)
        if len(runs) > 1:
            first = runs[0]
            across_schedules[flow] = max(between((flow, first), (flow, other))
                                         for other in runs[1:])

    across_flows = {}
    flows_present = sorted({f for f, _ in limits})
    for i, a in enumerate(flows_present):
        for b in flows_present[i + 1:]:
            shared = sorted({s for f, s in limits if f == a} & {s for f, s in limits if f == b})
            if shared:
                across_flows[f"{a}|{b}"] = between((a, shared[0]), (b, shared[0]))

    return {"limit_distance_across_schedules": across_schedules,
            "limit_distance_across_flows": across_flows}


# ---------------------------------------------------------------------------
# the studies
# ---------------------------------------------------------------------------

def _progress(args, label: str, started: float) -> None:
    """
    A line per cell, because a study can run for minutes and silence is
    indistinguishable from a hang -- which is how the first version of this file
    was debugged twice.
    """
    if getattr(args, "verbose", False):
        print("    %-46s %6.1fs" % (label, time.perf_counter() - started), flush=True)


def study_convergence(args) -> List[Dict[str, Any]]:
    """Descent profiles, energy to zero, and schedule independence."""
    rows, at = [], time.perf_counter()
    for seed in range(args.seeds):
        rows += run_cell(BASELINE, seed, list(FLOWS), list(SCHEDULES),
                         max_sweeps=args.max_sweeps, budget=args.budget)
        _progress(args, "seed %d" % seed, at)
    return rows


def study_scale(args) -> List[Dict[str, Any]]:
    """Steps to a fixed point against topology and size."""
    rows, at = [], time.perf_counter()
    for topology in ("path", "cycle", "star", "tree", "random", "complete"):
        for agents in (3, 5, 7, 9):
            params = dict(BASELINE, topology=topology, agents=agents, facts=3)
            for seed in range(args.seeds):
                rows += run_cell(params, seed, ["primal", "dual"], ["sync"],
                                 max_sweeps=args.max_sweeps, budget=args.budget)
            _progress(args, "%s n=%d" % (topology, agents), at)
    return rows


def study_abstraction(args) -> List[Dict[str, Any]]:
    """
    The coarseness sweep. `coarseness=1` is `f = id`, the case the demonstration
    runs today; above it the fibres are non-trivial and the two bisheaves come
    apart.
    """
    rows, at = [], time.perf_counter()
    for coarseness in (1, 2, 3, 6):
        # `parity` is deliberately absent. It is a legitimate abstraction and
        # `ensembles` still offers it, but a parity block leaves its stalks with
        # no product structure and no small cover, and the same cell that takes
        # 1.5s under `or` did not finish in 40s under `parity`. That is a finding
        # about the encoding, not a cell worth waiting for in every sweep.
        for aggregator in ("or", "and"):
            # Six facts, so that coarseness has somewhere to go; two labels, to
            # keep the stalk alphabet inside the size where the whole sweep
            # finishes. Both bisheaves, meet-aggregated: this is the report's
            # case (1) against case (2), which is what `legs` selects.
            params = dict(BASELINE, facts=6, labels=2, coarseness=coarseness,
                          aggregator=aggregator)
            for seed in range(args.seeds):
                rows += run_cell(params, seed, ["primal", "primal_co"],
                                 ["sync"], max_sweeps=args.max_sweeps,
                                 budget=args.budget)
            _progress(args, "coarseness %d by %s" % (coarseness, aggregator), at)
    return rows


def study_collapse(args) -> List[Dict[str, Any]]:
    """
    Where disagreement stops staying in the lattice. The encoding is the whole
    question: a Boolean fact contradicts into `bot`, a possibility set
    contradicts into the empty set.
    """
    rows, at = [], time.perf_counter()
    for encoding in ("boolean", "possibility"):
        for error in (0.0, 0.15, 0.3, 0.45, 0.6):
            params = dict(BASELINE, encoding=encoding, error=error, knowledge=0.7)
            for seed in range(args.seeds):
                try:
                    rows += run_cell(params, seed, ["primal"], ["sync"],
                                     max_sweeps=args.max_sweeps, budget=args.budget)
                except DegenerateEnsemble as exc:
                    # A cell where nothing ever disagrees has no instance to run;
                    # that is a reading of the cell, not a failure of the study.
                    rows.append({"study_note": str(exc), "params": dict(params),
                                 "seed": seed, "outcome": "degenerate"})
            _progress(args, "%s error=%.2f" % (encoding, error), at)
    return rows


def study_cost(args) -> List[Dict[str, Any]]:
    """
    What an exact-contract flow costs, with and without canonicalisation.

    The uncanonicalised arm is expected to hit its budget; that is the
    measurement (`doc/DUALITY.md` section 4), not a failed run.
    """
    rows, at = [], time.perf_counter()
    for canonicalise in (True, False):
        for agents in (3, 4, 5, 6):
            params = dict(BASELINE, agents=agents)
            for seed in range(max(3, args.seeds // 2)):
                rows += run_cell(params, seed, ["primal"], ["sync"],
                                 canonicalise=canonicalise,
                                 max_sweeps=args.max_sweeps, budget=args.budget)
            _progress(args, "n=%d canonicalise=%s" % (agents, canonicalise), at)
    return rows


STUDIES = {
    "convergence": study_convergence,
    "scale": study_scale,
    "abstraction": study_abstraction,
    "collapse": study_collapse,
    "cost": study_cost,
}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The lines worth printing, and the ones a reviewer should read first."""
    runs = [r for r in rows if r.get("kind") != "comparison"]
    measured = [r for r in runs if "invariants" in r]
    outcomes: Dict[str, int] = {}
    for row in runs:
        outcomes[row.get("outcome", "?")] = outcomes.get(row.get("outcome", "?"), 0) + 1

    violations = [
        {"study": r.get("study"), "flow": r.get("flow"), "schedule": r.get("schedule"),
         "seed": r.get("seed"), "failed": [k for k, v in r["invariants"].items()
                                           if v is False]}
        for r in measured
        if not all(v for k, v in r["invariants"].items() if isinstance(v, bool))
    ]

    settled = [r for r in measured if r["outcome"] == "converged"]
    return {
        "runs": len(runs),
        "measured": len(measured),
        "outcomes": outcomes,
        "all_invariants": not violations,
        "violations": violations,
        "mean_steps": round(sum(r["steps"] for r in settled) / len(settled), 3)
        if settled else None,
        "reached_a_section": sum(1 for r in settled if r["limit"]["is_section"]),
        "collapsed_somewhere": sum(1 for r in settled if r["limit"]["collapsed"]),
    }


def schedule_independence(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    RG Thm. 1 as a number rather than a claim: for each sheaf and each flow, the
    greatest distance between the limits two schedules reached. Every entry must
    be exactly zero -- the theorem says the schedule cannot change the answer,
    only how long it takes.
    """
    checked, disagreed = 0, []
    for row in rows:
        if row.get("kind") != "comparison":
            continue
        for flow, gap in row.get("limit_distance_across_schedules", {}).items():
            checked += 1
            if gap != 0.0:
                disagreed.append({"seed": row["seed"], "flow": flow, "distance": gap})
    return {"groups_checked": checked, "disagreements": disagreed}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--studies", type=str, default=",".join(STUDIES),
                        help="comma-separated subset of %s" % ",".join(STUDIES))
    parser.add_argument("--seeds", type=int, default=8, help="samples per cell")
    parser.add_argument("--max-sweeps", type=int, default=40)
    parser.add_argument("--budget", type=float, default=45.0,
                        help="wall-clock seconds per run before it is abandoned")
    parser.add_argument("--out", type=str, default="benchmarks")
    parser.add_argument("--verbose", action="store_true",
                        help="a line per cell, so a long study is observable")
    parser.add_argument("--quick", action="store_true",
                        help="two seeds and a short budget, for a smoke run")
    args = parser.parse_args()

    if args.quick:
        args.seeds = 2
        args.budget = min(args.budget, 15.0)

    chosen = [name.strip() for name in args.studies.split(",") if name.strip()]
    unknown = [name for name in chosen if name not in STUDIES]
    if unknown:
        parser.error("unknown studies %s; known: %s" % (unknown, sorted(STUDIES)))

    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    for name in chosen:
        at = time.perf_counter()
        produced = STUDIES[name](args)
        for row in produced:
            row["study"] = name
        rows += produced
        summary = summarise(produced)
        print("%-12s %3d runs  %5.1fs  invariants %s  outcomes %s"
              % (name, summary["runs"], time.perf_counter() - at,
                 "ok" if summary["all_invariants"] else "VIOLATED",
                 summary["outcomes"]))
        for violation in summary["violations"]:
            print("    violation: %s" % violation)

    overall = summarise(rows)
    independence = schedule_independence([r for r in rows if r.get("study") == "convergence"])

    record = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "settings": vars(args),
        "baseline": BASELINE,
        "summary": overall,
        "schedule_independence": independence,
        "runs": rows,
    }

    out_dir = (Path(__file__).parent / args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / ("convergence_%s.json" % stamp)
    out_path.write_text(json.dumps(record, indent=2))

    print()
    print("invariants held everywhere: %s" % overall["all_invariants"])
    print("schedule independence: %d groups checked, %d disagreed"
          % (independence["groups_checked"], len(independence["disagreements"])))
    print("Wrote %s (%.1fs)" % (out_path, record["elapsed_seconds"]))


if __name__ == "__main__":
    main()
