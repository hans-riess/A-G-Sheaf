"""
The convergence study, as figures.

Reads a JSON written by `exp/convergence.py` and writes one vector PDF per
figure (plus a PNG for looking at) into `doc/figures/`. Nothing is computed
here that the runner did not record: this file chooses what to show and how, and
a number that appears in a plot can be traced back to a run in the JSON.

    python exp/figures.py                       # the newest run in exp/benchmarks
    python exp/figures.py path/to/run.json --out doc/figures

Figures, and what each is evidence for:

    descent      the flows converge, and the primal descent is monotone
    energy       the Tarski-Dirichlet energy reaches exactly zero, at the sweep
                 the assignment becomes a global section
    steps        how the sweep count scales with topology and diameter
    schedules    RG Thm. 1: different schedules, different lengths, same limit
    abstraction  abstract agreement reached while concrete disagreement remains
    legs         where the two bisheaves start to disagree about the limit
    collapse     when disagreement leaves the lattice, by encoding
    cost         what an exact-contract sweep costs, canonicalised or not

The palette is the four-slot categorical order of the data-visualisation
reference, validated for adjacent-pair separation under protanopia,
deuteranopia and tritanopia. Two of its slots sit below 3:1 contrast on the
page, so every series is also labelled at the end of its line rather than
identified by colour alone. These are figures for a printed report, so they are
drawn for one surface only.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#: Categorical slots, in fixed order. Never cycled: a fifth series folds into
#: "other" or gets its own panel.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#c9c8c2"
SURFACE = "#fcfcfb"

#: The flows, in the order they are always drawn and coloured.
FLOW_ORDER = ["primal", "primal_co", "dual", "dual_kan"]
FLOW_LABEL = {
    "primal": "primal (kan)",
    "primal_co": "primal (co)",
    "dual": "dual (co)",
    "dual_kan": "dual (kan)",
}
FLOW_COLOR = dict(zip(FLOW_ORDER, SERIES))

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": RULE,
    "axes.labelcolor": INK,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "grid.color": "#e8e7e2",
    "grid.linewidth": 0.7,
})


# ---------------------------------------------------------------------------
# reading a run
# ---------------------------------------------------------------------------

def load(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def select(record, study=None, kind="run", **equals) -> List[Dict[str, Any]]:
    """
    Rows of one study, filtered by exact field matches. `kind="comparison"`
    selects the cross-flow rows instead of the per-run ones.
    """
    out = []
    for row in record["runs"]:
        if study is not None and row.get("study") != study:
            continue
        if (row.get("kind") == "comparison") != (kind == "comparison"):
            continue
        if all(row.get(key) == value for key, value in equals.items()):
            out.append(row)
    return out


def curves(rows, key: str) -> List[List[float]]:
    """One trace per run, for runs that produced one."""
    return [row["trace"][key] for row in rows if "trace" in row]


def padded(traces: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Traces of different lengths, squared off by holding the last value.

    Holding rather than masking is the honest padding here: a flow that has
    reached its fixed point stays there, so the extended value is what a longer
    run would have recorded, not a filler.
    """
    if not traces:
        return np.zeros((0, 0))
    width = max(len(t) for t in traces)
    return np.array([list(t) + [t[-1]] * (width - len(t)) for t in traces], dtype=float)


def band(ax, traces, color, label, alpha=0.15, label_at=1.0, label_dy=5):
    """
    Median curve with an interquartile band, and a direct label on the curve.

    `label_at` is where along the curve the label sits, as a fraction. Curves
    that converge to the same value -- which is the point of most of these
    figures -- would stack every label on one pixel if they were all placed at
    the end, so callers spread them across the part of the run where the series
    are still apart. `label_at=None` puts it at the elbow instead: the last step
    at which this curve was still moving, which is where a set of curves that
    differ only in *how long they take* are furthest apart.
    """
    grid = padded(traces)
    if not grid.size:
        return None
    steps = np.arange(grid.shape[1])
    median = np.median(grid, axis=0)
    ax.fill_between(steps, np.percentile(grid, 25, axis=0),
                    np.percentile(grid, 75, axis=0), color=color, alpha=alpha,
                    linewidth=0)
    ax.plot(steps, median, color=color, label=label)

    if label_at is None:
        moving = np.flatnonzero(np.abs(np.diff(median)) > 1e-12)
        where = int(moving[-1]) + 1 if moving.size else len(steps) - 1
    else:
        where = int(round(label_at * (len(steps) - 1)))
    ax.annotate(label, (steps[where], median[where]), textcoords="offset points",
                xytext=(4, label_dy), color=color, fontsize=7.5, va="bottom")
    return median


def spread(count: int) -> List[float]:
    """Label positions along a curve, evenly spaced and never at the very start."""
    if count <= 1:
        return [0.75]
    return [0.25 + 0.6 * index / (count - 1) for index in range(count)]


def finish(fig, name: str, out_dir: Path, caption: Optional[str] = None) -> Path:
    if caption:
        fig.text(0.0, -0.03, caption, fontsize=7.5, color=MUTED, ha="left", va="top",
                 wrap=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pdf"
    fig.savefig(path)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# the figures
# ---------------------------------------------------------------------------

def figure_descent(record, out_dir) -> Optional[Path]:
    """
    Distance to the fixed point, sweep by sweep, for the meet-aggregated flow
    and the join-aggregated one.

    The primal curve is the one to read first: because the iterates nest, it is
    not merely shrinking but monotone, and the runner asserts that at every
    step. Individual runs are drawn behind the median so the spread is visible
    without a second chart.
    """
    panels = [("primal", "meet-aggregated (primal, kan legs)"),
              ("dual", "join-aggregated (dual, co legs)")]
    available = [(flow, title) for flow, title in panels
                 if curves(select(record, "convergence", flow=flow, schedule="sync"),
                           "distance_to_limit")]
    if not available:
        return None

    fig, axes = plt.subplots(1, len(available), figsize=(7.2, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (flow, title) in zip(axes, available):
        rows = select(record, "convergence", flow=flow, schedule="sync")
        traces = curves(rows, "distance_to_limit")
        grid = padded(traces)
        for run in grid:
            ax.plot(np.arange(len(run)), run, color=FLOW_COLOR[flow], alpha=0.25,
                    linewidth=1.0)
        ax.plot(np.arange(grid.shape[1]), np.median(grid, axis=0),
                color=FLOW_COLOR[flow], linewidth=2.4, label="median")

        sections = [next((t for t, flag in enumerate(row["trace"]["is_section"]) if flag),
                         None) for row in rows if "trace" in row]
        sections = [s for s in sections if s is not None]
        if sections:
            ax.axvline(float(np.median(sections)), color=MUTED, linestyle=(0, (4, 3)),
                       linewidth=1.0)
            ax.annotate("median sweep\nreaching a section",
                        (float(np.median(sections)), ax.get_ylim()[1]),
                        textcoords="offset points", xytext=(5, -14),
                        fontsize=7.5, color=MUTED, va="top")

        ax.set_title(title, color=INK)
        ax.set_xlabel("sweep")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("distance to the limit")

    return finish(fig, "descent", out_dir,
                  "Total distance from each sweep's assignment to the fixed point it "
                  "reaches, over the ensemble; thin lines are individual runs, thick "
                  "the median. Zero is the fixed point.")


def figure_energy(record, out_dir) -> Optional[Path]:
    """
    The Tarski-Dirichlet energy, and the interfaces it is a sum over.

    The energy is the order-theoretic stand-in for the Dirichlet energy of the
    linear heat flow: a sum over interfaces of how far apart the two endpoints
    are once pushed onto the shared space. It is zero exactly on the global
    sections, so its arrival at zero and the assignment becoming a section are
    the same event -- which is the claim the right-hand panel makes visible.
    """
    rows = select(record, "convergence", schedule="sync")
    if not any("trace" in row for row in rows):
        return None

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    # Spread along the curve *and* stacked vertically: all four flows land on the
    # same value, so either dodge alone still collides.
    for index, (flow, at) in enumerate(zip(FLOW_ORDER, spread(len(FLOW_ORDER)))):
        subset = select(record, "convergence", flow=flow, schedule="sync")
        dy = 6 + 11 * index
        band(left, curves(subset, "dirichlet"), FLOW_COLOR[flow], FLOW_LABEL[flow],
             label_at=at, label_dy=dy)

        closed = curves(subset, "closed_interfaces")
        totals = [row["interfaces"] for row in subset if "trace" in row]
        if closed:
            fractions = [[c / total for c in trace]
                         for trace, total in zip(closed, totals)]
            band(right, fractions, FLOW_COLOR[flow], FLOW_LABEL[flow], label_at=at,
                 label_dy=-8 - 11 * index)

    left.set_title("disagreement across all interfaces", color=INK)
    left.set_ylabel("Tarski-Dirichlet energy")
    left.legend(fontsize=8, ncols=2, loc="upper right")
    right.set_title("interfaces that agree", color=INK)
    right.set_ylabel("fraction closed")
    right.set_ylim(0, 1.12)
    for ax in (left, right):
        ax.set_xlabel("sweep")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.margins(x=0.12)

    return finish(fig, "energy", out_dir,
                  "Median over the ensemble, shaded to the interquartile range. The "
                  "energy is zero exactly when every interface agrees, so the left "
                  "panel reaching zero and the right reaching one are one fact.")


def figure_steps(record, out_dir) -> Optional[Path]:
    """
    Sweeps to a fixed point, against the topology and against its diameter.

    Convergence here is finite-step rather than asymptotic (Tarski), so the
    quantity to plot is a count, not a rate. The diameter line is the natural
    reference: knowledge moves one hop per synchronous sweep, so a flow that had
    only to propagate would finish in about that many.
    """
    rows = [r for r in select(record, "scale", flow="primal") if r.get("steps") is not None]
    if not rows:
        return None

    by_topology = defaultdict(list)
    for row in rows:
        by_topology[row["params"]["topology"]].append(row["steps"])
    order = sorted(by_topology, key=lambda t: np.median(by_topology[t]))

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    box = left.boxplot([by_topology[t] for t in order], tick_labels=order,
                       patch_artist=True, widths=0.6, medianprops=dict(color=INK))
    for patch in box["boxes"]:
        patch.set(facecolor=SERIES[0], alpha=0.35, edgecolor=SERIES[0], linewidth=1.4)
    for whisker in box["whiskers"] + box["caps"]:
        whisker.set(color=SERIES[0], linewidth=1.2)
    left.set_title("sweeps to a fixed point, by topology", color=INK)
    left.set_ylabel("sweeps")
    left.tick_params(axis="x", labelrotation=20)

    rng = np.random.default_rng(0)
    for colour, flow in zip(SERIES, ("primal", "dual")):
        subset = [r for r in select(record, "scale", flow=flow) if r.get("steps") is not None]
        if not subset:
            continue
        x = np.array([r["diameter"] for r in subset], dtype=float)
        y = np.array([r["steps"] for r in subset], dtype=float)
        right.scatter(x + rng.uniform(-0.12, 0.12, x.size), y, s=18, color=colour,
                      alpha=0.65, edgecolor=SURFACE, linewidth=0.5,
                      label=FLOW_LABEL[flow])
    span = np.arange(1, max(r["diameter"] for r in rows) + 1)
    right.plot(span, span + 1, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    right.annotate("diameter + 1", (span[-1], span[-1] + 1), textcoords="offset points",
                   xytext=(-6, 6), fontsize=7.5, color=MUTED, ha="right")
    right.set_title("sweeps against graph diameter", color=INK)
    right.set_xlabel("diameter")
    right.set_ylabel("sweeps")
    right.legend(loc="upper left", fontsize=8)

    for ax in (left, right):
        ax.grid(axis="y")
        ax.set_axisbelow(True)

    return finish(fig, "steps", out_dir,
                  "Synchronous schedule. Convergence is finite-step, so the quantity "
                  "is a count of sweeps rather than a rate.")


def figure_schedules(record, out_dir) -> Optional[Path]:
    """
    Riess and Ghrist Theorem 1, drawn: the firing schedule changes how long the
    flow takes and not where it ends up.

    The left panel is the cost, the right panel the claim -- four schedules
    descending at different speeds to the same value. The distance between the
    limits themselves is recorded separately and is exactly zero, which the
    caption states because a chart of zeros shows nothing.
    """
    rows = [r for r in select(record, "convergence") if r.get("steps") is not None]
    if not rows:
        return None

    schedules = sorted({r["schedule"] for r in rows})
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.0))

    width = 0.8 / max(len(FLOW_ORDER), 1)
    for index, flow in enumerate(FLOW_ORDER):
        means, positions = [], []
        for slot, schedule in enumerate(schedules):
            steps = [r["steps"] for r in rows
                     if r["flow"] == flow and r["schedule"] == schedule]
            if steps:
                means.append(float(np.mean(steps)))
                positions.append(slot + index * width - 0.4 + width / 2)
        if means:
            left.bar(positions, means, width=width * 0.86, color=FLOW_COLOR[flow],
                     label=FLOW_LABEL[flow], linewidth=0)
    left.set_xticks(range(len(schedules)), [s.replace("_", "-") for s in schedules])
    left.tick_params(axis="x", labelrotation=12)
    left.set_title("sweeps to a fixed point, by schedule", color=INK)
    left.set_ylabel("mean sweeps")
    left.legend(fontsize=8, ncols=2)

    # Labelled at each curve's elbow -- where a set of curves that differ only in
    # how long they take are furthest apart -- and stacked vertically, because
    # three of the four schedules flatten at nearly the same sweep.
    for index, (colour, schedule) in enumerate(zip(SERIES, schedules)):
        subset = select(record, "convergence", flow="primal", schedule=schedule)
        band(right, curves(subset, "distance_to_limit"), colour,
             schedule.replace("_", "-"), alpha=0.10, label_at=None,
             label_dy=6 + 11 * index)
    right.set_title("the primal flow, under each schedule", color=INK)
    right.set_xlabel("sweep")
    right.set_ylabel("distance to the limit")
    right.margins(x=0.2)

    for ax in (left, right):
        ax.grid(axis="y")
        ax.set_axisbelow(True)

    independence = record.get("schedule_independence", {})
    return finish(fig, "schedules", out_dir,
                  "Every schedule reaches the same fixed point: across %d "
                  "(sheaf, flow) groups the greatest distance between two schedules' "
                  "limits was zero, with %d exceptions."
                  % (independence.get("groups_checked", 0),
                     len(independence.get("disagreements", []))))


def figure_abstraction(record, out_dir) -> Optional[Path]:
    """
    The figure the demonstration's redesign turns on.

    As the restriction coarsens, the flow keeps closing every interface -- the
    abstract disagreement is zero at every coarseness -- while the agents' own
    beliefs stay apart. That residue is the kernel of `f_!`: disagreement the
    shared vocabulary cannot express, and which the mission therefore tolerates.
    At coarseness 1 the restriction is a bijection and the two coincide, which is
    the case the demonstration runs today.
    """
    rows = [r for r in select(record, "abstraction", flow="primal") if "limit" in r]
    if not rows:
        return None

    aggregators = sorted({r["params"]["aggregator"] for r in rows})
    fig, axes = plt.subplots(1, len(aggregators), figsize=(7.2, 2.9), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, aggregator in zip(axes, aggregators):
        subset = [r for r in rows if r["params"]["aggregator"] == aggregator]
        coarseness = sorted({r["params"]["coarseness"] for r in subset})
        for colour, key, label in ((SERIES[0], "concrete_disagreement",
                                    "concrete (agents' own facts)"),
                                   (SERIES[2], "dirichlet",
                                    "abstract (shared vocabulary)")):
            means = [float(np.mean([r["limit"][key] for r in subset
                                    if r["params"]["coarseness"] == c]))
                     for c in coarseness]
            spread = [float(np.std([r["limit"][key] for r in subset
                                    if r["params"]["coarseness"] == c]))
                      for c in coarseness]
            ax.errorbar(coarseness, means, yerr=spread, color=colour, marker="o",
                        capsize=3, elinewidth=1.2, label=label)
            ax.annotate(label.split(" (")[0], (coarseness[-1], means[-1]),
                        textcoords="offset points", xytext=(5, 0), fontsize=8,
                        color=colour, va="center")
        ax.set_title(f"blocks summarised by {aggregator}", color=INK)
        ax.set_xlabel("coarseness (facts per block)")
        ax.set_xticks(coarseness)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.margins(x=0.25)
    axes[0].set_ylabel("disagreement at the limit")

    return finish(fig, "abstraction", out_dir,
                  "Mean over the ensemble, one standard deviation. Coarseness 1 is the "
                  "identity abstraction, where agreeing abstractly and agreeing "
                  "concretely are the same thing.")


def figure_legs(record, out_dir) -> Optional[Path]:
    """
    How far apart the two bisheaves' limits are, as the restriction coarsens.

    `legs="kan"` pushes by `f_!` and `legs="co"` by `f_*`, the report's case (1)
    and case (2). Along a bijection the two agree and the parameter is a no-op --
    the leftmost point, which is where the demonstration currently sits. The
    distance leaving zero is the moment `legs` acquires the meaning the report
    gives it.
    """
    rows = select(record, "abstraction", kind="comparison")
    pairs = [(r, r["limit_distance_across_flows"].get("primal|primal_co"))
             for r in rows]
    pairs = [(r, d) for r, d in pairs if d is not None]
    if not pairs:
        return None

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    every = sorted({r["params"]["coarseness"] for r, _ in pairs})

    aggregators = sorted({r["params"]["aggregator"] for r, _ in pairs})
    # The curves converge on the right, so the labels are dodged rather than
    # both parked on the last point.
    for colour, aggregator, dy in zip(SERIES, aggregators, (7, -12)):
        subset = [(r, d) for r, d in pairs if r["params"]["aggregator"] == aggregator]
        coarseness = sorted({r["params"]["coarseness"] for r, _ in subset})
        means = [float(np.mean([d for r, d in subset
                                if r["params"]["coarseness"] == c])) for c in coarseness]
        left.plot(coarseness, means, color=colour, marker="o", label=aggregator)
        left.annotate(aggregator, (coarseness[-1], means[-1]), textcoords="offset points",
                      xytext=(5, dy), fontsize=8, color=colour, va="center")
    left.set_title("distance between the two limits", color=INK)
    left.set_ylabel("distance between limits")

    # And what the difference costs. The right adjoint's guarantee leg is a
    # universal over the fibre -- every concrete state in the block must satisfy
    # it -- so a coarse block asks for more than any stalk can promise.
    runs = [r for r in select(record, "abstraction") if "limit" in r]
    for colour, flow in zip(SERIES, ("primal", "primal_co")):
        subset = [r for r in runs if r["flow"] == flow]
        fraction, present = [], []
        for c in every:
            cell = [r for r in subset if r["params"]["coarseness"] == c]
            if cell:
                present.append(c)
                fraction.append(sum(1 for r in cell if r["limit"]["collapsed"]) / len(cell))
        if present:
            right.plot(present, fraction, color=colour, marker="o", label=FLOW_LABEL[flow])
            right.annotate(FLOW_LABEL[flow], (present[-1], fraction[-1]),
                           textcoords="offset points", xytext=(5, 0), fontsize=8,
                           color=colour, va="center")
    right.set_title("runs where a stalk collapsed", color=INK)
    right.set_ylabel("fraction of runs")
    right.set_ylim(-0.05, 1.05)

    for ax in (left, right):
        ax.set_xlabel("coarseness (facts per block)")
        ax.set_xticks(every)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.margins(x=0.3)

    return finish(fig, "legs", out_dir,
                  "Left: zero at coarseness 1 — along a bijection, pushing by the left "
                  "Kan extension and by the right one give the same contract, so the "
                  "choice of bisheaf cannot matter. Right: above it, it matters a great "
                  "deal.")


def figure_collapse(record, out_dir) -> Optional[Path]:
    """
    How often disagreement leaves the lattice, by how a fact is valued.

    A Boolean fact contradicts into an unsatisfiable guarantee and takes its
    whole stalk to `bot`. A possibility set contradicts into the empty set,
    which is a state of the alphabet: the conflict is recorded, localised to the
    fact it is about, and nothing collapses. This is the encoding choice the
    demonstration already makes; the figure is what it buys.
    """
    rows = [r for r in select(record, "collapse") if "limit" in r]
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    every_error = sorted({r["params"]["error"] for r in rows})
    for colour, encoding in zip(SERIES, sorted({r["params"]["encoding"] for r in rows})):
        subset = [r for r in rows if r["params"]["encoding"] == encoding]
        errors = sorted({r["params"]["error"] for r in subset})
        fraction = []
        for error in errors:
            cell = [r for r in subset if r["params"]["error"] == error]
            fraction.append(sum(1 for r in cell if r["limit"]["collapsed"]) / len(cell))
        ax.plot(errors, fraction, color=colour, marker="o", label=encoding)
        ax.annotate(encoding, (errors[-1], fraction[-1]), textcoords="offset points",
                    xytext=(5, 0), fontsize=8, color=colour, va="center")

    ax.set_title("runs where some stalk reached a lattice extreme", color=INK)
    ax.set_xlabel("probability an agent's belief is wrong")
    ax.set_ylabel("fraction of runs")
    ax.set_ylim(-0.04, 1.08)
    ax.set_xticks(every_error)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.margins(x=0.14)

    return finish(fig, "collapse", out_dir,
                  "A collapsed stalk is one whose contract reached top or bottom; "
                  "`collapsed()` names which, and why.")


def figure_cost(record, out_dir) -> Optional[Path]:
    """
    What a sweep costs, with and without rewriting each stalk into canonical
    form.

    Every transport nests a quantifier layer, so an uncanonicalised flow carries
    its whole history in every formula and pays for it on every solver call. The
    right panel is the cause and the left is the effect. This is the measurement
    the demonstration needs before its control loop is asked to run on exact
    contracts.
    """
    rows = select(record, "cost")
    if not rows:
        return None

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for colour, flag, label, at in ((SERIES[0], True, "canonicalised", 0.55),
                                    (SERIES[1], False, "as transported", 0.3)):
        subset = [r for r in rows if r.get("canonicalise") is flag]
        seconds = [np.cumsum(r["seconds_per_step"]).tolist() for r in subset
                   if r.get("seconds_per_step")]
        sizes = [r["ast_size"] for r in subset if r.get("ast_size")]
        band(left, seconds, colour, label, alpha=0.12, label_at=at)
        band(right, sizes, colour, label, alpha=0.12, label_at=at)

    left.set_title("solver time", color=INK)
    left.set_ylabel("cumulative seconds")
    right.set_title("size of the largest stalk formula", color=INK)
    right.set_ylabel("AST nodes")
    right.set_yscale("log")
    for ax in (left, right):
        ax.set_xlabel("sweep")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.margins(x=0.22)

    timeouts = sum(1 for r in rows if r.get("outcome") == "timeout")
    note = ("No run was abandoned at its wall-clock budget." if not timeouts else
            "%d uncanonicalised run%s abandoned at the wall-clock budget and "
            "contribute%s only the sweeps finished before it."
            % (timeouts, " was" if timeouts == 1 else "s were",
               "s" if timeouts == 1 else ""))
    return finish(fig, "cost", out_dir,
                  "Median over the ensemble, shaded to the interquartile range. " + note)


FIGURES = {
    "descent": figure_descent,
    "energy": figure_energy,
    "steps": figure_steps,
    "schedules": figure_schedules,
    "abstraction": figure_abstraction,
    "legs": figure_legs,
    "collapse": figure_collapse,
    "cost": figure_cost,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", default=None,
                        help="a JSON from exp/convergence.py; default is the newest")
    parser.add_argument("--out", type=str, default="../doc/figures")
    parser.add_argument("--only", type=str, default=None,
                        help="comma-separated subset of %s" % ",".join(FIGURES))
    args = parser.parse_args()

    here = Path(__file__).parent
    if args.run:
        path = Path(args.run)
    else:
        candidates = sorted((here / "benchmarks").glob("convergence_*.json"))
        if not candidates:
            parser.error("no convergence_*.json in exp/benchmarks; run exp/convergence.py")
        path = candidates[-1]

    record = load(path)
    out_dir = (here / args.out).resolve() if not Path(args.out).is_absolute() \
        else Path(args.out)
    chosen = [n.strip() for n in args.only.split(",")] if args.only else list(FIGURES)

    print("reading %s (%d runs, %s)" % (path.name, record["summary"]["runs"],
                                        record["generated"]))
    for name in chosen:
        if name not in FIGURES:
            parser.error("unknown figure %r; known: %s" % (name, sorted(FIGURES)))
        written = FIGURES[name](record, out_dir)
        print("  %-12s %s" % (name, written.name if written else "skipped (no data)"))
    print("figures in %s" % out_dir)


if __name__ == "__main__":
    main()
