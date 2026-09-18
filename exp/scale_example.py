"""
One randomized instance of the demonstration construction, at network scale.

Where `convergence.py` runs a synthetic ensemble, this runner builds the very
sheaf the Robotarium demonstration runs -- `build_region_sheaf` over the 12x8
six-room world, mission contracts (reliance and exclusive claims) from planned
corridors, closed-form `R_!` transports along the region-summary restrictions --
but over a sampled world and a much larger communication network, and follows
one synchronous heat flow to its fixed point, recording per sweep:

    * the Tarski-Dirichlet energy, summed per (interface, region) on the
      shared interface alphabet -- zero exactly at a global section;
    * how many interfaces (and (interface, region) components) are closed;
    * the concrete disagreement the abstraction is tolerating, as tiles on
      which the two endpoints of an interface hold different beliefs.

The world sampler is `benchmark.sample_scenario`, so the instance is replayable
from its seed alone.

    exp/scale_example.py --agents 32 --seed 0

writes `scale_energy.pdf` (and `.png`) into `doc/ms4_report/figures/` and the
run's numbers into `exp/benchmarks/scale_example_a{agents}_s{seed}.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import sample_scenario
from agsheaf.gridworld import resolve_tile_labels, route_corridor
from agsheaf.measure import Alphabet, distance
from agsheaf.regions import parse_regions
from agsheaf.regionsheaf import _push_component, build_region_sheaf, sweep

# The demonstration's partition: six 4x4 rooms tiling the 12x8 arena, exactly
# the `regions:` block of `exp/worlds/primary.yaml`. The network scales; the world and
# its vocabulary stay the demonstration's.
GRID = (12, 8)
ROOMS = {
    "A": {"rect": [0, 4, 3, 7]}, "B": {"rect": [0, 0, 3, 3]},
    "C": {"rect": [4, 4, 7, 7]}, "D": {"rect": [4, 0, 7, 3]},
    "E": {"rect": [8, 4, 11, 7]}, "F": {"rect": [8, 0, 11, 3]},
}

# Figure surface and series, after `figures.py` -- but on plain white, for a
# printed page rather than a tinted card.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#c9c8c2"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": RULE, "axes.labelcolor": INK,
    "axes.facecolor": "#ffffff", "figure.facecolor": "#ffffff",
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 4,
})


def build_instance(agents: int, seed: int, knowledge: float, error: float,
                   extra_edges: int):
    """The sampled world and the demonstration sheaf over it, from one seed."""
    rng = np.random.default_rng(seed)
    scenario = sample_scenario(rng, grid_width_height=GRID, num_agents=agents,
                               knowledge=knowledge, error=error,
                               extra_edges=extra_edges)
    regions = parse_regions(ROOMS, GRID)

    # As in `run.py`: each agent knows its own assignment for certain, and the
    # corridors of the initial plans are the reliance of the initial mission
    # contracts. Other agents' start tiles are not marked unsafe -- this is an
    # information-flow experiment; nobody drives.
    beliefs = [dict(b) for b in scenario.beliefs]
    for i in range(agents):
        beliefs[i][(int(scenario.assignments[0, i]),
                    int(scenario.assignments[1, i]))] = "target"
    resolved = resolve_tile_labels(scenario.starts, GRID, scenario.assignments,
                                   label_beliefs=beliefs,
                                   mark_other_agents_unsafe=False)
    corridors = {
        i: route_corridor((int(scenario.starts[0, i]), int(scenario.starts[1, i])),
                          (int(scenario.assignments[0, i]), int(scenario.assignments[1, i])),
                          GRID, resolved[i])
        for i in range(agents)}

    sheaf = build_region_sheaf(beliefs, GRID, scenario.topology, regions,
                               corridors=corridors,
                               contracts_cfg={"reliance": True,
                                              "exclusive_claims": True})
    return scenario, regions, sheaf


def interface_alphabets(sheaf, regions):
    """One small `Alphabet` per (interface, region): the block the energy of
    that component is measured on. Built once; building enumerates."""
    alphabets = {}
    for u, v in sheaf.interfaces():
        for r in regions.names:
            variables = [var for var, _ in sheaf.graph["defs"][(u, v)][r]]
            alphabets[(u, v, r)] = Alphabet(variables)
    return alphabets


def measure_state(sheaf, regions, alphabets):
    """
    The flow's state, measured where the sheaf condition lives: both endpoints
    of every interface pushed by `R_!` onto the shared alphabet, per region.
    The same pushforwards decide closure (equality) and energy (distance), so
    the two numbers cannot drift apart.
    """
    energy = 0.0
    components_closed = 0
    interfaces_closed = 0
    concrete_beliefs = 0
    concrete_masks = 0
    for u, v in sheaf.interfaces():
        all_closed = True
        for r in regions.names:
            pu = _push_component(sheaf, u, v, r)
            pv = _push_component(sheaf, v, u, r)
            # Closure is semantic -- distance zero on the shared alphabet --
            # with AST equality as the fast path, so a canonicalisation quirk
            # can never report an agreeing interface as open.
            if pu == pv:
                components_closed += 1
                continue
            gap = distance(pu, pv, alphabets[(u, v, r)])
            if gap == 0.0:
                components_closed += 1
            else:
                all_closed = False
                energy += gap
        interfaces_closed += bool(all_closed)
        # Two concrete gauges per interface: tiles where the endpoints' held
        # beliefs (single-label pins) differ, and tiles where their possibility
        # masks differ. Exclusions travel through the interface, pins do not,
        # so the two can move differently under pure diffusion.
        bu, bv = sheaf.nodes[u]["beliefs"], sheaf.nodes[v]["beliefs"]
        concrete_beliefs += sum(1 for t in set(bu) | set(bv)
                                if bu.get(t) != bv.get(t))
        mu, mv = sheaf.nodes[u]["masks"], sheaf.nodes[v]["masks"]
        concrete_masks += sum(1 for t in set(mu) | set(mv)
                              if mu.get(t) != mv.get(t))
    return {"energy": energy, "components_closed": components_closed,
            "interfaces_closed": interfaces_closed,
            "concrete_disagreement": concrete_beliefs,
            "concrete_mask_disagreement": concrete_masks}


def run(args) -> dict:
    scenario, regions, sheaf = build_instance(args.agents, args.seed,
                                              args.knowledge, args.error,
                                              args.extra_edges)
    alphabets = interface_alphabets(sheaf, regions)
    edges = len(sheaf.interfaces())
    import networkx as nx
    diameter = nx.diameter(scenario.topology)

    rows = [dict(sweep=0, seconds=0.0, **measure_state(sheaf, regions, alphabets))]
    print(f"agents={args.agents} interfaces={edges} diameter={diameter} "
          f"components={edges * len(regions)}")
    print(f"sweep 0: {rows[0]}")

    for t in range(1, args.max_sweeps + 1):
        started = time.perf_counter()
        _, changed = sweep(sheaf)
        seconds = time.perf_counter() - started
        rows.append(dict(sweep=t, seconds=seconds,
                         **measure_state(sheaf, regions, alphabets)))
        print(f"sweep {t}: {rows[-1]}")
        if not changed:
            break

    section_sweep = next((row["sweep"] for row in rows
                          if row["interfaces_closed"] == edges), None)
    result = {
        "params": {"agents": args.agents, "seed": args.seed,
                   "knowledge": args.knowledge, "error": args.error,
                   "extra_edges": args.extra_edges, "grid": list(GRID),
                   "regions": sorted(ROOMS)},
        "interfaces": edges, "diameter": diameter,
        "components": edges * len(regions),
        "sweeps_to_fixed_point": rows[-1]["sweep"],
        "first_global_section": section_sweep,
        "seconds_per_sweep": {
            "mean": float(np.mean([r["seconds"] for r in rows[1:]])),
            "max": float(np.max([r["seconds"] for r in rows[1:]])),
        },
        "rows": rows,
    }
    return result


def draw(result: dict, out_dir: Path) -> Path:
    """Two panels, one instance: the energy of the flow, and the two levels of
    disagreement it separates."""
    rows = result["rows"]
    sweeps = [r["sweep"] for r in rows]
    energy = [r["energy"] for r in rows]
    closed = [r["interfaces_closed"] for r in rows]
    concrete = [r["concrete_mask_disagreement"] for r in rows]
    pins = [r["concrete_disagreement"] for r in rows]
    section = result["first_global_section"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 2.7))

    left.plot(sweeps, energy, color=SERIES[0], marker="o")
    # Symlog: the decay spans three decades before landing on *exactly* zero,
    # which a log axis cannot show and a linear axis hides in the first step.
    left.set_yscale("symlog", linthresh=0.003)
    left.set_ylim(bottom=-0.0002)
    left.set_xlabel("sweep")
    left.set_ylabel("Tarski–Dirichlet energy")
    left.set_title(f"{result['params']['agents']} agents, "
                   f"{result['interfaces']} interfaces, "
                   f"diameter {result['diameter']}")
    twin = left.twinx()
    twin.plot(sweeps, closed, color=SERIES[2])
    twin.set_ylabel("interfaces closed", color=SERIES[2])
    twin.tick_params(axis="y", labelcolor=SERIES[2])
    twin.spines.top.set_visible(False)
    twin.set_ylim(0, result["interfaces"] * 1.05)
    if section is not None:
        left.axvline(section, color=MUTED, lw=0.8, ls=":")
        left.annotate("global section", (section, 0.0015),
                      color=MUTED, fontsize=8, rotation=90,
                      ha="right", va="bottom", xytext=(-3, 0),
                      textcoords="offset points")

    right.plot(sweeps, energy, color=SERIES[0], marker="o")
    right.set_yscale("symlog", linthresh=0.003)
    right.set_ylim(bottom=-0.0002)
    right.set_xlabel("sweep")
    right.set_ylabel("abstract energy", color=SERIES[0])
    right.tick_params(axis="y", labelcolor=SERIES[0])
    right.set_title("abstract zero, concrete residue")
    rtwin = right.twinx()
    rtwin.plot(sweeps, concrete, color=SERIES[1])
    if pins != concrete:
        # Masks can tighten (exclusions travel) while pins cannot; when the
        # two gauges separate, show both. When they coincide, one line.
        rtwin.plot(sweeps, pins, color=SERIES[1], ls=":", lw=1.4)
        rtwin.annotate("masks", (sweeps[-1], concrete[-1]), color=SERIES[1],
                       fontsize=8, xytext=(-4, 5), textcoords="offset points",
                       ha="right")
        rtwin.annotate("held beliefs", (sweeps[-1], pins[-1]), color=SERIES[1],
                       fontsize=8, xytext=(-4, -11),
                       textcoords="offset points", ha="right")
    rtwin.set_ylabel("differing tiles, all interfaces", color=SERIES[1])
    rtwin.tick_params(axis="y", labelcolor=SERIES[1])
    rtwin.spines.top.set_visible(False)
    top = max(max(concrete), max(pins))
    rtwin.set_ylim(0, top * 1.1 if top else 1)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "scale_energy.pdf"
    fig.savefig(pdf)
    fig.savefig(out_dir / "scale_energy.png")
    plt.close(fig)
    return pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--knowledge", type=float, default=0.2)
    parser.add_argument("--error", type=float, default=0.1)
    parser.add_argument("--extra-edges", type=int, default=2)
    parser.add_argument("--max-sweeps", type=int, default=60)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "doc" / "ms4_report" / "figures")
    args = parser.parse_args()

    result = run(args)
    bench = Path(__file__).resolve().parent / "benchmarks"
    bench.mkdir(exist_ok=True)
    record = bench / f"scale_example_a{args.agents}_s{args.seed}.json"
    record.write_text(json.dumps(result, indent=2))
    pdf = draw(result, args.out)
    print(f"wrote {record}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
