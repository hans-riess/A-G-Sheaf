"""
Validates the convergence theory of the contract flow, from many initial contract assignments.

The test suite checks each theorem on hand-built fixture sheaves; the benchmark measures whether
communication *helps*. Neither answers the question a milestone reviewer actually asks: given the
sheaf the demonstration runs on, and an arbitrary starting assignment of contracts, does the flow
do what the theory says it does? This script answers it by sampling initial assignments and
checking every claim against each one.

Per trial, one belief sheaf is built and the flow is run under three schedules:

    decentralized   every agent broadcasts every round and all agents update from the same
                    snapshot -- lockstep, no agent seeing another's new value within a round
                    (`in_place=False`, `firing=None`). This is Algorithm 1 of the report.
    round-robin     one agent broadcasts per round, cycling: the maximally asynchronous live
                    schedule.
    random          each agent broadcasts independently with probability 1/2 per round, which is
                    live almost surely.

and five claims are checked:

    section         the limit is a global section (Hodge-Tarski)
    agree           all three schedules reach the *same* limit (schedule independence)
    monotone        every agent's final contract refines its initial one
    closed form     the limit equals the pointwise intersection of every agent's initial
                    possibility masks over the connected component -- the closed form this
                    sheaf's limit must take, computed independently of the flow
    no collapse     no stalk reached a lattice extreme; disagreement stayed in the lattice

The grid is kept small because the flow is run WITHOUT the mask-canonical re-encoding that
`gridsheaf.sweep` applies -- these are the library's own `ContractSheaf` primitives operating on
contracts directly, so that what is validated is the operator rather than the flattening.

Usage:

    python exp/validate_theory.py --trials 20
    python exp/validate_theory.py --trials 20 --topology path --agents 4
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np

from agsheaf.gridsheaf import (FULL_MASK, LABELS, build_belief_sheaf, decode_masks,
                               mask_labels)
from agsheaf.sheaf import random_firing, round_robin

#: Topologies to validate over, by name. Each takes the agent count and returns a graph whose
#: nodes are exactly range(N), which `build_belief_sheaf` requires.
TOPOLOGIES = {
    "path": nx.path_graph,
    "cycle": nx.cycle_graph,
    "star": lambda n: nx.star_graph(n - 1),
    "complete": nx.complete_graph,
}


def sample_beliefs(rng, tiles, num_agents, knowledge=0.35):
    """
    A random initial contract assignment, as one partial label map per agent.

    Each agent is given a belief about each tile with probability `knowledge`, drawn uniformly
    from the labels. Beliefs are NOT drawn from a shared ground truth: agents contradict each
    other freely, which is the case the flow has to survive and the one where the closed form
    below is interesting (a contested tile is an empty possibility set, not a failure).
    """
    beliefs = []
    for _ in range(num_agents):
        held = {}
        for tile in tiles:
            if rng.random() < knowledge:
                held[tile] = LABELS[int(rng.integers(len(LABELS)))]
        beliefs.append(held)
    return beliefs


def predicted_limit(beliefs, tiles):
    """
    The limit the flow must reach on a connected topology, computed without running it.

    Every restriction in this sheaf is the graph of a bijection, so transport is a semantic
    renaming and the Laplacian's meet is pointwise intersection of the neighbours' possibility
    masks. On a connected graph the flow therefore fuses every agent's knowledge into every
    stalk, and the fixed point is the same at all agents: the pointwise intersection of the
    initial masks. A tile some pair disagrees about intersects to the empty set.
    """
    limit = {}
    for tile in tiles:
        mask = FULL_MASK
        for held in beliefs:
            label = held.get(tile)
            if label is not None:
                mask &= 1 << LABELS.index(label)
        if mask != FULL_MASK:
            limit[tile] = mask
    return limit


def stalk_masks(sheaf, agent, tiles):
    """Agent `agent`'s current contract, decoded back to possibility masks."""
    return decode_masks(sheaf.contract(agent), sheaf.nodes[agent]['v'], tiles=tiles)


def run_trial(rng, grid_width_height, num_agents, topology_name, max_sweeps=64):
    """
    One initial assignment, run under all three schedules, with every claim checked.

    Returns a dict of booleans (the claims) plus the sweep counts, so a failure names which claim
    failed on which trial rather than only that something did.
    """
    width, height = grid_width_height
    tiles = [(x, y) for x in range(width) for y in range(height)]
    topology = TOPOLOGIES[topology_name](num_agents)
    beliefs = sample_beliefs(rng, tiles, num_agents)
    nodes = sorted(topology.nodes())

    def fresh():
        # build_belief_sheaf mutates the belief dicts it is handed, so each schedule gets its own
        import copy
        return build_belief_sheaf(copy.deepcopy(beliefs), grid_width_height, topology)

    initial = {i: dict(m) for i, m in
               ((i, stalk_masks(fresh(), i, tiles)) for i in nodes)}

    runs = {}
    sweeps = {}

    F = fresh()
    sweeps["decentralized"] = F.converge(in_place=False, max_sweeps=max_sweeps)
    runs["decentralized"] = F

    F = fresh()
    sweeps["round_robin"] = F.converge(firing_sequence=lambda: round_robin(nodes),
                                       max_sweeps=max_sweeps)
    runs["round_robin"] = F

    F = fresh()
    sweeps["random"] = F.converge(firing_sequence=lambda: random_firing(nodes, rng_shim(rng)),
                                  max_sweeps=max_sweeps)
    runs["random"] = F

    predicted = predicted_limit(beliefs, tiles)
    reference = runs["decentralized"]

    claims = {
        # Hodge-Tarski: the limit of every run is a global section
        "section": all(F.is_section() for F in runs.values()),
        # Schedule independence: all three limits agree, contract by contract, decided by z3
        "agree": all(reference.contract(i) == F.contract(i)
                     for name, F in runs.items() if name != "decentralized"
                     for i in nodes),
        # Monotonicity: knowledge only strengthens
        "monotone": all(F.contract(i).refines(F.initial(i))
                        for F in runs.values() for i in nodes),
        # The closed form, computed without running the flow
        "closed_form": all(stalk_masks(reference, i, tiles) == predicted for i in nodes),
        # Disagreement stayed in the lattice
        "no_collapse": all(not F.collapsed() for F in runs.values()),
    }

    contested = sum(1 for mask in predicted.values() if mask == 0)
    return {
        "claims": claims,
        "sweeps": sweeps,
        "diameter": nx.diameter(topology),
        "contested_tiles": contested,
        "constrained_tiles": len(predicted),
        "initial_held": sum(len(m) for m in initial.values()),
    }


class rng_shim:
    """`random_firing` wants an object with `.random()`; numpy Generators spell it `.random()`
    too, but returning a numpy float trips nothing here -- this exists only to make the
    dependency explicit."""
    def __init__(self, rng):
        self._rng = rng

    def random(self):
        return float(self._rng.random())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=20, help="initial assignments per topology")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--grid", type=str, default="4x3")
    parser.add_argument("--topologies", type=str, default="path,cycle,star,complete")
    parser.add_argument("--out", type=str, default="benchmarks")
    args = parser.parse_args()

    width, height = (int(part) for part in args.grid.lower().split("x"))
    topologies = [name.strip() for name in args.topologies.split(",")]

    started = time.perf_counter()
    cells = []
    for topology_name in topologies:
        trials = []
        for trial in range(args.trials):
            rng = np.random.default_rng(args.seed + trial)
            trials.append(run_trial(rng, (width, height), args.agents, topology_name))

        claim_names = list(trials[0]["claims"])
        summary = {name: sum(t["claims"][name] for t in trials) for name in claim_names}
        diameter = trials[0]["diameter"]
        sync_sweeps = [t["sweeps"]["decentralized"] for t in trials]
        cells.append({
            "topology": topology_name,
            "agents": args.agents,
            "diameter": diameter,
            "trials": len(trials),
            "passed": summary,
            "all_passed": all(count == len(trials) for count in summary.values()),
            "sweeps": {name: float(np.mean([t["sweeps"][name] for t in trials]))
                       for name in trials[0]["sweeps"]},
            "max_sync_sweeps": max(sync_sweeps),
            "within_diameter_bound": max(sync_sweeps) <= diameter + 1,
            "mean_contested_tiles": float(np.mean([t["contested_tiles"] for t in trials])),
        })

        cell = cells[-1]
        print("%-9s N=%d diam=%d  %d trials" % (topology_name, args.agents, diameter, len(trials)))
        for name in claim_names:
            print("    %-12s %d/%d" % (name, cell["passed"][name], len(trials)))
        print("    sweeps: decentralized %.2f, round-robin %.2f, random %.2f (bound %d, max %d)"
              % (cell["sweeps"]["decentralized"], cell["sweeps"]["round_robin"],
                 cell["sweeps"]["random"], diameter + 1, cell["max_sync_sweeps"]))
        print("    contested tiles per trial: %.1f" % cell["mean_contested_tiles"])
        print()

    record = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "settings": vars(args),
        "cells": cells,
        "all_passed": all(cell["all_passed"] for cell in cells),
    }

    out_dir = (Path(__file__).parent / args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("validation_%s.json" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_path.write_text(json.dumps(record, indent=2))
    print("all claims passed: %s" % record["all_passed"])
    print("Wrote %s (%.1fs)" % (out_path, record["elapsed_seconds"]))


if __name__ == "__main__":
    main()
