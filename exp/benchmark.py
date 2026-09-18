"""
Benchmarks the belief sheaf against no communication at all, over many worlds.

`sheaf_grid.py` and `greedy_grid.py` differ in exactly one thing -- whether a sweep of the
Tarski Laplacian runs before each grid step -- but they run one hand-written world, animate
it through the Robotarium and write videos, which answers "does it work here" rather than
"does it help". This runs the same two policies headlessly over randomly sampled worlds, so
the comparison is paired (identical world, identical starts, identical beliefs, one
difference) and repeatable.

The grid-level dynamics here are the experiments' own, imported rather than reimplemented:
`resolve_tile_labels` -> `choose_belief_aware_step` -> `score_robot_step`, with
`gridsheaf.communicate` in front of it under the sheaf policy. What is dropped is only the
physical layer -- the Robotarium integration, the barrier certificates, the video -- which
sits strictly below the tile abstraction and cannot change which tile a robot ends a step on.
`--replay` checks exactly that claim, by replaying the configured experiment here and
comparing against the log an animated run left behind.

Three policies are run on each sampled world:

    isolated     what greedy_grid.py does: every agent plans on its own beliefs forever.
    sheaf        what sheaf_grid.py does: one Laplacian sweep per grid step until the flow
                 settles, agents planning on the fused beliefs.
    omniscient   every agent given the ground truth outright. Not a policy anyone could run
                 -- it is the ceiling communication is trying to reach, and what makes a
                 score difference readable as a fraction of what was there to be had.

Usage:

    python exp/benchmark.py --episodes 30
    python exp/benchmark.py --episodes 20 --knowledge 0.1,0.3 --error 0.0,0.2
    python exp/benchmark.py --replay
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import time
from pathlib import Path
from typing import NamedTuple, Optional

import networkx as nx
import numpy as np

from agsheaf.gridsheaf import (build_belief_sheaf, communicate, forced_empty, observe_tiles,
                               parse_sweeps_per_step, reliance_contract, sheaf_state)
from agsheaf.gridworld import (GROUND_TRUTH_LABELS, choose_belief_aware_step,
                               generate_random_grid_coords, parse_assignments,
                               parse_communication_topology, parse_ground_truth,
                               parse_label_beliefs, parse_start_tiles, resolve_tile_labels,
                               route_corridor, score_robot_step)
from agsheaf.utils import load_experiment_config, run_output_path

#: The policies compared, in the order results are reported. "isolated" and "sheaf" are
#: greedy_grid.py and sheaf_grid.py; "omniscient" is the ceiling, not a runnable policy.
POLICIES = ("isolated", "sheaf", "omniscient")

#: Steps of nobody moving, at the end of a run that hit its cap, taken as a standoff rather
#: than as slow progress. Nothing in the planner is stochastic, so a state in which no agent
#: moves reproduces itself forever; a handful of steps is enough to tell it from a queue.
DEADLOCK_IDLE_STEPS = 5

#: What an agent believes about a tile it has been told nothing about, and whether the other
#: agents' current squares count as obstacles. These mirror exp/defaults.yaml, since the
#: point of the exercise is to benchmark the experiments rather than some near neighbour.
BELIEF_OPTIONS = dict(default_label="unknown", mark_other_agents_unsafe=False)


class Scenario(NamedTuple):
    """
    One world, fixed down to the starting squares, so that every policy is run on exactly
    the same problem and a difference in outcome can only come from the policy.

    `beliefs` is what the agents were *given*; an episode copies it before running, since
    the sheaf's sweeps fuse knowledge into these dicts in place.
    """
    grid_width_height: tuple[int, int]
    starts: np.ndarray
    assignments: np.ndarray
    ground_truth: dict[tuple[int, int], str]
    beliefs: list[dict[tuple[int, int], str]]
    topology: nx.Graph


def sample_scenario(
    rng: np.random.Generator,
    grid_width_height: tuple[int, int] = (9, 5),
    num_agents: int = 4,
    unsafe_fraction: float = 0.2,
    knowledge: float = 0.2,
    error: float = 0.0,
    extra_edges: int = 1
) -> Scenario:
    """
    Samples a world, an assignment, a partial and possibly wrong belief per agent, and a
    connected communication topology.

    The hazards are scattered rather than laid out as a wall, so no single agent's knowledge
    is decisive and what communication buys is the union of several partial views. Each agent
    is told about a `knowledge` fraction of the tiles, and a sampled belief is wrong with
    probability `error` -- wrong being a label drawn from the other two, so an error is a
    definite false claim rather than a shrug. That is the case the fusion has to survive: a
    confident neighbour that is wrong.

    The topology is a uniform spanning tree plus `extra_edges` chords, which keeps it
    connected -- knowledge that cannot reach an agent says nothing about whether sharing
    helps -- while varying the diameter, and so how many sweeps the flow needs.

    Args:
        rng (np.random.Generator): The generator to sample from, seeded by the caller.
        grid_width_height (tuple[int, int]): The width and height of the grid.
        num_agents (int): The number of agents.
        unsafe_fraction (float): Fraction of non-target tiles that are truly unsafe.
        knowledge (float): Fraction of tiles each agent is given a belief about.
        error (float): Probability that a given belief is a definite false label.
        extra_edges (int): Chords added to the spanning tree of the topology.
    """
    grid_width, grid_height = grid_width_height
    tiles = [(x, y) for x in range(grid_width) for y in range(grid_height)]

    # The assigned squares are targets in the world, which assert_assignments_are_targets
    # requires and scoring assumes, so they are drawn first and never made unsafe
    target_indices = rng.choice(len(tiles), size=num_agents, replace=False)
    targets = [tiles[index] for index in target_indices]

    remaining = [tile for tile in tiles if tile not in set(targets)]
    unsafe_count = int(round(unsafe_fraction * len(remaining)))
    unsafe_indices = rng.choice(len(remaining), size=unsafe_count, replace=False)
    unsafe = {remaining[index] for index in unsafe_indices}

    ground_truth = {tile: "unsafe" if tile in unsafe else "safe" for tile in tiles}
    for tile in targets:
        ground_truth[tile] = "target"

    # A belief is held about a tile with probability `knowledge`, and is then the truth
    # unless this is one of the `error` cases, in which case it is one of the other labels
    beliefs: list[dict[tuple[int, int], str]] = []
    for _ in range(num_agents):
        held = {}
        for tile in tiles:
            if rng.random() >= knowledge:
                continue
            truth = ground_truth[tile]
            if rng.random() < error:
                wrong = [label for label in GROUND_TRUTH_LABELS if label != truth]
                held[tile] = wrong[int(rng.integers(len(wrong)))]
            else:
                held[tile] = truth
        beliefs.append(held)

    topology = nx.random_labeled_tree(num_agents, seed=int(rng.integers(2**31)))
    non_edges = [pair for pair in nx.non_edges(topology)]
    for index in rng.choice(len(non_edges), size=min(extra_edges, len(non_edges)), replace=False):
        topology.add_edge(*non_edges[index])

    starts = generate_random_grid_coords(grid_width_height, num_agents, rng=rng)

    return Scenario(grid_width_height=grid_width_height,
                    starts=starts,
                    assignments=np.array([[tile[0] for tile in targets],
                                          [tile[1] for tile in targets]]),
                    ground_truth=ground_truth,
                    beliefs=beliefs,
                    topology=topology)


def belief_quality(beliefs: list[dict], ground_truth: dict) -> dict:
    """
    How much each agent holds and how much of it is true, averaged over agents.

    Coverage is the fraction of tiles an agent has any belief about and accuracy the
    fraction of those that match the world. Communication moves both: coverage up as
    knowledge arrives, accuracy up or down depending on whether what arrived was true. This
    is the measurement upstream of the score -- a policy can only route better if what it
    believes got better.

    Args:
        beliefs (list[dict]): Per-agent belief dicts, as the sheaf leaves them.
        ground_truth (dict): The world's true labeling.
    """
    tile_count = len(ground_truth)
    coverage, accuracy = [], []
    for held in beliefs:
        coverage.append(len(held) / tile_count)
        if held:
            accuracy.append(sum(1 for tile, label in held.items()
                                if ground_truth[tile] == label) / len(held))
    return {"coverage": float(np.mean(coverage)),
            "accuracy": float(np.mean(accuracy)) if accuracy else float("nan")}


def run_episode(
    scenario: Scenario,
    policy: str,
    max_steps: int = 60,
    sweeps_per_step: int = 1,
    legs: str = "kan",
    share_assignments: bool = True,
    observe_on_arrival: bool = False,
    mission_monitor: bool = True,
    tile_costs: Optional[dict] = None,
    tile_scores: Optional[dict] = None
) -> dict:
    """
    Runs one policy on one scenario, returning what it did.

    The loop is the experiments' loop: resolve each agent's labels, choose a belief-aware
    step, score it against the truth. Under "sheaf" a sweep of the Laplacian runs first,
    fusing each agent's beliefs with its neighbours' in place, and stops being run once the
    flow reports a fixed point, which the meet's monotonicity makes permanent.

    Args:
        scenario (Scenario): The world to run.
        policy (str): One of POLICIES.
        max_steps (int): Cap on grid steps, so an agent that cannot get through still ends.
        sweeps_per_step (int): Communication hops per grid step, under the sheaf policy.
        legs (str): Which adjoint bisheaf the section read-out uses, "kan" or "co".
        share_assignments (bool): Whether an agent seeds its own assigned square as a
            shareable target belief, as sheaf_grid.py does.
        observe_on_arrival (bool): Whether an agent learns the true label of the square it
            steps onto, as beliefs.observe_on_arrival makes it. Off for every sampled world,
            so the statistics measure the flow alone; --replay turns it on when the run being
            reproduced had it on, since otherwise the trajectories cannot match.
        mission_monitor (bool): Whether to check each agent's route reliance against its
            fused stalk. Diagnostic only -- it never changes a step.
        tile_costs (Optional[dict]): Cost of entering a tile, by believed label.
        tile_scores (Optional[dict]): Points for entering a tile, by true label.
    """
    assert policy in POLICIES, "Unknown policy %r; expected one of %s." % (policy, str(POLICIES))

    grid_width_height = scenario.grid_width_height
    number_of_agents = scenario.assignments.shape[1]
    ground_truth = scenario.ground_truth

    if policy == "omniscient":
        beliefs = [dict(ground_truth) for _ in range(number_of_agents)]
    else:
        beliefs = copy.deepcopy(scenario.beliefs)

    initial_quality = belief_quality(beliefs, ground_truth)

    belief_sheaf = None
    if policy == "sheaf":
        if share_assignments:
            for i in range(number_of_agents):
                beliefs[i][(int(scenario.assignments[0, i]),
                            int(scenario.assignments[1, i]))] = "target"
        if scenario.topology.number_of_edges() > 0:
            belief_sheaf = build_belief_sheaf(beliefs, grid_width_height, scenario.topology)

    belief_options = dict(label_beliefs=beliefs, **BELIEF_OPTIONS)

    coords = scenario.starts.copy()
    scores = np.zeros(number_of_agents)
    reached_target = np.zeros(number_of_agents, dtype=bool)

    settled = belief_sheaf is None
    settled_at_step = None
    rounds_run = 0   # grid steps that communicated at all
    sweeps_run = 0   # applications of the Laplacian, which is what a hop is
    conflict_count = 0
    communication_seconds = 0.0

    def count_sweep(*_):
        nonlocal sweeps_run
        sweeps_run += 1

    flagged = 0             # agent-steps whose corridor the fused knowledge rules out
    flagged_then_unsafe = 0 # ... of which the very next square really was unsafe
    contested_count = 0

    unsafe_entries = 0
    moves = 0
    step_count = 0
    positions = []

    # The step each agent first stood on its assigned square, or None if it never did. An
    # agent that has arrived is skipped by the planner from then on, so this is also where it
    # stayed. Scoring rewards a step onto a safe tile, so a run that stalls goes on earning:
    # the score of a whole run is only comparable between two runs that both ended by
    # arriving, and these are what say whether they did.
    arrival_steps: list[Optional[int]] = [None] * number_of_agents

    # Consecutive steps ending with nobody having moved. The planner has no protocol for two
    # agents wanting each other's square -- each holds position waiting for the other, which
    # is worth less than any move only while the other is there -- so a run can end in a
    # standoff no belief about the terrain can break. That is a property of the planner, not
    # of what the agents know, but knowing more can route two agents into the same corridor
    # and so provoke it, which is why it is measured rather than assumed away.
    idle_tail = 0

    while np.linalg.norm(coords - scenario.assignments) > 0 and step_count < max_steps:
        if not settled:
            started = time.perf_counter()
            conflicts, changed = communicate(belief_sheaf, sweeps=sweeps_per_step,
                                             on_sweep=count_sweep)
            communication_seconds += time.perf_counter() - started
            rounds_run += 1
            conflict_count += len(conflicts)
            settled = not changed
            if settled:
                settled_at_step = step_count

        tile_labels = resolve_tile_labels(coords, grid_width_height, scenario.assignments,
                                          **belief_options)
        steps = choose_belief_aware_step(coords, scenario.assignments, grid_width_height,
                                         tile_labels, tile_costs=tile_costs)

        if belief_sheaf is not None and mission_monitor:
            for i in range(number_of_agents):
                current_tile = (int(coords[0, i]), int(coords[1, i]))
                goal_tile = (int(scenario.assignments[0, i]), int(scenario.assignments[1, i]))
                if current_tile == goal_tile:
                    continue
                corridor = route_corridor(current_tile, goal_tile, grid_width_height,
                                          tile_labels[i], tile_costs=tile_costs)
                stalk_vars = belief_sheaf.nodes[i]['v']
                viability = belief_sheaf.contract(i).meet(reliance_contract(stalk_vars, corridor))
                contested = forced_empty(belief_sheaf.contract(i), stalk_vars, corridor)
                broken = [tile for tile in forced_empty(viability, stalk_vars, corridor)
                          if tile not in contested]
                if broken:
                    flagged += 1
                    next_tile = (int(steps[0, i]), int(steps[1, i]))
                    if ground_truth[next_tile] == "unsafe":
                        flagged_then_unsafe += 1

        step_scores = score_robot_step(steps, ground_truth, reached_target,
                                       tile_scores=tile_scores)
        scores += step_scores

        moved_this_step = 0
        for i in range(number_of_agents):
            tile = (int(steps[0, i]), int(steps[1, i]))
            if ground_truth[tile] == "unsafe":
                unsafe_entries += 1
            if tile != (int(coords[0, i]), int(coords[1, i])):
                moves += 1
                moved_this_step += 1
            if arrival_steps[i] is None and tile == (int(scenario.assignments[0, i]),
                                                     int(scenario.assignments[1, i])):
                arrival_steps[i] = step_count + 1
        idle_tail = 0 if moved_this_step else idle_tail + 1

        coords = steps

        # Each agent sees the square it has just stepped onto. What it sees for itself
        # overrides what it was told, so this reopens a settled flow -- exactly as it does in
        # the experiment drivers, which is what keeps a --replay faithful.
        if observe_on_arrival:
            observed = observe_tiles(
                beliefs,
                [(int(coords[0, i]), int(coords[1, i])) for i in range(number_of_agents)],
                ground_truth,
                sheaf=belief_sheaf)
            if any(record["changed"] for record in observed):
                settled = belief_sheaf is None

        positions.append([[int(coords[0, i]), int(coords[1, i])] for i in range(number_of_agents)])
        step_count += 1

    final_state = None
    if belief_sheaf is not None:
        state = sheaf_state(belief_sheaf, legs=legs)
        contested_count = sum(len(tiles) for tiles in state.contested.values())
        final_state = {
            "is_section": bool(state.is_section),
            "interfaces_agreeing": sum(1 for agrees in state.sections.values() if agrees),
            "interfaces": len(state.sections),
            "refines_initial": all(state.refines_initial.values()),
            "collapsed": state.collapsed,
        }

    arrived = [step for step in arrival_steps if step is not None]
    return {
        "policy": policy,
        "total_score": float(scores.sum()),
        "steps": step_count,
        "all_arrived": bool(np.linalg.norm(coords - scenario.assignments) == 0),
        "deadlocked": bool(step_count >= max_steps and idle_tail >= DEADLOCK_IDLE_STEPS),
        "agents_arrived": len(arrived) / number_of_agents,
        "mean_arrival_step": float(np.mean(arrived)) if arrived else float("nan"),
        "unsafe_entries": unsafe_entries,
        "unsafe_rate": unsafe_entries / max(1, number_of_agents * step_count),
        "moves": moves,
        "initial_belief": initial_quality,
        "final_belief": belief_quality(beliefs, ground_truth),
        "sweeps": sweeps_run,
        "communication_rounds": rounds_run,
        "settled_at_step": settled_at_step,
        "conflicts": conflict_count,
        "contested_tiles": contested_count,
        "communication_seconds": round(communication_seconds, 4),
        "monitor_flagged": flagged,
        "monitor_flagged_then_unsafe": flagged_then_unsafe,
        "sheaf": final_state,
        "positions": positions,
    }


def run_cell(
    episodes: int,
    seed: int,
    knowledge: float,
    error: float,
    **scenario_kwargs
) -> dict:
    """
    Runs every policy over `episodes` sampled worlds at one (knowledge, error) setting.

    Each episode's seed is derived from the cell's, so the worlds are the same across
    policies and reproducible across runs, and the comparison is paired: any difference in
    outcome comes from the policy and nothing else.

    Args:
        episodes (int): How many worlds to sample.
        seed (int): The base seed; episode k uses seed + k.
        knowledge (float): Fraction of tiles each agent is given a belief about.
        error (float): Probability that a given belief is a definite false label.
        **scenario_kwargs: Passed through to `sample_scenario` and `run_episode`.
    """
    episode_kwargs = {key: scenario_kwargs.pop(key)
                      for key in ("max_steps", "sweeps_per_step", "legs", "mission_monitor")
                      if key in scenario_kwargs}

    results = {policy: [] for policy in POLICIES}
    diameters = []
    for episode in range(episodes):
        rng = np.random.default_rng(seed + episode)
        scenario = sample_scenario(rng, knowledge=knowledge, error=error, **scenario_kwargs)
        diameters.append(nx.diameter(scenario.topology))
        for policy in POLICIES:
            outcome = run_episode(scenario, policy, **episode_kwargs)
            outcome.pop("positions")
            results[policy].append(outcome)

    return {"knowledge": knowledge, "error": error, "episodes": episodes, "seed": seed,
            "mean_topology_diameter": float(np.mean(diameters)),
            "episode_results": results,
            "summary": summarize(results)}


def summarize(results: dict) -> dict:
    """
    Aggregates one cell: per-policy means, and the paired sheaf-against-isolated comparison.

    Two footings are reported, because the run's own score is not comparable between runs of
    different lengths. Scoring pays a point for every step onto a safe tile, so a run that
    stalls short of its goals goes on earning: a policy can outscore another by wandering
    longer. `all` is every sampled world; `resolved` keeps only those in which all three
    policies got every agent home, where the score means what it is meant to mean. The
    length-free measurements -- how many agents arrived, how soon, and how often an agent
    stepped onto a truly unsafe tile -- are reported over every world regardless.

    Within a footing the comparison is given three ways. The mean difference says how much
    sharing was worth; the win/loss/tie split says how often it was worth anything, which a
    mean over a heavy-tailed difference hides; and gap closure says how much of the distance
    to omniscience it covered, the only one comparable across settings where the ceiling
    itself moves.

    Args:
        results (dict): Per-policy lists of episode outcomes.
    """
    def mean(policy, key):
        return float(np.nanmean([outcome[key] for outcome in results[policy]]))

    episodes = list(zip(results["sheaf"], results["isolated"], results["omniscient"]))
    resolved = [triple for triple in episodes if all(outcome["all_arrived"] for outcome in triple)]

    summary = {
        policy: {
            "score": mean(policy, "total_score"),
            "steps": mean(policy, "steps"),
            "agents_arrived": mean(policy, "agents_arrived"),
            "arrival_step": mean(policy, "mean_arrival_step"),
            "unsafe_entries": mean(policy, "unsafe_entries"),
            "unsafe_rate": mean(policy, "unsafe_rate"),
            "all_arrived": mean(policy, "all_arrived"),
            "deadlocked": mean(policy, "deadlocked"),
            "coverage": float(np.mean([outcome["final_belief"]["coverage"]
                                       for outcome in results[policy]])),
            "accuracy": float(np.mean([outcome["final_belief"]["accuracy"]
                                       for outcome in results[policy]])),
        }
        for policy in POLICIES
    }

    summary["sheaf"].update({
        "sweeps": mean("sheaf", "sweeps"),
        "communication_rounds": mean("sheaf", "communication_rounds"),
        "conflicts": mean("sheaf", "conflicts"),
        "contested_tiles": mean("sheaf", "contested_tiles"),
        "communication_seconds": mean("sheaf", "communication_seconds"),
        "sections_reached": float(np.mean([outcome["sheaf"]["is_section"]
                                           for outcome in results["sheaf"]
                                           if outcome["sheaf"] is not None] or [float("nan")])),
        "monitor_flagged": mean("sheaf", "monitor_flagged"),
        "monitor_flagged_then_unsafe": mean("sheaf", "monitor_flagged_then_unsafe"),
    })

    summary["sheaf_vs_isolated"] = {
        "all": _compare(episodes, "total_score"),
        "resolved": _compare(resolved, "total_score"),
        "resolved_episodes": len(resolved),
        # Entries are counted per agent-step as well as raw: a run that stalls keeps stepping,
        # so a raw count says as much about how long a run went on as about how safely it went
        "unsafe_rate": _compare(episodes, "unsafe_rate"),
        "unsafe_entries": _compare(episodes, "unsafe_entries"),
        "agents_arrived": _compare(episodes, "agents_arrived"),
    }
    return summary


def _compare(episodes: list, key: str) -> dict:
    """
    The paired sheaf-against-isolated comparison of one measurement, with omniscience as the
    ceiling it is read against.

    Args:
        episodes (list): (sheaf, isolated, omniscient) outcome triples for the same worlds.
        key (str): The measurement to compare.
    """
    if not episodes:
        return {"episodes": 0}

    differences = [sheaf[key] - isolated[key] for sheaf, isolated, _ in episodes]
    headroom = [omniscient[key] - isolated[key] for _, isolated, omniscient in episodes]

    # Only worlds where the ceiling is away from the floor say anything about how much of the
    # distance was covered; one the isolated policy already plays perfectly does not. Rooms of
    # both signs can still cancel to nothing across a cell, and a ratio to that is no number
    total_headroom = float(np.sum([room for room in headroom if room != 0]))
    covered = float(np.sum([difference for difference, room in zip(differences, headroom) if room != 0]))

    return {
        "episodes": len(episodes),
        "sheaf": float(np.mean([sheaf[key] for sheaf, _, _ in episodes])),
        "isolated": float(np.mean([isolated[key] for _, isolated, _ in episodes])),
        "omniscient": float(np.mean([omniscient[key] for _, _, omniscient in episodes])),
        "mean_difference": float(np.mean(differences)),
        "std_difference": float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0,
        "wins": sum(1 for difference in differences if difference > 0),
        "losses": sum(1 for difference in differences if difference < 0),
        "ties": sum(1 for difference in differences if difference == 0),
        "wilcoxon_p": _wilcoxon(differences),
        "gap_closure": covered / total_headroom if total_headroom else float("nan"),
        "headroom": float(np.mean(headroom)),
    }


def _wilcoxon(differences: list[float]) -> Optional[float]:
    """
    The two-sided Wilcoxon signed-rank p-value of the paired differences, or None when
    scipy is unavailable or every pair tied (the test has nothing to rank).
    """
    if not any(differences):
        return None
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    return float(wilcoxon(differences).pvalue)


def scenario_from_config(config_path: Path, defaults_path: Path) -> tuple[Scenario, dict]:
    """
    The configured experiment as a `Scenario`, so that the world exp/worlds/primary.yaml describes
    can be run here and checked against what an animated run of it recorded.

    Args:
        config_path (Path): The experiment configuration.
        defaults_path (Path): The general defaults it sits on top of.
    """
    config = load_experiment_config(config_path, defaults_path)
    gridworld_config = config["gridworld"]
    grid_width_height = (gridworld_config["grid_width"], gridworld_config["grid_height"])
    number_of_agents = gridworld_config["num_agents"]

    # Resolved as the drivers resolve it -- named by the experiment file, or sampled from the
    # seed -- since a replay that started the agents anywhere else could not match the run
    rng = np.random.default_rng(seed=config["seed"])
    starts = parse_start_tiles(config.get("starts"), number_of_agents, grid_width_height, rng=rng)

    return Scenario(
        grid_width_height=grid_width_height,
        starts=starts,
        assignments=parse_assignments(config.get("assignments"), number_of_agents, grid_width_height),
        ground_truth=parse_ground_truth(config.get("ground_truth"), grid_width_height),
        beliefs=parse_label_beliefs(config.get("beliefs") or {}, number_of_agents, grid_width_height),
        topology=parse_communication_topology(config.get("communication"), number_of_agents),
    ), config


def replay(experiments_dir: Path) -> dict:
    """
    Runs the configured experiment under both policies and checks each against the most
    recent log an animated run of that experiment wrote.

    This is what licenses everything else here: the benchmark drops the physical layer, and
    a step-for-step match against a run that kept it is the evidence that dropping it
    changes no outcome. A mismatch means the two have diverged and the numbers below are
    measuring something other than the experiments.

    Args:
        experiments_dir (Path): The directory holding config.yaml, defaults.yaml and logs/.
    """
    scenario, config = scenario_from_config(experiments_dir / "worlds" / "primary.yaml",
                                            experiments_dir / "defaults.yaml")
    sheaf_config = config.get("sheaf") or {}
    planning_config = config.get("planning") or {}
    scoring_config = config.get("scoring") or {}

    checks = {}
    for policy, experiment_name in (("isolated", "greedy_grid"), ("sheaf", "sheaf_grid")):
        outcome = run_episode(
            scenario, policy,
            max_steps=(config.get("goals") or {}).get("max_steps", 60),
            sweeps_per_step=parse_sweeps_per_step(sheaf_config.get("sweeps_per_step", 1),
                                                  sheaf_config.get("max_sweeps", 100)),
            legs=sheaf_config.get("legs", "kan"),
            share_assignments=sheaf_config.get("share_assignments", True),
            observe_on_arrival=(config.get("beliefs") or {}).get("observe_on_arrival", False),
            tile_costs=planning_config.get("tile_costs"),
            tile_scores=scoring_config.get("tile_scores"))

        logged = _latest_log(experiments_dir / "logs", experiment_name)
        check = {"policy": policy, "experiment": experiment_name,
                 "score": outcome["total_score"], "steps": outcome["steps"],
                 "log": None if logged is None else str(logged[0])}
        if logged is not None:
            path, record = logged
            recorded = [step["positions"] for step in record["steps"]]
            check.update({
                "logged_score": record["result"]["total_score"],
                "logged_steps": record["result"]["steps_taken"],
                "score_matches": record["result"]["total_score"] == outcome["total_score"],
                "positions_match": recorded == outcome["positions"],
            })
        checks[policy] = check
    return checks


def _latest_log(log_dir: Path, experiment_name: str):
    """
    The most recent run log written by `experiment_name`, as (path, record), or None. Logs
    are named by timestamp and carry the name of the script that wrote them, which is what
    tells one experiment's runs from the other's when both read the same configuration.
    """
    if not log_dir.is_dir():
        return None
    for path in sorted(log_dir.glob("*.json"), reverse=True):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if record.get("experiment", {}).get("name") == experiment_name and record.get("steps"):
            return path, record
    return None


def format_cell(cell: dict) -> str:
    """One cell of the sweep as a short table, the way it is read while running."""
    summary = cell["summary"]
    comparisons = summary["sheaf_vs_isolated"]
    lines = [
        "knowledge %.2f  error %.2f  (%d episodes, mean topology diameter %.2f)"
        % (cell["knowledge"], cell["error"], cell["episodes"], cell["mean_topology_diameter"]),
        "  %-11s %8s %8s %8s %8s %8s %8s %9s %9s"
        % ("policy", "score", "arrived", "arrival", "unsafe", "unsafe/s", "deadlock", "coverage", "accuracy"),
    ]
    for policy in POLICIES:
        row = summary[policy]
        lines.append("  %-11s %8.2f %8.2f %8.2f %8.2f %8.3f %8.2f %9.2f %9.2f"
                     % (policy, row["score"], row["agents_arrived"], row["arrival_step"],
                        row["unsafe_entries"], row["unsafe_rate"], row["deadlocked"],
                        row["coverage"], row["accuracy"]))

    for name, key in (("score (all worlds)", "all"),
                      ("score (both resolved)", "resolved"),
                      ("unsafe per agent-step", "unsafe_rate"),
                      ("unsafe entries", "unsafe_entries"),
                      ("agents arrived", "agents_arrived")):
        comparison = comparisons[key]
        if not comparison.get("episodes"):
            lines.append("  %-22s no episodes" % name)
            continue
        lines.append("  %-22s %+7.3f +- %5.3f  (%dW/%dL/%dT of %d, p=%s)  gap closed %s of %+.3f"
                     % (name, comparison["mean_difference"], comparison["std_difference"],
                        comparison["wins"], comparison["losses"], comparison["ties"],
                        comparison["episodes"],
                        "n/a" if comparison["wilcoxon_p"] is None else "%.4f" % comparison["wilcoxon_p"],
                        "n/a" if np.isnan(comparison["gap_closure"]) else "%.0f%%" % (100 * comparison["gap_closure"]),
                        comparison["headroom"]))

    lines.append("  flow: %.1f sweeps over %.1f rounds, %.1f conflicts, %.1f contested tiles, section reached in %.0f%% of runs, %.2fs of solving"
                 % (summary["sheaf"]["sweeps"], summary["sheaf"]["communication_rounds"],
                    summary["sheaf"]["conflicts"],
                    summary["sheaf"]["contested_tiles"], 100 * summary["sheaf"]["sections_reached"],
                    summary["sheaf"]["communication_seconds"]))
    lines.append("  monitor: %.2f corridors flagged per run, %.2f of them stepping onto a truly unsafe tile"
                 % (summary["sheaf"]["monitor_flagged"], summary["sheaf"]["monitor_flagged_then_unsafe"]))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=30, help="sampled worlds per cell")
    parser.add_argument("--seed", type=int, default=0, help="base seed; episode k uses seed + k")
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--grid", type=str, default="9x5")
    parser.add_argument("--knowledge", type=str, default="0.2",
                        help="comma-separated fractions of tiles each agent is told about")
    parser.add_argument("--error", type=str, default="0.0",
                        help="comma-separated probabilities that a belief is a false label")
    parser.add_argument("--unsafe-fraction", type=float, default=0.2)
    parser.add_argument("--extra-edges", type=int, default=1,
                        help="chords added to the topology's spanning tree")
    parser.add_argument("--sweeps-per-step", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--no-monitor", action="store_true",
                        help="skip the mission monitor, which is diagnostic and the slowest part")
    parser.add_argument("--out", type=str, default="benchmarks",
                        help="directory for the JSON result, relative to this file")
    parser.add_argument("--replay", action="store_true",
                        help="replay exp/worlds/primary.yaml here and check against the animated runs' logs")
    parser.add_argument("--summary", type=str, default=None,
                        help="reprint the tables of a saved result file instead of running anything")
    args = parser.parse_args()

    experiments_dir = Path(__file__).parent

    if args.summary:
        record = json.loads(Path(args.summary).read_text())
        print("%s, %ss, %s" % (record["generated"], record["elapsed_seconds"], record["settings"]))
        for cell in record["cells"]:
            print(format_cell(cell))
            print()
        return

    if args.replay:
        checks = replay(experiments_dir)
        for policy, check in checks.items():
            print("%s (%s): score %.1f in %d steps" % (policy, check["experiment"],
                                                       check["score"], check["steps"]))
            if check.get("log") is None:
                print("  no logged run to check against")
            else:
                print("  logged run %s: score %.1f in %d steps -- score %s, positions %s"
                      % (Path(check["log"]).name, check["logged_score"], check["logged_steps"],
                         "match" if check["score_matches"] else "DIFFER",
                         "match" if check["positions_match"] else "DIFFER"))
        return

    grid_width, grid_height = (int(part) for part in args.grid.lower().split("x"))
    knowledge_levels = [float(part) for part in args.knowledge.split(",")]
    error_levels = [float(part) for part in args.error.split(",")]

    started = time.perf_counter()
    cells = []
    for knowledge in knowledge_levels:
        for error in error_levels:
            cell = run_cell(args.episodes, args.seed, knowledge, error,
                            grid_width_height=(grid_width, grid_height),
                            num_agents=args.agents,
                            unsafe_fraction=args.unsafe_fraction,
                            extra_edges=args.extra_edges,
                            sweeps_per_step=args.sweeps_per_step,
                            max_steps=args.max_steps,
                            mission_monitor=not args.no_monitor)
            cells.append(cell)
            print(format_cell(cell))
            print()

    record = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "settings": vars(args),
        "policies": {
            "isolated": "greedy_grid.py: no communication",
            "sheaf": "sheaf_grid.py: %d Tarski Laplacian sweep(s) per grid step" % args.sweeps_per_step,
            "omniscient": "ground truth handed to every agent; the ceiling, not a policy",
        },
        "cells": cells,
    }

    out_dir = (experiments_dir / args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_output_path(out_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), "json")
    out_path.write_text(json.dumps(record, indent=2))
    print("Wrote %s (%.1fs)" % (out_path, record["elapsed_seconds"]))


if __name__ == "__main__":
    main()
