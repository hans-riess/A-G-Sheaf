# Architecture

## Repository layout

```
agsheaf/
├── bootstrap.py                 venv + submodule + editable installs
├── pyproject.toml               package metadata; declares the `agsheaf` console script
├── pytest.ini                   testpaths = test
├── src/agsheaf/                 the package
├── exp/                         runners, worlds, measurement campaigns
│   ├── defaults.yaml            how a run is conducted, for every world
│   ├── worlds/                  one file per world
│   ├── run.py                   the drive loop with a figure and a log behind it
│   ├── preview_frame.py         one frame, no run
│   ├── convergence.py           the randomized convergence campaign
│   ├── benchmark.py             sheaf vs. no-communication vs. omniscient
│   ├── validate_theory.py       the theory, sampled on the gridworld sheaf
│   ├── scale_example.py         the demonstration sheaf at network scale
│   ├── figures.py               convergence JSON -> PDF figures
│   ├── robot_icon.py            cosmetic robot reskin for local previews
│   └── robotarium/              the testbed submission pipeline
├── test/                        the pytest suite
├── libs/robotarium/             git submodule: robotarium_python_simulator
└── doc/                         this documentation
```

## The package, in layers

Read this bottom-up; each layer uses only the ones below it.

```
                      exp/run.py, exp/preview_frame.py,
                      exp/robotarium/experiment_template.py
                                    │
                                 mission          the drive loop, in one place
                                    │
                ┌───────────────────┼────────────────────┐
                │                   │                    │
           gridsheaf ──────────► regionsheaf          gridworld       log
        (f = id belief sheaf)  (region-abstracted    (physics, planner,   (JSON)
                │               mission sheaf)        rendering, config
                │                   │                 parsing)
                └────────┬──────────┘
                         │                            regions   worldgen
                    ┌────┴─────┐                    (partition) (world generator)
                    │          │
                  sheaf     product                 diagnostics  ensembles
             (ContractSheaf, (products of              │            │
              Laplacian,      lattices)                └─► measure ─┘
              harmonic flow)                                 │
                    │                                    simplify
                    └────────────► contracts ◄────────────────┘
                                  (the lattice,
                                   the Kan maps)
                                        │
                                       z3
```

### `contracts` — the lattice

An assume–guarantee contract is a pair `(a, g)` of Z3 predicates. `Contract`
implements `meet`, `join`, `refines`, `compose`, `quotient`, and the four
Kan-type embedding maps `lan`, `ran`, `pullback`, `dual_pullback`. A `Relation`
is a Z3 formula over a source and a target alphabet. `top()` and `bot()` are the
lattice extremes. Every predicate is decided by the solver; there is no
syntactic shortcut anywhere.

Crucially a contract denotes a *saturation class*, not a syntactic pair:
`(a, g)` and `(a, a ⇒ g)` are the same contract, and every operation goes
through the `sat_g` property rather than the raw `g`.

### `sheaf` — the sheaf and the flow

`ContractSheaf` **is** a `networkx.DiGraph`. Nodes carry their contract under
the attribute `c`; each directed edge `(u, v)` carries, under `rel`, the
`Relation` restricting `u` onto the shared edge space of the interface `{u, v}`.
An interface needs *both* directions, because the two endpoints restrict onto
the shared space by different relations — `add_interface` adds both.

On top of that: `transport` (parallel transport along a chosen bisheaf),
`laplacian`, the predicates `is_section` / `is_suffix` / `is_prefix` /
`is_harmonic` / `agrees_on`, and the flow `laplacian_update` / `converge`. The
firing-sequence generators `round_robin` and `random_firing` live here too.

### `product` — products of lattices

`ProductContract` and `ProductRelation` are one `Contract` (resp. `Relation`)
per component with every operation applied componentwise. They duck-type as
`Contract` and `Relation`, so everything `ContractSheaf` does works on them
unchanged. `regionsheaf` uses this to factor its stalks over the region
partition, which is what keeps every solver query inside one small alphabet.

### `simplify` and `measure` — keeping the flow finite and giving it a number

Each transport nests an existential inside a universal, and the flow composes
transports across sweeps, so a stalk's formula grows a quantifier layer per
sweep while saying the same thing. `simplify.qf` eliminates quantifiers when it
can *and checks the result*, returning its input unchanged when it cannot.
`measure.canonical` is the exact alternative for alphabets small enough to
tabulate: it rewrites both slots from what they denote.

`measure` also puts a metric on the lattice. `distance` is a genuine metric on
semantic classes, `divergence` its one-sided (Lawvere) half, and
`interval_volume` computes the same number a third way through the shipped
`meet`/`join` so the module stays honest about measuring *this* lattice.

### `diagnostics` — the predicates as magnitudes

Each yes/no answer in `sheaf` gets a number that is zero exactly where the
boolean was true:

| predicate | magnitude |
|---|---|
| `is_section(legs)` | `dirichlet(...) == 0` |
| `is_suffix(legs)` | `laplacian_residual(...) == 0` |
| `is_prefix(legs)` | `laplacian_residual(dual=True, ...) == 0` |
| `agrees_on(u, v, legs)` | `edge_disagreement(u, v, ...) == 0` |

`test_measure.py` asserts each pairing on every fixture sheaf.
`concrete_disagreement` is the one that is not a refinement of an existing
predicate: it measures two agents against each other in their *own* vocabulary
rather than in the shared one, and the gap between it and `edge_disagreement` is
the disagreement the abstraction tolerates.

### `ensembles` — random sheaves

A generator of random contract sheaves parametrised by topology, number of
facts, abstraction coarseness, fact encoding, knowledge and error rates. It
exists so the flow can be asked the same question many times; `exp/convergence.py`
is what asks.

### `regions`, `gridsheaf`, `regionsheaf` — the belief layer

`regions.Regions` is a named partition of the grid's tiles: the fibres of the
abstraction map. It imports neither `gridworld` nor z3.

`gridsheaf` builds a belief sheaf over the communication graph. Its alphabet is
powerset-valued: per agent and tile a 3-bit mask over `("target", "safe",
"unsafe")` naming the labels the agent still considers possible. Without a
region partition its restrictions are *bijections* — the f = id control arm.

`regionsheaf` is the demonstration proper: agent stalks speak per tile,
interface stalks speak per region, and the restriction is the graph of the
massively non-injective region-summary function. `gridsheaf`'s entry points
(`sweep`, `communicate`, `observe`, `observe_tiles`, `sheaf_state`,
`render_contracts`) dispatch to `regionsheaf` when the sheaf carries
`graph['mode'] == 'regions'`.

### `gridworld` — physics, planner, configuration, rendering

The largest module and the one with the widest job. `GridWorld` wraps the
Robotarium simulator with a tile abstraction and potential-field collision
avoidance. Alongside it: the belief-aware planner
(`choose_belief_aware_step`, `choose_agent_step`, `_compute_cost_to_go`,
`route_corridor`), the scorer (`score_robot_step`, `score_agent_step`),
`TileReservations` for asynchronous motion, every `parse_*` function the
configuration goes through, and every `initialize_*` function the figure is
built from.

### `mission` — the drive loop

One `Mission` class, used by all three runners. Agents plan a tile, drive to it,
communicate, and repeat until everyone is home or a budget runs out. The runners
supply what genuinely differs as callbacks: `on_tick` (draw and record poses),
`on_sweep` (what to do with a Laplacian iterate) and `on_hold` (stop the robots
and look). This loop used to be written out three times, which is how
`asynchronous_moves` came to work in one of them and be ignored by the other two.

### `log`, `utils`, `worldgen`, `cli`

`log.ExperimentLog` collects a run and writes it as one JSON file.
`utils` holds configuration composition (`load_experiment_config`,
`compose_config`, `merge_layers`, `apply_overrides`), output naming, and
`VideoSaver`. `worldgen` generates a demonstration world at a requested size.
`cli` is the `agsheaf` console entry point.

## How a run proceeds

```
  read config  ──►  parse world  ──►  build GridWorld  ──►  build belief sheaf
                                                                   │
  ┌────────────────────────────────────────────────────────────────┘
  │
  │  while not finished and budgets remain:
  │    ├─ agents that stopped moving release the tile behind them
  │    ├─ every `sweep_every_ticks` control iterations (async only):
  │    │     _run_sweeps(None)      one hop of the Laplacian, everyone broadcasting
  │    ├─ who may plan?   async: anyone idle.   sync: nobody until everyone is idle.
  │    ├─ for the planners:
  │    │     ├─ _run_sweeps(firing)          communicate before deciding
  │    │     ├─ mission monitor              assumption vs. fused stalk
  │    │     │     ├─ hazard/claim regions -> per-agent cost overlay
  │    │     │     └─ route_corridor + recommit -> retract stale claims, clear latch
  │    │     ├─ choose_agent_step            plan against the fused beliefs
  │    │     ├─ reservations.depart          book the tile
  │    │     └─ score / observe / log
  │    ├─ on_tick(coords, view)              draw, record poses
  │    └─ grid_world.update()                one 0.033 s control iteration
  │
  └──►  log.close(final_scores, steps_taken, all_arrived)
```

The identity that makes this a sheaf-theoretic demonstration rather than a
message-passing one: **one sweep of the Tarski Laplacian is exactly one hop of
communication**, and the fixed point the flow converges to is the greatest
global section below the initial assignment.

## Three deliberate non-abstractions

These look like duplication and are not. Each is documented in place; they are
gathered here so that a reader does not "fix" one.

1. **`exp/run.py`, `exp/preview_frame.py` and
   `exp/robotarium/experiment_template.py` each build their figure from the same
   rendering keys, in their own copy of the code.** Each of those files is a
   script that runs a whole experiment when imported, so there is nothing to
   import from either. A rendering key added to one belongs in all three.
   What they do *share* is the drive loop, which is why `agsheaf.mission` exists.

2. **`gridsheaf.LABELS` restates `gridworld.GROUND_TRUTH_LABELS`.**
   `gridsheaf` deliberately does not import `gridworld`, which would pull in the
   Robotarium stack at import time. A parity test in `test_gridsheaf.py` holds
   the two together.

3. **`regionsheaf` computes transport in closed form rather than calling
   `Contract.lan` / `.pullback`.** The generic Kan operators are correct here
   and build quantified formulas the flow then composes across sweeps; by the
   third sweep z3 cannot decide the tower. Along the graph of a total function
   both directions admit exact quantifier-free closed forms — pushforward as a
   finite image by model enumeration, pullback as substitution. A fixture test
   checks the closed forms against the generic operators.

## What is *not* in the package

* **The mathematics of the papers.** See [references.md](references.md).
* **`exp/` is not importable.** The runners are top-level script code that runs
  a whole experiment on import. `agsheaf.cli` runs them as subprocesses for
  exactly that reason.
* **`agsheaf.cli` and `agsheaf.worldgen` are not uploaded to the Robotarium.**
  `build_submission.py`'s `AGSHEAF_MODULES` lists what is:
  `gridworld`, `contracts`, `sheaf`, `product`, `regions`, `regionsheaf`,
  `gridsheaf`, `log`, `mission`. A submission carries a configuration baked into
  a module and never parses YAML or a command line.
