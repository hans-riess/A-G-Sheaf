# `agsheaf` documentation

Sheaf theory for multi-agent control and decision systems.

`agsheaf` has two halves, and it is worth knowing which one you are reading about.

**A library.** `agsheaf.contracts`, `agsheaf.sheaf`, `agsheaf.product`,
`agsheaf.measure`, `agsheaf.simplify`, `agsheaf.diagnostics` and
`agsheaf.ensembles` implement assume–guarantee contracts as a lattice, cellular
sheaves of those lattices over a graph, the Tarski Laplacian on such a sheaf, and
the harmonic flow that converges to its global sections. Nothing in that half
knows what a robot is. Every operation is decided by the Z3 SMT solver.

**A demonstration.** `agsheaf.gridworld`, `agsheaf.regions`,
`agsheaf.gridsheaf`, `agsheaf.regionsheaf`, `agsheaf.mission`,
`agsheaf.worldgen` and `agsheaf.log` put that library underneath a multi-robot
gridworld running on the Georgia Tech Robotarium simulator. Agents hold partial
and sometimes wrong beliefs about the tiles around them, communicate by running
the Laplacian flow over their communication graph, and plan against what they
have fused. Running one sweep *is* one hop of communication; that identity is the
whole point of the exercise.

## Where to start

| If you want to | read |
|---|---|
| install it and run something | [getting-started.md](getting-started.md) |
| know what the pieces are and how they fit | [architecture.md](architecture.md) |
| understand the mathematics | [MATH.md](MATH.md) |
| understand the two bisheaves and the abstraction | [DUALITY.md](DUALITY.md) |
| know why the implementation is believed correct | [correctness.md](correctness.md) |
| call the library from your own code | [library.md](library.md) |
| use the gridworld / belief-sheaf stack | [gridworld.md](gridworld.md) |
| write or edit a world file | [configuration.md](configuration.md) |
| drive the `agsheaf` command | [cli.md](cli.md) |
| run the measurement campaigns | [experiments.md](experiments.md) |
| build a Robotarium submission | [robotarium.md](robotarium.md) |
| read a run's JSON log | [logs.md](logs.md) |
| run or extend the test suite | [testing.md](testing.md) |
| look a term up | [glossary.md](glossary.md) |
| find the papers | [references.md](references.md) |
| contribute | [contributing.md](contributing.md) |

## A five-line tour

```python
import z3
from agsheaf import Contract, Relation, ContractSheaf

p1, p2, p3 = z3.Bools("p1 p2 p3")

F = ContractSheaf()
F.add_node(1, c=Contract(z3.BoolVal(True), z3.And(p1, p2)))
F.add_node(2, c=Contract(z3.BoolVal(True), z3.Or(p2, p3)))
F.add_interface(1, 2,
                Relation(z3.BoolVal(True), [p1, p2], [p2]),
                Relation(z3.BoolVal(True), [p2, p3], [p2]))

F.converge(verify=True)
assert F.is_section()
```

`converge` runs the harmonic flow to its fixed point. By the Hodge–Tarski
theorem that fixed point is the greatest global section below the assignment it
started from: the most each agent can say without contradicting what its
neighbours can see of it.

## Conventions used throughout

* **Agent indices are 0-based** everywhere they are reasoned about — in
  configuration, in logs, and as the sheaf's own vertices. `rendering.agent_label_offset`
  adds to them for display only, and the shipped worlds set it to `1`, so the
  figure labels agent `0` as "1".
* **Tiles are `(x, y)` pairs**, `(0, 0)` bottom-left, written `[x, y]` in YAML.
* **Grids are written highest row first** in logs and in the encoded label
  grids, so a printed grid reads the way the figure draws it.
* **Assertion messages are sentences.** The project's error strings say what
  failed, what was expected and what was received, in that order. See
  [contributing.md](contributing.md).
