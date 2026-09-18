# Duality and abstraction

Two questions this document answers:

1. When do the two adjoint bisheaves — `legs="kan"` and `legs="co"` — actually
   differ, and what does choosing one mean?
2. What does the region abstraction buy, and what does it cost?

They turn out to be the same question. The answer is: **both depend entirely on
how non-injective the restriction maps are.**

---

## 1. Two bisheaves

`agsheaf.contracts` implements four maps along a relation `R`, forming two
adjunctions (see [MATH.md §2](MATH.md#2-relations-and-the-four-embedding-maps)):

```
lan            ⊣  pullback          →  the "kan" bisheaf
dual_pullback  ⊣  ran               →  the "co" bisheaf
```

`ContractSheaf.transport(x, source, target, legs)` composes push-then-pull along
one of them:

| `legs` | push (restriction) | pull (coextension) |
|---|---|---|
| `"kan"` | `lan` — aggressive/maximal | `pullback` |
| `"co"` | `ran` — safe/conservative | `dual_pullback` |

These correspond to the two cases of the Milestone 3 framing (Naik et al. 2025
Def. 6): case (1) is the `lan`/`pullback` reading, case (2) the `ran`/
`dual_pullback` one.

**They cannot be mixed.** Pushing with `ran` and pulling with `pullback` needs
`pullback ⊣ ran`, which holds only for the graph of a total function, and
without it the Hodge–Lawvere theorems fail. `transport` accepts only the two
named pairings and raises on anything else.

### The `dual` flag is orthogonal to `legs`

`dual` selects how transported data is *aggregated* (meet vs. join);
`legs` selects *which bisheaf carries it*. All four combinations exist:

| flow | aggregation | bisheaf | what it is |
|---|---|---|---|
| `primal` | meet | `lan ⊣ pullback` | MS3 case (1) — the demonstration's flow |
| `primal_co` | meet | `dual_pullback ⊣ ran` | MS3 case (2) |
| `dual` | join | `dual_pullback ⊣ ran` | the dual with content |
| `dual_kan` | join | `lan ⊣ pullback` | Ghrist et al. Def. 7.3 Eq. (13) read literally |

`exp/convergence.py` runs all four. The last is a **labelled control**: over the
Boolean base with `W = ⊤` it carries no content (see
[MATH.md §5](MATH.md#the-prefix-condition-and-the-honest-caveat)), and running it
anyway is what turns that claim into something visible rather than asserted.

---

## 2. The degeneracy at `f = id`

**If every restriction map is a bijection, the two bisheaves coincide.**

Along the graph of a bijection, `lan` and `ran` agree — there is exactly one
source state per target state, so `∃x. R(x,y) ∧ φ(x)` and `∀x. R(x,y) ⇒ φ(x)`
pick out the same thing — and transport is a semantic renaming. Consequently:

* `legs="kan"` and `legs="co"` produce identical limits,
* the Laplacian's meet is exactly pointwise intersection of the neighbours'
  possibility masks,
* nothing about the choice of bisheaf can be *observed*,
* the "abstraction" forgets nothing, so abstract agreement and concrete
  agreement are the same number.

This is the situation in `gridsheaf.edge_relation`: the graph of the bijection
equating each tile's private mask with the edge copy. `regions.identity_regions`
(one region per tile) produces the same degeneracy in the region construction.

The f = id sheaf is **kept on purpose**, as the control arm of the abstraction
study (`agsheaf sims --no-abstraction`, or a world with no `regions:` block).
It is sound exactly there, it reproduces the pre-region demonstration verbatim,
and it is the thing the non-degenerate construction is measured against.

### The degeneracy table

| quantity | at `f = id` | at a genuine abstraction |
|---|---|---|
| `legs="kan"` vs `legs="co"` limit | identical | genuinely different |
| `edge_disagreement` vs `concrete_disagreement` | equal | `edge` can be 0 while `concrete` > 0 |
| the mask flatten (`decode` → re-`encode`) | sound | **unsound** — see §4 |
| `assumption_density = 0` (ensembles) | half of every distance is identically 0, `refines` is half-vacuous | same — this one is about the *assumption slot*, not the abstraction |

---

## 3. What the region abstraction is

`agsheaf.regionsheaf` replaces the bijections with the graph of a **region
summary function**, which is total and massively non-injective.

**Agent-side alphabet** (per agent `i`):

| variable | meaning |
|---|---|
| `S{i}_{x}_{y}` : BitVec(3) | the labels agent `i` still considers possible on tile `(x,y)` |
| `U{i}_{r}` : Bool | "my committed route passes through region `r`" |
| `O{i}_{j}_{r}` : Bool | "neighbour `j`'s route passes through `r`" — learned from the flow, never asserted first-hand |

One `O` variable **per neighbour**, not one "somebody else" bit: each interface
wires the bit to a different neighbour's claim, and a shared bit would receive
contradictory biconditionals from two interfaces and void the stalk.

**Interface alphabet** (per interface `{u,v}`, per region `r`, per label `l`):

| variable | defining term | meaning |
|---|---|---|
| `F{u}_{v}_{l}_{r}` | `⋁_{t ∈ r} (S_t & ~bit(l) == 0)` | *flagged*: some tile here has narrowed to within `l` |
| `X{u}_{v}_{l}_{r}` | `⋀_{t ∈ r} (S_t & bit(l) == 0)` | *excluded*: no tile here still admits `l` |
| `K{u}_{v}_{end}_{r}` | `U` at one end, `O` at the other | the route claim, crossed-wired |

### Why two polarities per label

The stalk guarantees are **upper bounds**, deliberately admitting the empty
possibility set (that is what keeps contradictions in-lattice). Under such an
encoding:

* a single "pinned here" predicate would never be entailed — the empty-set model
  escapes it;
* a single subset predicate would be escaped in the other polarity.

So `F` is what a pinned belief entails, `X` is what an all-clear belief entails,
and silence entails neither. Knowledge travels in both polarities; ignorance
stays silent.

A region whose fused stalk entails **both** `F` and `X` for one label is
contested at region level: "somewhere here the possibilities narrowed to within
`l`" and "nothing here admits `l`" at once, satisfied only by the empty set.
That is the per-tile empty mask one floor up, and it is still in-lattice — the
guarantee stays satisfiable throughout. `regionsheaf.contested_regions` reports
it.

### Biconditionals, not implications

`abstraction_relation` installs `edge_var == defining_term` for every interface
variable. Being biconditional keeps the relation the graph of a **total
function**, which is what collapses the four embedding maps to the adjoint
triple of Naik Def. 6, and which is what makes `regionsheaf`'s closed-form
transport exact (image + substitution) rather than an approximation.

The two endpoints of one interface install *different* relations — the `K`
wiring is mirrored — which is the mechanism by which one agent's claim (`U`)
reaches the other's expectation variables (`O`) via push-then-pull.

---

## 4. The kernel of `f_!` is the disagreement the mission tolerates

This is the point of the whole construction.

Which *tile* of a region carries a hazard flag is exactly what the interface
cannot see. Two agents whose tile beliefs differ **inside one region** push to
the same region summary, and the sheaf calls that agreement. So:

```
is_section  holds       while       concrete disagreement > 0
```

`diagnostics` gives both numbers. `edge_disagreement` measures the endpoints
after both have been pushed into the shared abstract vocabulary;
`concrete_disagreement` measures them against each other in their own, with one
side's contract relabelled into the other's copy (`diagnostics.relabel`). A flow
that closes the second while leaving the first positive has reached agreement at
the resolution the agents committed to communicate at, and no finer.

`RegionSheafState.concrete_disagreement` records it per interface as the number
of tiles the two endpoints' belief dicts differ on. **Nonzero while `is_section`
holds is the abstraction earning its keep.**

This is what the demonstration worlds are built around. From the footer of
`exp/worlds/variation.yaml`:

> Agent 4's TILE-level belief about the central wall is never corrected in the
> demonstration arm: it finishes the run still holding the wall in column 4 and
> still calling column 6 clear. The flow speaks the region vocabulary, and both
> columns lie in room D, so the error is invisible at the interface […] What the
> flow changes is where Agent 4 *goes*: routed by the network's regional
> knowledge and its neighbours' claims, it never enters the contested ground, and
> so never pays for an error it still holds. **The correction is behavioural, not
> doxastic.**

`agsheaf.worldgen`'s placement rules exist to make that happen by construction
rather than by luck: both walls interior to one region, partial walls with
staggered gaps, four long crossings and nobody parked. Each rule was arrived at
by watching a hand-placed world fail to produce the conflict it was meant to.

### Why the mask flatten is unsound here

`gridsheaf.sweep` decodes each fused stalk back to per-tile masks and re-encodes
it. That *flatten* is what keeps formulas from growing across sweeps, and it is
sound exactly along bijections, where a stalk's content is nothing but its masks.

Under a genuine abstraction it is not: a stalk holds region-level facts (`F`,
`X`, `K`) that no per-tile mask can express, and flattening would silently throw
them away. So `regionsheaf` keeps the flow state **exact** — each stalk
component is its own-knowledge formula plus a list of heard conjuncts, which is
literally the shape the meet produces — and the per-tile masks the planner and
renderer read are a **view decoded from that state, never written back**.

Formula growth is instead controlled by computing transport in closed form:

* **pullback is substitution** — `R*(A, G) = (A ∘ f, G ∘ f)`, exact for the
  graph of a total function, where `pullback` and `dual_pullback` coincide;
* **pushforward is a finite image** — the interface alphabet of one region is a
  handful of booleans, so `∃x. f(x) = y ∧ φ(x)` is exactly the disjunction of
  the reachable summaries, enumerable by repeated SAT with blocking clauses. The
  universal legs are the dual, not-image-of-not.

Every formula the flow ever holds is therefore quantifier-free over one region's
block. A fixture test in `test_regionsheaf.py` checks the closed forms against
the generic Kan operators for equivalence.

---

## 5. Coarseness as a dial

`agsheaf.ensembles` turns the abstraction into a *parameter*, so the redesign can
get evidence before it is built. Facts are partitioned into blocks of
`coarseness`, and the interface speaks about blocks rather than facts:

* `coarseness = 1` — each block is one fact, the restriction is a bijection,
  `f = id`. `legs` cannot matter.
* `coarseness > 1` — a genuine non-injective abstraction; the fibres are
  non-trivial and the kernel of `f_!` is the tolerated disagreement.

`aggregator` says how a block is summarised: `"or"` is the
"this region contains one" reading, `"and"` its dual, and `"parity"` the extreme
in which no single concrete fact is recoverable from the abstract one.

`exp/convergence.py --studies abstraction` sweeps `coarseness` with both
bisheaves and plots abstract agreement against concrete disagreement, and where
`legs` starts to matter. `exp/figures.py` draws it as the `abstraction` and
`legs` figures.

---

## 6. Open design questions

These are knobs rather than settled choices, and each is where the
demonstration's design is still open. They are enumerated here because
`ensembles` exposes each as a parameter precisely so that an ensemble can answer
them rather than a design note asserting them.

* **How coarse should the partition be?** Too fine and the abstraction tolerates
  nothing; too coarse and the interface cannot warn anybody about anything
  actionable. The demonstration uses six 4x4 rooms on a 12x8 grid.
* **Which aggregator?** `or` is what the demonstration uses ("some tile here is
  flagged"). `parity` is the adversarial extreme.
* **Where should the assumption live?** Currently: guarantees circulate,
  assumptions are monitored. See
  [MATH.md §9](MATH.md#9-what-the-demonstration-adds) for why saturating the
  assumption into the broadcast silences the network.
* **`assumption_density = 0` is a degenerate ensemble.** With every assumption
  `True`, half of every distance is identically zero and `refines` is
  half-vacuous. It is the `ensembles` default because it matches the
  demonstration's belief stalks, and it is a thing to vary deliberately when
  studying the assumption slot.
