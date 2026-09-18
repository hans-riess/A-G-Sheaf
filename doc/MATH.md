# The mathematics

This document states what the library implements and what each construction is
supposed to be. It is written to be read alongside the code: every section names
the module and function that carries it. Citations are keyed to
[references.md](references.md).

---

## 1. Contracts as a lattice

An **assume–guarantee contract** over an alphabet `X` is a pair `C = (A, G)` of
predicates over `X`. `A` says which environments the component will operate in;
`G` says what it promises in those environments.

`agsheaf.contracts.Contract` holds the pair. The two slots are *not*
symmetric — refinement is contravariant on assumptions and covariant on
guarantees:

```
C ≤ C'   iff   A' ⇒ A   and   satG ⇒ satG'
```

which is `Contract.refines`.

### Saturation

A contract denotes a class, not a syntactic pair. `(A, G)` and `(A, A ⇒ G)`
promise the same thing, because on an environment violating `A` the contract
promises nothing either way. The **saturated guarantee** is

```
satG = A ⇒ G
```

exposed as the `Contract.sat_g` property, and *every* operation in the library
goes through it rather than through the raw `g`. This is not cosmetic: `ran` and
`pullback` carry a universal quantifier on the guarantee leg, `∀` does not
distribute over `∨`, and feeding either an unsaturated representative gives a
genuinely different result — which breaks the adjunctions the Laplacian rests on.

### The lattice operations

| operation | assumption | guarantee | meaning |
|---|---|---|---|
| `meet` (`∧`) | `A ∨ A'` | `satG ∧ satG'` | intersection of viewpoints; greatest lower bound |
| `join` (`∨`) | `A ∧ A'` | `satG ∨ satG'` | least upper bound |
| `top()` (`⊤`) | `False` | `True` | identity for `meet`; every contract refines it; the **incompatible** contract |
| `bot()` (`⊥`) | `True` | `False` | identity for `join`; refines every contract; the **inconsistent** contract |

A vertex whose contract has reached `⊤` has localised an assumption failure; one
that has reached `⊥` has localised a guarantee conflict. `ContractSheaf.collapsed`
reports both, and localising *which* vertex failed is the practical argument for
a Laplacian formulation over a monolithic consistency check.

Two further operations are implemented from Pacti (Incer et al. 2025):

* **composition** `C ‖ C'`, Eq. (2): `A = (A ∧ A') ∨ ¬(satG ∧ satG')`,
  `G = satG ∧ satG'`. The guarantee is the same conjunction as `meet`; the
  assumption is *discharged*, because an environment need not satisfy both
  components' assumptions where their joint guarantee already establishes them.
  That discharge is exactly what separates composition from the lattice meet.
* **quotient** `C / C'`, Eq. (3): the largest contract whose composition with
  `C'` still refines `C` — the specification of the component missing from a
  design. It is right adjoint to composition, which is its defining property
  rather than a consequence, and `test_theorems.py` checks the adjunction
  directly.

### Consistency and compatibility

* `is_consistent()` — `satG` is satisfiable: the contract promises something
  achievable.
* `is_compatible()` — `A` is satisfiable: some environment can meet it.

---

## 2. Relations and the four embedding maps

A `Relation` is a Z3 formula `R(x, y)` together with a source alphabet `X` and a
target alphabet `Y`. The alphabets **may overlap**; a variable in both is read
as a coordinate the two sides hold in common, and the embedding maps leave it
free rather than quantifying over it (`contracts._bind`). Binding a shared
coordinate would silently compute the Kan extension of a *different* relation.

Four maps carry a contract along `R`:

| map | assumption leg | guarantee leg | direction |
|---|---|---|---|
| `lan` (Lan_R) | `∀x. R(x,y) ⇒ A(x)` | `∃x. R(x,y) ∧ satG(x)` | `X → Y` |
| `ran` (Ran_R) | `∃x. R(x,y) ∧ A(x)` | `∀x. R(x,y) ⇒ satG(x)` | `X → Y` |
| `pullback` (R\*) | `∃y. R(x,y) ∧ A(y)` | `∀y. R(x,y) ⇒ satG(y)` | `Y → X` |
| `dual_pullback` | `∀y. R(x,y) ⇒ A(y)` | `∃y. R(x,y) ∧ satG(y)` | `Y → X` |

`lan` pushes aggressively/maximally; `ran` pushes safely/conservatively.

### These form two adjunctions, not an adjoint triple

```
lan            ⊣  pullback           (X → Y  ⊣  Y → X)
dual_pullback  ⊣  ran                (Y → X  ⊣  X → Y)
```

An adjoint triple `Lan_R ⊣ R* ⊣ Ran_R` would require `pullback ⊣ ran`, and that
holds **only when `R` is the graph of a total function**. For a general relation
the two available adjunctions are `∃_R ⊣ ∀_{R'}` and `∃_{R'} ⊣ ∀_R` (with `R'`
the converse); `pullback` and `ran` share the `(A: ∃, G: ∀)` shape and so cannot
be adjoint to one another. The shapes are forced by the refinement order being
contravariant on assumptions and covariant on guarantees.

`test/conftest.py`'s **relation zoo** parametrises the theorem tests over
relations that are total / surjective / functional / injective in every
combination, which is what pins down precisely which shapes each claim needs.

When `R` *is* the graph of a total function — which is the case in
`regionsheaf`, where it is the region-summary map — the triple does hold, and
`lan`/`ran`/`pullback` collapse to the adjoint triple of Naik et al. (2025)
Def. 6. That is why `regionsheaf` can compute transport in closed form.

---

## 3. Cellular sheaves of contract lattices

A **cellular sheaf** `F` over a graph assigns:

* to each vertex `i` a stalk (here: a contract lattice over `i`'s alphabet),
* to each vertex–edge incidence `i ⊴ e` a restriction map.

`ContractSheaf` stores this as a directed graph. Node `i` carries its contract
under `c`; edge `(u, v)` carries under `rel` the relation restricting `u` onto
the shared edge space of `{u, v}`. Both orientations of every interface must be
present, which is what `add_interface` guarantees and what makes `DiGraph`
rather than `Graph` the right base class.

### Global sections

An assignment `x` (a 0-cochain) is a **global section** when, across every
interface, both endpoints push to the same contract on the shared edge space:

```
is_section(legs)   ⟺   for every {u, v}:   push(u → e)(x_u) == push(v → e)(x_v)
```

`agrees_on` is this localised to one interface, `section_edges` is the whole
per-interface picture, and `is_section` is the conjunction. A flow in progress
typically closes them one at a time, and naming *which* are closed is the
localisation a Laplacian formulation buys.

---

## 4. Parallel transport and the Tarski Laplacian

**Parallel transport** of `j`'s contract into `i`'s local space is: push up to
the shared edge via `j`'s restriction, then pull back down via `i`'s
(Ghrist et al. 2026, Def. 6.2). Two bisheaves are available, selected by `legs`:

```
legs="kan"   pullback ∘ lan             restriction ⊣ coextension
legs="co"    dual_pullback ∘ ran        coextension ⊣ restriction
```

Both are adjoint bisheaves in the sense of Def. 6.6, since both adjunctions hold
for an arbitrary relation. They are *different* bisheaves, and the adjunction
points the opposite way in each. **Mixing legs across the two** — pushing with
`ran`, pulling with `pullback` — is what breaks the Hodge–Lawvere theorems,
since `pullback ⊣ ran` needs the graph of a total function. `transport` refuses
anything but the two named pairings.

### The operator

```
(Lx)_i = ⋀_{j ∈ N(i) ∩ τ}  transport(j → i)(x_j)       [primal, Def. 6.3]
(Lx)_i = ⋁_{j ∈ N(i) ∩ τ}  transport(j → i)(x_j)       [dual, Def. 7.3]
```

`dual` selects the aggregation and `legs` selects the bisheaf; the two are
orthogonal. `ContractSheaf.laplacian` computes `(Lx)` for every node from a
**frozen** assignment `x`, so it is the operator itself rather than the effect
of a sweep. `τ` is the firing set (§6); `None` means all of `V` and recovers the
synchronous operator of Riess and Ghrist (2022) Eq. (4). A node with no firing
neighbour aggregates over nothing and collapses to the identity of the relevant
operation — `⊤` primal, `⊥` dual — which under `include_self` leaves it
untouched, as their Def. 5 requires. That falls out of the empty meet being `⊤`;
it needs no special case.

---

## 5. Hodge–Tarski, and what holds over *this* base

### The suffix condition

```
is_suffix(legs)   ⟺   x ≤ Lx
```

By the **Hodge–Lawvere theorem** (Ghrist et al. 2026, Thm. 6.10) these are the
weighted global sections. `ContractSheaf` carries no weight function — it is the
unweighted Boolean case `W = ⊤` — and every interface contributes both `(i, j)`
and `(j, i)`, so the suffix inequality holds in both directions at once and
collapses to equality. Over this base, therefore,

```
is_suffix(legs=L)   ⟺   is_section(legs=L)
```

and `test_theorems.py` checks that on every fixture sheaf.

### The prefix condition, and the honest caveat

```
is_prefix(legs)   ⟺   L_dual x ≤ x
```

What this means depends sharply on `legs`.

* **`legs="co"`** transports along `dual_pullback ⊣ ran`, whose adjunction runs
  the right way to transpose the condition. It comes out as exact `ran`-agreement
  across every interface — a genuine dual notion of consistency, and the one
  `is_section(legs="co")` tests.
* **`legs="kan"`** is Def. 7.3 Eq. (13) read literally: the same legs as the
  primal, aggregated by join. Here the adjunction points the wrong way to
  transpose `pullback(z) ≤ x_i`, so this is **not** an edge-agreement condition.

**Section 7 degenerates over a Boolean base.** Theorem 7.5 needs the linear
negation of a Girard quantale; over the Boolean base with `W = ⊤` the dualising
element is bottom, so Def. 7.4's cosection condition is vacuous. Prefer
`legs="co"` for a dual with content. `exp/convergence.py` runs `dual_kan` anyway
as a labelled control, so the claim can be *seen* rather than taken on trust.

Likewise `is_harmonic` (Thm. 7.6, both halves along the *same* bisheaf) is over
this base a strictly stronger condition than being a section rather than a
characterisation of one. Conjoining the two halves across different bisheaves is
meaningless, which is why `legs` is shared rather than free on each half.

---

## 6. The harmonic flow

### The update

```
x ← Lx ∧ x        [primal]          x ← L_dual x ∨ x        [dual]
```

This is the unweighted harmonic flow of Ghrist et al. (2026) Def. 6.13,
equivalently the heat flow `x ← (id ∧ L_t)x` of Riess and Ghrist (2022) Eq. (6).
It is `ContractSheaf.laplacian_update` with `include_self=True` (the default).

The `∧ x` matters. That form is **monotone** — decreasing primal, increasing
dual — so it cannot cycle, and it is what Proposition 6.15 requires: the flow
preserves each node's relationship to every global section, converging to the
**greatest section below the initial assignment** rather than to an arbitrary
one. Setting `include_self=False` gives the bare `x ← Lx`, which has the same
fixed points but is not monotone and *can* cycle.

### Update model

`in_place=False` reads every neighbour from a snapshot taken before the step,
which is Eq. (6) literally (Jacobi). `in_place=True` (the default) writes each
node as it goes, so a node computed later already sees its predecessors' new
values (Gauss–Seidel-like), and converges in fewer steps. Both are monotone,
both have identical fixed points, and Theorem 1 covers both.

Note that `gridsheaf.sweep` deliberately does *not* use `laplacian_update`: it
calls `laplacian` on a frozen assignment, because `in_place` would carry
knowledge more than one hop per sweep in DiGraph node order, and one sweep is
supposed to be one hop.

### Firing sequences and liveness

A **firing sequence** `τ` (Riess and Ghrist 2022, Def. 5) is a sequence of node
sets, one per step: who broadcasts. **Liveness** (their Assumption 2) requires
every node to fire infinitely often; it is the hypothesis of their Theorem 1, so
a schedule violating it carries no convergence guarantee. `sheaf.round_robin`
and `sheaf.random_firing` are live; `LivenessError` is raised when a schedule
starves a node.

**Theorem 1**: the sections are the time-independent solutions of the heat flow
for *any* firing sequence satisfying liveness. Its practical content is that the
answer does not depend on the schedule — which is why the sharpest available
test is to run several schedules and check they agree, and it is what
`test_async.py` and `exp/validate_theory.py` both do. The argument is
schedule-independent: if `x[t] ≥ y` for a section `y`, then `Ly ≥ y` and
monotonicity give `x[t+1] ≥ y` whatever subset was updated.

### Termination under a partial schedule

A step in which nothing changed proves only that the *firing* nodes had nothing
to say. `converge` declares a fixed point once every node has fired across an
**unbroken** run of steps in which nothing changed.

The run must be unbroken, and this is subtle. Crediting the nodes that fired in a
step that *did* change something is unsound: a node can broadcast and then change
within the same step — which happens whenever two adjacent nodes fire together —
so it sent a value it no longer holds. Round-robin never exposes this, since a
lone firing node has no firing neighbour and cannot change in the step it fires;
random firing does. `agsheaf.mission` carries the same logic at the run level,
in `_broadcast_since_change`.

### The Dirichlet energy

In the linear theory the quantity that falls to zero along the heat flow is the
Dirichlet energy `xᵀLx`, a sum over edges of how far apart the endpoints are
once pushed onto the shared space. Written that way it transports here
unchanged:

```
dirichlet(F, legs) = Σ_{interfaces} distance(push(u→e)(x_u), push(v→e)(x_v))
```

and it is zero exactly on the global sections, which is the sheaf condition.
There is no inner product here and no spectrum; what carries over is the
accounting, not the operator theory.

---

## 7. Measuring the lattice

Over a finite alphabet `Σ`, a contract denotes two subsets:

```
E(C) = ⟦A⟧          the environments it admits
M(C) = ⟦A ⇒ G⟧      the behaviors it permits
```

and refinement is inclusion, reversed on the first and forwards on the second.
With `μ` the (by default uniform) measure on `Σ`:

```
distance(C, C')   = ½ [ μ(E Δ E') + μ(M Δ M') ]
divergence(C, C') = ½ [ μ(E' \ E) + μ(M \ M') ]
```

`distance` is a metric on semantic classes, zero exactly on contracts z3 calls
equal. `divergence` vanishes exactly when `C` refines `C'`, so it is a **Lawvere
metric** (a category enriched in `[0, ∞]`) rather than a metric, and
`distance = divergence(C, C') + divergence(C', C)`.

The directed reading is the useful one along a flow: the primal flow descends,
so its iterates nest, so `distance(x_t, x_limit)` is not merely shrinking but
**monotone** — and a violation is a bug rather than slow convergence.
`divergence_profile` is the monotonicity certificate: for the primal flow,
`divergence_profile(x_next, x_now)` must be zero at every node, and a nonzero
entry names the node that moved the wrong way.

`interval_volume` computes the same number a third way, as `μ` of the lattice
interval `[C ∧ C', C ∨ C']` through the shipped `meet` and `join`. It must agree
with `distance`, and `test_measure.py` checks that it does — which is what keeps
the module honest about measuring the lattice the library implements rather than
a paraphrase of it.

Everything in `measure` enumerates `Σ`, so it is exact and exponential in the
alphabet. `Alphabet` refuses to build past `max_states` (default `2^16`) rather
than quietly taking a long time.

---

## 8. Products of sheaves

A product of complete lattices is a complete lattice under the componentwise
order, and a product of left adjoints is a left adjoint. A cellular sheaf whose
stalks are products and whose restriction maps are componentwise is exactly a
product of sheaves; its global sections are the products of the factors' global
sections, and the Tarski Laplacian of the product is the product of the factors'
Laplacians.

`agsheaf.product` makes that structural rather than hoped-for.
`regionsheaf`'s construction has no coupling between regions anywhere —
contracts, restrictions, meets and transports all factor over the partition —
so a stalk is a `ProductContract` (one `Contract` per region) and a restriction
is a `ProductRelation`. Everything `ContractSheaf` does works on them unchanged
by duck typing. What the factorization buys: every z3 query stays inside one
region's small alphabet, and agreement can be reported per `(interface, region)`.

What a product stalk is *not* is a single contract over the joined alphabet:
`Con(B₁ × B₂)` is strictly larger than `Con(B₁) × Con(B₂)`. The construction
never needs the extra room, and staying inside the product is what keeps the
flow's formulas from entangling regions as sweeps accumulate. `assemble` gives
the joined reading as a *view*, never as the state.

---

## 9. What the demonstration adds

See [DUALITY.md](DUALITY.md) for the abstraction and the two bisheaves in
detail. In brief:

**Possibility-valued facts.** Per agent and tile, a 3-bit mask over
`("target", "safe", "unsafe")`: the labels still considered possible. A held
belief is the **upper bound** `S & ~mask == 0` — a subset bound with no lower
bound. Guarantees are conjunctions of such bounds and assumptions are trivial
(a broadcast is an assertion, not a promise), so the contract meet conjoins
bounds, which is pointwise *intersection* of possibility sets.

Two agents disagreeing about a tile therefore fuse to `S ⊆ ∅`: a **satisfiable**
constraint, an element of the lattice, localised to the contested tile. Nothing
collapses to `⊥`, the flow stays exactly the Tarski Laplacian on the finite
product lattice of possibility masks, and `collapsed()` stays empty by
construction. Conflict handling is a read-out policy, not an algebraic repair.

Pinning the set *exactly* instead — asserting both that the believed label is
possible and that the others are not — would make the conjunction of two
contradicting beliefs unsatisfiable and take the whole stalk to `⊥`.
`ensembles` implements both, as the `"possibility"` and `"boolean"` encodings,
so the claim is something an ensemble can measure rather than a design note.

**Mission contracts.** An agent's route reliance — "the tiles I route through
keep a possibility outside hazard" — is the *assumption* of its mission
contract. `gridsheaf.reliance_contract` reifies it as a contract whose meet with
the fused stalk, read out through `forced_empty`, asks exactly whether the
network's knowledge still admits the mission's assumption.

**Why assumptions are monitored rather than circulated.** In `regionsheaf` the
guarantee is asserted unconditionally and is what the Laplacian transports; the
assumption is checked separately by `assumption_status`. The reason is
mechanical: contract saturation voids every promise on assumption-violating
states, so a guarantee conditioned on a nontrivial assumption entails almost
nothing under the existential push leg — any not-yet-refuted violation model
escapes it — and the network goes mute exactly where coordination is needed.
Measured, not hypothetical: with reliance saturated in, an agent relying on one
unknown tile transmits no knowledge at all.
