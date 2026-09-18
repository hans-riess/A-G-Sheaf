# The demo against the machinery: MS3 as the governing design

How the Robotarium demo's belief sheaf must change so that it demonstrates the
project's actual mathematics. First drafted as an audit against the whitepaper's
§6 (adjoint bisheaves over a quantale); re-anchored on the Milestone 3 report —
`doc/SEAMAN/past_milestones/HR0011-25-3-0235_MS3_Report.pdf` — after HR pointed
out that the original design intent was **levels of abstraction**: contract
embeddings (Galois connections) letting agents *reach agreement in the abstract
while keeping reasonably allowable disagreement over individual tiles*. That
intent subsumes the audit: every degeneracy catalogued below is one fact seen
from different sides.

## 1. What MS3 commits to

The pieces, with the report's numbering:

- **Behavior sheaf** (Def 10, Example 4): vertex stalks speak each agent's
  *concrete* language (grid-cell traces); edge stalks speak a *coarser shared*
  language (region-label traces); the restriction data is an **abstraction
  map** $f = F_{v \trianglelefteq e} : \mathcal{B}_v \to \mathcal{B}_e$.
- **Contract embeddings** (Def 6, Prop 2 — Naik et al.): $f$ induces the
  adjoint triple on contract lattices
  $f_! \dashv f^* \dashv f_*$, with
  $f_!(A,G) = (\forall_f A,\ \exists_f G)$ the *strongest abstract contract
  implementable*, $f_*(A,G) = (\exists_f A,\ \forall_f G)$ the *weakest
  abstract contract requiring*, and $f^*$ the preimage translation back down.
- **Four contract sheaves** (§2.4 table): push by $f_!$ (case 1) or $f_*$
  (case 2) from a behavior sheaf; pull by $f^*$ from a behavior cosheaf
  (cases 3, 4). Case (1) is the primary one: a section certifies that every
  agent's mission commitment, *translated into the shared vocabulary*, is
  mutually consistent.
- **The Tarski Laplacian** (Def 16) and the **primal/dual harmonic flows**
  (Eqs 21, 22), with Hodge–Tarski (Thm 1) and finite convergence (Cor 1).
- **Scope** ("Update on SEAMAN Scope"): Boolean / suplattice, deliberately.
  The quantale-enriched duality of the whitepaper's §6 is the *future*
  generalization (MS3 Remark 4), not the demonstrandum. The duality that is
  in scope is the Galois-connection one — Sup vs Inf, left vs right adjoints
  (MS3 Remark 1: "This duality is fundamental").

The crux, in one line: **the section condition lives at the edge's abstraction
level** — $(F_{u \trianglelefteq e})_! (\mathcal{C}_u) = (F_{v \trianglelefteq
e})_! (\mathcal{C}_v)$ (Eq 16) — so the kernel of $f_!$ is exactly the
disagreement the mission *tolerates*. Two agents whose tile beliefs differ
within a fiber of $f$ push to the same abstract contract, and the sheaf calls
that agreement, because at the shared vocabulary's resolution it is.

## 2. The diagnosis, unified: the demo runs $f = \mathrm{id}$

`gridsheaf.edge_relation` equates each tile's private mask with a fresh edge
copy — a bijection. The abstraction map is the identity, the fibers are
singletons, the kernel is trivial. Everything the earlier audit catalogued is
this one specialization:

| Degeneracy | As the $f = \mathrm{id}$ specialization | Receipt |
|---|---|---|
| Agreement means tile-by-tile identity; disagreement must be fought to ∅, never absorbed | trivial kernel of $f_!$ | `_compare_tile_labels`, the contested read-out |
| `sheaf.legs` is a no-op | $f_! = f_*$ when fibers are singletons | `test_gridsheaf.py::TestBisheafParity`, its own docstring |
| Transport is a renaming, not an embedding | $f^* f_! = \mathrm{id}$ along a bijection | `edge_relation` docstring |
| Unweighted, single-flow | the demo never leaves case (1) with $W=\top$ | `sheaf.py::is_suffix` docstring |
| `refines` half-vacuous, `collapsed()` never fires | $A \equiv \mathrm{True}$ at every stalk | `encode_masks` |

The last row is a *separate* degeneracy: abstraction alone does not populate
the assumption slot ($f_!$ of a trivial assumption is trivial). MS3 §3's own
framing — "the assumption of each agent's contract encodes the behaviors it
expects from its neighbors" — is the two-slot design (hearsay and expectations
→ A, first-hand knowledge and commitments → G), and it composes with the
abstraction change independently. Neither substitutes for the other.

## 3. Verified: the repo already implements the triple

The load-bearing fact, checked by running it (2026-08-10): **for a restriction
relation that is the graph of a function, `contracts.py`'s Kan operations are
Def 6.** `lan` $= f_!$, `ran` $= f_*$, `pullback` $= f^*$ — and `dual_pullback`
coincides with `pullback` on functions (both reduce to preimage), so the two
transports become

    legs="kan"   push f_!, pull f^*   =  MS3 case (1)  bottom-up commitments
    legs="co"    push f_*, pull f^*   =  MS3 case (2)  weakest-requirement

`sheaf.legs` acquires the report's own meaning at exactly the moment the
restriction stops being a bijection. No new operator code is needed; the change
is confined to *which relation* `build_belief_sheaf` installs on the edges.

The witness (runnable as-is against `src/`): one region of two tiles, edge
alphabet a single Boolean $H$ = "this region contains a hazard":

```python
import z3, sys; sys.path.insert(0, "src")
from agsheaf import Contract, Relation
UNSAFE = 4
H = z3.Bool("H")
S0, S1 = z3.BitVec("S_t0", 3), z3.BitVec("S_t1", 3)
f = Relation(H == z3.Or(S0 == UNSAFE, S1 == UNSAFE), [S0, S1], [H])

pins_t0 = Contract(z3.BoolVal(True), S0 == UNSAFE, vars=[S0, S1])
pins_t1 = Contract(z3.BoolVal(True), S1 == UNSAFE, vars=[S0, S1])
clean   = Contract(z3.BoolVal(True), z3.And(S0 != UNSAFE, S1 != UNSAFE), vars=[S0, S1])

pins_t0.lan(f) == pins_t1.lan(f)   # True  : allowable disagreement, edge agrees
pins_t0.lan(f) == clean.lan(f)     # False : region-level conflict, flow acts
pins_t0.lan(f).pullback(f).sat_g   # Or(S_t0 == 4, S_t1 == 4)
```

Three facts in those last lines:

1. Agents pinning **different** tiles push to the **same** abstract contract —
   the section holds while the tile grids differ. Agreement in the abstract,
   on shipped code.
2. Hazard vs region-clean push differently — the flow has something real to
   reconcile, and reconciles it at region granularity.
3. What $f^*$ sends back down is **disjunctive across the fiber** — "one of
   these tiles is the hazard" — which is precisely what the abstraction
   warrants and no more.

A second prototype (same session) showed disagreement about **goals** is
expressible the same way: allocation possibility-masks in the stalk alphabet,
both agents claiming a target → the component empties in one sweep, in-lattice;
a yield-and-reclaim then closes the section. Goal claims compose upward
naturally ("*some* agent covers the east region" is an abstract-level
sentence; which tile stays local).

## 4. The casualty: the mask-canonical flatten

Fact 3 above breaks an implementation invariant. `gridsheaf.sweep` decodes
every fused stalk to per-tile masks and re-encodes — the flatten that keeps
formulas from growing. A per-tile mask is a *product* over tiles; a
disjunction across a fiber is not product-shaped, so the flatten would
silently discard exactly the knowledge the abstraction produces. **The
flatten is only sound at $f = \mathrm{id}$.**

The principled fix is what MS3's own `harmonic_flow` (Listing 3) does: the
flow runs on **exact contracts**, convergence checked by contract equality —
which `ContractSheaf.laplacian_update` already implements with z3-decided
equality. The per-tile mask projection survives, demoted to what it honestly
is: the **renderer's and planner's lossy view** of the stalk. A view may
project; the flow may not. Formula growth then needs a canonicalization
story — candidate: quantifier elimination (`_dump/simplify.py` already wraps
z3's `qe` for exactly this shape) — and the solver cost per sweep needs
measuring before the Robotarium's control loop is asked to pay it.

## 5. What the demo becomes

Region boundaries drawn on the floor. Interface lines colored by **abstract**
agreement. The split-tile comparison view showing two agents still disagreeing
tile-by-tile **while the section is green**. Caption:

    sweep 4  ·  abstract: global section  ·  concrete: agents differ on 7 tiles

That single frame is the Galois connection — coordination without uniformity —
and it is honest: the disagreement on screen is precisely the kernel of the
abstraction the agents agreed to communicate through. A region-level conflict
(one agent believes a region clean, another believes it hazarded) shows as a
pink interface that the flow then closes, and what changes at the doubting
agent is a region-granular fact, not a copied tile grid.

## 6. The open modelling decisions

These are the design choices that *are* the mathematics of the demo, in
descending order of consequence:

1. **The region partition** — the fibers of $f$. What the mission's shared map
   is (rooms? corridors? sectors?), configured per experiment.
2. **The abstract vocabulary** — what a region may say (hazard-present,
   target-present, traversable-through, …). This choice *is* the choice of
   which disagreements are allowable.
3. **Which contract sheaf** — case (1) ($f_!$, bottom-up commitments) is the
   MS3 default and the recommendation; case (2) is one config key away once
   the relation is a real abstraction.
4. **Which flows** — primal (Eq 21) descending, dual (Eq 22) ascending, or
   both bracketing the section space on screen.
5. **Whether goals enter v1** — allocation claims at the abstract level, or
   held for the follow-on.
6. **The A-slot** — expectations/hearsay into assumptions (MS3 §3's framing),
   composable with all of the above, independently staged.

## 7. What the change touches

- `gridsheaf.build_belief_sheaf` / `edge_relation`: install the abstraction
  relation (graph of the region map) instead of the bijection; new `regions:`
  config block and vocabulary schema, rows in `doc/GRIDWORLD.md`.
- `gridsheaf.sweep`: exact-contract flow (§4); per-tile masks become a
  read-out; the keep-own and contested policies re-derive against the view,
  not the state.
- `test_gridsheaf.py::TestBisheafParity`: inverts from a description of the
  demo into a characterization of the degenerate case ($f$ bijective ⟺
  legs immaterial).
- `exp/validate_theory.py`: the closed form (limit = pointwise mask
  intersection) becomes an $f = \mathrm{id}$-only statement; the general
  claim is Cor 1's convergence to a section, checked by equality.
- Rendering: region overlay, abstract-agreement interface coloring, the
  two-level caption; benchmark baselines re-established.
- The motion-layer deadlock (head-on swap, 2026-08-10 debug session) is
  **orthogonal** and stays parked: it lives in `choose_belief_aware_step`,
  not in the sheaf.
