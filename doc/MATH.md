# Correctness against the references

Every mathematical claim the code makes is traceable to a numbered result in
`doc/references`, and every row below is a passing test. Where the code and a
source disagree, the disagreement is recorded rather than papered over.

Short names used throughout:

| Key | Source |
|---|---|
| **Naik** | Naik et al. (2025), *Contract Embeddings for Layered Control Architectures* |
| **RG** | Riess and Ghrist (2022), *Diffusion of Information on Networked Lattices by Gossip* |
| **Ghrist** | Ghrist et al. (2026), *Categorical Diffusion of Weighted Lattices* |
| **Pacti** | Incer et al. (2025), *Pacti: Assume-Guarantee Contracts for Efficient Compositional Analysis and Design* |

Two techniques carry the suite. *Symbolic contracts* (`conftest.symbolic`) have
uninterpreted predicates as their assumption and guarantee, so a property z3
discharges of one holds for **every** contract over that alphabet — these are
proofs, not samples. The *relation zoo* (`conftest.ZOO`) parametrises over
bijective, one-to-many, non-surjective, partial, many-to-one and empty
relations, because most of the subtle claims turn out to be conditions on the
*shape* of the relation rather than on contracts at all.

---

## The contract algebra

| Result | Claim | Test |
|---|---|---|
| Pacti Eq. (1) | refinement is the order whose glb is `meet` | `test_theorems.py::TestContractAlgebra::test_refinement_is_the_meet_order` |
| Naik §3.1 | `meet` is contract conjunction `(A₁∪A₂, G₁ˢᵃᵗ∩G₂ˢᵃᵗ)` | `::test_meet_is_naik_conjunction` |
| Naik §3.1 | consistency / compatibility | `test_contracts.py::TestTopBottom` |
| Pacti Eq. (2) | composition discharges assumptions the joint guarantee establishes | `test_contracts.py::TestCompositionAndQuotient` |
| Pacti Eq. (3) | **quotient is right adjoint to composition** | `::test_quotient_is_right_adjoint_to_composition` |
| — | composition is commutative, associative, monotone | `::test_composition_is_commutative_and_associative`, `::test_composition_is_monotone` |
| — | all six operations preserve saturation | `::test_operations_preserve_saturation` |

Saturation is applied on read (`Contract.sat_g`), so the last row is what makes
that design sound: were any operation to leave the saturated class, `sat_g`
would not be idempotent through the algebra and the Kan maps would silently
receive unsaturated input.

## The Kan maps

| Result | Claim | Test |
|---|---|---|
| **Naik Thm. 1**, RG Def. 4 / Lemma 2(3) | `lan ⊣ pullback`, for **every** relation | `TestAdjunctions::test_lan_left_adjoint_to_pullback` |
| — | `dual_pullback ⊣ ran`, for every relation | `::test_dual_pullback_left_adjoint_to_ran` |
| — | `pullback ⊣ ran` holds **iff** the relation is a total function | `::test_pullback_adjoint_to_ran_only_for_total_functions` |
| RG Lemma 2(1) | `pullback ∘ lan ⊒ id` | `::test_roundtrip_is_inflationary` |
| RG Lemma 1 | `pullback ∘ lan = id` iff total, functional **and** injective | `::test_roundtrip_is_identity_exactly_for_total_injections` |
| **Naik Def. 2** | conservative approximation ⟺ relation total and surjective | `::test_conservative_approximation_needs_total_and_surjective` |
| RG Def. 2 | `lan` preserves joins; `ran` preserves meets; `pullback` preserves meets; `dual_pullback` preserves joins | `TestSupMorphism` |
| — | `pullback` preserves joins (and `dual_pullback` meets) iff functional | `::test_pullback_preserves_joins_only_for_functions` |

`lan ⊣ pullback` **is** Naik Theorem 1. Unwinding Naik Def. 3 for a functional
relation, `lan` is the abstract embedding ⋏ = `(α_lb(A), α_ub(G))` and
`pullback` the concrete embedding ⋎, so `⋏(C) ⪯ C' ⟺ C ⪯ ⋎(C')` is exactly the
adjunction — and the implementation generalizes it from functions to arbitrary
relations.

Naik Def. 2's conservative-approximation condition (`γ_ub ⊆ γ_lb` and
`α_lb ⊆ α_ub`) turns out to be precisely totality and surjectivity of the
relation. `Relation` does not enforce either; the test records which shapes
satisfy them.

## The Laplacian and the flow

| Result | Claim | Test |
|---|---|---|
| Ghrist Def. 6.2 / 6.3, RG Eq. (4) | transport is coextension ∘ restriction; `L` aggregates by meet | `test_sheaf.py::TestTransport`, `::TestLaplacian` |
| Ghrist Lemma 6.5 | `L` is monotone | `test_theorems.py::TestHodgeLawvere::test_laplacian_is_monotone` |
| **Ghrist Thm. 6.10** | **suffix points = global sections** | `::test_suffix_points_are_global_sections` |
| Ghrist Def. 6.13, RG Eq. (6) | the flow descends and settles at a section | `::test_flow_descends`, `::test_flow_reaches_a_section` |
| Ghrist Prop. 6.15 | the flow never descends past a section below it | `::test_flow_never_descends_past_a_section_below_it` |
| Ghrist Prop. 6.15 | the limit is the **greatest** such section | `::test_limit_is_the_greatest_section_below_the_initial_cochain` |
| RG §IV | a contradictory interface collapses to `bot`, and `collapsed()` localises it | `::test_contradiction_collapses_to_bottom_and_is_localised` |

## Asynchrony

| Result | Claim | Test |
|---|---|---|
| RG Eq. (4) | firing everything is the synchronous Laplacian | `test_async.py::TestFiringSequences::test_all_fire_is_the_synchronous_laplacian` |
| RG Def. 5 | a node with no firing neighbour is untouched | `::test_node_with_no_firing_neighbour_is_untouched` |
| **RG Thm. 1** | every live schedule reaches the **same** fixed point | `TestScheduleIndependence` (round-robin, seeded random, both update models) |
| RG Eq. (6) | snapshot and in-place updates agree | `::test_snapshot_and_in_place_updates_agree` |
| RG Assumption 2 | a starved schedule is diagnosed, not spun on | `TestLiveness::test_starved_schedule_is_diagnosed` |
| Ghrist Prop. 6.15 | the greatest-section property survives asynchrony | `TestAsyncPreservesTheTheorems` |

## Measuring the flow

The tables above decide whether the flow reached a section. They cannot say how
far from one it started, nor which sweep closed most of the gap, and a
convergence study needs both. `measure.py` supplies the missing number.

Over a finite alphabet Σ a contract denotes a pair of subsets — the environments
`⟦A⟧` it admits and the behaviors `⟦A ⇒ G⟧` it permits, always read through
`sat_g` so it is the saturation class that is denoted. Refinement is inclusion,
reversed on the first slot. Under a measure `μ` on Σ:

    distance(C, C′)   = ½[ μ(E Δ E′) + μ(M Δ M′) ]
    divergence(C, C′) = ½[ μ(E′ \ E) + μ(M \ M′) ]

`distance` is a metric on semantic classes; `divergence` is its one-sided half,
zero exactly on refinement, and so a Lawvere metric — a category enriched in
[0, ∞] — rather than a metric. Every diagnostic in `diagnostics.py` is a
magnitude whose zero set is a predicate this library already decides, and each
row below is that correspondence being *checked* rather than assumed.

| Result | Claim | Test |
|---|---|---|
| — | `distance` is symmetric, bounded, triangular, and zero exactly where z3 calls two contracts equal | `test_measure.py::TestMetricAxioms` |
| Naik §3.1 | `divergence` is zero exactly where `Contract.refines` holds | `::TestDivergence::test_zero_divergence_is_refinement` |
| Pacti Eq. (1) | the same number is the measure of the lattice interval `[c₁ ⊓ c₂, c₁ ⊔ c₂]`, via the shipped `meet`/`join` | `::TestIntervalVolume` |
| **Ghrist Thm. 6.10** | `dirichlet = 0` ⟺ `is_section`, on every fixture sheaf and both bisheaves | `test_diagnostics.py::TestZeroSets` |
| Ghrist Thm. 6.10 | `edge_disagreement = 0` ⟺ `agrees_on`, interface by interface | `::test_edge_disagreement_is_zero_exactly_on_agreeing_interfaces` |
| Ghrist Def. 6.13 | `laplacian_residual = 0` ⟺ `is_suffix`, and dually ⟺ `is_prefix` | `::test_residual_is_zero_exactly_on_suffix_points`, `::test_dual_residual_is_zero_exactly_on_prefix_points` |
| Ghrist Def. 6.13 | the primal flow never ascends and the dual never descends, step by step | `::TestMonotonicity` |
| **Ghrist Prop. 6.15** | distance to the limit decreases **monotonically**, not merely eventually | `::test_distance_to_the_limit_decreases` |
| — | `canonical` preserves the contract, is idempotent, and lands saturated | `test_measure.py::TestCanonical` |
| RG §IV | a Boolean-valued fact collapses to `bot` on contradiction; a possibility-valued one records it in the lattice | `test_ensembles.py::TestEncodings` |
| **Naik Def. 6** | the ensemble's restrictions are graphs of total functions, for every aggregator | `::test_every_aggregator_gives_a_total_function` |
| — | at coarseness 1 the restriction is a bijection and `legs` is a no-op; above it the two bisheaves separate | `::TestAbstraction` |

`converge` takes an `on_step` observer for the same reason `communicate` does:
the flow overwrites each stalk in place, so that callback is the only moment an
intermediate iterate exists. It is also where a caller may rewrite each stalk
into an equivalent smaller form — the termination test is contract equality,
which cannot tell the difference (`test_diagnostics.py::TestObserver`).

### A trap in eliminating quantifiers

`simplify.qf` puts a formula in quantifier-free form, which is what keeps an
exact-contract flow from carrying its whole history in every stalk. The obvious
implementation is unsound, and was: a z3 `Goal` is a *satisfiability* problem, so
a tactic applied to one need only preserve equisatisfiability and may decide the
goal by choosing values for the free constants. On

    ForAll([p, q, r], Implies(e == Or(p, q), Or(p, q)))

whose quantifier-free equivalent is `e`, `Then(simplify, qe2)` returns the empty
goal — `True`, which is wrong at `e = False`. Every contract has free variables,
so this is the common case rather than a corner. `qf` therefore checks the
result against its input with the solver and returns the input unchanged unless
the answer is a clear yes; the input is always sound, so a failed check costs
speed and never correctness
(`test_measure.py::TestQuantifierElimination`).

## The belief sheaf over the gridworld

`gridsheaf.py` instantiates the machinery above for the knowledge problem of
`exp/sheaf_grid.py`: stalks are contracts over a powerset-valued
alphabet (a 3-bit possibility mask per tile), restrictions are equality
bijections onto fresh edge stalks, and one `sweep` is one hop of RG Eq. (6)
in Jacobi form. Disagreement fuses to the *satisfiable* constraint `S ⊆ ∅`
inside the lattice rather than collapsing a stalk to `bot`, so the flow never
leaves the finite product lattice of masks and the keep-own conflict policy is
purely a read-out.

| Result | Claim | Test |
|---|---|---|
| RG Eq. (6), Jacobi form | one sweep moves knowledge exactly one hop | `test_gridsheaf.py::TestDiffusion::test_one_sweep_is_one_hop` |
| Ghrist Def. 6.3 | the meet decodes to pointwise mask intersection | `::test_decode_of_the_raw_meet_matches_mask_intersection` |
| Ghrist Prop. 6.15 | the fixpoint is the component knowledge union, in ≤ diameter sweeps | `::test_fixpoint_is_the_component_knowledge_union`, `::test_flow_settles_in_diameter_sweeps` |
| Ghrist Thm. 6.10 | the fixpoint is a global section, even under disagreement | `::test_fixpoint_is_a_section_and_nothing_collapsed`, `TestConflicts::test_contested_stalks_stay_consistent_and_form_a_section` |
| monotonicity | knowledge only strengthens: `contract(i)` refines `initial(i)` throughout | `TestDiffusion::test_knowledge_only_strengthens` |
| in-lattice conflicts | a contested tile is the value `S ⊆ ∅`, never `bot`; `collapsed()` stays empty | `TestEncodeDecode::test_empty_mask_is_a_consistent_constraint`, `TestConflicts` |
| mission layer | reliance ⊓ fused stalk forces `∅` exactly on relied tiles the network rules out | `TestMissionLayer` |

### Observing the flow

The sheaf condition is a conjunction over interfaces, so an assignment below the
fixed point still *has* the interfaces it has already closed. `agrees_on`
localises `is_section` to one of them and `section_edges` reports them all;
`sheaf_state` snapshots the whole assignment between sweeps, and `communicate`
takes an `on_sweep` observer, which is the only place the intermediate iterates
exist. Together these are what let a run record the flow rather than only its
result — and what let a video draw an interface differently once it closes.

| Result | Claim | Test |
|---|---|---|
| Ghrist Thm. 6.10, localised | `is_section` ⟺ every interface agrees, and interfaces close one at a time | `TestSheafState::test_agrees_on_localises_is_section`, `::test_section_edges_agrees_with_is_section` |
| RG Eq. (6) | the observer sees each iterate as that sweep left it, the settling one included | `TestSweepObserver` |
| — | a snapshot reports possibility sets, contested tiles, and monotonicity without collapsing anything | `TestSheafState::test_state_reports_possibilities_and_contested` |

---

## Known gaps between code and source

### There is no weight function `W`

The papers are about `W`-weighted global sections over a general quantale `Q`.
This implementation is the unweighted Boolean case: `Q = B`, `W ≡ ⊤` on edges.
That is the setting of RG (whose Tarski Laplacian is Ghrist Example 6.11), and
over `B` the only meaningful weight on an existing edge is `⊤`, so nothing is
lost — but statements quantified over `W` specialize here, and two of them
degenerate.

### Over `B` with `W ≡ ⊤`, §7 of Ghrist et al. degenerates

For the Boolean quantale the dualizing element is `d = ⊥`, so `W^⊥ = [⊤, ⊥] = ⊥`
and Def. 7.4's cosection condition `F_{w◁e}(x_w) ⊴_⊥ F_{v◁e}(x_v)` is vacuous.
Thm. 7.6's `≃_{W ∧ W^⊥}` is likewise `≃_⊥`. So the parenthetical in Thm. 7.6 —
"when `W ≡ 1` and Q is affine, these are exactly the strict global sections" —
does not go through for `B`, and the dual theory of §7 carries no information
here.

Two consequences the tests record:

* `is_prefix(legs="kan")` — Def. 7.3 Eq. (13) read literally, same legs as the
  primal, aggregated by join — is **not** an edge-agreement condition. The
  adjunction points the wrong way to transpose it
  (`test_kan_prefix_is_not_an_agreement_condition`).
* `is_harmonic` is therefore *strictly stronger* than being a global section
  rather than a characterization of one. A strict section that is not harmonic
  witnesses the gap (`test_harmonic_is_strictly_stronger_than_being_a_section`).

Meanwhile `legs="co"` — transporting along `dual_pullback ⊣ ran` — gives a dual
with real content: its prefix points are exactly `ran`-agreement across every
interface (`test_co_prefix_points_are_ran_sections`). Both readings are
implemented and tested; `legs="co"` is the one to reach for when a dual notion
of consistency is actually wanted.

### The primal half already suffices

Because the base is Boolean, `W ≡ ⊤`, and every interface contributes both
orientations, Thm. 6.10's suffix inequality holds in both directions at once and
collapses to equality. Suffix points are therefore already the *strict* global
sections, with no need for the two-sided theorem:
`is_suffix(legs=L) ⟺ is_section(legs=L)` on every fixture sheaf.

### Not enforced at construction

`Relation` validates neither that the free variables of its expression lie in
`source ∪ target`, nor totality/surjectivity (Naik Def. 2). The suite *checks*
the latter without requiring it, since a non-total relation is still a
legitimate object — it simply does not induce a conservative approximation.

`Undecided` (a z3 `unknown`) is raised by `_is_valid`/`_is_satisfiable` but is
not plumbed through `Contract.__eq__`, `refines`, or `converge`, so a solver
timeout deep in a sweep surfaces without saying which node or which step it came
from. Nested quantifiers over unbounded integer arithmetic can reach this.
