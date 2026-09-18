"""
Theorem tests, each named for the result in `doc/references` it discharges.

Where a test uses `symbolic` contracts its conclusion is universally quantified
over contracts, so these are proofs rather than samples. Where it parametrises
over the relation zoo, the point is usually that the theorem's hypothesis is a
*shape* condition on the relation, and the test pins down exactly which shapes
satisfy it.

References:
  Naik    -- Naik et al. (2025), Contract Embeddings for Layered Control Architectures
  RG      -- Riess and Ghrist (2022), Diffusion of Information on Networked Lattices by Gossip
  Ghrist  -- Ghrist et al. (2026), Categorical Diffusion of Weighted Lattices
  Pacti   -- Incer et al. (2025), Pacti
"""

import pytest
import z3

from conftest import contracts_for, symbolic, ZOO, SHARED
from agsheaf.contracts import Contract, _bind, _is_valid, bot, top
from agsheaf.sheaf import CO, KAN


def refines_expr(c1, c2):
    """`c1.refines(c2)` as a formula rather than a decided boolean."""
    return z3.And(z3.Implies(c2.a, c1.a), z3.Implies(c1.sat_g, c2.sat_g))


def equals_expr(c1, c2):
    """`c1 == c2` as a formula rather than a decided boolean."""
    return z3.And(c1.a == c2.a, c1.sat_g == c2.sat_g)


# ---------------------------------------------------------------------------
# The contract algebra -- Naik Section 3.1, Pacti Eqs. (1)-(3)
# ---------------------------------------------------------------------------

class TestContractAlgebra:
    def test_refinement_is_the_meet_order(self):
        """
        Pacti Eq. (1). Refinement is decided by the closed form, so check it
        against the independent lattice characterisation: c1 <= c2 iff their
        greatest lower bound is c1.
        """
        c1, c2 = symbolic("r1", [z3.Bool("u")]), symbolic("r2", [z3.Bool("u")])
        assert _is_valid(refines_expr(c1, c2) == equals_expr(c1.meet(c2), c1))

    def test_meet_is_naik_conjunction(self):
        """Naik Section 3.1: A_and = A1 u A2, G_and = G1_sat n G2_sat."""
        u = z3.Bool("u")
        c1, c2 = symbolic("m1", [u]), symbolic("m2", [u])
        merged = c1.meet(c2)
        assert _is_valid(merged.a == z3.Or(c1.a, c2.a))
        assert _is_valid(merged.sat_g == z3.And(c1.sat_g, c2.sat_g))

    def test_quotient_is_right_adjoint_to_composition(self):
        """
        Pacti Eq. (3) defines the quotient as the largest c3 with
        c2 || c3 <= c1. That universal property is an adjunction, and it is the
        property the closed form has to earn.
        """
        u = z3.Bool("u")
        c1, c2, c3 = (symbolic(n, [u]) for n in ("q1", "q2", "q3"))
        assert _is_valid(refines_expr(c2.compose(c3), c1)
                         == refines_expr(c3, c1.quotient(c2)))

    def test_composition_counit(self):
        """c2 || (c1 / c2) <= c1 -- the quotient really is a solution."""
        u = z3.Bool("u")
        c1, c2 = symbolic("k1", [u]), symbolic("k2", [u])
        assert c2.compose(c1.quotient(c2)).refines(c1)

    def test_composition_is_commutative_and_associative(self):
        u = z3.Bool("u")
        c1, c2, c3 = (symbolic(n, [u]) for n in ("s1", "s2", "s3"))
        assert c1.compose(c2) == c2.compose(c1)
        assert c1.compose(c2).compose(c3) == c1.compose(c2.compose(c3))

    def test_composition_is_monotone(self):
        u = z3.Bool("u")
        c1, c2, c3 = (symbolic(n, [u]) for n in ("n1", "n2", "n3"))
        assert _is_valid(z3.Implies(refines_expr(c1, c2),
                                    refines_expr(c1.compose(c3), c2.compose(c3))))

    def test_composition_differs_from_meet(self):
        """
        Composition discharges assumptions the joint guarantee already
        establishes; the meet does not. If these coincided, one of them would
        be misdefined.
        """
        u = z3.Bool("u")
        c1, c2 = symbolic("d1", [u]), symbolic("d2", [u])
        assert not _is_valid(c1.compose(c2).a == c1.meet(c2).a)

    @pytest.mark.parametrize("op", ["meet", "join", "compose", "quotient"])
    def test_operations_preserve_saturation(self, op):
        """
        Saturation is applied on read, so every operation must land back in the
        saturated class; otherwise `sat_g` would not be idempotent through the
        algebra and the Kan maps would see unsaturated input.
        """
        u = z3.Bool("u")
        c1, c2 = symbolic("t1", [u]), symbolic("t2", [u])
        result = getattr(c1, op)(c2)
        assert _is_valid(result.g == result.sat_g)


# ---------------------------------------------------------------------------
# The Kan maps -- Naik Thm. 1, RG Def. 4 / Lemmas 1-2
# ---------------------------------------------------------------------------

class TestAdjunctions:
    def test_lan_left_adjoint_to_pullback(self, spec):
        """
        Naik Thm. 1: the abstract embedding refines C' iff C refines the
        concrete embedding. Unwinding Naik Def. 3, `lan` is the abstract
        embedding (alpha_lb on assumptions, alpha_ub on guarantees) and
        `pullback` the concrete one, so this is exactly that theorem -- and it
        holds for every shape of relation, not just the functional ones.

        Also RG Lemma 2(3), phi(p) <= q iff p <= phi+(q).
        """
        R = spec.relation
        cx, cy = contracts_for(spec)
        assert cx.refines(cx.lan(R).pullback(R)), "unit"
        assert cy.pullback(R).lan(R).refines(cy), "counit"

    def test_dual_pullback_left_adjoint_to_ran(self, spec):
        """The second adjunction, likewise unconditional on the relation."""
        R = spec.relation
        cx, cy = contracts_for(spec)
        assert cy.refines(cy.dual_pullback(R).ran(R)), "unit"
        assert cx.ran(R).dual_pullback(R).refines(cx), "counit"

    def test_pullback_adjoint_to_ran_only_for_total_functions(self, spec):
        """
        `pullback -| ran` is NOT a third adjunction. Both maps have the shape
        (A: exists, G: forall), so they cannot be adjoint unless the relation is
        the graph of a total function -- at which point `lan` and `ran` coincide
        anyway. The one-to-many relation breaks the unit and the partial one
        breaks the counit.
        """
        R = spec.relation
        cx, cy = contracts_for(spec)
        unit = cy.refines(cy.pullback(R).ran(R))
        counit = cx.ran(R).pullback(R).refines(cx)
        assert (unit and counit) == spec.total_function

    def test_roundtrip_is_inflationary(self, spec):
        """RG Lemma 2(1): phi+ . phi >= id, for every relation."""
        R = spec.relation
        cx, _ = contracts_for(spec)
        assert cx.refines(cx.lan(R).pullback(R))

    def test_roundtrip_is_identity_exactly_for_total_injections(self, spec):
        """
        RG Lemma 1: phi+ . phi = id when phi is join-preserving and injective.
        For a relation that means total, functional and injective -- a total
        injective function. Losing any one of the three makes the roundtrip
        strictly lossy, which is precisely the information a non-injective
        restriction discards.
        """
        R = spec.relation
        cx, _ = contracts_for(spec)
        exact = spec.total and spec.functional and spec.injective
        assert (cx == cx.lan(R).pullback(R)) == exact

    def test_conservative_approximation_needs_total_and_surjective(self, spec):
        """
        Naik Def. 2 requires gamma_ub(S) <= gamma_lb(S) and alpha_lb(S) <=
        alpha_ub(S). Reading those maps off `pullback` and `lan`, the first
        condition is totality of the relation and the second is surjectivity.
        Neither is checked at construction, so this records which zoo entries
        actually induce a conservative approximation in Naik's sense.
        """
        R = spec.relation
        src, tgt = R.source_vars, R.target_vars
        s_src = z3.Function("Cs", *[v.sort() for v in src], z3.BoolSort())(*src)
        s_tgt = z3.Function("Ct", *[v.sort() for v in tgt], z3.BoolSort())(*tgt)

        gamma_ub = _bind(z3.ForAll, tgt, src, z3.Implies(R.rel, s_tgt))
        gamma_lb = _bind(z3.Exists, tgt, src, z3.And(R.rel, s_tgt))
        alpha_lb = _bind(z3.ForAll, src, tgt, z3.Implies(R.rel, s_src))
        alpha_ub = _bind(z3.Exists, src, tgt, z3.And(R.rel, s_src))

        assert _is_valid(z3.Implies(gamma_ub, gamma_lb)) == spec.total
        assert _is_valid(z3.Implies(alpha_lb, alpha_ub)) == spec.surjective


class TestSupMorphism:
    """
    RG Def. 2 requires restriction maps to preserve joins, which is what makes
    the residual exist at all. Left adjoints preserve joins and right adjoints
    preserve meets; `pullback` is both, being a left adjoint to `ran`'s partner
    and a right adjoint to `lan`.
    """

    def test_lan_preserves_joins(self, spec):
        R = spec.relation
        c1 = symbolic(f"j1_{spec.name}", R.source_vars)
        c2 = symbolic(f"j2_{spec.name}", R.source_vars)
        assert c1.join(c2).lan(R) == c1.lan(R).join(c2.lan(R))

    def test_ran_preserves_meets(self, spec):
        R = spec.relation
        c1 = symbolic(f"w1_{spec.name}", R.source_vars)
        c2 = symbolic(f"w2_{spec.name}", R.source_vars)
        assert c1.meet(c2).ran(R) == c1.ran(R).meet(c2.ran(R))

    def test_pullback_preserves_meets(self, spec):
        """`pullback` is a right adjoint (of `lan`), so meets always survive."""
        R = spec.relation
        c1 = symbolic(f"b1_{spec.name}", R.target_vars)
        c2 = symbolic(f"b2_{spec.name}", R.target_vars)
        assert c1.meet(c2).pullback(R) == c1.pullback(R).meet(c2.pullback(R))

    def test_dual_pullback_preserves_joins(self, spec):
        """`dual_pullback` is a left adjoint (of `ran`), so joins always survive."""
        R = spec.relation
        c1 = symbolic(f"v1_{spec.name}", R.target_vars)
        c2 = symbolic(f"v2_{spec.name}", R.target_vars)
        assert c1.join(c2).dual_pullback(R) == c1.dual_pullback(R).join(c2.dual_pullback(R))

    def test_pullback_preserves_joins_only_for_functions(self, spec):
        """
        The converse direction is not free. `pullback`'s assumption leg is an
        exists, which does not distribute over the conjunction `join` puts
        there, so joins survive exactly when the relation is functional -- when
        there is only one y to choose. Dually for `dual_pullback` and meets.
        """
        R = spec.relation
        c1 = symbolic(f"f1_{spec.name}", R.target_vars)
        c2 = symbolic(f"f2_{spec.name}", R.target_vars)
        pb = c1.join(c2).pullback(R) == c1.pullback(R).join(c2.pullback(R))
        dpb = c1.meet(c2).dual_pullback(R) == c1.dual_pullback(R).meet(c2.dual_pullback(R))
        assert pb == spec.functional
        assert dpb == spec.functional

    def test_lan_does_not_preserve_meets(self):
        """
        A left adjoint need not preserve meets, and `lan` does not: its
        assumption leg is a forall, which does not distribute over the
        disjunction that `meet` puts there.
        """
        R = ZOO[1].relation  # one_to_many
        c1 = symbolic("nm1", R.source_vars)
        c2 = symbolic("nm2", R.source_vars)
        assert c1.meet(c2).lan(R) != c1.lan(R).meet(c2.lan(R))


class TestSharedAlphabet:
    """
    A variable in both alphabets is a coordinate the two sides share. The Kan
    maps must leave it free; binding it computes the extension of a different
    relation, which is what happened before `_bind`.
    """

    def test_shared_coordinates_stay_free(self):
        R = SHARED.relation
        c = Contract(z3.And(*R.source_vars[:2]), R.source_vars[2])
        pushed = c.lan(R)
        shared = R.source_vars[:2]
        for v in shared:
            assert v.decl() in [d for d in _decls(pushed.a)], f"{v} was bound away"

    def test_adjunction_survives_shared_coordinates(self):
        """The adjunction must hold for overlapping alphabets too."""
        R = SHARED.relation
        cx, cy = contracts_for(SHARED)
        assert cx.refines(cx.lan(R).pullback(R))
        assert cy.pullback(R).lan(R).refines(cy)


# ---------------------------------------------------------------------------
# The Laplacian -- Ghrist Section 6, RG Section IV
# ---------------------------------------------------------------------------

class TestHodgeLawvere:
    def test_laplacian_is_monotone(self, build_sheaf):
        """
        Ghrist Lemma 6.5: L is a Q-functor. Monotonicity is what makes the flow
        a descending chain rather than an arbitrary walk, so everything below
        rests on it. Checked by descending one node and confirming the whole
        image descends.
        """
        F = build_sheaf()
        x = F.assignment()
        y = dict(x)
        victim = next(iter(F.nodes()))
        y[victim] = bot()                     # bot refines everything
        Lx, Ly = F.laplacian(x), F.laplacian(y)
        assert all(Ly[i].refines(Lx[i]) for i in F.nodes())

    @pytest.mark.parametrize("legs", [KAN, CO])
    def test_suffix_points_are_global_sections(self, build_sheaf, legs):
        """
        Ghrist Thm. 6.10, the Hodge-Lawvere Theorem. Over this Boolean base
        with the trivial weighting, and with every interface contributing both
        orientations, the suffix inequality holds both ways and so collapses to
        equality on the edge stalk. Suffix points are therefore exactly the
        strict global sections -- before and after the flow alike.
        """
        F = build_sheaf()
        assert F.is_suffix(legs=legs) == F.is_section(legs=legs)
        F.converge(legs=legs, max_sweeps=50)
        assert F.is_suffix(legs=legs) == F.is_section(legs=legs)

    def test_flow_reaches_a_section(self, build_sheaf):
        """Ghrist Def. 6.13 / RG Eq. (6): the flow settles at a section."""
        F = build_sheaf()
        F.converge(verify=True, max_sweeps=50)
        assert F.is_section()

    def test_flow_descends(self, build_sheaf):
        """
        The flow is x <- Lx ^ x, so every node refines what it held before.
        Monotone descent is what rules out cycling.
        """
        F = build_sheaf()
        before = F.assignment()
        F.converge(max_sweeps=50)
        assert all(F.contract(i).refines(before[i]) for i in F.nodes())

    def test_flow_never_descends_past_a_section_below_it(self, build_sheaf):
        """
        Ghrist Prop. 6.15, the containment half: a section already below the
        initial assignment is still below the limit, so the flow cannot
        overshoot. `bot` everywhere is a section on any sheaf, giving a witness
        that exists on every fixture.
        """
        F = build_sheaf()
        witness = {i: bot() for i in F.nodes()}
        assert _is_section_of(F, witness), "bot everywhere should be a section"

        F.converge(max_sweeps=50)
        assert all(witness[i].refines(F.contract(i)) for i in F.nodes())

    def test_limit_is_the_greatest_section_below_the_initial_cochain(self):
        """
        Ghrist Prop. 6.15 with a witness that actually bites. On the single
        edge, x[0] = (xa > 0, true) and the flow settles at (xa > 0, xb > 0).
        The band 0 < x < 5 is a strictly smaller section that is still below
        x[0], so it must also be below the limit -- and the limit must not be
        below *it*, or the flow would have descended further than the greatest
        section allows.
        """
        from conftest import single_edge
        xa, xb = z3.Int("xa"), z3.Int("xb")
        F = single_edge()
        witness = {
            "a": Contract(z3.BoolVal(True), z3.And(xa > 0, xa < 5)),
            "b": Contract(z3.BoolVal(True), z3.And(xb > 0, xb < 5)),
        }
        assert _is_section_of(F, witness), "the band should be a section"
        assert all(witness[i].refines(F.contract(i)) for i in F.nodes()), \
            "the band should sit below the initial cochain"

        F.converge(max_sweeps=50)
        limit = F.assignment()
        assert all(witness[i].refines(limit[i]) for i in F.nodes()), \
            "the flow descended past a section below the initial cochain"
        assert not all(limit[i].refines(witness[i]) for i in F.nodes()), \
            "the limit should be the GREATEST such section, not just any"

    def test_fixed_point_is_stable(self, build_sheaf):
        """Once settled, further steps change nothing."""
        F = build_sheaf()
        F.converge(max_sweeps=50)
        assert F.laplacian_update() is True

    def test_contradiction_collapses_to_bottom_and_is_localised(self):
        """
        RG's practical argument for a Laplacian over a monolithic consistency
        check: an unsatisfiable interface drives the offending vertices to bot,
        and `collapsed` names them.
        """
        from conftest import contradictory
        F = contradictory()
        F.converge(max_sweeps=50)
        assert F.collapsed() == {"a": "inconsistent", "b": "inconsistent"}
        assert F.is_section(), "bot everywhere is still a section"


class TestDualLaplacian:
    """
    Ghrist Section 7. The two readings of the dual are genuinely different, and
    only one of them says anything over this base; these tests record which.
    """

    def test_co_prefix_points_are_ran_sections(self, build_sheaf):
        """
        With `legs="co"` the bisheaf is `dual_pullback -| ran`, whose adjunction
        transposes the prefix condition into exact `ran`-agreement across every
        interface. This is a dual notion of consistency with real content.
        """
        F = build_sheaf()
        assert F.is_prefix(legs=CO) == F.is_section(legs=CO)
        F.converge(dual=True, legs=CO, max_sweeps=50)
        assert F.is_prefix(legs=CO) == F.is_section(legs=CO)

    def test_kan_prefix_is_not_an_agreement_condition(self):
        """
        Def. 7.3 Eq. (13) read literally -- same legs as the primal, aggregated
        by join. Here the adjunction points the wrong way to transpose
        `pullback(z) <= x_i`, so prefix points are not characterised by edge
        agreement. That is the expected degeneracy, not a defect: Thm. 7.5 needs
        the linear negation of a Girard quantale, and over the Boolean base with
        W = top the dualising element is bottom, making Def. 7.4's cosection
        condition vacuous.
        """
        from conftest import private_variable
        F = private_variable()
        F.converge(max_sweeps=50)
        assert F.is_section(), "primal flow reaches a strict global section"
        assert not F.is_prefix(legs=KAN), \
            "if this ever passes, Eq. (13) has acquired content over B"

    def test_harmonic_is_strictly_stronger_than_being_a_section(self):
        """
        Ghrist Thm. 7.6 identifies the two-sidedly harmonic cochains with the
        strict global sections. That identification needs a Girard base with a
        non-degenerate weighting; here the primal half alone already pins down
        the sections, so `is_harmonic` is strictly stronger. A section that is
        not harmonic witnesses the gap.
        """
        from conftest import private_variable
        F = private_variable()
        F.converge(max_sweeps=50)
        assert F.is_section()
        assert not F.is_harmonic(legs=KAN)

    def test_harmonic_requires_a_single_bisheaf(self, build_sheaf):
        """
        `legs` is shared across both halves rather than free on each, because
        conjoining a suffix condition on one bisheaf with a prefix condition on
        another is meaningless. Whatever `is_harmonic` reports, it must equal
        the conjunction of its two halves read along the same legs.
        """
        F = build_sheaf()
        for legs in (KAN, CO):
            assert F.is_harmonic(legs=legs) == (F.is_suffix(legs=legs)
                                                and F.is_prefix(legs=legs))


def _is_section_of(F, assignment):
    """Whether `assignment` is a global section of `F`, leaving `F` untouched."""
    saved = F.assignment()
    try:
        for i, c in assignment.items():
            F.nodes[i]['c'] = c
        return F.is_section()
    finally:
        for i, c in saved.items():
            F.nodes[i]['c'] = c


def _decls(expr):
    """Function declarations occurring free in `expr`."""
    seen, out, stack = set(), [], [expr]
    while stack:
        e = stack.pop()
        if e.get_id() in seen:
            continue
        seen.add(e.get_id())
        if z3.is_quantifier(e):
            stack.append(e.body())
            continue
        if z3.is_app(e):
            if e.num_args() == 0:
                out.append(e.decl())
            stack.extend(e.children())
    return out
