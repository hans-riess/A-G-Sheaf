"""
The disagreement measure, checked against the things it is supposed to refine.

Every claim here has the same shape: a number this module computes must agree
with a decision z3 already makes. `distance` is zero exactly where
`Contract.__eq__` says equal, `divergence` is zero exactly where
`Contract.refines` says refines, `canonical` returns something `==` its input.
A measure that drifted from the lattice it measures would still produce smooth
plots, which is precisely why it has to be pinned to the solver rather than to
its own arithmetic.

References:
  RG      -- Riess and Ghrist (2022), Diffusion of Information on Networked Lattices by Gossip
  Ghrist  -- Ghrist et al. (2026), Categorical Diffusion of Weighted Lattices
"""

import itertools
import random

import numpy as np
import pytest
import z3

from agsheaf.contracts import Contract, Relation, bot, top
from agsheaf.measure import (Alphabet, AlphabetTooLarge, canonical, denotation,
                             distance, divergence, equivalent, interval_volume,
                             refines)
from agsheaf.simplify import ast_size, has_quantifier, qf


P, Q, R = z3.Bools("p q r")
ABC = [P, Q, R]


@pytest.fixture
def sigma():
    """Three Booleans: eight states, small enough to reason about by hand."""
    return Alphabet(ABC)


def random_contract(rng, variables=ABC):
    """A contract whose slots are random small clauses over `variables`."""
    def slot():
        chosen = rng.sample(list(variables), rng.randint(1, len(variables)))
        lits = [v if rng.random() < 0.5 else z3.Not(v) for v in chosen]
        if rng.random() < 0.15:
            return z3.BoolVal(rng.random() < 0.5)
        return z3.Or(*lits) if rng.random() < 0.5 else z3.And(*lits)
    return Contract(slot(), slot(), vars=variables)


def random_pairs(count=120, seed=20260811, variables=ABC):
    rng = random.Random(seed)
    return [(random_contract(rng, variables), random_contract(rng, variables))
            for _ in range(count)]


class TestAlphabet:
    def test_states_are_indexed_in_product_order(self, sigma):
        assert sigma.size == 8
        assert [str(v) for v in sigma.names] == ["p", "q", "r"]

    def test_models_matches_brute_force(self, sigma):
        """
        The whole module rests on `models`, so it is checked against the
        definition rather than against itself: substitute, ask z3, compare.
        """
        phi = z3.And(z3.Or(P, Q), z3.Not(z3.And(Q, R)))
        expected = []
        for state in itertools.product([False, True], repeat=3):
            grounded = z3.simplify(z3.substitute(
                phi, *[(v, z3.BoolVal(b)) for v, b in zip(ABC, state)]))
            expected.append(z3.is_true(grounded))
        assert list(sigma.models(phi)) == expected

    def test_a_formula_over_strangers_is_refused(self, sigma):
        """
        Measuring a formula with a free variable outside the alphabet would read
        that variable as universally quantified -- a different formula, silently.
        """
        with pytest.raises(ValueError, match="free in the formula"):
            sigma.models(z3.And(P, z3.Bool("elsewhere")))

    def test_an_alphabet_too_big_to_enumerate_is_refused(self):
        with pytest.raises(AlphabetTooLarge):
            Alphabet(z3.Bools(" ".join(f"b{i}" for i in range(20))), max_states=1024)

    def test_an_integer_needs_an_explicit_window(self):
        x = z3.Int("x")
        with pytest.raises(TypeError, match="not finite"):
            Alphabet([x])
        windowed = Alphabet([x], domains={"x": range(0, 4)})
        assert windowed.measure(windowed.models(x > 1)) == 0.5

    def test_weights_reweigh_the_measure(self):
        """
        A non-uniform measure changes magnitudes and nothing else: the sets are
        the same sets, so what is zero stays zero.
        """
        plain = Alphabet([P])
        skewed = Alphabet([P], weights=[3.0, 1.0])
        assert plain.measure(plain.models(P)) == 0.5
        assert skewed.measure(skewed.models(P)) == 0.25
        assert skewed.empty(skewed.models(z3.And(P, z3.Not(P))))


class TestQuantifierElimination:
    def test_qe_on_a_goal_is_not_equivalence_and_qf_catches_it(self):
        """
        The regression this guard exists for. A z3 `Goal` is a satisfiability
        problem, so a tactic may decide it by choosing values for the free
        constants; `Then(simplify, qe2)` returns the empty goal -- `True` -- for
        a formula that is false at `e = False`. Every contract passed through
        here has free variables, so an unchecked elimination would corrupt every
        measurement downstream rather than a rare one.
        """
        e = z3.Bool("e")
        phi = z3.ForAll(ABC, z3.Implies(e == z3.Or(P, Q), z3.Or(P, Q)))
        goal = z3.Goal()
        goal.add(phi)
        naive = z3.Then(z3.Tactic('simplify'), z3.Tactic('qe2'))(goal).as_expr()
        assert z3.is_true(naive), "the trap this test is about has changed shape"

        out = qf(phi)
        solver = z3.Solver()
        solver.add(out != phi)
        assert solver.check() == z3.unsat, f"qf returned a non-equivalent {out}"

    def test_a_transported_contract_is_measured_correctly(self):
        """
        The same trap seen through `models`: `lan` puts a `ForAll` on the
        assumption, and the pushed contract admits only `e = True`.
        """
        e = z3.Bool("e")
        c = Contract(z3.Or(P, Q), z3.And(Q, R), vars=ABC)
        pushed = c.lan(Relation(e == z3.Or(P, Q), ABC, [e]))
        edge = Alphabet([e])
        assert list(edge.models(pushed.a)) == [False, True]

    def test_qf_leaves_what_it_cannot_eliminate(self):
        """A failed elimination returns the input, which is always sound."""
        x, y = z3.Int("x"), z3.Int("y")
        phi = z3.ForAll([y], z3.Implies(y > x, y * y > x))
        assert qf(phi) is phi or not has_quantifier(qf(phi))


class TestMetricAxioms:
    def test_distance_is_symmetric_and_bounded(self, sigma):
        for c1, c2 in random_pairs():
            d = distance(c1, c2, sigma)
            assert 0.0 <= d <= 1.0
            assert d == distance(c2, c1, sigma)

    def test_zero_distance_is_z3_equality(self, sigma):
        """
        Identity of indiscernibles, against the solver rather than against the
        arithmetic: `distance == 0` must mean equal *as contracts*, saturation
        and all.
        """
        for c1, c2 in random_pairs():
            assert (distance(c1, c2, sigma) == 0.0) == (c1 == c2)

    def test_triangle_inequality(self, sigma):
        rng = random.Random(11)
        trios = [(random_contract(rng), random_contract(rng), random_contract(rng))
                 for _ in range(80)]
        for a, b, c in trios:
            assert distance(a, c, sigma) <= distance(a, b, sigma) + distance(b, c, sigma) + 1e-12

    def test_top_and_bottom_are_a_full_diameter_apart(self, sigma):
        assert distance(top(ABC), bot(ABC), sigma) == 1.0


class TestDivergence:
    def test_zero_divergence_is_refinement(self, sigma):
        """
        `divergence` is the Lawvere metric of the refinement order, so its zero
        set is the order itself -- checked against `Contract.refines`, which z3
        decides.
        """
        for c1, c2 in random_pairs():
            assert (divergence(c1, c2, sigma) == 0.0) == c1.refines(c2)
            assert refines(c1, c2, sigma) == c1.refines(c2)

    def test_the_two_directions_add_up_to_the_distance(self, sigma):
        for c1, c2 in random_pairs():
            assert divergence(c1, c2, sigma) + divergence(c2, c1, sigma) \
                == pytest.approx(distance(c1, c2, sigma))

    def test_divergence_is_directed(self, sigma):
        """A strict refinement is zero one way round and positive the other."""
        strict, loose = Contract(z3.BoolVal(True), z3.And(P, Q), vars=ABC), \
            Contract(z3.BoolVal(True), P, vars=ABC)
        assert strict.refines(loose)
        assert divergence(strict, loose, sigma) == 0.0
        assert divergence(loose, strict, sigma) > 0.0


class TestIntervalVolume:
    def test_interval_volume_is_the_distance(self, sigma):
        """
        Computed through the shipped `meet` and `join` rather than through the
        denotations directly, so agreement here is what ties the measure to
        `contracts.py`'s lattice instead of to a paraphrase of it.
        """
        for c1, c2 in random_pairs():
            assert interval_volume(c1, c2, sigma) == pytest.approx(distance(c1, c2, sigma))

    def test_the_interval_of_a_contract_with_itself_is_a_point(self, sigma):
        rng = random.Random(3)
        for _ in range(20):
            c = random_contract(rng)
            assert interval_volume(c, c, sigma) == 0.0


class TestCanonical:
    def test_canonical_preserves_the_contract(self, sigma):
        for c1, _ in random_pairs():
            assert canonical(c1, sigma) == c1

    def test_canonical_is_idempotent(self, sigma):
        for c1, _ in random_pairs(count=40):
            once = canonical(c1, sigma)
            assert equivalent(canonical(once, sigma), once, sigma)

    def test_canonical_is_saturated(self, sigma):
        """
        `[[A => G]]` always contains the complement of `[[A]]`, so rebuilding the
        second slot from what it denotes lands in the saturated class by
        construction rather than by a further step.
        """
        for c1, _ in random_pairs(count=40):
            assert canonical(c1, sigma).is_saturated()

    def test_canonical_shrinks_a_transported_contract(self):
        """
        What this is for. Transport nests a quantifier per sweep, and a stalk's
        formula grows while what it denotes does not; canonicalising takes it
        back to the size of its meaning. Without this an exact-contract flow
        does not finish (see `simplify`).
        """
        e = z3.Bool("e")
        c = Contract(z3.BoolVal(True), z3.And(P, Q), vars=ABC)
        rel = Relation(e == z3.Or(P, Q), ABC, [e])
        transported = c.lan(rel).pullback(rel)
        assert has_quantifier(transported.sat_g)

        reduced = canonical(transported, Alphabet(ABC))
        assert reduced == transported
        assert not has_quantifier(reduced.sat_g)
        assert ast_size(reduced.sat_g) < ast_size(transported.sat_g)

    def test_a_product_is_written_as_a_conjunction(self):
        """
        The factorisation that keeps canonical forms small: a set that is a
        product over its variables must not come back as a disjunction of its
        states. Six independent Booleans have 64 states; the conjunction has six
        conjuncts.
        """
        variables = z3.Bools("a0 a1 a2 a3 a4 a5")
        sigma = Alphabet(variables)
        product = z3.And(*[v if i % 2 else z3.Not(v) for i, v in enumerate(variables)])
        rebuilt = sigma.formula(sigma.models(product))
        assert ast_size(rebuilt) <= ast_size(product) + 2

    def test_a_non_product_is_still_exact(self):
        """
        Parity factorises over nothing at all, and must still come back right.

        It is the case the factorisation has to be *verified* rather than
        trusted: every pair of parity's coordinates is independent -- each
        pairwise shadow is the full square -- so the pairwise test puts all the
        axes in separate components, and only checking the product against the
        set catches that they are not jointly independent.
        """
        for width in (3, 5):
            variables = z3.Bools(" ".join(f"c{width}_{k}" for k in range(width)))
            sigma = Alphabet(variables)
            parity = variables[0]
            for v in variables[1:]:
                parity = z3.Xor(parity, v)
            rebuilt = sigma.formula(sigma.models(parity))
            assert np.array_equal(sigma.models(rebuilt), sigma.models(parity)), width

    def test_arbitrary_sets_round_trip(self):
        """
        `formula` is the inverse of `models` on *any* set, not only on the
        product-shaped ones a stalk usually denotes. Sets this size force the
        recursive cover rather than the flat normal form, which is the path the
        dual flow takes.
        """
        variables = z3.Bools("d0 d1 d2 d3 d4 d5")
        sigma = Alphabet(variables)
        rng = np.random.default_rng(20260811)
        for density in (0.05, 0.3, 0.5, 0.9):
            wanted = rng.random(sigma.size) < density
            rebuilt = sigma.formula(wanted)
            assert np.array_equal(sigma.models(rebuilt), wanted), density


class TestDenotation:
    def test_the_denotation_is_of_the_saturation_class(self, sigma):
        """`(a, g)` and `(a, a => g)` are the same contract and must denote the
        same pair of sets."""
        raw = Contract(z3.Or(P, Q), R, vars=ABC)
        saturated = Contract(raw.a, raw.sat_g, vars=ABC)
        assert np.array_equal(denotation(raw, sigma).permits,
                              denotation(saturated, sigma).permits)

    def test_permits_contains_the_rejected_environments(self, sigma):
        """
        The structural fact `canonical` leans on: nothing outside the assumption
        can violate the guarantee, so `[[A => G]]` covers the complement of
        `[[A]]`.
        """
        rng = random.Random(5)
        for _ in range(30):
            d = denotation(random_contract(rng), sigma)
            assert not (~d.admits & ~d.permits).any()
