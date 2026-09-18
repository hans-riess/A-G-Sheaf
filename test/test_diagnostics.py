"""
The sheaf-level diagnostics, checked against the predicates they refine.

Each diagnostic is a magnitude whose zero set is supposed to be an existing
yes/no answer. That correspondence is the whole claim, so it is tested on every
fixture sheaf, in both bisheaves, and at both ends of the flow -- an assignment
before the flow has done anything and the fixed point it settles at.

The fixtures speak integer arithmetic, which has no counting measure, so they
are measured over an explicit window (`Alphabet`'s `domains`). Distances are
therefore relative to that window: two contracts differing only outside it read
as equal. That does not weaken these tests -- a diagnostic that disagreed with
its predicate would do so inside the window as readily as outside -- but it is
why the numbers below are not comparable across fixtures.

References:
  RG      -- Riess and Ghrist (2022), Diffusion of Information on Networked Lattices by Gossip
  Ghrist  -- Ghrist et al. (2026), Categorical Diffusion of Weighted Lattices
"""

import random

import pytest
import z3

from conftest import BUILDERS
from agsheaf.contracts import Contract
from agsheaf.diagnostics import (Alphabets, concrete_disagreement, dirichlet,
                                 distance_profile, divergence_profile,
                                 edge_disagreement, edge_energies,
                                 laplacian_residual, node_variables, relabel)
from agsheaf.ensembles import random_sheaf
from agsheaf.measure import canonical
from agsheaf.sheaf import CO, KAN

#: The integer fixtures use small values; this window covers them with room to
#: spare on both sides, so a contract that constrains nothing inside it really
#: does constrain nothing there.
WINDOW = range(-4, 16)


def alphabets(sheaf):
    """`Alphabets` for a fixture sheaf, windowing whatever integers it holds."""
    names = {}
    for i in sheaf.nodes():
        for v in node_variables(sheaf, i):
            names[str(v)] = WINDOW
    for u, v in sheaf.interfaces():
        for var in sheaf.restriction(u, v).target_vars:
            names[str(var)] = WINDOW
    return Alphabets(sheaf, domains=names)


def canonicalising(sheaf, alphabets_):
    """
    An `on_step` callback rewriting every stalk into its canonical form. What
    the convergence runner does, and what keeps a flow over a coarse abstraction
    from accumulating a quantifier layer per sweep.
    """
    def rewrite(step, unchanged):
        for i in sheaf.nodes():
            sheaf.nodes[i]['c'] = canonical(sheaf.contract(i), alphabets_.node(i))
    return rewrite


def guaranteeing(var):
    """A contract guaranteeing one Boolean, for the relabelling tests."""
    return Contract(z3.BoolVal(True), var, vars=[var])


@pytest.fixture(params=sorted(BUILDERS), ids=lambda n: n)
def fixture_sheaf(request):
    return BUILDERS[request.param]()


class TestZeroSets:
    """
    Each diagnostic against the predicate it refines, before and after the flow.
    """

    @pytest.mark.parametrize("legs", [KAN, CO])
    def test_dirichlet_is_zero_exactly_on_sections(self, fixture_sheaf, legs):
        F, A = fixture_sheaf, alphabets(fixture_sheaf)
        assert (dirichlet(F, A, legs=legs) == 0.0) == F.is_section(legs=legs)
        F.converge(max_sweeps=50)
        assert (dirichlet(F, A, legs=legs) == 0.0) == F.is_section(legs=legs)

    def test_the_primal_flow_drives_the_energy_to_zero(self, fixture_sheaf):
        """Ghrist Thm. 6.10 with a magnitude: the flow closes every interface."""
        F, A = fixture_sheaf, alphabets(fixture_sheaf)
        F.converge(max_sweeps=50)
        assert dirichlet(F, A) == 0.0
        assert F.is_section()

    @pytest.mark.parametrize("legs", [KAN, CO])
    def test_edge_disagreement_is_zero_exactly_on_agreeing_interfaces(
            self, fixture_sheaf, legs):
        F, A = fixture_sheaf, alphabets(fixture_sheaf)
        for u, v in F.interfaces():
            assert (edge_disagreement(F, u, v, A, legs=legs) == 0.0) \
                == F.agrees_on(u, v, legs=legs)

    def test_edge_energies_sum_to_the_dirichlet_energy(self, fixture_sheaf):
        F, A = fixture_sheaf, alphabets(fixture_sheaf)
        assert sum(edge_energies(F, A).values()) == pytest.approx(dirichlet(F, A))

    def test_residual_is_zero_exactly_on_suffix_points(self, fixture_sheaf):
        F, A = fixture_sheaf, alphabets(fixture_sheaf)
        assert (laplacian_residual(F, A) == 0.0) == F.is_suffix()
        F.converge(max_sweeps=50)
        assert (laplacian_residual(F, A) == 0.0) == F.is_suffix()

    def test_dual_residual_is_zero_exactly_on_prefix_points(self, fixture_sheaf):
        """
        The dual with content: `legs="co"` transports along `dual_pullback -| ran`,
        whose prefix points are exact `ran`-agreement (`doc/MATH.md`).
        """
        F, A = fixture_sheaf, alphabets(fixture_sheaf)
        assert (laplacian_residual(F, A, dual=True, legs=CO) == 0.0) == F.is_prefix(legs=CO)
        F.converge(dual=True, legs=CO, max_sweeps=50)
        assert (laplacian_residual(F, A, dual=True, legs=CO) == 0.0) == F.is_prefix(legs=CO)


class TestMonotonicity:
    def test_the_primal_flow_never_moves_a_stalk_upwards(self, fixture_sheaf):
        """
        Ghrist Def. 6.13 as a certificate rather than a claim: every step must
        leave `divergence(x_next, x_now)` at zero on every node, since the flow
        descends. A nonzero entry names the node that moved the wrong way.
        """
        F, A = fixture_sheaf, alphabets(fixture_sheaf)
        previous = F.assignment()

        def check(step, unchanged):
            nonlocal previous
            current = F.assignment()
            assert all(d == 0.0 for d in divergence_profile(F, A, current, previous).values()), \
                f"the primal flow ascended at step {step}"
            previous = current

        F.converge(max_sweeps=50, on_step=check)

    def test_the_dual_flow_never_moves_a_stalk_downwards(self, fixture_sheaf):
        F, A = fixture_sheaf, alphabets(fixture_sheaf)
        previous = F.assignment()

        def check(step, unchanged):
            nonlocal previous
            current = F.assignment()
            assert all(d == 0.0 for d in divergence_profile(F, A, previous, current).values()), \
                f"the dual flow descended at step {step}"
            previous = current

        F.converge(dual=True, legs=CO, max_sweeps=50, on_step=check)

    def test_distance_to_the_limit_decreases(self, fixture_sheaf):
        """
        The plot the whole study rests on. Because the iterates nest, the
        distance to the fixed point is not merely shrinking but monotone --
        so a curve that ever rises is a bug, not slow convergence.
        """
        F, A = fixture_sheaf, alphabets(fixture_sheaf)
        trace = [F.assignment()]
        F.converge(max_sweeps=50, on_step=lambda step, still: trace.append(F.assignment()))
        target = trace[-1]
        curve = [sum(distance_profile(F, A, x, target).values()) for x in trace]
        assert curve[-1] == 0.0
        assert all(curve[t] >= curve[t + 1] - 1e-12 for t in range(len(curve) - 1)), curve


class TestObserver:
    def test_on_step_sees_every_iterate_including_the_settling_one(self):
        """
        The callback must fire once per step taken, the step that changes
        nothing included -- that last one is the fixed point, and a trace
        missing it has no limit to measure against.
        """
        F = BUILDERS["path"]()
        seen = []
        steps = F.converge(max_sweeps=50, on_step=lambda step, still: seen.append((step, still)))
        assert [step for step, _ in seen] == list(range(steps + 1))
        assert seen[-1][1] is True
        assert not any(still for _, still in seen[:-1])

    def test_a_semantics_preserving_callback_does_not_change_the_limit(self):
        """
        What licenses canonicalising from inside the flow: the termination test
        is contract equality, which cannot tell a formula from an equivalent
        one, so rewriting each stalk between steps changes the cost and not the
        answer.
        """
        plain = BUILDERS["shared_alphabet"]()
        plain.converge(max_sweeps=50)

        rewritten = BUILDERS["shared_alphabet"]()
        A = alphabets(rewritten)

        def canonicalise(step, still):
            for i in rewritten.nodes():
                rewritten.nodes[i]['c'] = canonical(rewritten.contract(i), A.node(i))

        rewritten.converge(max_sweeps=50, on_step=canonicalise)
        assert all(plain.contract(i) == rewritten.contract(i) for i in plain.nodes())


class TestConcreteAgainstAbstract:
    """
    The measurement the demonstration's redesign turns on: agreement in the
    shared vocabulary against agreement in the agents' own (`doc/DUALITY.md`).
    """

    def test_relabel_reads_a_contract_in_another_agents_copy(self):
        xa, xb = z3.Bools("a_0 b_0")
        renamed = relabel(guaranteeing(xa), [xa], [xb])
        assert renamed == guaranteeing(xb)

    def test_relabel_refuses_alphabets_of_different_size(self):
        xa, xb = z3.Bools("a_0 b_0")
        with pytest.raises(ValueError, match="different size"):
            relabel(guaranteeing(xa), [xa], [xa, xb])

    def test_a_bijective_restriction_leaves_no_concrete_disagreement(self):
        """
        At `coarseness=1` the abstraction is the identity, its fibres are
        singletons, and agreeing abstractly *is* agreeing concretely. This is
        the case the demonstration currently runs.
        """
        F = random_sheaf(agents=4, topology="path", facts=3, coarseness=1,
                         knowledge=0.7, error=0.4, rng=random.Random(4))
        A = Alphabets(F)
        F.converge(max_sweeps=50, on_step=canonicalising(F, A))
        assert dirichlet(F, A) == 0.0
        assert all(concrete_disagreement(F, u, v, A) == 0.0 for u, v in F.interfaces())

    def test_a_coarse_restriction_tolerates_concrete_disagreement(self):
        """
        Above `coarseness=1` the fibres are non-trivial, and the flow closes the
        interfaces while the agents still hold different concrete beliefs. That
        residue is the kernel of `f_!` -- the disagreement the shared vocabulary
        cannot see and the mission therefore tolerates.
        """
        found = False
        for seed in range(12):
            F = random_sheaf(agents=4, topology="path", facts=4, coarseness=4,
                             aggregator="or", knowledge=0.7, error=0.4,
                             rng=random.Random(seed))
            A = Alphabets(F)
            F.converge(max_sweeps=50, on_step=canonicalising(F, A))
            assert dirichlet(F, A) == 0.0, "the flow must still reach a section"
            if any(concrete_disagreement(F, u, v, A) > 0.0 for u, v in F.interfaces()):
                found = True
                break
        assert found, "no coarse instance kept a concrete disagreement past the section"
