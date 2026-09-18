"""
The random sheaf generator: that what it produces is a sheaf, and that its
parameters mean what the study reads them as meaning.

A generator is easy to get subtly wrong in ways no downstream plot reveals -- an
abstraction that is secretly still a bijection, a "disagreement" knob that
produces agreement, an ensemble that has quietly degenerated into one instance.
Each of those would show up as a clean, wrong figure. These tests pin the
parameters to the properties the convergence study attributes to them.

References:
  Naik    -- Naik et al. (2025), Contract Embeddings for Layered Control Architectures
  RG      -- Riess and Ghrist (2022), Diffusion of Information on Networked Lattices by Gossip
"""

import random

import networkx as nx
import pytest
import z3

from agsheaf.contracts import Contract, _is_valid
from agsheaf.diagnostics import Alphabets, dirichlet
# `_contract` is private, and reaching for it is deliberate: these tests build
# the two-agent instances the generator would only reach by chance.
from agsheaf.ensembles import (AGGREGATORS, ENCODINGS, TOPOLOGIES, _contract,
                               abstraction, blocks_of, random_sheaf)
from agsheaf.measure import Alphabet, canonical
from agsheaf.sheaf import CO, KAN, ContractSheaf


def canonicalising(sheaf, alphabets):
    def rewrite(step, unchanged):
        for i in sheaf.nodes():
            sheaf.nodes[i]['c'] = canonical(sheaf.contract(i), alphabets.node(i))
    return rewrite


class TestWellFormed:
    @pytest.mark.parametrize("topology", sorted(TOPOLOGIES))
    def test_every_topology_gives_a_connected_sheaf_on_range_n(self, topology):
        F = random_sheaf(agents=5, topology=topology, facts=3,
                         rng=random.Random(1))
        assert sorted(F.nodes()) == list(range(5))
        assert nx.is_connected(nx.Graph(F.to_undirected()))

    def test_every_interface_carries_both_directions(self):
        """
        `add_interface` puts both in; a generator that added only one would make
        `neighbors` empty on one side and the Laplacian would silently return
        the lattice identity there (`sheaf.ContractSheaf`).
        """
        F = random_sheaf(agents=4, topology="cycle", facts=3, rng=random.Random(2))
        for u, v in F.interfaces():
            assert F.has_edge(u, v) and F.has_edge(v, u)
            assert [str(x) for x in F.restriction(u, v).target_vars] \
                == [str(x) for x in F.restriction(v, u).target_vars]

    def test_nodes_carry_their_variables(self):
        """
        `diagnostics` needs an alphabet for a stalk that has collapsed, and a
        `bot()` contract mentions no variables at all.
        """
        F = random_sheaf(agents=3, facts=4, encoding="boolean", rng=random.Random(3))
        for i in F.nodes():
            assert len(F.nodes[i]['vars']) == 4

    def test_provenance_is_recorded(self):
        F = random_sheaf(agents=3, topology="star", facts=3, coarseness=2,
                         knowledge=0.4, error=0.2, rng=random.Random(4))
        assert F.graph["topology"] == "star"
        assert F.graph["coarseness"] == 2
        assert F.graph["diameter"] == 2
        assert F.graph["interfaces"] == len(F.interfaces())
        assert len(F.graph["ground_truth"]) == 3
        assert F.graph["attempts"] >= 1


class TestDisagreementIsGuaranteed:
    def test_the_starting_assignment_is_never_already_a_section(self):
        """
        The rejection sampling that keeps the ensemble from measuring nothing:
        without it a large share of every cell converges in zero sweeps.
        """
        for seed in range(20):
            F = random_sheaf(agents=4, facts=3, knowledge=0.5, error=0.3,
                             rng=random.Random(seed))
            assert not F.is_section()

    def test_a_cell_with_nothing_to_do_is_reported_rather_than_returned(self):
        """An agent that knows nothing can disagree with no one."""
        with pytest.raises(RuntimeError, match="nothing for"):
            random_sheaf(agents=3, facts=3, knowledge=0.0, error=0.0,
                         rng=random.Random(5), attempts=8)

    def test_disagreement_can_be_waived(self):
        F = random_sheaf(agents=3, facts=3, knowledge=0.0, error=0.0,
                         require_disagreement=False, rng=random.Random(5))
        assert F.is_section()


class TestAbstraction:
    def test_coarseness_one_is_a_bijection_and_the_legs_coincide(self):
        """
        Naik Def. 6 read backwards: when the restriction is a bijection, `f_!`
        and `f_*` agree, so the two bisheaves cannot be told apart. This is the
        `f = id` degeneracy of `doc/DUALITY.md` -- reproduced here on purpose, as
        the baseline the coarser cases are compared against.
        """
        F = random_sheaf(agents=4, topology="path", facts=3, coarseness=1,
                         rng=random.Random(6))
        assert F.section_edges(legs=KAN) == F.section_edges(legs=CO)
        x = F.assignment()
        for u, v in F.interfaces():
            assert F.transport(x, u, v, legs=KAN) == F.transport(x, u, v, legs=CO)

    def test_a_coarse_restriction_separates_the_legs(self):
        """
        And the point of the parameter: above `coarseness=1` the fibres are
        non-trivial and `f_!` stops agreeing with `f_*`, which is what makes
        `sheaf.legs` mean the report's case (1) against case (2).
        """
        F = random_sheaf(agents=4, topology="path", facts=4, coarseness=4,
                         knowledge=0.8, error=0.3, rng=random.Random(7))
        x = F.assignment()
        assert any(F.transport(x, u, v, legs=KAN) != F.transport(x, u, v, legs=CO)
                   for u, v in F.interfaces())

    def test_blocks_partition_the_facts(self):
        assert [list(b) for b in blocks_of(5, 2)] == [[0, 1], [2, 3], [4]]
        assert [list(b) for b in blocks_of(4, 1)] == [[0], [1], [2], [3]]
        with pytest.raises(ValueError, match="at least 1"):
            blocks_of(4, 0)

    @pytest.mark.parametrize("aggregator", sorted(AGGREGATORS))
    def test_every_aggregator_gives_a_total_function(self, aggregator):
        """
        The restriction has to be the graph of a *function* for `lan` to be
        Naik's `f_!`: every concrete state must have exactly one abstract image.
        """
        scheme = ENCODINGS["boolean"](3)
        node = scheme.stalk_vars("u", 4)
        shared = scheme.shared_vars("u", "v", len(blocks_of(4, 2)))
        rel = abstraction(scheme, node, shared, 4, 2, aggregator)

        solver = z3.Solver()
        other = [z3.Bool(f"{v}_alt") for v in shared]
        solver.add(rel.rel)
        solver.add(z3.substitute(rel.rel, *zip(shared, other)))
        solver.add(z3.Or(*[a != b for a, b in zip(shared, other)]))
        assert solver.check() == z3.unsat, "two abstract images for one concrete state"


class TestEncodings:
    def test_the_boolean_encoding_collapses_a_contradiction(self):
        """
        RG section IV: contradicting guarantees meet at an unsatisfiable one, the
        stalk reaches `bot`, and `collapsed()` localises it. Faithful, and
        contagious -- one contradicted fact takes the whole stalk with it.
        """
        F = contradicting("boolean")
        A = Alphabets(F)
        F.converge(max_sweeps=20, on_step=canonicalising(F, A))
        assert F.collapsed() == {0: "inconsistent", 1: "inconsistent"}
        assert F.is_section(), "bot everywhere is still a section"

    def test_the_possibility_encoding_keeps_a_contradiction_in_the_lattice(self):
        """
        What `gridsheaf` does instead, and what `doc/MATH.md` claims for it: a
        contested fact fuses to the *empty possibility set*, which is a state of
        the alphabet rather than a failure of it. Nothing collapses, and the
        disagreement stays localised to the one fact it is about.
        """
        F = contradicting("possibility")
        A = Alphabets(F)
        F.converge(max_sweeps=20, on_step=canonicalising(F, A))
        assert F.collapsed() == {}
        assert F.is_section()
        assert dirichlet(F, A) == 0.0

        # and the conflict is *recorded*, not merely survived: the fused stalk
        # entails that no label remains possible for the fact they fought over.
        empty_set = z3.And(*[z3.Not(v) for v in F.nodes[0]['vars']])
        assert _is_valid(z3.Implies(F.contract(0).sat_g, empty_set))

    def test_an_ignorant_agent_sits_at_the_top_of_its_lattice(self):
        """
        Knowing nothing must be the identity for the meet, or an agent joining
        the network would weaken everyone it talks to.
        """
        scheme = ENCODINGS["possibility"](3)
        variables = scheme.stalk_vars(0, 2)
        empty = _contract(scheme, variables, {}, {})
        sigma = Alphabet(variables)
        assert canonical(empty, sigma) == Contract(z3.BoolVal(True), z3.BoolVal(True),
                                                   vars=variables)


def contradicting(encoding):
    """
    Two agents, one fact, opposite beliefs -- the smallest instance in which the
    two encodings disagree about what a conflict costs.
    """
    scheme = ENCODINGS[encoding](3)
    F = ContractSheaf()
    variables = {}
    for i, value in ((0, 0), (1, 1)):
        variables[i] = scheme.stalk_vars(i, 1)
        held = {0: bool(value)} if encoding == "boolean" else {0: value}
        F.add_node(i, c=_contract(scheme, variables[i], held, {}), vars=variables[i])
    shared = scheme.shared_vars(0, 1, 1)
    F.add_interface(0, 1,
                    abstraction(scheme, variables[0], shared, 1, 1, "or"),
                    abstraction(scheme, variables[1], shared, 1, 1, "or"))
    return F
