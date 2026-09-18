"""
Mechanics of `ContractSheaf`: construction, accessors, transport, and the flow's
bookkeeping. The mathematical content lives in `test_theorems.py` and
`test_async.py`; this file covers the API those rest on, including the error
paths, since a silently malformed sheaf is the failure mode that costs most.
"""

import pytest
import z3

from conftest import (contradictory, disconnected, path, private_variable,
                      single_edge, star, triangle)
from agsheaf.contracts import Contract, Relation, bot, top
from agsheaf.sheaf import CO, KAN, ContractSheaf


class TestAddInterface:
    def test_adds_both_orientations(self):
        """
        Each endpoint restricts onto the shared edge by a different relation, so
        an interface needs both directed edges. `neighbors` follows successors
        only, so a missing reverse edge would silently empty the aggregation.
        """
        F = single_edge()
        assert F.has_edge("a", "b") and F.has_edge("b", "a")
        assert set(F.neighbors("a")) == {"b"}
        assert set(F.neighbors("b")) == {"a"}

    def test_rejects_mismatched_edge_alphabets(self):
        """
        Both endpoints must land in the *same* edge stalk, or transport would
        compose two relations that never meet.
        """
        xa, xb, e1, e2 = z3.Int("xa"), z3.Int("xb"), z3.Int("e1"), z3.Int("e2")
        F = ContractSheaf()
        F.add_node("a", c=top())
        F.add_node("b", c=top())
        with pytest.raises(ValueError, match="edge variable mismatch"):
            F.add_interface("a", "b",
                            Relation(e1 == xa, [xa], [e1]),
                            Relation(e2 == xb, [xb], [e2]))

    def test_alphabet_comparison_is_by_name_not_identity(self):
        """
        Separately constructed z3 constants of the same name denote the same
        variable, and the check must see that.
        """
        xa, xb = z3.Int("xa"), z3.Int("xb")
        F = ContractSheaf()
        F.add_node("a", c=top())
        F.add_node("b", c=top())
        F.add_interface("a", "b",
                        Relation(z3.Int("e") == xa, [xa], [z3.Int("e")]),
                        Relation(z3.Int("e") == xb, [xb], [z3.Int("e")]))
        assert F.has_edge("a", "b")


class TestAccessors:
    def test_contract_returns_the_node_datum(self):
        F = single_edge()
        assert isinstance(F.contract("a"), Contract)

    def test_contract_on_unknown_node_raises(self):
        with pytest.raises(KeyError, match="no node"):
            single_edge().contract("nope")

    def test_contract_on_node_without_one_raises_with_a_hint(self):
        F = ContractSheaf()
        F.add_node("bare")
        with pytest.raises(KeyError, match="carries no contract"):
            F.contract("bare")

    def test_restriction_returns_the_edge_relation(self):
        F = single_edge()
        assert isinstance(F.restriction("a", "b"), Relation)

    def test_restriction_hints_when_only_the_reverse_edge_exists(self):
        """
        The likeliest construction mistake is `add_edge` in one direction, so
        the error names the fix rather than just reporting a missing key.
        """
        xa, xb, e = z3.Int("xa"), z3.Int("xb"), z3.Int("e")
        F = ContractSheaf()
        F.add_node("a", c=top())
        F.add_node("b", c=top())
        F.add_edge("b", "a", rel=Relation(e == xb, [xb], [e]))
        with pytest.raises(KeyError, match="add_interface"):
            F.restriction("a", "b")

    def test_restriction_between_non_adjacent_nodes_raises(self):
        with pytest.raises(KeyError, match="no edge"):
            star().restriction("a", "b")


class TestAssignmentBookkeeping:
    def test_assignment_is_a_snapshot(self):
        F = single_edge()
        x = F.assignment()
        F.converge()
        assert x["b"] == Contract(z3.BoolVal(True), z3.BoolVal(True))

    def test_initial_is_the_assignment_as_given(self):
        F = single_edge()
        before = F.assignment()
        F.converge(max_sweeps=50)
        assert all(F.initial(i) == before[i] for i in F.nodes())

    def test_initial_survives_many_steps(self):
        """It is snapshotted on the first step only, never refreshed."""
        F = triangle()
        before = F.assignment()
        for _ in range(5):
            F.laplacian_update()
        assert all(F.initial(i) == before[i] for i in F.nodes())

    def test_reset_restores_the_initial_assignment(self):
        F = single_edge()
        before = F.assignment()
        F.converge(max_sweeps=50)
        F.reset()
        assert all(F.contract(i) == before[i] for i in F.nodes())

    def test_initial_falls_back_for_untouched_nodes(self):
        F = single_edge()
        assert F.initial("a") == F.contract("a")

    def test_collapsed_is_empty_for_a_healthy_sheaf(self):
        assert single_edge().collapsed() == {}

    def test_collapsed_localises_the_conflict(self):
        F = contradictory()
        F.converge(max_sweeps=50)
        assert F.collapsed() == {"a": "inconsistent", "b": "inconsistent"}

    def test_collapsed_reports_incompatibility(self):
        F = ContractSheaf()
        F.add_node("x", c=top())
        assert F.collapsed() == {"x": "incompatible"}


class TestTransport:
    def test_lands_in_the_target_alphabet(self):
        F = single_edge()
        moved = F.transport(F.assignment(), "a", "b")
        assert moved.vars == frozenset({"xb"})

    def test_renames_the_guarantee_across_the_interface(self):
        F = single_edge()
        moved = F.transport(F.assignment(), "a", "b")
        assert moved == Contract(z3.BoolVal(True), z3.Int("xb") > 0)

    def test_carries_the_assumption_too(self):
        xa = z3.Int("xa")
        F = single_edge()
        F.nodes["a"]["c"] = Contract(xa < 5, xa > 0)
        moved = F.transport(F.assignment(), "a", "b")
        xb = z3.Int("xb")
        assert moved == Contract(xb < 5, xb > 0)

    def test_direction_matters(self):
        xb = z3.Int("xb")
        F = single_edge(b_guarantee=xb < 7)
        moved = F.transport(F.assignment(), "b", "a")
        assert moved == Contract(z3.BoolVal(True), z3.Int("xa") < 7)

    def test_legs_agree_when_the_restriction_is_bijective(self):
        """With nothing hidden from the edge, `lan` and `ran` coincide."""
        F = single_edge()
        x = F.assignment()
        assert F.transport(x, "a", "b", legs=KAN) == F.transport(x, "a", "b", legs=CO)

    def test_legs_diverge_when_a_variable_is_private(self):
        """
        `a` promises something about a variable the edge cannot see. The Kan
        legs push what the edge *can* witness; the co legs demand it hold for
        every hidden value, which nothing guarantees.
        """
        F = private_variable()
        x = F.assignment()
        kan = F.transport(x, "a", "b", legs=KAN)
        co = F.transport(x, "a", "b", legs=CO)
        assert kan == Contract(z3.BoolVal(True), z3.Bool("xb"))
        assert co == bot()
        assert kan != co

    def test_unknown_legs_rejected(self):
        F = single_edge()
        with pytest.raises(ValueError, match="legs must be"):
            F.transport(F.assignment(), "a", "b", legs="sideways")

    def test_transport_between_non_adjacent_nodes_raises(self):
        F = star()
        with pytest.raises(KeyError):
            F.transport(F.assignment(), "a", "b")


class TestSectionOfAFrozenAssignment:
    """
    The interface predicates take an `x` like every other predicate here. The
    flow overwrites each stalk in place, so an observer recording the trajectory
    holds iterates the sheaf has already left behind; without this it could not
    ask whether any of them was a section.
    """

    def test_x_defaults_to_the_current_assignment(self):
        F = path()
        assert F.is_section(x=F.assignment()) == F.is_section()
        assert F.section_edges(x=F.assignment()) == F.section_edges()

    def test_a_past_iterate_can_still_be_asked_about(self):
        F = path()
        before = F.assignment()
        assert not F.is_section(x=before)

        F.converge(max_sweeps=50)
        assert F.is_section()
        assert not F.is_section(x=before), \
            "the settled sheaf answered for itself instead of for the assignment given"

    def test_the_limit_read_back_agrees_with_the_sheaf(self):
        F = path()
        F.converge(max_sweeps=50)
        assert F.is_section(x=F.assignment())
        assert all(F.agrees_on(u, v, x=F.assignment()) for u, v in F.interfaces())


class TestLaplacian:
    def test_reads_a_frozen_assignment(self):
        """
        L is the operator at a given cochain, so every node must read the same
        `x` -- otherwise it would be reporting the effect of a sweep.
        """
        F = path()
        x = F.assignment()
        first = F.laplacian(x)
        assert all(F.laplacian(x)[i] == first[i] for i in F.nodes())

    def test_isolated_node_gets_the_lattice_identity(self):
        F = disconnected()
        assert F.laplacian()["solo"] == top()
        assert F.laplacian(dual=True)["solo"] == bot()

    def test_include_self_leaves_an_isolated_node_alone(self):
        F = disconnected()
        assert F.laplacian(include_self=True)["solo"] == F.contract("solo")

    def test_meets_every_neighbour(self):
        F = star()
        xc = z3.Int("xc")
        assert F.laplacian()["c"] == Contract(z3.BoolVal(True),
                                              z3.And(xc > 0, xc < 10))

    def test_unknown_legs_rejected(self):
        with pytest.raises(ValueError, match="legs must be"):
            single_edge().laplacian(legs="sideways")


class TestFlow:
    def test_update_reports_whether_anything_changed(self):
        F = single_edge()
        assert F.laplacian_update() is False
        F.converge(max_sweeps=50)
        assert F.laplacian_update() is True

    def test_update_tightens_the_free_endpoint(self):
        F = single_edge()
        F.laplacian_update()
        assert F.contract("b") == Contract(z3.BoolVal(True), z3.Int("xb") > 0)

    def test_result_refines_the_original(self):
        F = single_edge()
        before = F.assignment()
        F.laplacian_update()
        assert F.contract("b").refines(before["b"])
        assert not before["b"].refines(F.contract("b"))

    def test_converge_returns_zero_when_already_settled(self):
        F = single_edge()
        F.converge(max_sweeps=50)
        F2 = single_edge()
        F2.nodes["a"]["c"] = F.contract("a")
        F2.nodes["b"]["c"] = F.contract("b")
        assert F2.converge(max_sweeps=50) == 0

    def test_converge_verify_accepts_a_genuine_fixed_point(self):
        single_edge().converge(verify=True, max_sweeps=50)

    def test_dual_flow_weakens(self):
        """The join-aggregated flow ascends where the primal descends."""
        xb = z3.Int("xb")
        F = single_edge(a_guarantee=z3.BoolVal(True), b_guarantee=xb > 0)
        before = F.assignment()
        F.laplacian_update(dual=True, legs=CO)
        assert before["b"].refines(F.contract("b"))

    def test_unknown_legs_rejected(self):
        with pytest.raises(ValueError, match="legs must be"):
            single_edge().laplacian_update(legs="sideways")

    def test_budget_exhaustion_reports_non_convergence(self):
        with pytest.raises(RuntimeError, match="no fixed point"):
            path().converge(max_sweeps=1)
