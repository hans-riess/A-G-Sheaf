"""
Asynchronous updates: Riess and Ghrist (2022), Definition 5 (the asynchronous
Tarski Laplacian), Assumption 2 (liveness), and Theorem 1.

Theorem 1 says the sections are the time-independent solutions of the heat flow
for *any* firing sequence satisfying liveness. Its practical content is that the
answer does not depend on the schedule, so the sharpest test available is to run
several schedules and check they agree -- which is what most of this file does.
"""

import random

import pytest
import z3

from conftest import BUILDERS, path, single_edge, triangle
from agsheaf.contracts import Contract, Relation
from agsheaf.sheaf import (CO, KAN, ContractSheaf, LivenessError, random_firing,
                           round_robin)


def settle(build, **kwargs):
    """Runs `build()` to a fixed point and returns the sheaf."""
    F = build()
    F.converge(max_sweeps=400, **kwargs)
    return F


def same_fixed_point(f, g):
    """Semantic equality of two settled assignments, decided by the solver."""
    return all(f.contract(i) == g.contract(i) for i in f.nodes())


class TestFiringSequences:
    def test_all_fire_is_the_synchronous_laplacian(self, build_sheaf):
        """
        RG Eq. (4). Firing everything is the synchronous Tarski Laplacian, so
        `firing=None` and `firing=V` must agree exactly.
        """
        F = build_sheaf()
        x = F.assignment()
        every = F.laplacian(x, firing=list(F.nodes()))
        default = F.laplacian(x)
        assert all(every[i] == default[i] for i in F.nodes())

    def test_node_with_no_firing_neighbour_is_untouched(self):
        """
        RG Def. 5 restricts the aggregation to N_i n tau_t, so a node none of
        whose neighbours broadcast has nothing to fuse and must hold still.
        """
        F = single_edge()
        before = F.assignment()
        F.laplacian_update(firing=set())
        assert all(F.contract(i) == before[i] for i in F.nodes())

    def test_only_listeners_of_firing_nodes_move(self):
        """When only `a` broadcasts, `b` may move but `a` itself may not."""
        F = single_edge()
        before = F.assignment()
        F.laplacian_update(firing={"a"})
        assert F.contract("a") == before["a"]
        assert F.contract("b").refines(before["b"])
        assert not before["b"].refines(F.contract("b"))


class TestScheduleIndependence:
    """Riess and Ghrist Theorem 1: the limit does not depend on the schedule."""

    @pytest.mark.parametrize("name", sorted(BUILDERS))
    def test_round_robin_agrees_with_all_fire(self, name):
        build = BUILDERS[name]
        reference = settle(build)
        async_ = settle(build, firing_sequence=lambda: round_robin(
            sorted(build().nodes(), key=str)))
        assert same_fixed_point(reference, async_)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_random_schedules_agree_with_all_fire(self, seed):
        reference = settle(triangle)
        rng = random.Random(seed)
        async_ = settle(triangle,
                        firing_sequence=lambda: random_firing(list("abc"), rng))
        assert same_fixed_point(reference, async_)
        assert async_.is_section()

    @pytest.mark.parametrize("name", sorted(BUILDERS))
    def test_snapshot_and_in_place_updates_agree(self, name):
        """
        `in_place=False` is RG Eq. (6) literally -- every neighbour read from
        the previous step. `in_place=True` lets a node see its predecessors'
        new values within the step. Different trajectories, same limit.
        """
        build = BUILDERS[name]
        assert same_fixed_point(settle(build), settle(build, in_place=False))

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_asynchrony_and_snapshotting_compose(self, seed):
        """The two axes are independent, so all four combinations must agree."""
        reference = settle(triangle)
        for in_place in (True, False):
            rng = random.Random(seed)
            F = settle(triangle, in_place=in_place,
                       firing_sequence=lambda: random_firing(list("abc"), rng))
            assert same_fixed_point(reference, F)

    def test_in_place_is_not_slower(self):
        """
        The reason to keep the in-place default: seeing a predecessor's new
        value within the step can only bring the descent forward, never delay
        it.
        """
        F, G = triangle(), triangle()
        assert F.converge(max_sweeps=400) <= G.converge(max_sweeps=400, in_place=False)


class TestLiveness:
    def test_starved_schedule_is_diagnosed(self):
        """
        RG Assumption 2 is the hypothesis of Theorem 1, so a schedule violating
        it carries no guarantee. Spinning to the sweep limit and blaming
        divergence would be the wrong diagnosis -- name the starved nodes.
        """
        def only_a():
            while True:
                yield frozenset({"a"})

        with pytest.raises(LivenessError) as excinfo:
            triangle().converge(firing_sequence=only_a, max_sweeps=20)
        message = str(excinfo.value)
        assert "'b'" in message and "'c'" in message
        assert "'a'" not in message

    def test_truncated_budget_reports_the_nodes_it_never_reached(self):
        """
        The honest limit of the diagnostic: `round_robin` is live, but two
        steps cannot reach three nodes, so `c` looks starved. From inside a
        truncated run a starving schedule and a budget too small to cover a
        round are indistinguishable, and the message says so rather than
        picking one.
        """
        with pytest.raises(LivenessError) as excinfo:
            triangle().converge(firing_sequence=lambda: round_robin(list("abc")),
                                max_sweeps=2)
        assert "'c'" in str(excinfo.value)
        assert "max_sweeps" in str(excinfo.value)

    def test_partial_schedule_does_not_declare_victory_early(self):
        """
        A step in which nothing changed proves only that the firing nodes had
        nothing to say. Convergence may be declared only once every node has
        fired since the last change -- otherwise `round_robin` would stop after
        its first quiet step, well short of a section.
        """
        F = triangle()
        F.converge(firing_sequence=lambda: round_robin(list("abc")), max_sweeps=400)
        assert F.is_section()
        assert F.laplacian_update() is True


class TestAsyncPreservesTheTheorems:
    @pytest.mark.parametrize("name", sorted(BUILDERS))
    def test_async_limit_is_a_section(self, name):
        build = BUILDERS[name]
        F = settle(build, firing_sequence=lambda: round_robin(
            sorted(build().nodes(), key=str)), verify=True)
        assert F.is_section()
        assert F.is_suffix()

    def test_async_flow_still_descends(self):
        F = triangle()
        before = F.assignment()
        F.converge(firing_sequence=lambda: round_robin(list("abc")), max_sweeps=400)
        assert all(F.contract(i).refines(before[i]) for i in F.nodes())

    def test_greatest_section_property_survives_asynchrony(self):
        """Ghrist Prop. 6.15 under a partial schedule (see `test_theorems`)."""
        xa, xb = z3.Int("xa"), z3.Int("xb")
        witness = {
            "a": Contract(z3.BoolVal(True), z3.And(xa > 0, xa < 5)),
            "b": Contract(z3.BoolVal(True), z3.And(xb > 0, xb < 5)),
        }
        F = single_edge()
        F.converge(firing_sequence=lambda: round_robin(["a", "b"]), max_sweeps=400)
        assert all(witness[i].refines(F.contract(i)) for i in F.nodes())


class TestTerminationCriterion:
    """
    `converge` may declare a fixed point only after an unbroken run of quiet
    steps covering every node.

    The nodes that fired in a step which *changed* something earn no credit: a
    node can broadcast and then change within that same step -- it aggregates
    over its own firing neighbours -- so what it sent is already stale and its
    neighbours have not seen its current contract. Counting those firings lets
    the flow stop short of a section.

    Round-robin cannot expose this, since a lone firing node has no firing
    neighbour and so never changes in the step it fires; that is why the
    schedule-independence tests above passed while the criterion was unsound.
    The exposing case is the opposite extreme -- a full broadcast, then a
    partial schedule -- and `path` is the smallest fixture that shows it,
    because information needs two hops to cross it.
    """

    @staticmethod
    def _all_then_round_robin():
        """Everyone broadcasts once, then one node at a time. Live throughout."""
        yield frozenset("abc")
        while True:
            for node in "abc":
                yield frozenset({node})

    def test_a_full_broadcast_then_a_partial_schedule_still_reaches_a_section(self):
        """
        The regression. Under this schedule the pre-fix `converge` returned at
        step 1: the opening broadcast changed something and credited all three
        nodes, so the first quiet step satisfied "everyone has fired since the
        last change" while `c` had not yet received what `b` learned.
        """
        F = path()
        F.converge(firing_sequence=self._all_then_round_robin, max_sweeps=400)
        assert F.laplacian_update() is True, \
            "converge returned a point a further round still moves"
        assert F.is_section()

    def test_the_limit_matches_the_all_fire_limit(self):
        """Schedule independence, on the schedule that used to violate it."""
        reference = settle(path)
        F = path()
        F.converge(firing_sequence=self._all_then_round_robin, max_sweeps=400)
        assert same_fixed_point(reference, F)
