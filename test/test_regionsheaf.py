"""
The region-abstracted mission sheaf: alphabets differ, agreement is measured
in the interface vocabulary, and the mission contracts drive replanning.

The load-bearing claims, each pinned here:

  * in-fiber tolerance -- agents pinning different tiles of one region push
    the same summary, so the section holds over concrete disagreement;
  * cross-fiber content -- a hazarded region pushes differently from a clean
    one, the flow reconciles them, and the contradiction is absorbed as a
    region-level contested marker, in-lattice, with no stalk collapsing;
  * the closed-form transport (image + substitution) is the generic Kan
    composite, not an approximation of it;
  * claims travel the crossed wiring (U at one end to O at the other), and
    recommitment retracts them without ever contradicting a stalk;
  * the two bisheaves genuinely differ -- the degeneracy doc/DUALITY.md
    catalogued is gone.
"""

import z3
import networkx as nx
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agsheaf.regions import Regions, identity_regions, parse_regions
from agsheaf.gridsheaf import FULL_MASK, UNSAFE_BIT, build_belief_sheaf, communicate, sheaf_state
from agsheaf import regionsheaf as rs
from agsheaf.gridworld import route_corridor


GRID = (4, 2)
TWO_REGIONS = {"west": {"rect": [0, 0, 1, 1]}, "east": {"rect": [2, 0, 3, 1]}}


def build(beliefs, corridors=None, contracts_cfg=None, regions_cfg=None):
    regions = parse_regions(regions_cfg or TWO_REGIONS, GRID)
    topology = nx.Graph()
    topology.add_nodes_from(range(len(beliefs)))
    topology.add_edges_from((i, i + 1) for i in range(len(beliefs) - 1))
    return build_belief_sheaf([dict(b) for b in beliefs], GRID, topology,
                              regions=regions, corridors=corridors,
                              contracts_cfg=contracts_cfg)


def guarantees_sat(sheaf):
    """No stalk component's guarantee may ever become unsatisfiable."""
    regions = sheaf.graph['regions']
    for i in sheaf.nodes():
        for r in regions.names:
            solver = z3.Solver()
            solver.add(sheaf.nodes[i]['c'][r].sat_g)
            assert solver.check() == z3.sat, f"node {i} region {r} guarantee unsat"


class TestRegions:
    def test_partition_must_cover_the_grid(self):
        with pytest.raises(AssertionError, match="do not cover"):
            Regions({"west": [(0, 0)]}, GRID)

    def test_partition_must_not_overlap(self):
        with pytest.raises(AssertionError, match="must not overlap"):
            Regions({"a": [(0, 0)], "b": [(0, 0)]}, (1, 1))

    def test_rect_and_list_specs_agree(self):
        by_rect = parse_regions({"all": {"rect": [0, 0, 3, 1]}}, GRID)
        by_list = parse_regions({"all": [[x, y] for x in range(4) for y in range(2)]}, GRID)
        assert set(by_rect.tiles("all")) == set(by_list.tiles("all"))

    def test_identity_partition_is_identity(self):
        assert identity_regions(GRID).is_identity()
        assert not parse_regions(TWO_REGIONS, GRID).is_identity()

    def test_missing_block_falls_back_to_identity(self):
        assert parse_regions(None, GRID).is_identity()

    def test_region_of_and_regions_of(self):
        regions = parse_regions(TWO_REGIONS, GRID)
        assert regions.region_of((0, 1)) == "west"
        assert regions.regions_of([(0, 0), (3, 1), (1, 1)]) == ["west", "east"]


class TestInFiberTolerance:
    """Different tiles, same region, same summary: the kernel of the abstraction."""

    def test_section_holds_over_concrete_disagreement(self):
        # The two agents pin DIFFERENT west tiles hazardous. At the interface's
        # resolution both say exactly "west has a hazard": the section condition
        # holds from the start, while the belief dicts genuinely differ.
        sheaf = build([{(0, 0): "unsafe"}, {(0, 1): "unsafe"}])
        state = sheaf_state(sheaf)
        assert state.sections[(0, 1)]
        assert state.is_section
        assert state.concrete_disagreement[(0, 1)] == 2

    def test_same_summary_means_equal_pushforwards(self):
        sheaf = build([{(0, 0): "unsafe"}, {(0, 1): "unsafe"}])
        push_0 = rs._push_component(sheaf, 0, 1, "west")
        push_1 = rs._push_component(sheaf, 1, 0, "west")
        assert push_0 == push_1


class TestCrossFiberContent:
    """A hazarded region against a clean one: the flow has something to reconcile."""

    def setup_method(self):
        self.sheaf = build([
            {(2, 0): "unsafe"},
            {(2, 0): "safe", (2, 1): "safe", (3, 0): "safe", (3, 1): "safe"},
        ])

    def test_pushforwards_differ_before_the_flow(self):
        assert not sheaf_state(self.sheaf).sections[(0, 1)]

    def test_flow_reconciles_and_localises(self):
        conflicts, changed = communicate(self.sheaf, sweeps=10)
        assert changed
        state = sheaf_state(self.sheaf)
        assert state.is_section
        # The disagreement is held in-lattice: agent 0's pinned tile against
        # agent 1's all-clear meets to the empty possibility set at the tile...
        assert (2, 0) in state.contested[0]
        # ...and to the flagged-and-excluded marker at the region.
        assert "unsafe" in state.contested_regions[0].get("east", [])
        assert state.collapsed == {}
        guarantees_sat(self.sheaf)

    def test_region_hearsay_reaches_the_planner_readout(self):
        communicate(self.sheaf, sweeps=10)
        # Agent 1 now knows east is hazard-flagged somewhere, without any tile
        # of its own view being pinned hazardous.
        assert "east" in rs.hazard_regions(self.sheaf, 1)

    def test_monotone_refinement_under_the_flow(self):
        communicate(self.sheaf, sweeps=10)
        state = sheaf_state(self.sheaf)
        assert all(state.refines_initial.values())


class TestClosedFormTransport:
    """The image/substitution transport IS the generic Kan composite."""

    def setup_method(self):
        self.sheaf = build(
            [{(0, 0): "unsafe", (2, 0): "safe"}, {(0, 1): "safe"}],
            corridors={0: [(1, 0), (2, 0)], 1: [(1, 1)]},
        )

    @pytest.mark.parametrize("legs", ["kan", "co"])
    def test_transport_component_matches_generic(self, legs):
        x = self.sheaf.assignment()
        generic = self.sheaf.transport(x, 0, 1, legs=legs)
        for r in self.sheaf.graph['regions'].names:
            closed = rs.transport_component(self.sheaf, 0, 1, r, x=x, legs=legs)
            assert closed == generic[r], f"transport differs on {r} ({legs})"

    @pytest.mark.parametrize("legs", ["kan", "co"])
    def test_push_component_matches_generic_pushforward(self, legs):
        generic = self.sheaf.pushforward(0, 1, legs=legs)
        for r in self.sheaf.graph['regions'].names:
            closed = rs._push_component(self.sheaf, 0, 1, r, legs=legs)
            assert closed == generic[r], f"pushforward differs on {r} ({legs})"

    def test_sweep_is_the_heat_flow(self):
        # One canonical sweep against the operator applied directly: the state
        # the sweep leaves is x ^ Lx, componentwise, up to z3 equality.
        x = self.sheaf.assignment()
        L = self.sheaf.laplacian(x, include_self=True)
        rs.sweep(self.sheaf)
        for i in self.sheaf.nodes():
            for r in self.sheaf.graph['regions'].names:
                assert self.sheaf.nodes[i]['c'][r] == L[i][r], \
                    f"sweep result differs from x ^ Lx at node {i} region {r}"

    def test_legs_genuinely_differ(self):
        # Along a non-injective restriction lan and ran no longer coincide:
        # the degeneracy the delivered demo ran under is gone.
        kan = rs._push_component(self.sheaf, 0, 1, "west", legs="kan")
        co = rs._push_component(self.sheaf, 0, 1, "west", legs="co")
        assert not (kan == co)


class TestClaims:
    """Route commitments travel the crossed wiring and retract cleanly."""

    def setup_method(self):
        # Agent 0's corridor crosses both regions; agent 1 stays east.
        self.sheaf = build(
            [{}, {}],
            corridors={0: [(1, 0), (2, 0), (3, 0)], 1: [(3, 1)]},
        )

    def test_claims_propagate_into_expectations(self):
        communicate(self.sheaf, sweeps=10)
        assert rs.expected_claims(self.sheaf, 1) == {"west": [0], "east": [0]}
        assert rs.expected_claims(self.sheaf, 0) == {"east": [1]}

    def test_mutual_claim_is_flagged_for_both(self):
        communicate(self.sheaf, sweeps=10)
        assert rs.assumption_status(self.sheaf, 0).claimed == {"east": [1]}
        assert rs.assumption_status(self.sheaf, 1).claimed == {"east": [0]}

    def test_recommit_retracts_claims_without_contradiction(self):
        communicate(self.sheaf, sweeps=10)
        assert rs.recommit(self.sheaf, 1, [(1, 1), (0, 1)])  # east -> west
        communicate(self.sheaf, sweeps=10)
        assert rs.expected_claims(self.sheaf, 0) == {"west": [1]}
        guarantees_sat(self.sheaf)
        assert sheaf_state(self.sheaf).is_section

    def test_corridor_change_within_regions_is_not_an_epoch(self):
        communicate(self.sheaf, sweeps=10)
        heard_before = {r: list(self.sheaf.nodes[0]['state'][r]['heard'])
                        for r in self.sheaf.graph['regions'].names}
        # A different corridor through the same regions: the commitment the
        # network sees is unchanged, so nothing is purged and nothing reopens.
        assert not rs.recommit(self.sheaf, 0, [(1, 1), (2, 1), (3, 0)])
        for r, heard in heard_before.items():
            assert self.sheaf.nodes[0]['state'][r]['heard'] == heard
        assert self.sheaf.nodes[0]['route'] == [(1, 1), (2, 1), (3, 0)]

    def test_arrived_agent_withdraws_by_recommitting_empty(self):
        communicate(self.sheaf, sweeps=10)
        assert rs.recommit(self.sheaf, 0, [])
        communicate(self.sheaf, sweeps=10)
        assert rs.expected_claims(self.sheaf, 1) == {}


class TestAssumptionMonitor:
    def test_region_hearsay_warns_without_naming_the_tile(self):
        # Agent 1 knows agent 0's corridor tile is hazardous. What crosses the
        # interface is "east has a hazard somewhere" -- exactly what the
        # abstraction warrants and no more -- so the monitor's region read-out
        # fires (and the planner's overlay with it) while the per-tile
        # violation deliberately does not: the fused guarantee cannot name the
        # tile, because the emptied-set model is always an alternative witness.
        sheaf = build(
            [{}, {(2, 0): "unsafe", (2, 1): "safe", (3, 0): "safe", (3, 1): "safe"}],
            corridors={0: [(2, 0), (3, 0)], 1: []},
        )
        assert rs.hazard_regions(sheaf, 0) == []
        communicate(sheaf, sweeps=10)
        assert "east" in rs.hazard_regions(sheaf, 0)
        assert rs.assumption_status(sheaf, 0).hazards == {}

    def test_own_pin_on_the_corridor_is_a_named_violation(self):
        # First-hand knowledge does pin: an agent whose own belief flags its
        # corridor tile has a refuted reliance conjunct with the tile named.
        sheaf = build([{(2, 0): "unsafe"}, {}], corridors={0: [(2, 0), (3, 0)], 1: []})
        status = rs.assumption_status(sheaf, 0)
        assert "east" in status.violated
        assert status.hazards == {"east": [(2, 0)]}

    def test_claim_conflict_is_a_named_violation(self):
        # Exclusivity is refuted crisply: the neighbor's claim arrives as an
        # entailed expectation literal, which contradicts the "-O" conjunct.
        sheaf = build([{}, {}],
                      corridors={0: [(3, 0)], 1: [(3, 1)]})
        communicate(sheaf, sweeps=10)
        status = rs.assumption_status(sheaf, 0)
        assert "east" in status.violated
        assert status.claimed == {"east": [1]}

    def test_mission_contract_carries_both_slots(self):
        sheaf = build([{(0, 0): "safe"}, {}], corridors={0: [(1, 0)], 1: []})
        c = rs.mission_contract(sheaf, 0, "west")
        assert not z3.is_true(z3.simplify(c.a))   # reliance + exclusivity
        assert not z3.is_true(z3.simplify(c.g))   # knowledge + route pin


class TestObservation:
    def test_observation_overrides_hearsay(self):
        sheaf = build([
            {(2, 0): "unsafe"},
            {(2, 0): "safe", (2, 1): "safe", (3, 0): "safe", (3, 1): "safe"},
        ])
        communicate(sheaf, sweeps=10)
        assert (2, 0) in sheaf_state(sheaf).contested[0]
        # Agent 0 drives over the tile and sees it is safe: the contested tile
        # resolves for it, first-hand knowledge beating what it was told.
        assert rs.observe(sheaf, 0, (2, 0), "safe")
        assert sheaf.nodes[0]['beliefs'][(2, 0)] == "safe"
        assert sheaf.nodes[0]['masks'][(2, 0)] != 0
        guarantees_sat(sheaf)


class TestPlannerOverlay:
    def test_extra_costs_steer_the_corridor_around_a_region(self):
        grid = (4, 2)
        labels = {(x, y): "safe" for x in range(4) for y in range(2)}
        straight = route_corridor((0, 0), (3, 0), grid, labels)
        assert (1, 0) in straight and (2, 0) in straight
        overlay = {(1, 0): 50.0, (2, 0): 50.0}
        detour = route_corridor((0, 0), (3, 0), grid, labels, extra_costs=overlay)
        assert (1, 0) not in detour or (2, 0) not in detour
        assert detour[-1] == (3, 0)


class TestLegacyDispatch:
    def test_no_regions_builds_the_legacy_sheaf(self):
        topology = nx.Graph()
        topology.add_nodes_from([0, 1])
        topology.add_edge(0, 1)
        sheaf = build_belief_sheaf([{(0, 0): "unsafe"}, {}], GRID, topology)
        assert sheaf.graph.get('mode') is None
        conflicts, changed = communicate(sheaf, sweeps=5)
        assert changed
        # Legacy limit: pointwise mask intersection -- agent 1 learned the pin.
        assert sheaf.nodes[1]['masks'][(0, 0)] == UNSAFE_BIT

    def test_contracts_require_a_partition(self):
        topology = nx.Graph()
        topology.add_nodes_from([0, 1])
        topology.add_edge(0, 1)
        with pytest.raises(AssertionError, match="regions partition"):
            build_belief_sheaf([{}, {}], GRID, topology, corridors={0: []})
