"""
The belief sheaf over the gridworld communication topology (`agsheaf.gridsheaf`).

What is being tested is the reduction of communication to the Tarski Laplacian:
one sweep is one hop, the meet is pointwise intersection of possibility masks,
disagreement empties a mask *inside* the lattice instead of collapsing a stalk,
and the keep-own conflict policy lives entirely in the read-out projection. The
mission layer's reliance contracts get the same ∅ read-out through a bona fide
contract meet.
"""

import networkx as nx
import pytest
import z3

from agsheaf.gridsheaf import (CONVERGE, EMPTY_MASK, FULL_MASK, LABEL_BITS,
                               LABELS, SAFE_BIT, TARGET_BIT, UNSAFE_BIT,
                               Conflict, beliefs_to_masks, build_belief_sheaf,
                               communicate, decode_masks, edge_relation,
                               edge_vars, encode_masks, forced_empty,
                               grid_tiles, mask_label, mask_labels, observe,
                               observe_tiles, parse_sweeps_per_step,
                               reliance_contract, sheaf_state, sweep, tile_vars)

#: A 3x1 corridor: tiles (0,0), (1,0), (2,0). Small enough to enumerate, long
#: enough that a path sheaf over it has a two-hop diameter.
GRID = (3, 1)
LEFT, MID, RIGHT = (0, 0), (1, 0), (2, 0)


def path_sheaf(*beliefs):
    """A path sheaf 0 - 1 - ... - n-1 over GRID with the given belief dicts."""
    topology = nx.Graph()
    topology.add_nodes_from(range(len(beliefs)))
    topology.add_edges_from((i, i + 1) for i in range(len(beliefs) - 1))
    return build_belief_sheaf(list(beliefs), GRID, topology)


class TestEncodeDecode:
    def test_roundtrip(self):
        v = tile_vars("t", GRID)
        masks = {LEFT: UNSAFE_BIT, RIGHT: SAFE_BIT}
        assert decode_masks(encode_masks(masks, v), v) == masks

    def test_empty_masks_encode_the_trivial_contract(self):
        v = tile_vars("t", GRID)
        c = encode_masks({}, v)
        assert c.is_consistent()
        assert decode_masks(c, v) == {}

    def test_beliefs_to_masks_drops_unknown(self):
        """"unknown" is the absence of a constraint, not a label with a bit."""
        masks = beliefs_to_masks({LEFT: "unsafe", MID: "unknown"})
        assert masks == {LEFT: UNSAFE_BIT}

    def test_empty_mask_is_a_consistent_constraint(self):
        """
        The load-bearing property of the powerset alphabet: a contested tile
        is the *value* S = ∅, not an unsatisfiable contract, so ex falso never
        poisons the other tiles.
        """
        v = tile_vars("t", GRID)
        c = encode_masks({MID: EMPTY_MASK, RIGHT: SAFE_BIT}, v)
        assert c.is_consistent()
        assert decode_masks(c, v) == {MID: EMPTY_MASK, RIGHT: SAFE_BIT}

    def test_tiles_restricts_the_scan(self):
        v = tile_vars("t", GRID)
        c = encode_masks({LEFT: UNSAFE_BIT, RIGHT: SAFE_BIT}, v)
        assert decode_masks(c, v, tiles=[LEFT]) == {LEFT: UNSAFE_BIT}

    def test_mask_label_reads_singletons_only(self):
        assert [mask_label(LABEL_BITS[l]) for l in LABELS] == list(LABELS)
        assert mask_label(EMPTY_MASK) is None
        assert mask_label(FULL_MASK) is None
        assert mask_label(TARGET_BIT | SAFE_BIT) is None


class TestBuildBeliefSheaf:
    def test_nodes_carry_contract_masks_and_the_callers_dicts(self):
        beliefs = [{LEFT: "unsafe"}, {}]
        F = path_sheaf(*beliefs)
        assert F.nodes[0]["masks"] == {LEFT: UNSAFE_BIT}
        # The very dict object, not a copy: sweeps mutate it in place, which
        # is what lets the gridworld renderer and planner see communication.
        assert F.nodes[0]["beliefs"] is beliefs[0]
        assert F.nodes[0]["own"] == beliefs[0] and F.nodes[0]["own"] is not beliefs[0]
        assert F.contract(0).is_consistent()

    def test_initial_is_snapshotted_at_build(self):
        """
        The custom `sweep` never calls `laplacian_update`, whose lazy c_init
        snapshot the base class relies on, so build must set it eagerly for
        `initial` and the refines diagnostics to mean anything.
        """
        F = path_sheaf({LEFT: "unsafe"}, {})
        sweep(F)
        assert decode_masks(F.initial(1), F.nodes[1]["v"]) == {}

    def test_both_orientations_share_the_edge_alphabet(self):
        F = path_sheaf({}, {})
        rel_01, rel_10 = F.restriction(0, 1), F.restriction(1, 0)
        assert (set(map(str, rel_01.target_vars))
                == set(map(str, rel_10.target_vars)))
        assert F.has_edge(0, 1) and F.has_edge(1, 0)

    def test_isolated_agent_is_a_node(self):
        topology = nx.Graph([(0, 1)])
        topology.add_node(2)
        F = build_belief_sheaf([{}, {}, {LEFT: "safe"}], GRID, topology)
        assert set(F.nodes()) == {0, 1, 2}
        assert list(F.neighbors(2)) == []

    def test_mismatched_agents_raise(self):
        with pytest.raises(ValueError, match="do not match"):
            build_belief_sheaf([{}, {}], GRID, nx.Graph([(0, 1), (1, 2)]))


class TestDiffusion:
    def test_one_sweep_is_one_hop(self):
        """
        The confirmed semantics: knowledge moves exactly one interface per
        sweep. A Gauss-Seidel style in-place update would leak it further in
        DiGraph node order; this is the regression against that.
        """
        F = path_sheaf({LEFT: "unsafe"}, {}, {})
        _, changed = sweep(F)
        assert changed
        assert F.nodes[1]["masks"] == {LEFT: UNSAFE_BIT}
        assert F.nodes[2]["masks"] == {}
        _, changed = sweep(F)
        assert changed
        assert F.nodes[2]["masks"] == {LEFT: UNSAFE_BIT}

    def test_fixpoint_is_the_component_knowledge_union(self):
        F = path_sheaf({LEFT: "unsafe"}, {}, {RIGHT: "safe"})
        conflicts, changed = communicate(F, sweeps=10)
        assert changed and conflicts == []
        union = {LEFT: UNSAFE_BIT, RIGHT: SAFE_BIT}
        assert all(F.nodes[i]["masks"] == union for i in F.nodes())
        assert all(F.nodes[i]["beliefs"] == {LEFT: "unsafe", RIGHT: "safe"}
                   for i in F.nodes())

    def test_fixpoint_is_a_section_and_nothing_collapsed(self):
        F = path_sheaf({LEFT: "unsafe"}, {}, {RIGHT: "safe"})
        communicate(F, sweeps=10)
        assert F.is_section()
        assert F.collapsed() == {}

    def test_flow_settles_in_diameter_sweeps(self):
        F = path_sheaf({LEFT: "unsafe"}, {}, {})
        _, first = sweep(F)
        _, second = sweep(F)
        _, third = sweep(F)
        assert first and second and not third

    def test_knowledge_only_strengthens(self):
        """Monotonicity: every node's contract refines its initial one."""
        F = path_sheaf({LEFT: "unsafe"}, {}, {RIGHT: "safe"})
        communicate(F, sweeps=10)
        assert all(F.contract(i).refines(F.initial(i)) for i in F.nodes())

    def test_decode_of_the_raw_meet_matches_mask_intersection(self):
        """
        The semantic claim under the whole construction: transport along the
        equality bijections is a renaming, so the Laplacian's contract meet
        decodes to pointwise intersection of the neighbours' masks.
        """
        F = path_sheaf({MID: "safe"}, {MID: "safe", LEFT: "unsafe"})
        fused = F.laplacian(F.assignment(), include_self=True)
        assert (decode_masks(fused[0], F.nodes[0]["v"])
                == {MID: SAFE_BIT, LEFT: UNSAFE_BIT})

    def test_interface_insertion_order_does_not_matter(self):
        beliefs_a = [{LEFT: "unsafe"}, {}, {RIGHT: "safe"}]
        beliefs_b = [{LEFT: "unsafe"}, {}, {RIGHT: "safe"}]
        forward = nx.Graph()
        forward.add_nodes_from(range(3))
        forward.add_edges_from([(0, 1), (1, 2)])
        backward = nx.Graph()
        backward.add_nodes_from(range(3))
        backward.add_edges_from([(2, 1), (1, 0)])
        Fa = build_belief_sheaf(beliefs_a, GRID, forward)
        Fb = build_belief_sheaf(beliefs_b, GRID, backward)
        communicate(Fa, sweeps=10)
        communicate(Fb, sweeps=10)
        assert all(Fa.nodes[i]["masks"] == Fb.nodes[i]["masks"] for i in range(3))


class TestBisheafParity:
    """
    The restrictions here are graphs of bijections -- each tile's private mask equated with
    its edge copy -- and along one of those `lan` and `ran` agree, so transport is a
    renaming whichever adjoint bisheaf carries it. That is the claim `edge_relation` rests
    on, and what makes the sheaf's `legs` setting immaterial for a belief sheaf while
    remaining load-bearing for a sheaf whose restrictions lose information.
    """

    def test_transport_agrees_along_both_bisheaves(self):
        F = path_sheaf({LEFT: "unsafe"}, {MID: "safe"}, {})
        x = F.assignment()
        assert F.transport(x, 0, 1, legs="kan") == F.transport(x, 0, 1, legs="co")

    def test_the_laplacian_agrees_along_both_bisheaves(self):
        F = path_sheaf({LEFT: "unsafe"}, {MID: "safe"}, {RIGHT: "target"})
        x = F.assignment()
        kan = F.laplacian(x, include_self=True, legs="kan")
        co = F.laplacian(x, include_self=True, legs="co")
        assert all(kan[i] == co[i] for i in F.nodes())

    def test_the_section_read_out_agrees_along_both_bisheaves(self):
        """
        Which matters because the flow transports along "kan" whatever a run configures,
        while `sheaf_state` reads sections along the configured legs: the two only stay in
        step because the restrictions make them the same question.
        """
        F = path_sheaf({LEFT: "unsafe"}, {}, {MID: "unsafe"})
        for _ in range(4):
            assert F.section_edges(legs="kan") == F.section_edges(legs="co")
            sweep(F)


class TestConflicts:
    def test_disagreement_empties_the_mask_and_both_keep_their_own(self):
        F = path_sheaf({MID: "target"}, {MID: "unsafe"})
        conflicts, _ = sweep(F)
        assert F.nodes[0]["masks"][MID] == EMPTY_MASK
        assert F.nodes[1]["masks"][MID] == EMPTY_MASK
        assert F.nodes[0]["beliefs"][MID] == "target"
        assert F.nodes[1]["beliefs"][MID] == "unsafe"
        assert set(conflicts) == {
            Conflict(0, 1, MID, "target", "unsafe"),
            Conflict(1, 0, MID, "unsafe", "target"),
        }

    def test_contested_stalks_stay_consistent_and_form_a_section(self):
        """
        Agree-to-disagree is a genuine global section: both endpoints hold
        S = ∅ at the contested tile, so they agree on the edge stalk, and no
        contract ever becomes inconsistent -- the conflict lives inside the
        lattice, not in the solver's ⊥.
        """
        F = path_sheaf({MID: "target"}, {MID: "unsafe"})
        _, _ = sweep(F)
        _, changed = sweep(F)
        assert not changed
        assert F.collapsed() == {}
        assert F.is_section()

    def test_bystander_with_no_opinion_stays_unknown(self):
        F = path_sheaf({MID: "target"}, {}, {MID: "unsafe"})
        communicate(F, sweeps=10)
        assert F.nodes[1]["masks"][MID] == EMPTY_MASK
        assert MID not in F.nodes[1]["beliefs"]
        assert F.nodes[1]["beliefs"] == {}

    def test_non_clashing_knowledge_still_diffuses_through_a_conflict(self):
        """A contested tile does not stop the rest of a broadcast."""
        F = path_sheaf({MID: "target"}, {MID: "unsafe", RIGHT: "safe"})
        sweep(F)
        assert F.nodes[0]["masks"][RIGHT] == SAFE_BIT
        assert F.nodes[0]["beliefs"][RIGHT] == "safe"

    def test_conflicts_are_reported_once_not_every_sweep(self):
        F = path_sheaf({MID: "target"}, {MID: "unsafe"})
        first, _ = sweep(F)
        second, _ = sweep(F)
        assert first and not second

    def test_propagated_conflicts_attribute_as_conflict(self):
        """
        A bystander between two disagreeing agents receives singletons from
        both on the same sweep; the agents themselves later receive the
        bystander's already-emptied mask, attributed as "conflict".
        """
        F = path_sheaf({MID: "target"}, {}, {MID: "unsafe"})
        all_conflicts, _ = communicate(F, sweeps=10)
        bystander = {c for c in all_conflicts if c.agent == 1}
        assert bystander == {
            Conflict(1, 0, MID, "unknown", "target"),
            Conflict(1, 2, MID, "unknown", "unsafe"),
        }
        endpoint = {c for c in all_conflicts if c.agent == 0}
        assert endpoint == {Conflict(0, 1, MID, "target", "conflict")}


class TestMissionLayer:
    def test_clear_corridor_is_viable(self):
        F = path_sheaf({LEFT: "unsafe"}, {RIGHT: "safe"})
        communicate(F, sweeps=10)
        v = F.nodes[0]["v"]
        corridor = [MID, RIGHT]
        viability = F.contract(0).meet(reliance_contract(v, corridor))
        assert forced_empty(viability, v, corridor) == []

    def test_learned_wall_breaks_the_corridor(self):
        """
        The mission-layer story end to end: agent 1 routes through LEFT, which
        only agent 0 knows is unsafe. Before communication the reliance meet
        is fine; after one hop the fused stalk rules the tile out.
        """
        F = path_sheaf({LEFT: "unsafe"}, {})
        v = F.nodes[1]["v"]
        corridor = [LEFT, MID]
        before = F.contract(1).meet(reliance_contract(v, corridor))
        assert forced_empty(before, v, corridor) == []
        sweep(F)
        after = F.contract(1).meet(reliance_contract(v, corridor))
        assert forced_empty(after, v, corridor) == [LEFT]

    def test_reliance_over_no_tiles_is_trivial(self):
        v = tile_vars("t", GRID)
        c = reliance_contract(v, [])
        assert c.is_consistent()
        assert forced_empty(c, v, grid_tiles(GRID)) == []


class TestDegenerate:
    def test_edgeless_topology_settles_immediately(self):
        topology = nx.Graph()
        topology.add_nodes_from(range(2))
        F = build_belief_sheaf([{LEFT: "unsafe"}, {}], GRID, topology)
        conflicts, changed = communicate(F)
        assert conflicts == [] and not changed
        assert F.nodes[0]["masks"] == {LEFT: UNSAFE_BIT}
        assert F.nodes[1]["masks"] == {}

    def test_isolated_agent_is_untouched_while_the_rest_converge(self):
        topology = nx.Graph([(0, 1)])
        topology.add_node(2)
        F = build_belief_sheaf([{LEFT: "unsafe"}, {}, {RIGHT: "safe"}],
                               GRID, topology)
        communicate(F, sweeps=10)
        assert F.nodes[1]["masks"] == {LEFT: UNSAFE_BIT}
        assert F.nodes[2]["masks"] == {RIGHT: SAFE_BIT}

    def test_communicate_crosses_as_many_hops_as_asked(self):
        F = path_sheaf({LEFT: "unsafe"}, {}, {})
        _, changed = communicate(F, sweeps=2)
        assert changed
        assert F.nodes[2]["masks"] == {LEFT: UNSAFE_BIT}


class TestObservation:
    """
    What an agent sees for itself, which is the one thing that enters a stalk
    from outside the flow. The property under test is that it lands in *all* of
    the node state the sweep keeps mutually consistent -- masks, contract,
    beliefs, own -- so that the next sweep diffuses it rather than reading
    around it, and that it overrides rather than meets: first-hand beats
    hearsay, even where hearsay has already emptied the tile.
    """

    def test_observation_becomes_the_agents_belief(self):
        F = path_sheaf({}, {})
        assert observe(F, 0, MID, "unsafe")
        assert F.nodes[0]["masks"][MID] == UNSAFE_BIT
        assert F.nodes[0]["beliefs"][MID] == "unsafe"
        assert F.nodes[0]["own"][MID] == "unsafe"

    def test_observation_keeps_the_contract_in_step_with_the_masks(self):
        """
        `sweep` fuses contracts and decodes back to masks, so an observation
        written to one and not the other would simply be undone by the next
        sweep.
        """
        F = path_sheaf({LEFT: "safe"}, {})
        observe(F, 0, MID, "unsafe")
        node = F.nodes[0]
        assert decode_masks(node["c"], node["v"]) == node["masks"]
        assert node["masks"] == {LEFT: SAFE_BIT, MID: UNSAFE_BIT}

    def test_observation_diffuses_on_the_next_sweep(self):
        F = path_sheaf({}, {}, {})
        observe(F, 0, MID, "unsafe")
        communicate(F, sweeps=10)
        assert F.nodes[2]["beliefs"][MID] == "unsafe"

    def test_seeing_it_beats_being_told_otherwise(self):
        """
        The observer keeps what it saw and the neighbour keeps what it holds,
        so the tile reads as contested at both -- until the neighbour goes and
        looks for itself, at which point they agree on the truth.
        """
        F = path_sheaf({MID: "unsafe"}, {MID: "unsafe"})
        observe(F, 0, MID, "safe")
        communicate(F, sweeps=4)
        assert F.nodes[0]["masks"][MID] == EMPTY_MASK
        assert F.nodes[0]["beliefs"][MID] == "safe"
        assert F.nodes[1]["beliefs"][MID] == "unsafe"

        observe(F, 1, MID, "safe")
        communicate(F, sweeps=4)
        assert F.nodes[0]["beliefs"][MID] == "safe"
        assert F.nodes[1]["beliefs"][MID] == "safe"

    def test_observation_reopens_a_settled_flow(self):
        F = path_sheaf({}, {})
        _, changed = communicate(F, sweeps=10)
        assert not changed
        observe(F, 0, MID, "safe")
        _, changed = communicate(F, sweeps=10)
        assert changed

    def test_seeing_what_is_already_believed_changes_nothing(self):
        F = path_sheaf({MID: "safe"}, {})
        assert not observe(F, 0, MID, "safe")

    def test_unknown_is_not_observable(self):
        F = path_sheaf({}, {})
        with pytest.raises(ValueError):
            observe(F, 0, MID, "unknown")

    def test_a_tile_off_the_grid_is_rejected(self):
        F = path_sheaf({}, {})
        with pytest.raises(ValueError):
            observe(F, 0, (9, 9), "safe")

    def test_observe_tiles_reports_what_was_news(self):
        F = path_sheaf({MID: "safe"}, {})
        truth = {LEFT: "safe", MID: "safe", RIGHT: "unsafe"}
        records = observe_tiles([F.nodes[0]["beliefs"], F.nodes[1]["beliefs"]],
                                [MID, RIGHT], truth, sheaf=F)
        assert records == [
            {"agent": 0, "tile": [1, 0], "label": "safe", "changed": False},
            {"agent": 1, "tile": [2, 0], "label": "unsafe", "changed": True},
        ]
        assert F.nodes[1]["masks"][RIGHT] == UNSAFE_BIT

    def test_observe_tiles_without_a_sheaf_writes_the_belief_dicts(self):
        """The baseline run has no sheaf, and senses the same world."""
        beliefs = [{}, {MID: "unsafe"}]
        records = observe_tiles(beliefs, [LEFT, MID],
                                {LEFT: "safe", MID: "safe", RIGHT: "safe"})
        assert beliefs == [{LEFT: "safe"}, {MID: "safe"}]
        assert [record["changed"] for record in records] == [True, True]


class TestSweepBudget:
    def test_a_number_of_sweeps_is_that_many(self):
        assert parse_sweeps_per_step(3, 100) == 3

    def test_converge_resolves_to_the_bound(self):
        assert parse_sweeps_per_step(CONVERGE, 100) == 100

    @pytest.mark.parametrize("configured", [0, -1, 1.5, "one", None, True])
    def test_bad_budgets_are_rejected(self, configured):
        """True is an int in Python; a YAML `sweeps_per_step: true` is a typo."""
        with pytest.raises(AssertionError):
            parse_sweeps_per_step(configured, 100)

    def test_converging_reaches_the_same_fixed_point_as_sweeping_by_hand(self):
        F = path_sheaf({LEFT: "unsafe"}, {}, {})
        communicate(F, sweeps=parse_sweeps_per_step(CONVERGE, 100))
        assert F.is_section()
        assert F.nodes[2]["masks"] == {LEFT: UNSAFE_BIT}


class TestSheafState:
    """
    The snapshot a run records between sweeps, and the per-interface reading of
    the sheaf condition it rests on. What is being tested is that the flow's
    progress is *observable* at the granularity the Laplacian works at: an
    interface at a time, rather than only "section" or "not section" at the end.
    """

    def test_interfaces_lists_each_once(self):
        F = path_sheaf({}, {}, {})
        assert len(list(F.edges())) == 4
        assert sorted(sorted(pair) for pair in F.interfaces()) == [[0, 1], [1, 2]]

    def test_agrees_on_localises_is_section(self):
        """
        The near interface closes a sweep before the far one, so the assignment
        is not a section while one of its interfaces already is. Reporting only
        `is_section` would lose exactly that.
        """
        F = path_sheaf({LEFT: "unsafe"}, {}, {})
        sweep(F)
        assert F.agrees_on(0, 1) and not F.agrees_on(1, 2)
        assert not F.is_section()

        sweep(F)
        assert F.agrees_on(0, 1) and F.agrees_on(1, 2)
        assert F.is_section()

    def test_section_edges_agrees_with_is_section(self):
        F = path_sheaf({LEFT: "unsafe"}, {MID: "safe"}, {})
        for _ in range(4):
            assert F.is_section() == all(F.section_edges().values())
            sweep(F)

    def test_state_reports_possibilities_and_contested(self):
        F = path_sheaf({MID: "safe"}, {MID: "unsafe"}, {})
        sweep(F)
        state = sheaf_state(F)
        assert state.possibilities[0][MID] == ()
        assert state.contested[0] == [MID]
        assert state.contested[2] == []
        # The disagreement is in the lattice, so no stalk has collapsed
        assert state.collapsed == {}
        assert all(state.refines_initial.values())

    def test_state_keys_interfaces_by_sorted_pair(self):
        F = path_sheaf({}, {}, {})
        assert sorted(sheaf_state(F).sections) == [(0, 1), (1, 2)]

    def test_state_is_section_matches_the_sheaf(self):
        F = path_sheaf({LEFT: "unsafe"}, {}, {})
        for _ in range(3):
            state = sheaf_state(F)
            assert state.is_section == F.is_section()
            assert state.is_section == all(state.sections.values())
            sweep(F)

    def test_mask_labels_reads_a_mask_as_the_set_it_denotes(self):
        assert mask_labels(EMPTY_MASK) == ()
        assert mask_labels(SAFE_BIT) == ("safe",)
        assert mask_labels(FULL_MASK) == LABELS
        assert mask_labels(TARGET_BIT | UNSAFE_BIT) == ("target", "unsafe")


class TestSweepObserver:
    def test_on_sweep_sees_every_iterate_in_order(self):
        F = path_sheaf({LEFT: "unsafe"}, {}, {})
        seen = []
        communicate(F, sweeps=3, on_sweep=lambda i, c, changed: seen.append((i, changed)))
        assert seen == [(0, True), (1, True), (2, False)]

    def test_on_sweep_sees_the_state_that_sweep_left(self):
        """
        The observer runs between sweeps, which is the only place the
        intermediate iterates exist: `communicate` returns just the aggregate.
        """
        F = path_sheaf({LEFT: "unsafe"}, {}, {})
        states = []
        communicate(F, sweeps=2, on_sweep=lambda i, c, changed: states.append(sheaf_state(F)))
        assert states[0].possibilities[1][LEFT] == ("unsafe",)
        assert LEFT not in states[0].possibilities[2]
        assert states[1].possibilities[2][LEFT] == ("unsafe",)

    def test_on_sweep_fires_for_the_settling_sweep_too(self):
        F = path_sheaf({}, {}, {})
        seen = []
        communicate(F, sweeps=5, on_sweep=lambda i, c, changed: seen.append(changed))
        assert seen == [False]


class TestLogIntegration:
    def test_record_step_carries_conflicts_and_mission(self, tmp_path):
        import json
        import numpy as np
        from agsheaf.log import ExperimentLog

        log = ExperimentLog(tmp_path / "run.json", {"experiment_name": "t"},
                            GRID, 2, record_poses=False)
        conflict = Conflict(0, 1, MID, "target", "unsafe")
        log.record_step(0, np.zeros((2, 2), dtype=int),
                        conflicts=[{**conflict._asdict(),
                                    "tile": list(conflict.tile)}],
                        mission={"1": {"corridor": [list(MID)],
                                       "broken": [list(MID)]}})
        log.record_step(1, np.zeros((2, 2), dtype=int))
        log.close()

        record = json.loads((tmp_path / "run.json").read_text())
        assert record["steps"][0]["conflicts"][0]["incoming"] == "unsafe"
        assert record["steps"][0]["mission"]["1"]["broken"] == [[1, 0]]
        assert "conflicts" not in record["steps"][1]
        assert "mission" not in record["steps"][1]

    def test_encoded_state_carries_the_contracts_when_asked(self):
        """
        The stalk grid says what a local section means; this says what it is.
        Both, so a run states its assume/guarantee propositions rather than
        leaving them to be reconstructed from the possibility sets.
        """
        from agsheaf.gridsheaf import render_contracts
        from agsheaf.log import encode_sheaf_state

        F = path_sheaf({MID: "unsafe"}, {})
        sweep(F)
        encoded = encode_sheaf_state(sheaf_state(F), GRID, contracts=render_contracts(F))

        assert set(encoded["contracts"]) == {"0", "1"}
        for lines in encoded["contracts"].values():
            assert lines[0] == "Contract("
            # the assumption is True at every belief stalk, by construction
            assert lines[1].strip() == "A: True"
            assert any(line.strip().startswith("G:") for line in lines)

    def test_contracts_are_left_out_unless_asked_for(self):
        F = path_sheaf({MID: "unsafe"}, {})
        from agsheaf.log import encode_sheaf_state
        assert "contracts" not in encode_sheaf_state(sheaf_state(F), GRID)

    def test_rendered_contract_is_what_printing_one_gives(self):
        from agsheaf.gridsheaf import render_contracts

        F = path_sheaf({MID: "unsafe"}, {})
        assert render_contracts(F)[0] == repr(F.contract(0)).splitlines()

    def test_encoded_state_writes_stalks_and_read_outs(self):
        from agsheaf.log import (CONTESTED_CHARACTER, SEVERAL_CHARACTER,
                                 encode_sheaf_state)

        F = path_sheaf({MID: "safe"}, {MID: "unsafe"}, {RIGHT: "target"})
        sweep(F)
        encoded = encode_sheaf_state(sheaf_state(F), GRID)

        # A contested tile is what a labeling cannot express: agent 0 keeps its
        # own "safe" there, and only the stalk says the set has emptied. Agent
        # 2's target is still one hop away from agent 0, so its tile is blank.
        assert encoded["stalks"]["0"] == [".%s." % CONTESTED_CHARACTER]
        assert encoded["stalks"]["1"] == [".%st" % CONTESTED_CHARACTER]
        assert encoded["contested"]["0"] == [list(MID)]
        assert encoded["sections"] == {"0-1": False, "1-2": False}
        assert encoded["is_section"] is False
        assert encoded["collapsed"] == {}
        assert SEVERAL_CHARACTER not in "".join(encoded["stalks"]["0"])

        # At the fixed point the contested tile is a section like any other:
        # every stalk holds the same emptied set, so the interfaces close
        communicate(F, sweeps=4)
        settled = encode_sheaf_state(sheaf_state(F), GRID)
        assert settled["sections"] == {"0-1": True, "1-2": True}
        assert settled["is_section"] is True
        assert settled["collapsed"] == {}

    def test_encoded_state_leaves_unconstrained_tiles_unknown(self):
        from agsheaf.log import LABEL_CHARACTERS, encode_sheaf_state

        F = path_sheaf({}, {}, {})
        encoded = encode_sheaf_state(sheaf_state(F), GRID)
        assert encoded["stalks"]["1"] == [LABEL_CHARACTERS["unknown"] * GRID[0]]

    def test_record_sheaf_carries_the_structure_and_the_first_iterate(self, tmp_path):
        import json
        from agsheaf.log import ExperimentLog, encode_sheaf_state

        F = path_sheaf({LEFT: "unsafe"}, {}, {})
        log = ExperimentLog(tmp_path / "run.json", {"experiment_name": "t"},
                            GRID, 3, record_poses=False)
        log.record_sheaf(interfaces=[sorted(pair) for pair in F.interfaces()],
                         stalk_alphabet="one possibility set per tile",
                         legs="kan", sweeps_per_step=1,
                         initial_state=encode_sheaf_state(sheaf_state(F), GRID))
        log.close()

        record = json.loads((tmp_path / "run.json").read_text())
        assert record["sheaf"]["interfaces"] == [[0, 1], [1, 2]]
        assert record["sheaf"]["legs"] == "kan"
        assert record["sheaf"]["initial"]["is_section"] is False

    def test_a_run_with_no_sheaf_records_none(self, tmp_path):
        import json
        import numpy as np
        from agsheaf.log import ExperimentLog

        log = ExperimentLog(tmp_path / "run.json", {}, GRID, 2, record_poses=False)
        log.record_step(0, np.zeros((2, 2), dtype=int))
        log.close()

        record = json.loads((tmp_path / "run.json").read_text())
        assert "sheaf" not in record
        assert "sweeps" not in record["steps"][0]
        assert "settled" not in record["steps"][0]

    def test_a_run_that_drew_cleanly_records_no_projection_failures(self, tmp_path):
        import json
        import numpy as np
        from agsheaf.log import ExperimentLog

        log = ExperimentLog(tmp_path / "run.json", {}, GRID, 2, record_poses=False)
        log.record_step(0, np.zeros((2, 2), dtype=int))
        log.close()

        record = json.loads((tmp_path / "run.json").read_text())
        assert "projection_failures" not in record

    def test_a_run_whose_projection_broke_says_so(self, tmp_path):
        """
        A run is not invalidated by its drawing failing -- nothing recorded here is computed
        from what was drawn -- but its video shows less than its numbers do, and the record is
        what keeps the two from being read as if they agreed.
        """
        import json
        import numpy as np
        from agsheaf.log import ExperimentLog

        log = ExperimentLog(tmp_path / "run.json", {}, GRID, 2, record_poses=False)
        log.record_step(0, np.zeros((2, 2), dtype=int))
        log.close(final_scores=np.array([3.0, 4.0]), steps_taken=1, all_arrived=True,
                  projection_failures={"the belief tiles": {
                      "error": "RuntimeError: the projector fell over",
                      "traceback": "Traceback (most recent call last): ...",
                      "count": 1220}})

        record = json.loads((tmp_path / "run.json").read_text())
        assert record["projection_failures"]["the belief tiles"]["count"] == 1220
        assert "RuntimeError" in record["projection_failures"]["the belief tiles"]["error"]

        # The run still stands, and is still scored
        assert record["result"]["all_arrived"] is True
        assert record["result"]["total_score"] == 7.0


class TestGridworldParity:
    def test_labels_mirror_ground_truth_labels(self):
        """
        gridsheaf restates the label tuple rather than importing gridworld
        (whose Robotarium import would weigh down the contract test suite);
        this is the one test that pays that import to keep the two in step.
        """
        from agsheaf.gridworld import GROUND_TRUTH_LABELS
        assert LABELS == GROUND_TRUTH_LABELS
