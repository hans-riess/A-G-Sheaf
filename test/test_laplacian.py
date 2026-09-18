"""
The worked example, end to end: two contracts over overlapping alphabets,
diffused to agreement by the harmonic flow.

This is the example `tutorial.ipynb` walks through. Node 1 speaks about
{p1, p2, p3} and node 2 about {p1, p2, p4}; the interface between them is an
edge stalk {p1, p2, e} in which e couples p3 to p4. The two nodes therefore
*share* the coordinates p1 and p2, which the Kan maps must leave free -- writing
the interface as a direct node-to-node relation over the same constants would
quantify them away and diffuse a different problem entirely.
"""

import z3

from agsheaf import Contract, Relation, ContractSheaf
from agsheaf.sheaf import round_robin


P1, P2, P3, P4 = z3.Bools("p1 p2 p3 p4")
E = z3.Bool("e")


def build():
    """Node 1 over {p1,p2,p3}, node 2 over {p1,p2,p4}, coupled through e."""
    F = ContractSheaf()
    F.add_node(1, c=Contract(z3.And(P1, P2), P3))
    F.add_node(2, c=Contract(P4, z3.Or(P1, P2)))
    F.add_interface(1, 2,
                    Relation(E == P3, [P1, P2, P3], [P1, P2, E]),
                    Relation(E == P4, [P1, P2, P4], [P1, P2, E]))
    return F


class TestWorkedExample:
    def test_starts_disagreeing(self):
        F = build()
        assert not F.is_section()
        assert not F.is_suffix()

    def test_flow_reaches_a_section(self):
        """
        `verify=True` asserts the suffix condition against a frozen assignment
        once the flow settles; `is_section` then confirms the Hodge-Lawvere
        identification independently.
        """
        F = build()
        F.converge(verify=True)
        assert F.is_section()
        assert F.is_suffix()

    def test_shared_coordinates_are_not_quantified_away(self):
        """
        The regression this file exists to catch. Pushing node 1 onto the edge
        binds p3 -- the coordinate the edge replaces -- and nothing else; p1 and
        p2 belong to both alphabets and must survive as free variables.
        """
        F = build()
        pushed = F.contract(1).lan(F.restriction(1, 2))
        body = str(pushed.a)
        assert body.startswith("ForAll(p3"), body
        assert "p1" in body and "p2" in body

    def test_result_refines_the_initial_assignment(self):
        F = build()
        before = F.assignment()
        F.converge()
        assert all(F.contract(i).refines(before[i]) for i in F.nodes())

    def test_asynchronous_schedule_agrees(self):
        """Riess and Ghrist Thm. 1: the schedule does not change the answer."""
        sync = build()
        sync.converge()
        async_ = build()
        async_.converge(firing_sequence=lambda: round_robin([1, 2]), max_sweeps=50)
        assert all(sync.contract(i) == async_.contract(i) for i in sync.nodes())
