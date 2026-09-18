"""
Shared fixtures for the contract/sheaf test suite.

Two ideas carry most of the weight here.

*Symbolic contracts.* Rather than sampling concrete assumptions and guarantees,
`symbolic` builds a contract whose legs are uninterpreted predicates. A property
asserted of such a contract and discharged by z3 holds for *every* contract over
that alphabet, so these are proofs rather than spot checks.

*The relation zoo.* Nearly every subtle claim about the Kan maps is really a
claim about the shape of the relation -- total, surjective, functional,
injective. Parametrising over the zoo is what distinguishes the two adjunctions
that hold universally from the one that needs a total function.
"""

import random

import pytest
import z3

from agsheaf.contracts import Contract, Relation
from agsheaf.sheaf import ContractSheaf


# ---------------------------------------------------------------------------
# symbolic contracts
# ---------------------------------------------------------------------------

def symbolic(name, variables):
    """
    A contract over `variables` whose assumption and guarantee are uninterpreted
    predicates. Properties proved of it are universally quantified over
    contracts.
    """
    sorts = [v.sort() for v in variables]
    a = z3.Function(f"A_{name}", *sorts, z3.BoolSort())
    g = z3.Function(f"G_{name}", *sorts, z3.BoolSort())
    return Contract(a(*variables), g(*variables))


@pytest.fixture
def sym():
    """Factory for `symbolic`, so tests can name several independent contracts."""
    return symbolic


# ---------------------------------------------------------------------------
# the relation zoo
# ---------------------------------------------------------------------------

X, Y = z3.Int("x"), z3.Int("y")
BA, BP, BE = z3.Bool("ba"), z3.Bool("bp"), z3.Bool("be")


class Spec:
    """A relation together with the shape facts the theorems turn on."""

    def __init__(self, name, relation, *, total, surjective, functional,
                 injective, disjoint=True):
        self.name = name
        self.relation = relation
        self.total = total
        self.surjective = surjective
        self.functional = functional
        self.injective = injective
        self.disjoint = disjoint

    @property
    def total_function(self):
        return self.total and self.functional

    def __repr__(self):
        return self.name


ZOO = [
    Spec("bijection", Relation(Y == X + 1, [X], [Y]),
         total=True, surjective=True, functional=True, injective=True),

    Spec("one_to_many", Relation(z3.And(Y >= X, Y <= X + 2), [X], [Y]),
         total=True, surjective=True, functional=False, injective=False),

    # y == 2x over the integers: every x has an image, but odd y have no preimage.
    Spec("non_surjective", Relation(Y == 2 * X, [X], [Y]),
         total=True, surjective=False, functional=True, injective=True),

    # undefined below the origin in both directions.
    Spec("partial", Relation(z3.And(Y == X, X > 0), [X], [Y]),
         total=False, surjective=False, functional=True, injective=True),

    # the restriction of a stalk carrying a private variable the edge cannot see.
    Spec("many_to_one", Relation(BE == BA, [BA, BP], [BE]),
         total=True, surjective=True, functional=True, injective=False),

    Spec("empty", Relation(z3.BoolVal(False), [X], [Y]),
         total=False, surjective=False, functional=True, injective=True),
]

#: Alphabets that overlap: `p1, p2` are coordinates the two sides hold in
#: common, so the Kan maps must leave them free. This is the shape the tutorial
#: uses, and the one that silently computed the wrong extension before `_bind`.
P1, P2, P3, P4 = z3.Bools("p1 p2 p3 p4")
SHARED = Spec("shared_alphabet",
              Relation(z3.Implies(P3, P4), [P1, P2, P3], [P1, P2, P4]),
              total=True, surjective=True, functional=False, injective=False,
              disjoint=False)


@pytest.fixture(params=ZOO, ids=lambda s: s.name)
def spec(request):
    """Every relation in the zoo, one test per shape."""
    return request.param


@pytest.fixture(params=[s for s in ZOO if s.disjoint], ids=lambda s: s.name)
def disjoint_spec(request):
    """Zoo entries whose alphabets do not overlap."""
    return request.param


def contracts_for(spec, tag=""):
    """A symbolic contract on each side of `spec`'s relation."""
    r = spec.relation
    return (symbolic(f"src_{spec.name}{tag}", r.source_vars),
            symbolic(f"tgt_{spec.name}{tag}", r.target_vars))


# ---------------------------------------------------------------------------
# sheaf builders
# ---------------------------------------------------------------------------

def _int_interface(F, u, v, xu, xv, e):
    """Adds the interface {u, v} whose edge stalk equates both endpoints."""
    F.add_interface(u, v, Relation(e == xu, [xu], [e]), Relation(e == xv, [xv], [e]))


def single_edge(a_guarantee=None, b_guarantee=None):
    """Two nodes over one integer interface; `a` constrains, `b` starts free."""
    xa, xb, e = z3.Int("xa"), z3.Int("xb"), z3.Int("e")
    F = ContractSheaf()
    F.add_node("a", c=Contract(z3.BoolVal(True),
                               xa > 0 if a_guarantee is None else a_guarantee))
    F.add_node("b", c=Contract(z3.BoolVal(True),
                               z3.BoolVal(True) if b_guarantee is None else b_guarantee))
    _int_interface(F, "a", "b", xa, xb, e)
    return F


def path():
    """a -- b -- c, so information must cross two hops to reach `c`."""
    xa, xb, xc = z3.Int("xa"), z3.Int("xb"), z3.Int("xc")
    eab, ebc = z3.Int("eab"), z3.Int("ebc")
    F = ContractSheaf()
    F.add_node("a", c=Contract(z3.BoolVal(True), xa > 0))
    F.add_node("b", c=Contract(z3.BoolVal(True), z3.BoolVal(True)))
    F.add_node("c", c=Contract(z3.BoolVal(True), xc < 10))
    _int_interface(F, "a", "b", xa, xb, eab)
    _int_interface(F, "b", "c", xb, xc, ebc)
    return F


def star():
    """Centre `c` with leaves `a`, `b`, each interface with its own edge stalk."""
    xa, xb, xc = z3.Int("xa"), z3.Int("xb"), z3.Int("xc")
    eac, ebc = z3.Int("eac"), z3.Int("ebc")
    F = ContractSheaf()
    F.add_node("a", c=Contract(z3.BoolVal(True), xa > 0))
    F.add_node("b", c=Contract(z3.BoolVal(True), xb < 10))
    F.add_node("c", c=Contract(z3.BoolVal(True), z3.BoolVal(True)))
    _int_interface(F, "a", "c", xa, xc, eac)
    _int_interface(F, "b", "c", xb, xc, ebc)
    return F


def triangle():
    """A cycle, so the flow must reach agreement around a loop."""
    xa, xb, xc = z3.Int("xa"), z3.Int("xb"), z3.Int("xc")
    eab, ebc, eac = z3.Int("eab"), z3.Int("ebc"), z3.Int("eac")
    F = ContractSheaf()
    F.add_node("a", c=Contract(z3.BoolVal(True), xa > 0))
    F.add_node("b", c=Contract(z3.BoolVal(True), xb < 10))
    F.add_node("c", c=Contract(z3.BoolVal(True), z3.BoolVal(True)))
    _int_interface(F, "a", "b", xa, xb, eab)
    _int_interface(F, "b", "c", xb, xc, ebc)
    _int_interface(F, "a", "c", xa, xc, eac)
    return F


def disconnected():
    """An edge plus an isolated node, which no transport can ever reach."""
    F = single_edge()
    F.add_node("solo", c=Contract(z3.BoolVal(True), z3.Int("xs") > 0))
    return F


def private_variable():
    """
    `a` guarantees something about a variable the edge cannot see, so its
    restriction is non-injective and `lan` differs from `ran`.
    """
    xa, pa, xb, e = z3.Bool("xa"), z3.Bool("pa"), z3.Bool("xb"), z3.Bool("e")
    F = ContractSheaf()
    F.add_node("a", c=Contract(z3.BoolVal(True), z3.And(xa, pa)))
    F.add_node("b", c=Contract(z3.BoolVal(True), z3.BoolVal(True)))
    F.add_interface("a", "b",
                    Relation(e == xa, [xa, pa], [e]),
                    Relation(e == xb, [xb], [e]))
    return F


def contradictory():
    """Endpoints promise incompatible things; the flow must collapse to bottom."""
    xa, xb, e = z3.Bool("xa"), z3.Bool("xb"), z3.Bool("e")
    F = ContractSheaf()
    F.add_node("a", c=Contract(z3.BoolVal(True), xa))
    F.add_node("b", c=Contract(z3.BoolVal(True), z3.Not(xb)))
    F.add_interface("a", "b", Relation(e == xa, [xa], [e]), Relation(e == xb, [xb], [e]))
    return F


def shared_alphabet_sheaf():
    """
    The tutorial's example in the sheaf's own convention: node 1 over
    {p1, p2, p3}, node 2 over {p1, p2, p4}, and an edge stalk {p1, p2, e}
    coupling p3 to p4. `p1, p2` are shared coordinates and must stay free.
    """
    e = z3.Bool("e")
    F = ContractSheaf()
    F.add_node(1, c=Contract(z3.And(P1, P2), P3))
    F.add_node(2, c=Contract(P4, z3.Or(P1, P2)))
    F.add_interface(1, 2,
                    Relation(e == P3, [P1, P2, P3], [P1, P2, e]),
                    Relation(e == P4, [P1, P2, P4], [P1, P2, e]))
    return F


#: Builders whose sheaves the flow can settle, keyed by name for parametrisation.
BUILDERS = {
    "single_edge": single_edge,
    "path": path,
    "star": star,
    "triangle": triangle,
    "disconnected": disconnected,
    "private_variable": private_variable,
    "contradictory": contradictory,
    "shared_alphabet": shared_alphabet_sheaf,
}


@pytest.fixture(params=sorted(BUILDERS), ids=lambda n: n)
def build_sheaf(request):
    """Every sheaf builder, as a zero-argument factory returning a fresh sheaf."""
    return BUILDERS[request.param]


@pytest.fixture
def rng():
    """A seeded generator, so a failing schedule is reproducible."""
    return random.Random(20260802)
