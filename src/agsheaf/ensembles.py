"""
Random contract sheaves, for asking the flow the same question many times.

The test suite runs the flow on eight hand-built sheaves and checks that each
theorem holds. That establishes correctness and says nothing about *typical*
behavior: how many sweeps a sheaf of this size usually needs, how the limit
degrades as agents disagree more, where the flow stops being able to reconcile
anything. Those are distributional questions, and they need a distribution.

The generator is built around four modelling commitments, each of which is a
knob rather than a hard-coded choice, because each is exactly where the
demonstration's design is still open (`doc/DUALITY.md` section 6).

*Copies of one vocabulary.* There are `facts` facts about the world. Agent `i`
reasons about its own private copies of them. Nothing in the sheaf relates one
agent's copy to another's -- stalks are private, and the only channel between
them is the interface. But an outside observer knows position `k` means the same
fact everywhere, which is what makes `diagnostics.concrete_disagreement` a
question that can be asked. This mirrors `gridsheaf.tile_vars`, where each agent
holds its own bit-vector per tile.

*Abstraction, with coarseness as a parameter.* The facts are partitioned into
blocks of `coarseness`, and the interface speaks about blocks rather than facts.
`coarseness=1` makes the restriction a bijection, which is `f = id`: the case the
demonstration currently runs, and the case in which `legs="kan"` and `legs="co"`
cannot differ. Above 1 the restriction is a genuine non-injective abstraction,
the fibres are non-trivial, and the kernel of `f_!` is the disagreement the
mission tolerates. Sweeping this parameter is how the redesign gets evidence
before it is built.

*Two slots, from two sources.* Following the Milestone 3 framing, an agent's
guarantee is what it knows first-hand and its assumption is what it expects of
its neighbours: each fact is *known* with probability `knowledge`, and each fact
it does not know is *expected* with probability `assumption_density`. Every
belief is the truth, perturbed with probability `error`, so agents contradict
each other and the flow has something to reconcile. At `assumption_density=0`
every assumption is `True`, half of every distance is identically zero, and
`refines` is half-vacuous -- the degeneracy `doc/DUALITY.md` records in its table.

*What a contradiction does.* This is the `encoding` parameter, and it is the
sharpest of the four:

    "boolean"      a fact is one Boolean. Two agents asserting `x` and `~x` meet
                   at an unsatisfiable guarantee, so the stalk collapses to
                   `bot` and `collapsed()` names it. This is Riess and Ghrist
                   section IV behavior, and it is contagious: one contradicted
                   fact takes the whole stalk down with it.
    "possibility"  a fact is a possibility set over `labels` labels, one Boolean
                   per label, and a belief pins it to a singleton. Two agents
                   pinning different labels meet at the *empty* possibility set,
                   which is a perfectly satisfiable state of the alphabet. The
                   disagreement is recorded in the lattice, localised to the one
                   fact, and nothing collapses.

The second is what `gridsheaf` does with its bit-vector masks, and the reason
`doc/MATH.md` can claim the demonstration's conflicts "stay in the lattice".
Having both here turns that claim into something an ensemble can measure rather
than a design note.

Finally, the generator rejects starting assignments that are *already* global
sections. Without that, a substantial share of every parameter cell converges in
zero sweeps and the ensemble measures nothing; it was 5 cells out of 8 in the
prototype that motivated this module. `attempts` in the returned metadata says
how hard it had to look, which is itself a reading of how contested a cell is.
"""

import random
from typing import Any, Dict, List, Optional, Sequence

import networkx as nx
import z3

from .contracts import Contract, Relation
from .sheaf import ContractSheaf


# ---------------------------------------------------------------------------
# topologies
# ---------------------------------------------------------------------------

class DegenerateEnsemble(RuntimeError):
    """
    Every draw from this parameter cell was already a global section, so there is
    nothing for the flow to do. Its own exception type because a study wants to
    record the cell and carry on, and catching a bare `RuntimeError` there would
    also swallow a real failure.
    """


def _random_tree(n: int, rng: random.Random) -> nx.Graph:
    """A uniformly-attached random tree on `range(n)`: connected, n-1 edges."""
    g = nx.empty_graph(n)
    for i in range(1, n):
        g.add_edge(i, rng.randrange(i))
    return g


def _random_connected(n: int, rng: random.Random, density: float) -> nx.Graph:
    """
    A random connected graph: a spanning tree, plus each remaining pair with
    probability `density`.

    Built this way rather than by resampling `gnp_random_graph` until it comes
    out connected, because a disconnected sheaf is not a harder instance of the
    same question -- knowledge cannot cross components at all, so its limit is a
    different object. The `disconnected` test fixture covers that case
    deliberately; here it would only be noise.
    """
    g = _random_tree(n, rng)
    for i in range(n):
        for j in range(i + 1, n):
            if not g.has_edge(i, j) and rng.random() < density:
                g.add_edge(i, j)
    return g


#: Communication topologies, by name. Each takes the agent count, the
#: generator's rng and the density, and returns a connected graph on `range(n)`.
TOPOLOGIES = {
    "path": lambda n, rng, density: nx.path_graph(n),
    "cycle": lambda n, rng, density: nx.cycle_graph(n),
    "star": lambda n, rng, density: nx.star_graph(n - 1),
    "complete": lambda n, rng, density: nx.complete_graph(n),
    "tree": lambda n, rng, density: _random_tree(n, rng),
    "random": _random_connected,
}


def _parity(*args):
    """`Xor` folded over a block; z3's is strictly binary."""
    out = args[0]
    for arg in args[1:]:
        out = z3.Xor(out, arg)
    return out


#: How a block of concrete facts is summarised by the abstract fact the
#: interface speaks. `or` is the "this region contains one" reading of
#: `doc/DUALITY.md`; `parity` is the extreme in which no single concrete fact is
#: recoverable from the abstract one.
AGGREGATORS = {"or": z3.Or, "and": z3.And, "parity": _parity}


def blocks_of(facts: int, coarseness: int) -> List[range]:
    """The fibres of the abstraction: consecutive runs of `coarseness` facts."""
    if coarseness < 1:
        raise ValueError(f"coarseness must be at least 1, got {coarseness}")
    return [range(start, min(start + coarseness, facts))
            for start in range(0, facts, coarseness)]


# ---------------------------------------------------------------------------
# encodings
# ---------------------------------------------------------------------------

class Encoding:
    """
    How a fact becomes variables, a belief becomes a formula, and a block of
    facts becomes what the interface can say about it.

    Everything the generator does that is not graph theory goes through here, so
    that adding a way of valuing a fact does not touch the sampling, the
    topologies, or the abstraction.
    """

    name = None
    width = 1          #: variables per fact

    def values(self) -> Sequence:
        """The values a fact can take in the ground truth."""
        raise NotImplementedError

    def perturb(self, value, rng: random.Random):
        """`value` misremembered -- some other value, uniformly."""
        others = [v for v in self.values() if v != value]
        return rng.choice(others) if others else value

    def stalk_vars(self, agent: Any, facts: int) -> List[z3.BoolRef]:
        raise NotImplementedError

    def shared_vars(self, u: Any, v: Any, blocks: int) -> List[z3.BoolRef]:
        raise NotImplementedError

    def belief(self, variables: Sequence, fact: int, value) -> z3.BoolRef:
        """What an agent asserts when it holds that `fact` has `value`."""
        raise NotImplementedError

    def clauses(self, node_vars: Sequence, shared: Sequence, facts: int,
                coarseness: int, fold) -> List[z3.BoolRef]:
        """The graph of the abstraction map, one equation per shared variable."""
        raise NotImplementedError


class BooleanEncoding(Encoding):
    """One Boolean per fact; a belief is a literal; a contradiction is `bot`."""

    name = "boolean"
    width = 1

    def values(self):
        return (False, True)

    def stalk_vars(self, agent, facts):
        return [z3.Bool(f"x{agent}_{k}") for k in range(facts)]

    def shared_vars(self, u, v, blocks):
        lo, hi = sorted((str(u), str(v)))
        return [z3.Bool(f"h{lo}_{hi}_{b}") for b in range(blocks)]

    def belief(self, variables, fact, value):
        return variables[fact] if value else z3.Not(variables[fact])

    def clauses(self, node_vars, shared, facts, coarseness, fold):
        return [shared[b] == fold(*[node_vars[k] for k in block])
                for b, block in enumerate(blocks_of(facts, coarseness))]


class PossibilityEncoding(Encoding):
    """
    One Boolean per (fact, label), read as "this label is still possible here".

    A belief pins the set to a singleton; ignorance leaves it full; two agents
    pinning different labels meet at the empty set, which is a state of the
    alphabet rather than a failure of it. This is `gridsheaf`'s possibility
    masks with the bit-vector spelled out as its bits, which is what keeps the
    formulas Boolean and therefore cheap to measure.
    """

    name = "possibility"

    def __init__(self, labels: int = 3):
        if labels < 2:
            raise ValueError(f"a possibility set needs at least two labels, got {labels}")
        self.labels = labels
        self.width = labels

    def values(self):
        return tuple(range(self.labels))

    def stalk_vars(self, agent, facts):
        return [z3.Bool(f"x{agent}_{k}_{l}")
                for k in range(facts) for l in range(self.labels)]

    def shared_vars(self, u, v, blocks):
        lo, hi = sorted((str(u), str(v)))
        return [z3.Bool(f"h{lo}_{hi}_{b}_{l}")
                for b in range(blocks) for l in range(self.labels)]

    def belief(self, variables, fact, value):
        """
        "this fact is `value`" said as an *upper* bound: every other label is
        impossible. Deliberately not also asserting that `value` itself is
        possible.

        That asymmetry is the whole mechanism. `gridsheaf.encode_masks` writes
        the same thing as `S & ~mask == 0`, a subset bound with no lower bound,
        and it is why two agents naming different labels meet at `S subset {}`
        -- a state of the alphabet -- instead of at `False`. Pin the set exactly
        and the conjunction of two beliefs is unsatisfiable, the stalk collapses
        to `bot`, and the encoding stops being different from the Boolean one.
        """
        base = fact * self.labels
        return z3.And(*[z3.Not(variables[base + l])
                        for l in range(self.labels) if l != value])

    def clauses(self, node_vars, shared, facts, coarseness, fold):
        out = []
        for b, block in enumerate(blocks_of(facts, coarseness)):
            for l in range(self.labels):
                out.append(shared[b * self.labels + l]
                           == fold(*[node_vars[k * self.labels + l] for k in block]))
        return out


#: Encodings by name; `labels` is read only by "possibility".
ENCODINGS = {
    "boolean": lambda labels: BooleanEncoding(),
    "possibility": lambda labels: PossibilityEncoding(labels),
}


def abstraction(encoding: Encoding, node_vars: Sequence, shared: Sequence,
                facts: int, coarseness: int, aggregator: str = "or") -> Relation:
    """
    The restriction of one endpoint onto an interface: the graph of the
    abstraction map sending a concrete state to what the shared vocabulary can
    say about it.

    A function's graph, so `lan` is `f_!` and `pullback` is `f^*` in the sense of
    Naik Def. 6 -- which is what makes `sheaf.legs` mean the Milestone 3 report's
    case (1) and case (2) rather than two spellings of the same thing. At
    `coarseness=1` each block is a single fact and this is a bijection.
    """
    if aggregator not in AGGREGATORS:
        raise ValueError(f"aggregator must be one of {sorted(AGGREGATORS)}, got {aggregator!r}")
    clauses = encoding.clauses(node_vars, shared, facts, coarseness,
                               AGGREGATORS[aggregator])
    return Relation(clauses[0] if len(clauses) == 1 else z3.And(*clauses),
                    list(node_vars), list(shared))


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def _beliefs(rng: random.Random, encoding: Encoding, truth: Sequence,
             knowledge: float, error: float, assumption_density: float):
    """
    One agent's split of the world into what it commits to and what it relies on.

    Returns `(known, expected)`, each a dict from fact index to the value the
    agent holds -- the real one, or a perturbed one with probability `error`. The
    two are disjoint: an agent does not assume of its neighbours what it already
    knows itself.
    """
    known, expected = {}, {}
    for k, actual in enumerate(truth):
        held = actual if rng.random() >= error else encoding.perturb(actual, rng)
        if rng.random() < knowledge:
            known[k] = held
        elif rng.random() < assumption_density:
            expected[k] = held
    return known, expected


def _contract(encoding: Encoding, variables: Sequence, known: Dict[int, Any],
              expected: Dict[int, Any]) -> Contract:
    """
    Guarantee from first-hand knowledge, assumption from expectations of others.

    An agent holding nothing guarantees `True` -- it promises no more than the
    ambient world -- and expecting nothing assumes `True`, admitting every
    environment. Both are the identity for their slot, so an ignorant agent sits
    at the top of the lattice and constrains no neighbour, which is what it
    should do.
    """
    def conjoin(held):
        if not held:
            return z3.BoolVal(True)
        parts = [encoding.belief(variables, k, value) for k, value in sorted(held.items())]
        return parts[0] if len(parts) == 1 else z3.And(*parts)

    return Contract(conjoin(expected), conjoin(known), vars=variables)


def random_sheaf(agents: int = 4, topology: str = "path", facts: int = 4,
                 coarseness: int = 1, aggregator: str = "or",
                 encoding: str = "possibility", labels: int = 3,
                 knowledge: float = 0.5, error: float = 0.25,
                 assumption_density: float = 0.0, density: float = 0.3,
                 rng: Optional[random.Random] = None,
                 require_disagreement: bool = True,
                 attempts: int = 64) -> ContractSheaf:
    """
    One sample from the ensemble; see the module docstring for what the
    parameters mean.

    Every node carries its variables under ``vars`` as well as its contract, so
    that `diagnostics` can measure a stalk that has collapsed to a lattice
    extreme -- a `bot()` contract mentions no variables at all, and a sheaf that
    knew only about contracts could not say what alphabet it collapsed over.

    The sample's provenance lands in ``F.graph``: every parameter, the ground
    truth, and how many draws it took to find a starting assignment that was not
    already a section. With `require_disagreement` and no such draw inside
    `attempts`, this raises rather than quietly returning a trivial instance --
    an ensemble that silently degenerates is worse than one that stops.
    """
    if topology not in TOPOLOGIES:
        raise ValueError(f"topology must be one of {sorted(TOPOLOGIES)}, got {topology!r}")
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {sorted(ENCODINGS)}, got {encoding!r}")
    if agents < 2:
        raise ValueError(f"need at least two agents to have an interface, got {agents}")
    rng = rng or random.Random()

    scheme = ENCODINGS[encoding](labels)
    graph = TOPOLOGIES[topology](agents, rng, density)
    nodes = sorted(graph.nodes())
    blocks = len(blocks_of(facts, coarseness))

    for draw in range(1, attempts + 1):
        truth = [rng.choice(scheme.values()) for _ in range(facts)]
        sheaf = ContractSheaf()
        variables = {}
        for i in nodes:
            variables[i] = scheme.stalk_vars(i, facts)
            known, expected = _beliefs(rng, scheme, truth, knowledge, error,
                                       assumption_density)
            sheaf.add_node(i, c=_contract(scheme, variables[i], known, expected),
                           vars=variables[i], known=known, expected=expected)
        for u, v in graph.edges():
            shared = scheme.shared_vars(u, v, blocks)
            sheaf.add_interface(
                u, v,
                abstraction(scheme, variables[u], shared, facts, coarseness, aggregator),
                abstraction(scheme, variables[v], shared, facts, coarseness, aggregator))

        if not require_disagreement or not sheaf.is_section():
            sheaf.graph.update(
                topology=topology, agents=agents, facts=facts,
                coarseness=coarseness, aggregator=aggregator, encoding=encoding,
                labels=labels, knowledge=knowledge, error=error,
                assumption_density=assumption_density, density=density,
                ground_truth=list(truth), attempts=draw,
                diameter=nx.diameter(graph), interfaces=graph.number_of_edges())
            return sheaf

    raise DegenerateEnsemble(
        f"no starting assignment disagreed anywhere in {attempts} draws at "
        f"agents={agents}, facts={facts}, knowledge={knowledge}, error={error}. "
        f"Every draw was already a global section, so this cell has nothing for "
        f"the flow to do -- raise `error` or `knowledge`, or pass "
        f"require_disagreement=False to accept it."
    )
