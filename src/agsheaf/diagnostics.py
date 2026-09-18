"""
Numbers for a sheaf in flight.

`ContractSheaf` answers yes/no questions -- is this a section, is this a suffix
point, which interfaces agree. Those are the right questions for a theorem and
the wrong ones for a plot: they cannot say how far from a section an assignment
is, nor which sweep closed most of the gap. This module reads the same
questions through `measure`, so each of them acquires a magnitude that is zero
exactly where the boolean was true.

    is_section(legs)          <-->  dirichlet(...) == 0
    is_suffix(legs)           <-->  laplacian_residual(...) == 0
    is_prefix(legs)           <-->  laplacian_residual(dual=True, ...) == 0
    agrees_on(u, v, legs)     <-->  edge_disagreement(u, v, ...) == 0

Each pairing is asserted in `test_measure.py` on every fixture sheaf, because a
diagnostic that disagrees with the predicate it is supposed to refine is worse
than no diagnostic.

`dirichlet` is the one worth a name. In the linear theory the quantity that
falls to zero along the heat flow is the Dirichlet energy `x^T L x`, a sum over
edges of how far apart the endpoints are once pushed onto the shared space.
Written that way it transports to this setting unchanged -- sum over interfaces
of the distance between the two pushforwards -- and it is zero exactly on the
global sections, which is the sheaf condition. There is no inner product here
and no spectrum; what carries over is the accounting, not the operator theory.

The last function is the one the demonstration is really about.
`concrete_disagreement` measures two agents against each other in their *own*
vocabulary, while `edge_disagreement` measures them in the shared one. When the
restriction is an abstraction rather than a bijection the two are different
numbers, and their difference is the disagreement the mission tolerates -- the
kernel of `f_!` (see `doc/DUALITY.md`). At `f = id` they coincide, which is the
degenerate case the demonstration currently runs.
"""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import z3

from .contracts import Contract
from .measure import Alphabet, distance, divergence
from .sheaf import CO, KAN, ContractSheaf


class Alphabets:
    """
    One `Alphabet` per node and per interface of a sheaf, built once and reused.

    Building an alphabet enumerates it, so this exists to keep that from
    happening per sweep. It also fixes *which* variables a node's alphabet
    consists of, which the sheaf itself never records: a node carries a
    contract, and a contract carries at most a set of variable names.

    A node's variables are looked for in three places, in order:

      1. the node's ``vars`` attribute, if the builder set one -- the only
         source that survives a stalk whose contract has collapsed to a lattice
         extreme, and the one `ensembles` uses;
      2. the source alphabet of its restrictions, which every node incident to
         an interface has;
      3. the constants of its *initial* contract -- the assignment as given,
         before any transport could have introduced others.

    An isolated node whose initial contract mentions nothing gets the empty
    alphabet, over which every contract is either the top or the bottom one.
    That is a faithful answer to a degenerate question, not a failure.
    """

    def __init__(self, sheaf: ContractSheaf, domains: Optional[Mapping] = None,
                 max_states: int = 1 << 16):
        self.sheaf = sheaf
        self.domains = domains
        self.max_states = max_states
        self._nodes: Dict[Any, Alphabet] = {}
        self._edges: Dict[Tuple[Any, Any], Alphabet] = {}

    def node(self, i: Any) -> Alphabet:
        """The alphabet of node `i`'s stalk."""
        if i not in self._nodes:
            self._nodes[i] = Alphabet(node_variables(self.sheaf, i),
                                      domains=self.domains, max_states=self.max_states)
        return self._nodes[i]

    def edge(self, u: Any, v: Any) -> Alphabet:
        """
        The alphabet of the interface {u, v}. Keyed unordered: both endpoints
        restrict onto the *same* shared space, which `add_interface` enforces.
        """
        for key in ((u, v), (v, u)):
            if key in self._edges:
                return self._edges[key]
        self._edges[(u, v)] = Alphabet(self.sheaf.restriction(u, v).target_vars,
                                       domains=self.domains, max_states=self.max_states)
        return self._edges[(u, v)]


def node_variables(sheaf: ContractSheaf, i: Any) -> Sequence:
    """The z3 variables of node `i`'s stalk; see `Alphabets` for the search order."""
    declared = sheaf.nodes[i].get('vars')
    if declared is not None:
        return list(declared)

    found, seen = [], set()
    for j in sheaf.neighbors(i):
        for v in sheaf.restriction(i, j).source_vars:
            if str(v) not in seen:
                seen.add(str(v))
                found.append(v)
    if found:
        return found

    initial = sheaf.initial(i)
    for expr in (initial.a, initial.sat_g):
        for v in _constants_in(expr):
            if str(v) not in seen:
                seen.add(str(v))
                found.append(v)
    return sorted(found, key=str)


def _constants_in(expr):
    out, stack, seen = [], [expr], set()
    while stack:
        e = stack.pop()
        if e.get_id() in seen:
            continue
        seen.add(e.get_id())
        if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            out.append(e)
        stack.extend(e.children())
    return out


# ---------------------------------------------------------------------------
# disagreement across an interface
# ---------------------------------------------------------------------------

def edge_disagreement(sheaf: ContractSheaf, u: Any, v: Any, alphabets: Alphabets,
                      legs: str = KAN, x: Optional[Mapping] = None) -> float:
    """
    How far apart the two endpoints of one interface are, measured on the shared
    space.

    Zero exactly when `sheaf.agrees_on(u, v, legs)` -- and it compares the very
    contracts that predicate compares, since both go through
    `ContractSheaf.pushforward`. A diagnostic that recomputed the pushforward
    its own way could drift from the predicate it is supposed to refine.
    """
    return distance(sheaf.pushforward(u, v, legs=legs, x=x),
                    sheaf.pushforward(v, u, legs=legs, x=x),
                    alphabets.edge(u, v))


def edge_energies(sheaf: ContractSheaf, alphabets: Alphabets, legs: str = KAN,
                  x: Optional[Mapping] = None) -> Dict[Tuple[Any, Any], float]:
    """
    `edge_disagreement` for every interface, keyed as `interfaces()` keys them.
    The numeric counterpart of `section_edges`: which interfaces are closed, and
    how far from closed the rest are.
    """
    return {(u, v): edge_disagreement(sheaf, u, v, alphabets, legs=legs, x=x)
            for u, v in sheaf.interfaces()}


def dirichlet(sheaf: ContractSheaf, alphabets: Alphabets, legs: str = KAN,
              x: Optional[Mapping] = None) -> float:
    """
    The Tarski-Dirichlet energy: the total disagreement across all interfaces.
    Zero exactly on the global sections of the `legs` bisheaf.

    Summed rather than averaged, so that it is extensive in the interfaces the
    way the sheaf condition is a conjunction over them; divide by
    `len(sheaf.interfaces())` for a per-interface figure.
    """
    return sum(edge_energies(sheaf, alphabets, legs=legs, x=x).values())


# ---------------------------------------------------------------------------
# distance from a fixed point
# ---------------------------------------------------------------------------

def laplacian_residual(sheaf: ContractSheaf, alphabets: Alphabets,
                       dual: bool = False, legs: str = KAN,
                       x: Optional[Mapping] = None) -> float:
    """
    How far the assignment is from being a fixed point of the Laplacian:

        primal   sum_i divergence(x_i, (L x)_i)        zero iff a suffix point
        dual     sum_i divergence((L_dual x)_i, x_i)   zero iff a prefix point

    Directed rather than symmetric, because the conditions are: `is_suffix` asks
    `x <= Lx` and `is_prefix` asks `L_dual x <= x`, and `divergence` is exactly
    the failure of one of those. A symmetric distance here would report the slack
    in a satisfied inequality as if it were a defect.
    """
    if x is None:
        x = sheaf.assignment()
    L = sheaf.laplacian(x, dual=dual, legs=legs)
    if dual:
        return sum(divergence(L[i], x[i], alphabets.node(i)) for i in sheaf.nodes())
    return sum(divergence(x[i], L[i], alphabets.node(i)) for i in sheaf.nodes())


def distance_profile(sheaf: ContractSheaf, alphabets: Alphabets,
                     x: Mapping, y: Mapping) -> Dict[Any, float]:
    """Per-node `distance` between two assignments over the same sheaf."""
    return {i: distance(x[i], y[i], alphabets.node(i)) for i in sheaf.nodes()}


def divergence_profile(sheaf: ContractSheaf, alphabets: Alphabets,
                       x: Mapping, y: Mapping) -> Dict[Any, float]:
    """
    Per-node `divergence` from `x` to `y`, which is zero at every node exactly
    when `x` refines `y` pointwise.

    This is the monotonicity certificate of the flow: for the primal flow every
    step must satisfy `divergence_profile(x_next, x_now) == 0` everywhere, and
    for the dual flow the same with the arguments swapped. A nonzero entry names
    the node that moved the wrong way.
    """
    return {i: divergence(x[i], y[i], alphabets.node(i)) for i in sheaf.nodes()}


# ---------------------------------------------------------------------------
# abstract agreement against concrete disagreement
# ---------------------------------------------------------------------------

def relabel(contract: Contract, source: Sequence, target: Sequence) -> Contract:
    """
    The same contract read in another agent's copy of the same vocabulary,
    renaming `source[k]` to `target[k]`.

    This is meaningful only because the two alphabets are *copies* -- agent u's
    `k`-th fact and agent v's `k`-th fact are the same fact about the world,
    held privately. Nothing in the sheaf knows that; it is a fact about how the
    ensemble was built, and the sheaf is right not to know it, since a stalk's
    variables are private by construction. The renaming is what lets an outside
    observer ask a question the agents cannot ask each other directly: do you
    two actually believe the same thing?
    """
    if len(source) != len(target):
        raise ValueError(
            f"cannot relabel between alphabets of different size: "
            f"{len(source)} vs {len(target)}")
    pairs = [(s, t) for s, t in zip(source, target) if not s.eq(t)]
    if not pairs:
        return contract
    return Contract(z3.substitute(contract.a, *pairs),
                    z3.substitute(contract.sat_g, *pairs),
                    vars=[str(t) for t in target])


def concrete_disagreement(sheaf: ContractSheaf, u: Any, v: Any,
                          alphabets: Alphabets,
                          x: Optional[Mapping] = None) -> float:
    """
    How far apart two agents are in their own concrete vocabulary, with `v`'s
    contract relabelled into `u`'s copy.

    Contrast `edge_disagreement`, which measures them after both have been
    pushed into the shared abstract vocabulary. A flow that closes the second
    while leaving the first positive has reached agreement at the resolution the
    agents committed to communicate at, and no finer -- which is the point of
    the abstraction, and shows up here as the gap between two numbers rather
    than as a caveat in prose.

    Requires both nodes to carry a ``vars`` attribute listing their variables in
    the same order; positions are what identifies a fact across agents.
    """
    su, sv = sheaf.nodes[u].get('vars'), sheaf.nodes[v].get('vars')
    if su is None or sv is None:
        raise KeyError(
            f"nodes {u!r} and {v!r} must both carry a 'vars' attribute for their "
            f"vocabularies to be identified position by position")
    cu = sheaf.contract(u) if x is None else x[u]
    cv = sheaf.contract(v) if x is None else x[v]
    return distance(cu, relabel(cv, list(sv), list(su)), alphabets.node(u))


def concrete_energy(sheaf: ContractSheaf, alphabets: Alphabets,
                    x: Optional[Mapping] = None) -> float:
    """`concrete_disagreement` summed over interfaces -- `dirichlet`'s concrete twin."""
    return sum(concrete_disagreement(sheaf, u, v, alphabets, x=x)
               for u, v in sheaf.interfaces())
