"""
The region-abstracted mission sheaf: contracts per tile, agreement per region.

This is the non-degenerate construction `doc/DUALITY.md` calls for, replacing
the delivered demo's bijective restrictions (`gridsheaf.edge_relation`, kept
there as the control arm). Agent stalks speak a concrete per-tile language;
edge stalks speak a coarser per-region language; the restriction maps push by
`lan` (MS3 case (1)), and the sheaf condition -- both endpoints pushing to the
same contract -- is checked in the *interface* alphabet. Two agents whose tile
beliefs differ inside one region push to the same region summary: the kernel
of the abstraction is exactly the disagreement the mission tolerates.

The alphabets. Locally, agent `i` assigns a 3-bit possibility mask `S{i}_x_y`
to every tile (as in `gridsheaf`), a Bool `U{i}_r` per region ("my committed
route passes through r"), and a Bool `O{i}_{j}_r` per neighbor and region
("agent j's route passes through r" -- learned from the flow, never asserted
first-hand). On the interface {i, j}, per region and per label: a flagged bit
`F` and an excluded bit `X` (see `interface_vars` for why knowledge needs
both polarities over an upper-bound encoding), and one claim bit per endpoint
`K..{i}_r`, `K..{j}_r`. The restriction relation is the graph of the total
function sending a concrete state to its region summary (`summary_defs`).
Because it is a total function, `lan`/`ran`/`pullback` collapse to the
adjoint triple of MS3 Def. 6, and because it is massively non-injective, the
two bisheaves genuinely differ.

The mission contract of agent `i` at region `r` is the two-slot pair

    A_r = "every corridor tile of mine in r keeps a possibility outside
           hazard" and, when r is on my route, "no neighbor routes through r"
    G_r = "the world is as I know it in r" and "U_r == (my route crosses r)"

with a division of labor this module is deliberate about. The *guarantee* is
asserted unconditionally in the stalk and is what the Laplacian transports:
facts and commitments circulate. The *assumption* is monitored, not
saturated into the circulating guarantee -- because contract saturation voids
every promise on assumption-violating states, a guarantee conditioned on a
nontrivial assumption entails almost nothing under the existential push leg
(any not-yet-refuted violation model escapes it), and the network would go
mute exactly where coordination is needed. Measured, not hypothetical: with
reliance saturated in, an agent relying on one unknown tile transmits no
knowledge at all. So the sheaf circulates assertions; `assumption_status`
checks each agent's mission assumption against its fused guarantee (the same
meet-then-read-out pattern as the legacy mission monitor, now over claims as
well as hazards); a violation triggers replanning and `recommit`; and the
full two-slot contract is still a first-class object -- `mission_contract`
for the stalk-level pair, `interface_contract` for its pushforward, both
legs, onto an interface.

Stalks are `ProductContract`s -- one contract per region -- and restriction
maps are `ProductRelation`s. Nothing in the construction couples regions, so
the sheaf is a product of per-region sheaves; the Tarski theory applies
verbatim (products of suplattices are suplattices), every solver query stays
inside one region's small alphabet, and agreement is reported per (interface,
region). See `agsheaf.product`.

The flow state is exact, not flattened: each stalk component is kept as its
own-knowledge formula plus a list of heard conjuncts -- the shape the meet
produces -- with transported contracts computed in *closed form* (pushforward
as a finite image over the interface booleans, pullback as substitution; see
the closed-form section) so every formula the flow ever holds is
quantifier-free over one region's block. The per-tile masks the planner and
renderer read are a view decoded from this state, never written back: the
mask flatten of `gridsheaf.sweep` is only sound along bijections
(doc/DUALITY.md section 4), and here the restrictions are not.
"""

import z3
import networkx as nx
from typing import (Any, Dict, Iterable, List, Mapping, NamedTuple,
                    Optional, Sequence, Tuple)

from .contracts import Contract, Relation, Undecided, _is_satisfiable, _is_valid
from .product import ProductContract, ProductRelation
from .regions import Regions
from .sheaf import ContractSheaf
from .gridsheaf import (EMPTY_MASK, FULL_MASK, LABEL_BITS, UNSAFE_BIT,
                        Conflict, Tile, beliefs_to_masks, mask_label,
                        mask_labels, tile_vars)

#: Sheaves built here mark themselves so `gridsheaf`'s entry points dispatch.
MODE = "regions"


# ---------------------------------------------------------------------------
# variables
# ---------------------------------------------------------------------------

def route_vars(name: Any, regions: Regions) -> Dict[str, z3.BoolRef]:
    """`U{name}_{r}` per region: the agent's own route commitment bits."""
    return {r: z3.Bool(f"U{name}_{r}") for r in regions.names}


def expectation_vars(name: Any, neighbors: Iterable[Any],
                     regions: Regions) -> Dict[Any, Dict[str, z3.BoolRef]]:
    """
    `O{name}_{j}_{r}` per neighbor and region: whether agent `name` knows
    neighbor `j`'s route to pass through `r`. One variable per neighbor rather
    than one "somebody else" bit, because each interface wires the bit to a
    *different* neighbor's claim; a shared bit would receive contradictory
    biconditionals from two interfaces and void the stalk.
    """
    return {j: {r: z3.Bool(f"O{name}_{j}_{r}") for r in regions.names}
            for j in neighbors}


def interface_vars(u: Any, v: Any, regions: Regions) -> Dict[str, Any]:
    """
    The shared edge alphabet of the interface {u, v}: per region and per
    label, a *flagged* summary bit ``F`` ("some tile's possibilities have
    narrowed to within this label" -- which includes the emptied, contested
    tile) and an *excluded* bit ``X`` ("no tile here still admits this
    label"), plus one claim bit ``K`` per endpoint. Named over the sorted
    pair so both endpoints agree on the alphabet whichever way the interface
    is added.

    Two polarities per label because the stalk guarantees are upper bounds,
    deliberately admitting the empty possibility set (that is what keeps
    contradictions in-lattice): a single "pinned here" predicate would never
    be entailed -- the empty-set model escapes it -- and a single subset
    predicate would be escaped in the other polarity. ``F`` is what a pinned
    belief entails, ``X`` is what an all-clear belief entails, silence
    entails neither, and a region whose fused stalk entails both ``F`` and
    ``X`` for one label is contested at region level -- the same
    contradiction-as-a-value the per-tile empty mask records, one floor up
    (`contested_regions`).
    """
    lo, hi = sorted((u, v), key=str)
    return {
        'F': {label: {r: z3.Bool(f"F{lo}_{hi}_{label}_{r}") for r in regions.names}
              for label in LABEL_BITS},
        'X': {label: {r: z3.Bool(f"X{lo}_{hi}_{label}_{r}") for r in regions.names}
              for label in LABEL_BITS},
        'K': {end: {r: z3.Bool(f"K{lo}_{hi}_{end}_{r}") for r in regions.names}
              for end in (lo, hi)},
    }


def _block_vars(data: Mapping, r: str, regions: Regions) -> List:
    """Agent-side alphabet of one region: its tiles' masks, U_r, and the O_jr."""
    return ([data['v'][t] for t in regions.tiles(r)]
            + [data['uvar'][r]]
            + [data['ovar'][j][r] for j in data['ovar']])


# ---------------------------------------------------------------------------
# the restriction relation
# ---------------------------------------------------------------------------

def summary_defs(node: Any, neighbor: Any, data: Mapping, ifc: Mapping,
                 regions: Regions) -> Dict[str, List[Tuple[z3.BoolRef, z3.BoolRef]]]:
    """
    Per region, the interface variables paired with their defining terms over
    the node's block -- the region-summary function f_{node <| e} written out:
    per label l with bit b,

        F_l_r  :=  some tile t of r has S_t & ~b == 0   (narrowed to within l)
        X_l_r  :=  every tile t of r has S_t & b == 0   (l no longer admitted)
        K[node]_r      :=  U{node}_r              (own claim, under own name)
        K[neighbor]_r  :=  O{node}_{neighbor}_r   (expectation, under theirs)

    This is the single source of truth for the abstraction: the relation the
    sheaf installs is its graph (`abstraction_relation`), and the closed-form
    transport (`_push_component` / `_pull_component`) evaluates it directly.
    """
    S = data['v']
    out = {}
    for r in regions.names:
        tiles = regions.tiles(r)
        defs: List[Tuple[z3.BoolRef, z3.BoolRef]] = []
        for label, bit in LABEL_BITS.items():
            defs.append((ifc['F'][label][r],
                         z3.Or([S[t] & (FULL_MASK & ~bit) == 0 for t in tiles])))
            defs.append((ifc['X'][label][r],
                         z3.And([S[t] & bit == 0 for t in tiles])))
        defs.append((ifc['K'][node][r], data['uvar'][r]))
        defs.append((ifc['K'][neighbor][r], data['ovar'][neighbor][r]))
        out[r] = defs
    return out


def abstraction_relation(node: Any, neighbor: Any, data: Mapping,
                         ifc: Mapping, regions: Regions) -> ProductRelation:
    """
    The graph of the region-summary function f_{node <| e} for the interface
    {node, neighbor}: per region, the conjunction of ``edge_var == term``
    biconditionals from `summary_defs`.

    Biconditionals, not implications: the summary is exact, so pinned beliefs
    entail F, all-clear beliefs entail X, silence entails neither -- knowledge
    is transmitted in both polarities and ignorance stays silent -- and the
    relation stays the graph of a total function, which is what collapses the
    four embedding maps to MS3's adjoint triple (Def. 6). It is massively
    non-injective: which tile carries the flag is exactly what the interface
    cannot see, so tile-level disagreements inside one region push to the
    same summary and the sheaf calls that agreement.

    The two endpoints of one interface install *different* relations (the K
    wiring is mirrored), which is the mechanism by which one agent's claim
    reaches the other's expectation variables via push-then-pull.

    The source alphabet lists every variable of the node's region block,
    including expectation bits about *other* neighbors; those do not occur in
    the relation and are simply private to this interface, quantified away by
    the embedding maps like any private variable.
    """
    all_defs = summary_defs(node, neighbor, data, ifc, regions)
    comps = {}
    for r in regions.names:
        defs = all_defs[r]
        comps[r] = Relation(z3.And([var == term for var, term in defs]),
                            _block_vars(data, r, regions),
                            [var for var, _ in defs])
    return ProductRelation(comps)


# ---------------------------------------------------------------------------
# the contracts
# ---------------------------------------------------------------------------

def _own_guarantee(data: Mapping, r: str, regions: Regions) -> z3.BoolRef:
    """
    The agent's first-hand guarantee over one region block: an upper bound
    ``S_t & ~mask == 0`` per believed tile ("the world is as I know it"), and
    when the agent holds a committed route, the pin ``U_r == (route crosses
    r)`` in both polarities -- publishing "I will not cross r" is as much a
    commitment as the opposite. Asserted unconditionally: this is what the
    flow circulates.
    """
    S = data['v']
    masks = beliefs_to_masks(data['own'])
    bounds = [S[t] & (FULL_MASK & ~masks[t]) == 0
              for t in regions.tiles(r) if t in masks and masks[t] != FULL_MASK]
    if data['route'] is not None:
        bounds.append(data['uvar'][r] == z3.BoolVal(r in data['route_regions']))
    return z3.And(bounds) if bounds else z3.BoolVal(True)


def _own_assumption(data: Mapping, r: str, regions: Regions) -> z3.BoolRef:
    """
    The agent's mission assumption over one region block: every corridor tile
    of its route lying in `r` keeps a possibility outside hazard (`reliance`;
    violated both by a tile pinned hazardous and by one contested down to the
    empty set), and when `r` is on the agent's route, no neighbor routes
    through it (`exclusive_claims`). Region-local by design: contract
    semantics voids promises on assumption-violating states, so a
    cross-region assumption would let a violation anywhere void readings
    everywhere.

    This is the monitored slot of the mission contract -- `assumption_status`
    checks it against the fused guarantee -- not a conjunct of what the flow
    circulates; see the module docstring for why saturating it into the
    broadcast would silence the network.
    """
    cfg = data['contracts_cfg']
    conjuncts = []
    if data['route'] is not None and cfg.get('reliance', True):
        tiles = set(regions.tiles(r))
        conjuncts += [data['v'][t] & (FULL_MASK & ~UNSAFE_BIT) != 0
                      for t in data['route'] if t in tiles]
    if (data['route'] is not None and cfg.get('exclusive_claims', True)
            and r in data['route_regions']):
        conjuncts += [z3.Not(data['ovar'][j][r]) for j in data['ovar']]
    return z3.And(conjuncts) if conjuncts else z3.BoolVal(True)


def _component(data: Mapping, r: str, regions: Regions) -> Contract:
    """
    One region's stalk contract, rebuilt from the canonical state: the own
    guarantee conjoined with the transported `heard` conjuncts -- literally
    the shape the meet produces, so rebuilding from the lists is the meet,
    not an approximation of it. `heard` entries are (source, term) pairs; the
    provenance is what `recommit`'s epoch purge selects on, and only the
    terms enter the formula.
    """
    state = data['state'][r]
    terms = [term for _, term in state['heard']]
    g = state['own_g'] if not terms else z3.And([state['own_g']] + terms)
    return Contract(z3.BoolVal(True), g, vars=_block_vars(data, r, regions))


def _rebuild(data: Mapping, regions: Regions) -> None:
    """Recomputes the node's `ProductContract` from its canonical state."""
    data['c'] = ProductContract({r: _component(data, r, regions)
                                 for r in regions.names})


def mission_contract(sheaf: ContractSheaf, agent: Any, r: str) -> Contract:
    """
    The agent's two-slot mission contract at one region: the monitored
    assumption paired with the asserted guarantee. This is the A/G object the
    run logs and renders -- the stalk circulates its guarantee, the monitor
    checks its assumption, `recommit` rewrites both.
    """
    data = sheaf.nodes[agent]
    state = data['state'][r]
    return Contract(state['own_a'], state['own_g'],
                    vars=_block_vars(data, r, sheaf.graph['regions']))


# ---------------------------------------------------------------------------
# building the sheaf
# ---------------------------------------------------------------------------

def build_region_sheaf(beliefs: Sequence[Dict[Tile, str]],
                       grid_width_height: Tuple[int, int],
                       topology: nx.Graph,
                       regions: Regions,
                       corridors: Optional[Mapping[Any, Sequence[Tile]]] = None,
                       contracts_cfg: Optional[Mapping[str, Any]] = None
                       ) -> ContractSheaf:
    """
    The mission sheaf of a run: one product stalk per agent over the
    communication topology, with `abstraction_relation` restrictions onto a
    fresh per-region interface alphabet per edge.

    `corridors` maps each agent to the tile path its plan relies on
    (`gridworld.route_corridor`); omitting it, or omitting an agent, builds a
    belief-only stalk with trivial assumptions and no route commitment.
    `contracts_cfg` holds the `contracts:` block -- `reliance` and
    `exclusive_claims`, both defaulting on.

    Each node carries the canonical flow state (see `sweep`) plus the same
    view attributes the legacy sheaf keeps -- `masks`, the caller's `beliefs`
    dict (projected into in place), and `own` -- so the planner and renderer
    read communication the same way in both modes.
    """
    if sorted(topology.nodes()) != list(range(len(beliefs))):
        raise ValueError(
            f"topology nodes {sorted(topology.nodes())} do not match "
            f"{len(beliefs)} agents; expected exactly range({len(beliefs)})"
        )
    if regions.grid_width_height != tuple(grid_width_height):
        raise ValueError(
            f"regions partition a {regions.grid_width_height} grid but the "
            f"world is {tuple(grid_width_height)}"
        )
    corridors = dict(corridors or {})
    contracts_cfg = dict(contracts_cfg or {})

    sheaf = ContractSheaf()
    sheaf.graph['mode'] = MODE
    sheaf.graph['regions'] = regions
    sheaf.graph['contracts_cfg'] = contracts_cfg

    for i in sorted(topology.nodes()):
        neighbors = sorted(topology.neighbors(i))
        route = corridors.get(i)
        data = {
            'v': tile_vars(i, grid_width_height),
            'uvar': route_vars(i, regions),
            'ovar': expectation_vars(i, neighbors, regions),
            'beliefs': beliefs[i],
            'own': dict(beliefs[i]),
            'masks': beliefs_to_masks(beliefs[i]),
            'route': None if route is None else [tuple(map(int, t)) for t in route],
            'contracts_cfg': contracts_cfg,
        }
        data['route_regions'] = ([] if data['route'] is None
                                 else regions.regions_of(data['route']))
        data['state'] = {}
        for r in regions.names:
            data['state'][r] = {'own_a': _own_assumption(data, r, regions),
                                'own_g': _own_guarantee(data, r, regions),
                                'heard': []}
        _rebuild(data, regions)
        data['c_init'] = data['c']
        sheaf.add_node(i, **data)

    for u, w in topology.edges():
        ifc = interface_vars(u, w, regions)
        sheaf.graph.setdefault('ifc', {})[tuple(sorted((u, w)))] = ifc
        sheaf.graph.setdefault('defs', {})
        sheaf.graph['defs'][(u, w)] = summary_defs(u, w, sheaf.nodes[u], ifc, regions)
        sheaf.graph['defs'][(w, u)] = summary_defs(w, u, sheaf.nodes[w], ifc, regions)
        sheaf.add_interface(
            u, w,
            abstraction_relation(u, w, sheaf.nodes[u], ifc, regions),
            abstraction_relation(w, u, sheaf.nodes[w], ifc, regions),
        )
    return sheaf


# ---------------------------------------------------------------------------
# closed-form transport along the graph of the summary function
# ---------------------------------------------------------------------------
#
# The generic Kan operators (`Contract.lan` / `.pullback`, applied by
# `ContractSheaf.transport`) are correct here but build quantified formulas,
# and the flow composes them across sweeps: each hop wraps the neighbour's
# previous heard terms in a fresh quantifier layer, and by the third sweep z3
# is being asked about towers it cannot decide in reasonable time. For the
# graph of a total function both directions admit exact closed forms that
# stay quantifier-free:
#
#   * pullback is substitution:  R*(A, G) = (A o f, G o f) -- replace each
#     interface variable by its defining term;
#   * pushforward is a finite image: the interface alphabet of one region is
#     a handful of booleans, so Exists x (f(x) = y ^ phi(x)) is exactly the
#     disjunction of the reachable summaries, enumerable by repeated SAT with
#     blocking clauses (at most 2^|defs| of them, in practice few); the
#     universal legs are the dual, not-image-of-not.
#
# A fixture test checks these against the generic operators for equivalence;
# the flow itself uses only the closed forms, so every formula it ever holds
# is quantifier-free over one region's block.

def _image(phi: z3.BoolRef, defs: Sequence[Tuple[z3.BoolRef, z3.BoolRef]]) -> z3.BoolRef:
    """
    The exact image of `phi` under the summary function, as a formula over
    the interface variables: the disjunction of every summary some model of
    `phi` maps to, found by model enumeration with blocking clauses on the
    defining terms.
    """
    solver = z3.Solver()
    solver.add(phi)
    disjuncts: List[z3.BoolRef] = []
    for _ in range((1 << len(defs)) + 1):
        verdict = solver.check()
        if verdict == z3.unsat:
            break
        if verdict != z3.sat:
            raise Undecided(f"z3 returned unknown enumerating an image: "
                            f"{solver.reason_unknown()}")
        model = solver.model()
        bits = [z3.is_true(model.eval(term, model_completion=True))
                for _, term in defs]
        disjuncts.append(z3.And([var if b else z3.Not(var)
                                 for (var, _), b in zip(defs, bits)]))
        solver.add(z3.Not(z3.And([term if b else z3.Not(term)
                                  for (_, term), b in zip(defs, bits)])))
    else:
        raise RuntimeError("image enumeration outran the abstract state space, "
                           "which cannot happen: blocking clauses must exhaust it")
    return z3.simplify(z3.Or(disjuncts)) if disjuncts else z3.BoolVal(False)


def _push_component(sheaf: ContractSheaf, u: Any, v: Any, r: str, legs: str = "kan",
                    x: Optional[Mapping] = None) -> Contract:
    """
    Node `u`'s region-r component pushed onto the interface {u, v} alphabet:
    the closed form of `pushforward` restricted to one region. `legs="kan"`
    pushes by lan (assumption not-image-of-not, guarantee image), `legs="co"`
    by ran (the dual pairing). Cached on the component's interned ASTs, so a
    component the flow has not changed is never re-enumerated -- across sweeps
    and across grid steps.

    The cache entry holds the keyed expressions and re-checks them with
    ``.eq()`` on every hit: z3 recycles AST ids after garbage collection
    (the same hazard `measure.models` documents), and a cache keyed on a
    bare id has been observed serving one epoch's pushed image for a
    different epoch's contract -- the recycled id landed on the new formula
    and the flow transported retracted claims back in.
    """
    c = (x[u] if x is not None else sheaf.contract(u))[r]
    defs = sheaf.graph['defs'][(u, v)][r]
    cache = sheaf.graph.setdefault('_push_cache', {})
    key = (u, v, r, legs, c.a.get_id(), c.sat_g.get_id())
    hit = cache.get(key)
    if hit is not None:
        held_a, held_g, out = hit
        if held_a.eq(c.a) and held_g.eq(c.sat_g):
            return out
    if legs == "co":
        a = _image(c.a, defs)
        g = z3.simplify(z3.Not(_image(z3.Not(c.sat_g), defs)))
    else:
        a = z3.simplify(z3.Not(_image(z3.Not(c.a), defs)))
        g = _image(c.sat_g, defs)
    out = Contract(a, g, vars=[var for var, _ in defs])
    cache[key] = (c.a, c.sat_g, out)
    return out


def _pull_component(edge_contract: Contract,
                    defs: Sequence[Tuple[z3.BoolRef, z3.BoolRef]],
                    block: List) -> Contract:
    """
    An interface contract pulled back into a node's block: substitution of
    each interface variable by its defining term -- exact for the graph of a
    total function, where pullback and dual pullback coincide.
    """
    pairs = [(var, term) for var, term in defs]
    return Contract(z3.simplify(z3.substitute(edge_contract.a, *pairs)),
                    z3.simplify(z3.substitute(edge_contract.sat_g, *pairs)),
                    vars=block)


def transport_component(sheaf: ContractSheaf, source: Any, target: Any, r: str,
                        x: Optional[Mapping] = None, legs: str = "kan") -> Contract:
    """
    One region of `ContractSheaf.transport(source -> target)`, in closed
    form: push `source`'s component to the interface, pull it into `target`'s
    block. Semantically identical to the generic Kan composite (a fixture
    test holds the two together); structurally quantifier-free.
    """
    pushed = _push_component(sheaf, source, target, r, legs=legs, x=x)
    data = sheaf.nodes[target]
    return _pull_component(pushed, sheaf.graph['defs'][(target, source)][r],
                           _block_vars(data, r, sheaf.graph['regions']))


def interface_contract(sheaf: ContractSheaf, u: Any, v: Any, r: str) -> Contract:
    """
    Node `u`'s *mission* contract at region `r` pushed onto the interface
    {u, v} by lan: assumption leg "every summary all of whose realisations
    satisfy my assumption", guarantee leg "every summary consistent with what
    I assert". This is the broadcast message as a person would read it --
    R_!(A, G) of the plan's interpretation table -- computed for display and
    logging; the flow itself circulates the stalk (whose slotted assumption
    is trivial; the module docstring explains the division).
    """
    data = sheaf.nodes[u]
    state = data['state'][r]
    defs = sheaf.graph['defs'][(u, v)][r]
    c = mission_contract(sheaf, u, r)
    return Contract(z3.simplify(z3.Not(_image(z3.Not(c.a), defs))),
                    _image(c.sat_g, defs),
                    vars=[var for var, _ in defs])


# ---------------------------------------------------------------------------
# canonicalisation helpers -- all semantics-preserving
# ---------------------------------------------------------------------------

def _canon(expr: z3.BoolRef) -> z3.BoolRef:
    """
    Cheap deterministic normalisation. Deliberately NOT quantifier
    elimination: measured on this construction (2026-08-11), `simplify.qf`'s
    qe2 takes 5-30 seconds per transported leg and *grows* the result an
    order of magnitude, while the closed-form transport keeps every formula
    quantifier-free by construction and z3 decides everything downstream in
    milliseconds. `z3.simplify` is deterministic, so fixed-point detection by
    interned-AST identity (`_absorb`) holds.
    """
    return z3.simplify(expr)


def _conjuncts(expr: z3.BoolRef) -> List[z3.BoolRef]:
    """Top-level And flattened to a list, dropping literal Trues."""
    out, stack = [], [expr]
    while stack:
        e = stack.pop()
        if z3.is_and(e):
            stack.extend(e.children())
        elif not z3.is_true(e):
            out.append(e)
    return out


def _absorb(heard: List[Tuple[Any, z3.BoolRef]], source: Any,
            new: z3.BoolRef) -> bool:
    """
    Adds `(source, new)` to `heard` unless an identical term (same interned
    AST) is already present. Returns whether anything was added. Identity,
    not equivalence: transported formulas are rebuilt deterministically from
    the frozen assignment, so at a fixed point they re-arrive as the very
    same AST and the state provably stops changing.
    """
    if any(t.get_id() == new.get_id() for _, t in heard):
        return False
    heard.append((source, new))
    return True


def _prune_heard(state: Dict) -> None:
    """
    Drops heard conjuncts entailed by the rest of the component -- including
    the own layer, which is what absorbs an agent's own information echoed
    back through a neighbor. Entailed conjuncts only, so the contract's
    denotation is untouched; `Undecided` keeps the term.
    """
    kept: List[Tuple[Any, z3.BoolRef]] = []
    remaining = list(state['heard'])
    while remaining:
        source, term = remaining.pop(0)
        rest = [state['own_g']] + [t for _, t in kept] + [t for _, t in remaining]
        try:
            entailed = _is_valid(z3.Implies(z3.And(rest), term))
        except (Undecided, z3.Z3Exception):
            entailed = False
        if not entailed:
            kept.append((source, term))
    state['heard'] = kept


# ---------------------------------------------------------------------------
# the view: masks decoded from the exact state
# ---------------------------------------------------------------------------

def _decode_region(data: Mapping, r: str, regions: Regions) -> Dict[Tile, int]:
    """
    The strongest per-tile possibility masks the region component entails.
    A solver `unknown` keeps the bit: lose information rather than invent it.
    """
    state = data['state'][r]
    solver = z3.Solver()
    solver.add(state['own_g'])
    for _, term in state['heard']:
        solver.add(term)
    out: Dict[Tile, int] = {}
    for tile in regions.tiles(r):
        mask = EMPTY_MASK
        for bit in LABEL_BITS.values():
            solver.push()
            solver.add(data['v'][tile] & bit != 0)
            if solver.check() != z3.unsat:
                mask |= bit
            solver.pop()
        if mask != FULL_MASK:
            out[tile] = mask
    return out


def _project_beliefs(data: Mapping, tiles: Iterable[Tile],
                     new_masks: Mapping[Tile, int]) -> None:
    """
    The keep-own projection of `gridsheaf.sweep`, unchanged: singleton masks
    become labels, an emptied mask keeps the agent's own original label (or
    reverts to unknown if it held none), anything else says nothing.
    """
    beliefs, own = data['beliefs'], data['own']
    for tile in tiles:
        label = mask_label(new_masks.get(tile, FULL_MASK))
        if label is not None:
            beliefs[tile] = label
        elif tile in own:
            beliefs[tile] = own[tile]
        else:
            beliefs.pop(tile, None)


# ---------------------------------------------------------------------------
# the flow
# ---------------------------------------------------------------------------

def sweep(sheaf: ContractSheaf, firing: Optional[Iterable[Any]] = None,
          legs: str = "kan") -> Tuple[List[Conflict], bool]:
    """
    One communication step of the region sheaf: every agent fuses its firing
    neighbours' transported contracts, moving knowledge exactly one hop.

    The operator is the Tarski heat flow x <- (id ^ L)x in Jacobi form -- all
    transports are taken from the frozen assignment -- realised on the
    canonical state directly: a transported component's saturated guarantee
    contributes its conjuncts to the heard list, which is exactly what
    `Contract.meet` produces over trivial assumption slots, with dedupe and
    entailment pruning applied on the way in. Both are semantics-preserving,
    so this *is* the flow of `ContractSheaf.laplacian` with `include_self`,
    kept in a shape whose growth is bounded; a fixture test checks the two
    against each other.

    The per-tile masks and the caller's belief dicts are then re-decoded for
    every region whose component changed -- as a view of the state, never
    written back into it. Newly emptied masks are attributed to the differing
    neighbours in the returned `Conflict` list, as in the legacy sweep.

    Returns (conflicts, changed); unchanged means every term re-arrived as an
    AST already held, which only happens when the frozen assignment was a
    fixed point, so `changed` False is a sound stop signal.
    """
    regions: Regions = sheaf.graph['regions']
    x = sheaf.assignment()
    active = None if firing is None else set(firing)
    masks_before = {i: dict(sheaf.nodes[i]['masks']) for i in sheaf.nodes()}

    conflicts: List[Conflict] = []
    changed = False
    for i in sheaf.nodes():
        data = sheaf.nodes[i]
        broadcasters = [j for j in sheaf.neighbors(i)
                        if active is None or j in active]
        incoming = {(j, r): transport_component(sheaf, j, i, r, x=x, legs=legs)
                    for j in broadcasters for r in regions.names}
        touched: List[str] = []
        for r in regions.names:
            state = data['state'][r]
            region_touched = False
            for j in broadcasters:
                for term in _conjuncts(_canon(incoming[(j, r)].sat_g)):
                    region_touched |= _absorb(state['heard'], j, term)
            if region_touched:
                _prune_heard(state)
                touched.append(r)
        if not touched:
            continue
        changed = True
        _rebuild(data, regions)

        new_masks = dict(data['masks'])
        for r in touched:
            decoded = _decode_region(data, r, regions)
            for tile in regions.tiles(r):
                if tile in decoded:
                    new_masks[tile] = decoded[tile]
                else:
                    new_masks.pop(tile, None)

        contested = [t for t in sorted(new_masks)
                     if new_masks.get(t, FULL_MASK) == EMPTY_MASK
                     and masks_before[i].get(t, FULL_MASK) != EMPTY_MASK]
        if contested:
            conflicts.extend(_attribute(sheaf, i, incoming, contested, regions))

        tiles_touched = [t for r in touched for t in regions.tiles(r)]
        data['masks'] = new_masks
        _project_beliefs(data, tiles_touched, new_masks)

    return conflicts, changed


def _attribute(sheaf: ContractSheaf, i: Any, incoming: Mapping,
               contested: Sequence[Tile], regions: Regions) -> List[Conflict]:
    """
    Log-only attribution of newly contested tiles, as in the legacy sweep:
    decode each neighbour's transported component at just those tiles and name
    the proposals that differ from the agent's own label.
    """
    conflicts = []
    data = sheaf.nodes[i]
    own = data['own']
    for (j, r), inc in incoming.items():
        tiles = [t for t in contested if regions.region_of(t) == r]
        if not tiles:
            continue
        solver = z3.Solver()
        solver.add(inc.sat_g)
        for tile in tiles:
            mask = EMPTY_MASK
            for bit in LABEL_BITS.values():
                solver.push()
                solver.add(data['v'][tile] & bit != 0)
                if solver.check() != z3.unsat:
                    mask |= bit
                solver.pop()
            if mask == FULL_MASK:
                continue
            proposal = mask_label(mask) or "conflict"
            own_label = own.get(tile, "unknown")
            if proposal != own_label:
                conflicts.append(Conflict(i, j, tile, own_label, proposal))
    return conflicts


# ---------------------------------------------------------------------------
# out-of-flow updates: observation and recommitment
# ---------------------------------------------------------------------------

def observe(sheaf: ContractSheaf, agent: Any, tile: Tile, label: str) -> bool:
    """
    Records a first-hand observation into an agent's stalk: what the agent
    sees for itself overrides what it was told.

    As in the legacy sheaf this is not a step of the flow -- it can weaken a
    stalk, and it makes a fixed point no longer final. The override is
    region-granular here: the own layer of the tile's region is rebuilt around
    the observation, and heard conjuncts of that region that have become
    inconsistent with the new first-hand knowledge are dropped -- hearsay
    survives an observation exactly as far as it is compatible with it.

    Returns whether anything changed; a caller latched on the flow being
    settled must clear the latch when this returns True.
    """
    regions: Regions = sheaf.graph['regions']
    data = sheaf.nodes[agent]
    if label not in LABEL_BITS:
        raise ValueError(f"{label!r} is not an observable label; expected one of {sorted(LABEL_BITS)}")
    if tile not in data['v']:
        raise ValueError(f"tile {tile} is not on the grid agent {agent}'s stalk is over")

    mask = LABEL_BITS[label]
    if (data['masks'].get(tile) == mask and data['beliefs'].get(tile) == label
            and data['own'].get(tile) == label):
        return False

    data['own'][tile] = label
    data['beliefs'][tile] = label
    data['masks'] = {**data['masks'], tile: mask}

    r = regions.region_of(tile)
    state = data['state'][r]
    state['own_g'] = _own_guarantee(data, r, regions)
    state['own_a'] = _own_assumption(data, r, regions)
    state['heard'] = [(source, term) for source, term in state['heard']
                      if _sat_or_keep(z3.And(state['own_g'], term))]
    _rebuild(data, regions)
    return True


def _sat_or_keep(expr: z3.BoolRef) -> bool:
    """Satisfiability with `unknown` counting as satisfiable: keep, don't drop."""
    try:
        return _is_satisfiable(expr)
    except (Undecided, z3.Z3Exception):
        return True


def recommit(sheaf: ContractSheaf, agent: Any,
             corridor: Sequence[Tile]) -> bool:
    """
    Replaces an agent's mission commitment: a new corridor, hence new route
    regions, a new route guarantee and a new mission assumption.

    This is the second out-of-flow update, the one an assumption violation
    triggers (see `assumption_status`): the agent replans, recommits, and the
    flow then re-closes the sections its new claims opened.

    A recommitment starts a new *epoch*, and the purge is by provenance, not
    by inspecting formulas. Within one epoch every stalk only strengthens, so
    successive pushed images nest and heard terms can never contradict one
    another; everything that can go stale when agent `a` changes commitment
    is either held *by* `a` (facts derived jointly with its old commitment,
    including echoes of that commitment through a neighbor) or held by a
    neighbor and *sourced from* `a` (the pushed images that carried the old
    claims, whose cubes entangle them with world facts). So: `a` drops all
    its hearsay -- its own knowledge is untouched, and the flow re-delivers
    the network's within a diameter of sweeps, which is the honest price of
    changing one's plan -- and each neighbor drops exactly the terms sourced
    from `a`. Claims never travel further than one hop (the expectation
    variables are private to their interface), so no third node can hold
    stale commitment information. An earlier version purged by scanning
    formulas for claim variables; z3's simplifier entangles claim bits with
    world bits differently from run to run, and under-purging left jointly
    stale conjunctions that voided stalks. Provenance is granularity-proof.

    Returns whether anything changed; the settled latch must be cleared when
    it did, since both this agent's stalk and its neighbors' may have moved.
    """
    regions: Regions = sheaf.graph['regions']
    data = sheaf.nodes[agent]
    corridor = [tuple(map(int, t)) for t in corridor]
    route_regions = regions.regions_of(corridor)
    if data['route'] == corridor and data['route_regions'] == route_regions:
        return False

    if data['route'] is not None and data['route_regions'] == route_regions:
        # The corridor moved but its regions did not -- the ordinary case of an agent
        # advancing along its own route. Nothing the flow circulates depends on the
        # tiles themselves (the route pins and claims are per region), so only the
        # monitored assumption needs refreshing: no epoch, no purge, no reopened
        # sections. Returning False tells the caller the network saw nothing.
        data['route'] = corridor
        for r in regions.names:
            data['state'][r]['own_a'] = _own_assumption(data, r, regions)
        return False

    data['route'] = corridor
    data['route_regions'] = route_regions
    for r in regions.names:
        state = data['state'][r]
        state['own_a'] = _own_assumption(data, r, regions)
        state['own_g'] = _own_guarantee(data, r, regions)
        state['heard'] = []
    _rebuild(data, regions)
    _refresh_view(data, regions)

    for j in sheaf.neighbors(agent):
        neighbor = sheaf.nodes[j]
        touched = False
        for r in regions.names:
            state = neighbor['state'][r]
            kept = [(source, term) for source, term in state['heard']
                    if source != agent]
            if len(kept) != len(state['heard']):
                state['heard'] = kept
                touched = True
        if touched:
            _rebuild(neighbor, regions)
            _refresh_view(neighbor, regions)
    return True


def _refresh_view(data: Mapping, regions: Regions) -> None:
    """
    Re-decodes the whole per-tile view from the state and re-projects it into
    the caller's belief dict. Needed after an out-of-flow retraction, which
    can *weaken* the state: the incremental view updates in `sweep` only ever
    tighten.
    """
    new_masks: Dict[Tile, int] = {}
    for r in regions.names:
        new_masks.update(_decode_region(data, r, regions))
    data['masks'] = new_masks
    _project_beliefs(data, [t for r in regions.names for t in regions.tiles(r)],
                     new_masks)


# ---------------------------------------------------------------------------
# read-outs
# ---------------------------------------------------------------------------

def _entails(premise: z3.BoolRef, fact: z3.BoolRef) -> bool:
    """Entailment with `unknown` counting as not entailed: report only knowledge."""
    try:
        return not _is_satisfiable(z3.And(premise, z3.Not(fact)))
    except (Undecided, z3.Z3Exception):
        return False


def expected_claims(sheaf: ContractSheaf, agent: Any) -> Dict[str, List[Any]]:
    """
    Per region, the neighbours whose routes the agent's fused stalk *entails*
    pass through it -- what the flow has taught it about who is coming.
    """
    regions: Regions = sheaf.graph['regions']
    data = sheaf.nodes[agent]
    out: Dict[str, List[Any]] = {}
    for r in regions.names:
        g = data['c'][r].sat_g
        claimants = [j for j in data['ovar'] if _entails(g, data['ovar'][j][r])]
        if claimants:
            out[r] = claimants
    return out


def hazard_regions(sheaf: ContractSheaf, agent: Any) -> List[str]:
    """
    The regions whose fused guarantee entails "some tile here is
    hazard-flagged" (its possibilities have narrowed to within {unsafe}) --
    region-level hearsay the abstraction warrants without naming the tile.
    This is what the planner's cost overlay reads: an agent warned at region
    granularity routes around the region, which is exactly the behavior the
    interface vocabulary licenses.
    """
    regions: Regions = sheaf.graph['regions']
    data = sheaf.nodes[agent]
    out = []
    for r in regions.names:
        clause = z3.Or([data['v'][t] & (FULL_MASK & ~UNSAFE_BIT) == 0
                        for t in regions.tiles(r)])
        if _entails(data['c'][r].sat_g, clause):
            out.append(r)
    return out


def contested_regions(sheaf: ContractSheaf, agent: Any) -> Dict[str, List[str]]:
    """
    Per region, the labels for which the fused guarantee entails both the
    flagged and the excluded summary -- "somewhere here the possibilities have
    narrowed to within l" and "nothing here admits l" at once. Only the empty
    possibility set satisfies both, so this is the region-level contested
    marker: the network's knowledge about the region is contradictory without
    the contradiction being localised to any one tile. The per-tile analogue
    (an emptied mask) is what `sweep` reports as `Conflict`s; this is the
    same value one abstraction level up, and it is in-lattice -- the guarantee
    stays satisfiable throughout.
    """
    regions: Regions = sheaf.graph['regions']
    data = sheaf.nodes[agent]
    out: Dict[str, List[str]] = {}
    for r in regions.names:
        g = data['c'][r].sat_g
        tiles = regions.tiles(r)
        labels = []
        for label, bit in LABEL_BITS.items():
            flagged = z3.Or([data['v'][t] & (FULL_MASK & ~bit) == 0 for t in tiles])
            excluded = z3.And([data['v'][t] & bit == 0 for t in tiles])
            if _entails(g, flagged) and _entails(g, excluded):
                labels.append(label)
        if labels:
            out[r] = labels
    return out


class AssumptionStatus(NamedTuple):
    """
    One agent's mission assumption checked against its fused stalk, per
    region: `violated` lists the regions where the fused guarantee refutes the
    assumption outright, `hazards` names the corridor tiles the network has
    flagged hazardous, and `claimed` the route regions with the neighbours
    known to route through them. Checked against the *current commitment's*
    assumption (`own_a`): this is the meet-then-read-out pattern of the
    legacy mission monitor, generalised from corridor hazards to claims.
    """
    agent: Any
    violated: List[str]
    hazards: Dict[str, List[Tile]]
    claimed: Dict[str, List[Any]]


def assumption_status(sheaf: ContractSheaf, agent: Any) -> AssumptionStatus:
    """The mission monitor's read-out; see `AssumptionStatus`."""
    regions: Regions = sheaf.graph['regions']
    data = sheaf.nodes[agent]
    violated: List[str] = []
    hazards: Dict[str, List[Tile]] = {}
    claimed: Dict[str, List[Any]] = {}
    route = data['route'] or []
    for r in regions.names:
        g = data['c'][r].sat_g
        own_a = data['state'][r]['own_a']
        if not z3.is_true(own_a) and _entails(g, z3.Not(own_a)):
            violated.append(r)
        tiles = set(regions.tiles(r))
        pinned = [t for t in route if t in tiles
                  and _entails(g, data['v'][t] & (FULL_MASK & ~UNSAFE_BIT) == 0)]
        if pinned:
            hazards[r] = pinned
        if r in data['route_regions']:
            claimants = [j for j in data['ovar']
                         if _entails(g, data['ovar'][j][r])]
            if claimants:
                claimed[r] = claimants
    return AssumptionStatus(agent, violated, hazards, claimed)


# ---------------------------------------------------------------------------
# the snapshot
# ---------------------------------------------------------------------------

def render_contracts(sheaf: ContractSheaf) -> Dict[Any, List[str]]:
    """
    Each agent's mission contracts as the assume/guarantee pairs they are, per
    region, rendered as lines for the log. The regions an agent has nothing to
    say about -- trivial assumption, trivial guarantee -- are omitted, which is
    most of them; a contract that assumes and asserts nothing is not worth two
    kilobytes a sweep.
    """
    regions: Regions = sheaf.graph['regions']
    out: Dict[Any, List[str]] = {}
    for i in sheaf.nodes():
        lines: List[str] = []
        for r in regions.names:
            c = mission_contract(sheaf, i, r)
            a, g = z3.simplify(c.a), z3.simplify(c.g)
            if z3.is_true(a) and z3.is_true(g):
                continue
            lines.append(f"[{r}]")
            lines.extend(f"  A: {a}".splitlines())
            lines.extend(f"  G: {g}".splitlines())
        out[i] = lines
    return out


class RegionSheafState(NamedTuple):
    """
    A snapshot of the region sheaf between sweeps: the per-tile view
    (`possibilities`, `contested`), the region-level contested markers
    (`contested_regions`: labels both flagged and excluded at once -- the
    contradiction the abstraction holds without localising it to a tile; the
    per-tile `contested` often stays empty here precisely because the
    interface vocabulary absorbs tile-level conflicts), the sheaf condition
    measured in the interface alphabet (`sections` per interface,
    `section_regions` naming which regions agree on each, `is_section` for
    all of them), the claims picture, per-agent monotonicity
    (`refines_initial`, componentwise), the concrete disagreement each
    interface tolerates (`concrete_disagreement`: tiles on which the two
    endpoints' belief dicts differ -- nonzero while `is_section` holds is the
    abstraction earning its keep), and `collapsed`, expected empty since
    disagreement is recorded in-lattice.
    """
    possibilities: Dict[Any, Dict[Tile, Tuple[str, ...]]]
    contested: Dict[Any, List[Tile]]
    contested_regions: Dict[Any, Dict[str, List[str]]]
    sections: Dict[Tuple[Any, Any], bool]
    section_regions: Dict[Tuple[Any, Any], Dict[str, bool]]
    is_section: bool
    claims: Dict[Any, Dict[str, Any]]
    refines_initial: Dict[Any, bool]
    concrete_disagreement: Dict[Tuple[Any, Any], int]
    collapsed: Dict[Any, str]


def sheaf_state(sheaf: ContractSheaf, legs: str = "kan") -> RegionSheafState:
    """Snapshots the region sheaf as a `RegionSheafState`."""
    regions: Regions = sheaf.graph['regions']

    sections: Dict[Tuple[Any, Any], bool] = {}
    section_regions: Dict[Tuple[Any, Any], Dict[str, bool]] = {}
    disagreement: Dict[Tuple[Any, Any], int] = {}
    for u, v in sheaf.interfaces():
        key = tuple(sorted((u, v)))
        per_region = {}
        for r in regions.names:
            push_u = _push_component(sheaf, u, v, r, legs=legs)
            push_v = _push_component(sheaf, v, u, r, legs=legs)
            per_region[r] = bool(push_u == push_v)
        section_regions[key] = per_region
        sections[key] = all(per_region.values())
        bu, bv = sheaf.nodes[u]['beliefs'], sheaf.nodes[v]['beliefs']
        disagreement[key] = sum(1 for t in set(bu) | set(bv)
                                if bu.get(t) != bv.get(t))

    claims = {}
    for i in sheaf.nodes():
        data = sheaf.nodes[i]
        entry: Dict[str, Any] = {'committed': list(data['route_regions'])}
        expected = expected_claims(sheaf, i)
        if expected:
            entry['expected'] = expected
        claims[i] = entry

    return RegionSheafState(
        possibilities={i: {tile: mask_labels(mask)
                           for tile, mask in sheaf.nodes[i]['masks'].items()}
                       for i in sheaf.nodes()},
        contested={i: sorted(tile for tile, mask in sheaf.nodes[i]['masks'].items()
                             if mask == EMPTY_MASK)
                   for i in sheaf.nodes()},
        contested_regions={i: contested_regions(sheaf, i) for i in sheaf.nodes()},
        sections=sections,
        section_regions=section_regions,
        is_section=all(sections.values()),
        claims=claims,
        refines_initial={i: bool(sheaf.contract(i).refines(sheaf.initial(i)))
                         for i in sheaf.nodes()},
        concrete_disagreement=disagreement,
        collapsed=sheaf.collapsed(),
    )
