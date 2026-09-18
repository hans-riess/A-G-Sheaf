"""
A belief sheaf over the gridworld's communication topology.

Each agent holds partial, possibly wrong beliefs about tile labels (see
`gridworld.parse_label_beliefs`). This module encodes those beliefs as the
stalks of a `ContractSheaf` over the communication graph, so that running the
sheaf's Laplacian flow *is* the agents communicating: one sweep moves knowledge
exactly one hop.

The alphabet is powerset-valued. Per agent and tile there is a 3-bit vector
`S{i}_{x}_{y}` -- the set of labels the agent still considers possible, one bit
per label in `LABELS`. A held belief is the upper bound ``S & ~mask == 0``
(``S`` is contained in the believed singleton); an unheld tile is unconstrained.
Guarantees are conjunctions of such bounds, assumptions are trivial -- a
broadcast is an assertion, not a promise -- so the contract meet conjoins
bounds, which is pointwise intersection of possibility sets. Two agents
disagreeing about a tile therefore fuse to ``S ⊆ ∅``: a *satisfiable*
constraint, an element of the lattice, localised to the contested tile. Nothing
ever collapses to the inconsistent contract, so the flow stays exactly the
Tarski Laplacian on the finite product lattice of possibility masks, and
`ContractSheaf.collapsed` stays empty by construction. Conflict handling is a
read-out policy (see `sweep`), not an algebraic repair.

`sheaf_state` snapshots the whole assignment between sweeps -- possibility sets,
which interfaces have closed, whether the collection of them is a global section
-- and `communicate` takes an observer called on each iterate, which is the only
place the intermediate ones exist. That is what lets a run record the flow rather
than only the beliefs it leaves behind.

`observe` is the one way anything from outside the sheaf gets in: an agent that
drives over a tile sees what is really there, and what it sees overrides what it
was told. That is not a step of the flow -- it can weaken a stalk, and it makes a
fixed point no longer final -- so it is kept separate from `sweep` and its
consequences are spelled out there rather than smuggled into the Laplacian.

The assume/guarantee structure proper is exercised one level up: an agent's
route reliance -- the assumption of its mission contract, "the tiles I route
through admit a non-unsafe possibility" -- is reified by `reliance_contract`
and checked against the fused stalk with a contract meet plus the same ∅
read-out (`forced_empty`).

This module deliberately does not import `agsheaf.gridworld`, which pulls in
the Robotarium stack at import time; `LABELS` restates
`gridworld.GROUND_TRUTH_LABELS`, with a parity test in `test_gridsheaf.py`.
"""

import z3
import networkx as nx
from typing import (Any, Callable, Dict, Iterable, List, Mapping, NamedTuple,
                    Optional, Sequence, Tuple)
from agsheaf import Contract,Relation,ContractSheaf

Tile = Tuple[int, int]

#: Labels a belief can pin a tile to, in bit order. "unknown" is not a label
#: but the absence of a constraint, so it has no bit.
LABELS = ("target", "safe", "unsafe")

MASK_WIDTH = len(LABELS)
LABEL_BITS = {label: 1 << position for position, label in enumerate(LABELS)}
TARGET_BIT, SAFE_BIT, UNSAFE_BIT = (LABEL_BITS[label] for label in LABELS)
FULL_MASK = (1 << MASK_WIDTH) - 1
EMPTY_MASK = 0

#: Singleton masks back to their labels; any other mask has no label reading.
_SINGLETON_LABELS = {bit: label for label, bit in LABEL_BITS.items()}

#: Configured in place of a sweep count to run the flow to its fixed point.
CONVERGE = "converge"


def mask_label(mask: int) -> Optional[str]:
    """The label of a singleton possibility mask, or None for any other mask."""
    return _SINGLETON_LABELS.get(mask)


def mask_labels(mask: int) -> Tuple[str, ...]:
    """
    Every label a possibility mask still admits, in `LABELS` order: the mask
    read as the set it denotes rather than as bits. `FULL_MASK` gives all of
    `LABELS`, `EMPTY_MASK` the empty tuple -- the contested tile.

    This is the form anything outside this module should read masks in. It is
    what lets `log.py` record a stalk without restating the bit order, the way
    it would have to if the state travelled as integers.
    """
    return tuple(label for label, bit in LABEL_BITS.items() if mask & bit)


def grid_tiles(grid_width_height: Tuple[int, int]) -> List[Tile]:
    """
    Every (X,Y) tile of the grid, in a fixed order. The order is what lines
    the variable lists of `edge_relation` up across an interface, so every
    variable dict in this module is built by iterating it.
    """
    width, height = grid_width_height
    return [(x, y) for x in range(width) for y in range(height)]


def tile_vars(name: Any, grid_width_height: Tuple[int, int]) -> Dict[Tile, z3.BitVecRef]:
    """
    An agent's private copy of the world description: one possibility mask
    `S{name}_{x}_{y}` per tile. Stalks own their variables; agents talk only
    through the shared edge variables of `edge_vars`.
    """
    return {(x, y): z3.BitVec(f"S{name}_{x}_{y}", MASK_WIDTH)
            for x, y in grid_tiles(grid_width_height)}


def edge_vars(u: Any, v: Any, grid_width_height: Tuple[int, int]) -> Dict[Tile, z3.BitVecRef]:
    """
    The shared edge stalk of the interface {u, v}: fresh masks
    `E{u}_{v}_{x}_{y}`, named over the sorted pair so both endpoints agree on
    the alphabet whichever way the interface is added.
    """
    lo, hi = sorted((u, v), key=str)
    return {(x, y): z3.BitVec(f"E{lo}_{hi}_{x}_{y}", MASK_WIDTH)
            for x, y in grid_tiles(grid_width_height)}


def beliefs_to_masks(beliefs: Mapping[Tile, str]) -> Dict[Tile, int]:
    """
    A label-belief dict (the `parse_label_beliefs` shape) as singleton
    possibility masks. Entries labelled "unknown" hold nothing and are
    dropped, matching the absence-is-unknown convention of the belief dicts.
    """
    return {tile: LABEL_BITS[label]
            for tile, label in beliefs.items() if label in LABEL_BITS}


def encode_masks(masks: Mapping[Tile, int],
                 vars: Dict[Tile, z3.BitVecRef]) -> Contract:
    """
    The stalk contract of a possibility assignment: assumption trivially true,
    guarantee the conjunction of upper bounds ``S_t & ~mask == 0`` over every
    tile whose mask says anything (`FULL_MASK` is no constraint; `EMPTY_MASK`
    forces ``S_t == 0``, the in-lattice record of a contested tile). The
    alphabet is the full variable list, so `Contract._same_alphabet` guards
    every meet against a transported contract from the wrong stalk.
    """
    bounds = [vars[tile] & (FULL_MASK & ~mask) == 0
              for tile, mask in masks.items() if mask != FULL_MASK]
    return Contract(z3.BoolVal(True),
                    z3.And(bounds) if bounds else z3.BoolVal(True),
                    vars=list(vars.values()))


def decode_masks(contract: Contract, vars: Dict[Tile, z3.BitVecRef],
                 tiles: Optional[Iterable[Tile]] = None) -> Dict[Tile, int]:
    """
    The strongest possibility mask the contract entails per tile: a bit
    survives iff the saturated guarantee admits a model with that label still
    possible. Tiles decoding to `FULL_MASK` (nothing entailed) are omitted, so
    the result is in the same partial shape `encode_masks` consumes.

    A solver answer of `unknown` keeps the bit -- losing information rather
    than inventing it. `tiles` restricts the scan; the sweep passes only the
    tiles some participant constrains, which is what keeps decoding cheap.
    """
    solver = z3.Solver()
    solver.add(contract.sat_g)
    out: Dict[Tile, int] = {}
    for tile in (vars.keys() if tiles is None else tiles):
        mask = EMPTY_MASK
        for bit in LABEL_BITS.values():
            solver.push()
            solver.add(vars[tile] & bit != 0)
            if solver.check() != z3.unsat:
                mask |= bit
            solver.pop()
        if mask != FULL_MASK:
            out[tile] = mask
    return out


def edge_relation(node_vars: Dict[Tile, z3.BitVecRef],
                  shared_vars: Dict[Tile, z3.BitVecRef]) -> Relation:
    """
    The restriction of a stalk onto a shared edge stalk: the graph of the
    bijection equating each tile's private mask with the edge copy. Along the
    graph of a bijection `lan` and `ran` agree and transport is a semantic
    renaming, so the Laplacian's meet is exactly pointwise intersection of the
    neighbours' possibility masks.
    """
    tiles = list(node_vars.keys())
    return Relation(z3.And([shared_vars[t] == node_vars[t] for t in tiles]),
                    [node_vars[t] for t in tiles],
                    [shared_vars[t] for t in tiles])


def build_belief_sheaf(beliefs: Sequence[Dict[Tile, str]],
                       grid_width_height: Tuple[int, int],
                       topology: nx.Graph,
                       regions=None,
                       corridors=None,
                       contracts_cfg=None) -> ContractSheaf:
    """
    The belief sheaf of a run: one stalk per agent over the communication
    topology (`gridworld.parse_communication_topology`, whose nodes are
    exactly range(N)), with `edge_relation` restrictions onto a fresh edge
    stalk per interface.

    With `regions` (a `regions.Regions` partition) this instead builds the
    region-abstracted mission sheaf of `agsheaf.regionsheaf` -- per-tile agent
    alphabets, per-region interface alphabets, two-slot contracts -- and every
    entry point of this module (`sweep`, `communicate`, `observe`,
    `observe_tiles`, `sheaf_state`) dispatches on which construction it is
    handed. Without `regions` the construction below is the delivered f = id
    demonstration, kept verbatim as the degenerate control arm
    (doc/DUALITY.md): restrictions are bijections and the flow is the mask
    flatten, which is sound exactly there. `corridors` and `contracts_cfg`
    are the mission-contract inputs of `regionsheaf.build_region_sheaf` and
    are meaningless without a partition.

    Each node i carries:

      ``c``, ``c_init``  the stalk contract (and its snapshot, so
                         `ContractSheaf.initial` and `refines` diagnostics
                         work under the custom `sweep`);
      ``v``              the tile-variable dict;
      ``masks``          the canonical possibility state `sweep` maintains;
      ``beliefs``        THE CALLER'S label dict -- `sweep` projects fused
                         knowledge into it in place, which is what lets the
                         gridworld renderer and planner see communication
                         happen without any change on their side;
      ``own``            a frozen copy of the label dict as given, the
                         reference for the keep-own conflict policy.
    """
    if regions is not None:
        from .regionsheaf import build_region_sheaf
        return build_region_sheaf(beliefs, grid_width_height, topology,
                                  regions, corridors=corridors,
                                  contracts_cfg=contracts_cfg)
    assert corridors is None and contracts_cfg is None, \
        "Failed to build the belief sheaf. corridors and contracts_cfg belong to the region construction; pass a regions partition with them."

    if sorted(topology.nodes()) != list(range(len(beliefs))):
        raise ValueError(
            f"topology nodes {sorted(topology.nodes())} do not match "
            f"{len(beliefs)} agents; expected exactly range({len(beliefs)})"
        )
    sheaf = ContractSheaf()
    for i in sorted(topology.nodes()):
        v = tile_vars(i, grid_width_height)
        masks = beliefs_to_masks(beliefs[i])
        c = encode_masks(masks, v)
        sheaf.add_node(i, c=c, c_init=c, v=v, masks=masks,
                       beliefs=beliefs[i], own=dict(beliefs[i]))
    for u, w in topology.edges():
        shared = edge_vars(u, w, grid_width_height)
        sheaf.add_interface(u, w,
                            edge_relation(sheaf.nodes[u]['v'], shared),
                            edge_relation(sheaf.nodes[w]['v'], shared))
    return sheaf


class Conflict(NamedTuple):
    """
    One agent's record of a tile whose fused possibility set emptied: the
    network told it something irreconcilable with what it (or another
    neighbour) holds. `own` is the label the agent keeps under the keep-own
    policy ("unknown" when it held nothing); `incoming` is the differing label
    the neighbour proposed, or "conflict" when the neighbour itself carried an
    already-contested tile.
    """
    agent: Any
    neighbor: Any
    tile: Tile
    own: str
    incoming: str


def _attribute_conflicts(sheaf: ContractSheaf, x: Mapping, i: Any,
                         contested: Sequence[Tile]) -> List[Conflict]:
    """
    Log-only attribution: decode each neighbour's transported contract at just
    the contested tiles and name the proposals that differ from the agent's
    own label. Never touches sheaf state.
    """
    conflicts = []
    own = sheaf.nodes[i]['own']
    v = sheaf.nodes[i]['v']
    for j in sheaf.neighbors(i):
        incoming_masks = decode_masks(sheaf.transport(x, j, i), v, tiles=contested)
        for tile in contested:
            mask = incoming_masks.get(tile, FULL_MASK)
            if mask == FULL_MASK:
                continue
            incoming = mask_label(mask) or "conflict"
            own_label = own.get(tile, "unknown")
            if incoming != own_label:
                conflicts.append(Conflict(i, j, tile, own_label, incoming))
    return conflicts


def sweep(sheaf: ContractSheaf,
          firing: Optional[Iterable[Any]] = None) -> Tuple[List[Conflict], bool]:
    """
    One communication step: every agent fuses its firing neighbours' broadcasts,
    moving knowledge exactly one hop.

    `firing` is the set tau_t of Riess and Ghrist (2022), Def 5 -- the agents
    that broadcast this step. `None`, the default, fires everyone and recovers
    the synchronous sweep of their Eq. (4). An agent with no firing neighbour
    has nothing to fuse and is left untouched, which falls out of `laplacian`
    under `include_self` rather than needing to be special-cased here.

    Their Theorem 1 is what makes a partial firing set safe: the sections are
    the fixed points of the flow for any schedule satisfying liveness, so who
    broadcasts when changes the route to the answer and not the answer. What it
    does not do is excuse a schedule that starves an agent -- see
    `ContractSheaf.converge`, which raises rather than reporting a fixed point
    the flow never established.

    The operator is `ContractSheaf.laplacian` with `include_self` on a frozen
    assignment -- the heat flow x <- (id ∧ L)x of Riess and Ghrist (2022),
    Eq. (6), in Jacobi form. `laplacian_update` is deliberately not used: its
    `in_place` default lets later nodes read earlier nodes' updated contracts,
    which carries knowledge more than one hop per sweep in DiGraph node order,
    and its solver-equality change detection is redundant once the state is
    mask-canonical.

    After fusing, each node's result is decoded back to masks (the flatten
    that keeps formulas from growing across sweeps), projected into the
    caller's belief dict -- singleton masks become labels; an emptied mask
    keeps the agent's own original label (the keep-own policy), or reverts to
    unknown if it held none -- and re-encoded as the node's contract. Newly
    contested tiles are attributed to the differing neighbours in the returned
    `Conflict` list.

    Returns (conflicts, changed); `changed` False means the flow has reached
    its fixed point, which the meet's monotonicity makes permanent.
    """
    if sheaf.graph.get('mode') == 'regions':
        from . import regionsheaf
        return regionsheaf.sweep(sheaf, firing=firing)

    x = sheaf.assignment()
    masks_before = {i: sheaf.nodes[i]['masks'] for i in sheaf.nodes()}
    fused = sheaf.laplacian(x, include_self=True, firing=firing)

    conflicts: List[Conflict] = []
    changed = False
    for i in sheaf.nodes():
        data = sheaf.nodes[i]
        candidates = set(masks_before[i])
        for j in sheaf.neighbors(i):
            candidates |= set(masks_before[j])
        tiles = sorted(candidates)

        new_masks = decode_masks(fused[i], data['v'], tiles=tiles)
        if new_masks == masks_before[i]:
            continue
        changed = True

        contested = [t for t in tiles
                     if new_masks.get(t, FULL_MASK) == EMPTY_MASK
                     and masks_before[i].get(t, FULL_MASK) != EMPTY_MASK]
        if contested:
            conflicts.extend(_attribute_conflicts(sheaf, x, i, contested))

        beliefs, own = data['beliefs'], data['own']
        for tile in tiles:
            label = mask_label(new_masks.get(tile, FULL_MASK))
            if label is not None:
                beliefs[tile] = label
            elif tile in own:
                beliefs[tile] = own[tile]
            else:
                beliefs.pop(tile, None)

        data['masks'] = new_masks
        data['c'] = encode_masks(new_masks, data['v'])

    return conflicts, changed


def parse_sweeps_per_step(sweeps_per_step: Any, max_sweeps: int = 100) -> int:
    """
    How many sweeps one grid step runs, resolved from what the configuration said.

    A number is that many communication hops before the agents move. The literal
    "converge" instead runs the flow to its fixed point first, so that agents step on
    everything the network between them knows rather than on what has reached them so
    far. That needs no separate loop: `communicate` stops early at the fixed point, so
    converging is running it with a bound large enough never to be the thing that stops
    it. `max_sweeps` is that bound, and a safety net rather than a parameter -- the flow
    closes in at most diameter + 1 sweeps, which `exp/validate_theory.py` checks.

    Booleans are rejected rather than read as the ints Python makes them: a YAML
    `sweeps_per_step: true` is a mistake, not a request for one sweep.

    Args:
        sweeps_per_step: The configured value, a positive number of sweeps or "converge".
        max_sweeps (int): The bound "converge" resolves to.
    """
    if sweeps_per_step == CONVERGE:
        assert _is_positive_int(max_sweeps), "Failed to parse the sheaf configuration. The max_sweeps entry must be a positive number of sweeps, received %r." % (max_sweeps,)
        return max_sweeps

    assert _is_positive_int(sweeps_per_step), "Failed to parse the sheaf configuration. The sweeps_per_step entry must be a positive number of sweeps or \"%s\", received %r." % (CONVERGE, sweeps_per_step)
    return sweeps_per_step


def _is_positive_int(value: Any) -> bool:
    """A number of sweeps: an int, and not one of the bools that are ints."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def communicate(sheaf: ContractSheaf, sweeps: int = 1,
                on_sweep: Optional[Callable[[int, List[Conflict], bool], None]] = None,
                firing: Optional[Iterable[Any]] = None
                ) -> Tuple[List[Conflict], bool]:
    """
    Runs up to `sweeps` sweeps -- so many communication hops -- stopping early
    at a fixed point. Returns the accumulated conflicts and whether anything
    changed at all; once this reports False the caller can stop calling, since
    the flow is monotone and a fixed point is final.

    `on_sweep(index, conflicts, changed)` is called after each sweep, the
    last one included, with the sweep's index within this call. It is how a
    caller observes the individual Laplacian iterates rather than only their
    aggregate -- `sheaf_state(sheaf)` inside the callback snapshots the
    assignment between hops, which is what a per-sweep log or a video frame
    per iterate needs.

    `firing` restricts who broadcasts, and is held fixed across the sweeps of
    one call -- a caller wanting a different set per hop calls this once per
    hop. See `sweep`.
    """
    all_conflicts: List[Conflict] = []
    any_changed = False
    for index in range(sweeps):
        conflicts, changed = sweep(sheaf, firing=firing)
        all_conflicts.extend(conflicts)
        any_changed |= changed
        if on_sweep is not None:
            on_sweep(index, conflicts, changed)
        if not changed:
            break
    return all_conflicts, any_changed


def observe(sheaf: ContractSheaf, agent: Any, tile: Tile, label: str) -> bool:
    """
    Records a first-hand observation into an agent's stalk: the agent has been to `tile`
    and seen for itself that it is `label`.

    An observation is not a step of the flow. Everything a stalk holds by the time it has
    communicated arrived by hearsay, and the Laplacian can only ever narrow that; what an
    agent sees for itself arrives from outside the sheaf and overrides. So this writes the
    tile down as a singleton in all four of the pieces of node state `sweep` keeps mutually
    consistent -- the canonical `masks`, the contract `c` they encode, the caller's
    `beliefs` dict the planner and renderer read, and `own`, the keep-own reference.

    Writing `own` is what makes the override hold up against the network. A neighbour still
    broadcasting the old label meets it to the empty possibility set on both sides, and
    `sweep`'s keep-own policy then hands each of them its own back: the observer keeps what
    it saw, the neighbour keeps what it was told, and the tile reads as contested at both
    until the neighbour goes and looks for itself.

    Two consequences are worth naming, since neither is a fault. An observation can replace
    one singleton with another, or give a contested tile a label back, so it can *weaken* a
    stalk; `ContractSheaf.initial` is deliberately left as the 0-cochain the run started
    from, so a `refines(initial)` diagnostic reading False afterwards is the record that
    this happened rather than a broken invariant. And a fixed point stops being final: a
    caller that has latched on `communicate` reporting no change must clear that latch when
    this returns True, or the thing observed will never be told to anybody.

    Returns whether this changed anything.

    Args:
        sheaf (ContractSheaf): The belief sheaf whose stalk is written.
        agent: The observing node, an agent index.
        tile (Tile): The (X,Y) tile observed.
        label (str): What it was seen to be, one of `LABELS`.
    """
    if sheaf.graph.get('mode') == 'regions':
        from . import regionsheaf
        return regionsheaf.observe(sheaf, agent, tile, label)

    data = sheaf.nodes[agent]
    if label not in LABEL_BITS:
        raise ValueError(f"{label!r} is not an observable label; expected one of {LABELS}")
    if tile not in data['v']:
        raise ValueError(f"tile {tile} is not on the grid agent {agent}'s stalk is over")

    mask = LABEL_BITS[label]
    if (data['masks'].get(tile) == mask and data['beliefs'].get(tile) == label
            and data['own'].get(tile) == label):
        return False

    # masks is replaced rather than mutated, as in `sweep`, so that a snapshot taken of it
    # stays the state it was taken of
    data['masks'] = {**data['masks'], tile: mask}
    data['beliefs'][tile] = label
    data['own'][tile] = label
    data['c'] = encode_masks(data['masks'], data['v'])
    return True


def observe_tiles(label_beliefs: Sequence[Dict[Tile, str]],
                  tiles_by_agent: Sequence[Tile],
                  ground_truth: Mapping[Tile, str],
                  sheaf: Optional[ContractSheaf] = None) -> List[Dict[str, Any]]:
    """
    Every agent looks at the tile it is standing on and writes down what is really there.

    This is the whole of what navigation teaches an agent: it learns the ground it has
    driven over, one tile per step, and nothing about ground it has only routed past. An
    agent that crosses a tile it held no belief about therefore stops guessing at it, and
    -- once the flow runs again -- so do the agents it can talk to.

    Works with or without a sheaf, so the communicating run and the baseline sense the same
    world: given one, each observation goes through `observe` and into the stalk; without
    one, it is written straight into the belief dict, which is all there is to write.
    Tiles are taken as (X,Y) pairs rather than the gridworld's 2xN array so that this
    module stays free of numpy and of `agsheaf.gridworld`.

    Returns one record per agent -- ``{"agent", "tile", "label", "changed"}`` -- for the
    log, and for the caller to see whether there is anything new to communicate.

    Args:
        label_beliefs (Sequence[Dict[Tile, str]]): Per-agent belief dicts, which with a
            sheaf are the very dicts its stalks were built over.
        tiles_by_agent (Sequence[Tile]): The (X,Y) tile each agent is now on.
        ground_truth (Mapping[Tile, str]): The world's true labeling, what is seen.
        sheaf (Optional[ContractSheaf]): The belief sheaf, when the run has one.
    """
    records = []
    for agent, tile in enumerate(tiles_by_agent):
        tile = (int(tile[0]), int(tile[1]))
        label = ground_truth[tile]
        if label not in LABEL_BITS:
            raise ValueError(f"tile {tile} is truly {label!r}, which is not observable; "
                             f"expected one of {LABELS}")

        if sheaf is not None:
            changed = observe(sheaf, agent, tile, label)
        else:
            changed = label_beliefs[agent].get(tile) != label
            label_beliefs[agent][tile] = label

        records.append({"agent": agent, "tile": list(tile), "label": label,
                        "changed": bool(changed)})
    return records


class SheafState(NamedTuple):
    """
    A snapshot of the whole sheaf between sweeps: the 0-cochain as possibility
    sets, plus the sheaf-theoretic read-outs of where the flow has got to.

    `possibilities` is per agent the tiles it constrains, each as the set of
    labels still admitted (`mask_labels`); a tile the agent leaves free is
    absent, so this is the same partial shape the masks are kept in.
    `contested` names, per agent, the tiles whose set has emptied.

    `sections` says which *interfaces* have closed -- both endpoints pushing to
    the same contract on their shared edge stalk -- and `is_section` whether
    all of them have, i.e. whether the collection of local contracts is a
    global section. Below the fixed point the two differ: the flow closes
    interfaces one at a time, and naming which is exactly what the Laplacian
    formulation buys over a monolithic consistency check.

    `refines_initial` is the monotonicity invariant per agent (knowledge only
    ever strengthens), and `collapsed` is `ContractSheaf.collapsed` -- expected
    to stay empty here, since disagreement is recorded in-lattice as the empty
    possibility set rather than by collapsing a stalk.
    """
    possibilities: Dict[Any, Dict[Tile, Tuple[str, ...]]]
    contested: Dict[Any, List[Tile]]
    sections: Dict[Tuple[Any, Any], bool]
    is_section: bool
    refines_initial: Dict[Any, bool]
    collapsed: Dict[Any, str]


def render_contracts(sheaf: ContractSheaf) -> Dict[Any, List[str]]:
    """
    Each local section as the contract it actually is, rendered the way printing
    one renders it: the assume/guarantee pair, both simplified, split into lines.

    A stalk is recorded elsewhere as a grid of possibility sets, which is what
    the agent *means* but not what the sheaf *holds*. This is the object itself,
    so a run can be read as the assume/guarantee statement it makes at each
    agent -- and, since the log carries every iterate, at each application of
    the Laplacian.

    Two things about the rendering are worth knowing before reading one. The
    assumption is `True` at every belief stalk by construction (`encode_masks`):
    a broadcast is an assertion, not a promise, so the content is all in the
    guarantee. And the guarantee is printed saturated (a => g) and simplified,
    which rewrites each mask bound `S & ~mask == 0` into one `Extract` equality
    per excluded bit -- the same proposition as the bound, spelled out bit by
    bit rather than as a mask.

    Lines rather than one string, as everything else multi-line in a log is, so
    the JSON stays readable; "\\n".join(...) puts it back.

    A region sheaf renders as its per-region *mission* contracts instead --
    the two-slot A/G pairs the monitor checks -- see
    `regionsheaf.render_contracts`.

    Args:
        sheaf (ContractSheaf): The belief sheaf to render the stalks of.
    """
    if sheaf.graph.get('mode') == 'regions':
        from . import regionsheaf
        return regionsheaf.render_contracts(sheaf)
    return {node: repr(sheaf.contract(node)).splitlines() for node in sheaf.nodes()}


def sheaf_state(sheaf: ContractSheaf, legs: str = "kan") -> SheafState:
    """
    Snapshots `sheaf` as a `SheafState`. Reads the canonical mask state `sweep`
    maintains, so this is only meaningful for a sheaf `build_belief_sheaf`
    made; interfaces are keyed by their sorted pair, so a key is the same
    whichever orientation was added first.

    Cheap enough to call after every sweep: the mask-canonical re-encoding
    keeps each stalk a short conjunction of bitvector bounds, so the interface
    equalities are small quantified-BV queries rather than the growing
    formulas an unflattened flow would accumulate.

    A region sheaf snapshots as the richer `regionsheaf.RegionSheafState`
    (per-region sections, claims, concrete disagreement) rather than this
    module's `SheafState`.
    """
    if sheaf.graph.get('mode') == 'regions':
        from . import regionsheaf
        return regionsheaf.sheaf_state(sheaf, legs=legs)

    # is_section is read off the interfaces rather than asked of the sheaf again: it is the
    # same conjunction, and computing it twice would double the solver work for nothing
    sections = {tuple(sorted((u, v))): sheaf.agrees_on(u, v, legs=legs)
                for u, v in sheaf.interfaces()}
    return SheafState(
        possibilities={i: {tile: mask_labels(mask)
                           for tile, mask in sheaf.nodes[i]['masks'].items()}
                       for i in sheaf.nodes()},
        contested={i: sorted(tile for tile, mask in sheaf.nodes[i]['masks'].items()
                             if mask == EMPTY_MASK)
                   for i in sheaf.nodes()},
        sections=sections,
        is_section=all(sections.values()),
        refines_initial={i: bool(sheaf.contract(i).refines(sheaf.initial(i)))
                         for i in sheaf.nodes()},
        collapsed=sheaf.collapsed(),
    )


def reliance_contract(vars: Dict[Tile, z3.BitVecRef], tiles: Iterable[Tile],
                      forbidden: int = UNSAFE_BIT) -> Contract:
    """
    An agent's route reliance, reified as a contract: every relied tile keeps
    a possibility outside `forbidden`. This is the assumption of the agent's
    mission contract ("assuming my corridor is passable, I reach my goal")
    written in the guarantee slot, so that meeting it with the fused stalk and
    reading off `forced_empty` asks exactly whether the network's knowledge
    still admits the mission's assumption.
    """
    bounds = [vars[tile] & forbidden == 0 for tile in tiles]
    return Contract(z3.BoolVal(True),
                    z3.And(bounds) if bounds else z3.BoolVal(True),
                    vars=list(vars.values()))


def forced_empty(contract: Contract, vars: Dict[Tile, z3.BitVecRef],
                 tiles: Iterable[Tile]) -> List[Tile]:
    """
    The tiles whose possibility set the contract forces to ∅ -- the read-out
    that makes the empty mask observable. Applied to
    ``fused.meet(reliance_contract(...))`` it names exactly the relied tiles
    the fused knowledge rules out; a solver `unknown` counts as not forced.
    """
    solver = z3.Solver()
    solver.add(contract.sat_g)
    out = []
    for tile in tiles:
        solver.push()
        solver.add(vars[tile] != 0)
        if solver.check() == z3.unsat:
            out.append(tile)
        solver.pop()
    return out
