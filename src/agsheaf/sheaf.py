import networkx as nx
from .contracts import Contract, Relation, top, bot
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

#: The two adjoint bisheaves a `ContractSheaf` can transport along. `KAN` is the
#: restriction/coextension pair `lan -| pullback` of Ghrist et al. (2026)
#: Def 6.2; `CO` is the opposite pairing `dual_pullback -| ran`.
KAN = "kan"
CO = "co"


class LivenessError(RuntimeError):
    """
    A firing sequence starved some node. Riess and Ghrist (2022), Assumption 2
    requires every node to fire infinitely often; it is the hypothesis of their
    Theorem 1, so a schedule violating it carries no convergence guarantee.
    """


def round_robin(nodes: Sequence) -> Iterator[frozenset]:
    """
    Fires one node per step, cycling forever. The simplest live firing sequence
    (Riess and Ghrist 2022, Assumption 2), and the maximally asynchronous one.
    """
    order = list(nodes)
    if not order:
        return
    t = 0
    while True:
        yield frozenset({order[t % len(order)]})
        t += 1


def random_firing(nodes: Sequence, rng, p: float = 0.5) -> Iterator[frozenset]:
    """
    Fires each node independently with probability `p` per step, which is live
    almost surely. `rng` is any object with a `random()` method (e.g.
    `random.Random(seed)`); seed it to keep tests reproducible.

    Never yields the empty set -- a step in which nothing fires is a step in
    which nothing can happen, and would only burn the sweep budget.
    """
    order = list(nodes)
    if not order:
        return
    while True:
        active = frozenset(v for v in order if rng.random() < p)
        yield active if active else frozenset({order[int(rng.random() * len(order))]})


class ContractSheaf(nx.DiGraph):
    """
    A cellular sheaf of contract lattices over a graph.

    This is a `networkx.DiGraph` with two conventions:

      * each node carries its local contract under the attribute ``c``;
      * each *directed* edge ``(u, v)`` carries, under the attribute ``rel``,
        the `Relation` restricting node ``u`` onto the shared edge space of the
        interface ``{u, v}``.

    A `DiGraph` (rather than a `Graph`) is required precisely because of the
    second convention: the two endpoints of an interface restrict onto the
    shared space by *different* relations, so each interface needs both
    ``(u, v)`` and ``(v, u)`` present, holding ``u``'s and ``v``'s restriction
    respectively. `add_interface` adds both at once; forgetting the reverse
    edge silently empties `neighbors`, which would make the Laplacian return
    the lattice identity everywhere.

    Because the sheaf *is* the graph, all of networkx is available directly::

        F = ContractSheaf()
        F.add_node(1, c=Contract(z3.And(p1, p2), p3))
        F.add_node(2, c=Contract(p4, z3.Or(p1, p2)))
        F.add_interface(1, 2, rel_1, rel_2)

        F.converge(verify=True)
        assert F.is_section()

    Every method takes a ``dual`` flag selecting the primal (left Kan / meet)
    or dual (right Kan / join) reading of the same construction.
    """

    def add_interface(self, u: Any, v: Any, rel_u: Relation, rel_v: Relation) -> None:
        """
        Adds the interface {u, v} as the pair of directed edges carrying each
        endpoint's restriction onto the shared edge space.
        """
        if set(map(str, rel_u.target_vars)) != set(map(str, rel_v.target_vars)):
            raise ValueError(
                f"edge variable mismatch at interface {{{u!r}, {v!r}}}: "
                f"{u} targets {rel_u.target_vars}, {v} targets {rel_v.target_vars}"
            )
        self.add_edge(u, v, rel=rel_u)
        self.add_edge(v, u, rel=rel_v)

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------

    def contract(self, i: Any) -> Contract:
        """
        The contract carried by node `i` (its ``c`` attribute).
        """
        if i not in self:
            raise KeyError(f"no node {i!r} in this sheaf")
        try:
            return self.nodes[i]['c']
        except KeyError:
            raise KeyError(
                f"node {i!r} carries no contract; add it with "
                f"add_node({i!r}, c=Contract(...))"
            ) from None

    def restriction(self, u: Any, v: Any) -> Relation:
        """
        The relation restricting node `u` onto the shared edge space of the
        interface {u, v} -- i.e. the ``rel`` attribute of the directed edge
        (u, v).
        """
        if not self.has_edge(u, v):
            hint = (
                f"; the interface {{{u!r}, {v!r}}} has only its reverse edge, so "
                f"both directions are needed -- use add_interface"
                if self.has_edge(v, u) else ""
            )
            raise KeyError(f"no edge ({u!r}, {v!r}) in this sheaf{hint}")
        try:
            return self.edges[u, v]['rel']
        except KeyError:
            raise KeyError(
                f"edge ({u!r}, {v!r}) carries no relation; add it with "
                f"add_edge({u!r}, {v!r}, rel=Relation(...))"
            ) from None

    def initial(self, i: Any) -> Contract:
        """
        The contract node `i` carried before the first sweep -- its ``c_init``
        attribute, snapshotted lazily on the first sweep that touches the node.
        Falls back to the current contract for a node no sweep has reached yet,
        so this is always the assignment as given.
        """
        if i not in self:
            raise KeyError(f"no node {i!r} in this sheaf")
        data = self.nodes[i]
        return data['c_init'] if 'c_init' in data else self.contract(i)

    def reset(self) -> None:
        """
        Restores every node's contract to its `initial` value, discarding the
        result of any sweeps so far.
        """
        for i in self.nodes():
            self.nodes[i]['c'] = self.initial(i)

    def assignment(self) -> Dict[Any, Contract]:
        """A frozen copy of the current 0-cochain."""
        return {i: self.contract(i) for i in self.nodes()}

    def collapsed(self) -> Dict[Any, str]:
        """
        Vertices whose contract has degenerated to a lattice extreme, and how.
        Localising *which* vertex failed is the practical argument for a
        Laplacian formulation over a monolithic consistency check.
        """
        out = {}
        for i in self.nodes():
            c = self.contract(i)
            if not c.is_consistent():
                out[i] = 'inconsistent'
            elif not c.is_compatible():
                out[i] = 'incompatible'
        return out

    # ------------------------------------------------------------------
    # the Laplacian
    # ------------------------------------------------------------------

    def transport(self, x: Mapping, source: Any, target: Any,
                   legs: str = KAN) -> Contract:
        """
        Parallel transport of `source`'s contract, read from the assignment
        `x`, into `target`'s local space: push it up to the shared edge via
        `source`'s restriction, then pull it back down via `target`'s
        (Ghrist et al. 2026, Def 6.2).

        `legs` selects which bisheaf does the carrying:

            "kan"   pullback . lan          restriction |- coextension
            "co"    dual_pullback . ran     coextension |- restriction

        Both are adjoint bisheaves in the sense of Def 6.6 -- `lan -| pullback`
        and `dual_pullback -| ran` both hold for an arbitrary relation -- but
        they are *different* bisheaves, and the adjunction points the opposite
        way in each. Mixing legs across the two (pushing with `ran`, pulling
        with `pullback`) is what breaks the Hodge-Lawvere theorems, since
        `pullback -| ran` holds only when the relation is the graph of a total
        function.

        Note this is independent of `dual`, which selects only how the
        transported data is aggregated; see `laplacian`.
        """
        c = x[source]
        rel_source = self.restriction(source, target)
        rel_target = self.restriction(target, source)

        if legs == CO:
            return c.ran(rel_source).dual_pullback(rel_target)
        if legs != KAN:
            raise ValueError(f"legs must be {KAN!r} or {CO!r}, got {legs!r}")
        return c.lan(rel_source).pullback(rel_target)

    def laplacian(self, x: Optional[Mapping] = None, dual: bool = False,
                  legs: str = KAN, include_self: bool = False,
                  firing: Optional[Iterable] = None) -> Dict[Any, Contract]:
        """
        (Lx)_i for every node, computed from a FROZEN assignment `x`:

            (Lx)_i = Meet_{j in N(i) ^ tau} transport(j -> i)(x_j)   [primal]
            (Lx)_i = Join_{j in N(i) ^ tau} transport(j -> i)(x_j)   [dual]

        Every node reads the same `x`, so this is the operator L itself rather
        than the effect of a sweep.

        `dual` selects the aggregation (weighted meet vs par-weighted join,
        Def 6.3 vs Def 7.3) and `legs` selects the bisheaf; the two are
        orthogonal. With `legs="kan"`, `dual=True` is Eq. (13) read literally --
        the same legs as the primal, aggregated the other way round.

        `firing` restricts the aggregation to broadcasting neighbours, i.e. the
        set tau_t of Riess and Ghrist (2022), Def 5; `None` means all of V and
        recovers the synchronous Laplacian of their Eq. (4). A node with no
        firing neighbour aggregates over nothing and so collapses to the
        identity of the relevant lattice operation (`top` primal, `bot` dual) --
        which under `include_self` leaves it untouched, as Def 5 requires.
        """
        if x is None:
            x = self.assignment()
        active = None if firing is None else set(firing)
        out = {}
        for i in self.nodes():
            acc = x[i] if include_self else (bot() if dual else top())
            for j in self.neighbors(i):
                if active is not None and j not in active:
                    continue
                inc = self.transport(x, j, i, legs=legs)
                acc = acc.join(inc) if dual else acc.meet(inc)
            out[i] = acc
        return out

    def is_suffix(self, x: Optional[Mapping] = None, legs: str = KAN) -> bool:
        """
        x <= Lx, for the meet-aggregated Laplacian (Ghrist et al. 2026, Def 3.1).

        By the Hodge-Lawvere Theorem (Thm 6.10) these are the weighted global
        sections. This sheaf carries no weight function -- it is the unweighted
        Boolean case W = top -- and every interface contributes both (i, j) and
        (j, i), so the suffix inequality holds in both directions at once and
        collapses to equality. Over this base, therefore,

            is_suffix(legs=L)  <=>  is_section(legs=L)

        which `test_theorems.py` checks on every fixture sheaf.
        """
        if x is None:
            x = self.assignment()
        L = self.laplacian(x, legs=legs)
        return all(x[i].refines(L[i]) for i in self.nodes())

    def is_prefix(self, x: Optional[Mapping] = None, legs: str = KAN) -> bool:
        """
        L_dual x <= x, for the join-aggregated Laplacian (Thm 7.5).

        What this means depends on `legs`, and the two readings differ sharply:

        `legs="co"` transports along the `dual_pullback -| ran` bisheaf, whose
        adjunction runs the right way to transpose this condition. It comes out
        as exact `ran`-agreement across every interface -- a genuine dual
        notion of consistency, and the one `is_section(legs="co")` tests.

        `legs="kan"` is Def 7.3 Eq. (13) read literally: the same legs as the
        primal, aggregated by join. Here the adjunction points the wrong way to
        transpose `pullback(z) <= x_i`, so this is *not* an edge-agreement
        condition. That is expected rather than a defect: Thm 7.5 needs the
        linear negation of a Girard quantale, and over the Boolean base with
        W = top the dualising element is bottom, so Def 7.4's cosection
        condition is vacuous. Prefer `legs="co"` for a dual with content.
        """
        if x is None:
            x = self.assignment()
        L = self.laplacian(x, dual=True, legs=legs)
        return all(L[i].refines(x[i]) for i in self.nodes())

    def is_harmonic(self, x: Optional[Mapping] = None, legs: str = KAN) -> bool:
        """
        Both halves of the Two-sided Hodge Theorem (Thm 7.6): a suffix point of
        the meet-aggregated Laplacian and a prefix point of the join-aggregated
        one, *along the same bisheaf*.

        Conjoining the two across different bisheaves is meaningless, which is
        why `legs` is shared rather than free on each half. Over this Boolean
        base the primal half already pins down the strict global sections (see
        `is_suffix`), so this is a strictly stronger condition than being a
        section, not a characterisation of one -- Thm 7.6's identification needs
        a Girard base with a non-degenerate weighting.
        """
        if x is None:
            x = self.assignment()
        return self.is_suffix(x, legs=legs) and self.is_prefix(x, legs=legs)

    def interfaces(self) -> List[Tuple[Any, Any]]:
        """
        Each interface once, as the (u, v) pair whose direction was added
        first. `add_interface` puts both directed edges in, so iterating
        `edges()` would visit every interface twice; this is what any
        per-interface diagnostic wants to iterate instead.
        """
        seen = set()
        out = []
        for i, j in self.edges():
            if (j, i) in seen:
                continue
            seen.add((i, j))
            out.append((i, j))
        return out

    def pushforward(self, u: Any, v: Any, legs: str = KAN,
                    x: Optional[Mapping] = None) -> Contract:
        """
        Node `u`'s contract pushed onto the shared edge space of the interface
        {u, v}, by the restriction of the `legs` bisheaf (`lan` for "kan", `ran`
        for "co"). The first leg of `transport`, on its own -- and the thing the
        sheaf condition compares.
        """
        if legs not in (KAN, CO):
            raise ValueError(f"legs must be {KAN!r} or {CO!r}, got {legs!r}")
        c = self.contract(u) if x is None else x[u]
        r = self.restriction(u, v)
        return c.ran(r) if legs == CO else c.lan(r)

    def agrees_on(self, u: Any, v: Any, legs: str = KAN,
                  x: Optional[Mapping] = None) -> bool:
        """
        Whether the single interface {u, v} agrees: both endpoints push to the
        same contract on the shared edge space.

        This is `is_section` localised to one interface -- the sheaf condition
        is a conjunction over interfaces, so an assignment that is not yet a
        global section still *has* the interfaces it has already closed, and
        naming them is the point of a Laplacian formulation. A flow in progress
        typically closes them one at a time.

        `x` asks about a frozen assignment rather than the sheaf's current one,
        as every other predicate here does. The flow overwrites each stalk in
        place, so an observer recording the trajectory holds iterates the sheaf
        has already left behind, and this is how it asks about them.
        """
        return self.pushforward(u, v, legs=legs, x=x) \
            == self.pushforward(v, u, legs=legs, x=x)

    def section_edges(self, legs: str = KAN,
                      x: Optional[Mapping] = None) -> Dict[Tuple[Any, Any], bool]:
        """
        `agrees_on` for every interface, keyed by the pair `interfaces` gives.
        Unlike `is_section` this never short-circuits: it is the diagnostic
        view, saying *which* interfaces are closed rather than whether all are.
        """
        return {(u, v): self.agrees_on(u, v, legs=legs, x=x)
                for u, v in self.interfaces()}

    def is_section(self, legs: str = KAN, x: Optional[Mapping] = None) -> bool:
        """
        Checks whether the assignment is a global section: across every
        interface, both endpoints agree once pushed onto the shared edge space
        by the restriction of the `legs` bisheaf (`lan` for "kan", `ran` for
        "co"). Use `section_edges` for which interfaces those are.
        """
        if legs not in (KAN, CO):
            raise ValueError(f"legs must be {KAN!r} or {CO!r}, got {legs!r}")
        return all(self.agrees_on(u, v, legs=legs, x=x) for u, v in self.interfaces())

    # ------------------------------------------------------------------
    # the flow
    # ------------------------------------------------------------------

    def laplacian_update(self, dual: bool = False, legs: str = KAN,
                         include_self: bool = True,
                         firing: Optional[Iterable] = None,
                         in_place: bool = True) -> bool:
        """
        Applies one step of the harmonic flow, returning True iff nothing
        changed.

        With `include_self` (the default) this is the unweighted harmonic flow
        of Ghrist et al. (2026), Definition 6.13, equivalently the heat flow
        x <- (id ^ L_t) x of Riess and Ghrist (2022), Eq. (6):

            x <- Lx ^ x      [primal]        x <- L_dual x v x      [dual]

        That form is monotone -- decreasing primal, increasing dual -- so it
        cannot cycle, and it is what Proposition 6.15 requires: the flow
        preserves each node's relationship to every global section, converging
        to the greatest section below the initial assignment rather than to an
        arbitrary one. Setting `include_self=False` gives the bare x <- Lx,
        which has the same fixed points but is not monotone and can cycle.

        `firing` is the set tau_t of broadcasting nodes (Riess and Ghrist 2022,
        Def 5); `None` fires everything, recovering their synchronous Eq. (4).
        A node none of whose neighbours fire aggregates over nothing and is
        left untouched, as Def 5 requires -- this falls out of `include_self`
        and the empty meet being `top`, so it needs no special case.

        `in_place` selects the update model. False reads every neighbour from a
        snapshot taken before the step, which is Eq. (6) literally. True (the
        default) writes each node as it goes, so a node computed later already
        sees its predecessors' new values -- a pull-style asynchronous update
        that converges in fewer steps. Both are monotone and have identical
        fixed points, and Theorem 1 of Riess and Ghrist covers both, since its
        conclusion holds for any firing sequence satisfying liveness.

        Either way this returns only "did anything change": a criterion like
        `is_suffix` is a statement about the operator L at a given cochain and
        must be evaluated against a frozen assignment, never mid-step.
        """
        if legs not in (KAN, CO):
            raise ValueError(f"legs must be {KAN!r} or {CO!r}, got {legs!r}")
        unchanged = True
        active = None if firing is None else set(firing)
        source = self.nodes(data='c') if in_place else self.assignment()

        for i in self.nodes():
            current = self.contract(i)
            acc = current if include_self else (bot() if dual else top())
            for j in self.neighbors(i):
                if active is not None and j not in active:
                    continue
                inc = self.transport(source, j, i, legs=legs)
                acc = acc.join(inc) if dual else acc.meet(inc)

            unchanged &= (current == acc)

            # Preserve the assignment as given before the first step
            # overwrites it; see `initial` and `reset`.
            self.nodes[i].setdefault('c_init', current)
            self.nodes[i]['c'] = acc

        return bool(unchanged)

    def converge(self, dual: bool = False, legs: str = KAN,
                 include_self: bool = True,
                 firing_sequence: Optional[Iterable] = None,
                 in_place: bool = True,
                 max_sweeps: int = 100, verify: bool = False,
                 on_step: Optional[Callable[[int, bool], None]] = None) -> int:
        """
        Runs the harmonic flow to a fixed point, returning the number of steps
        taken before it settled (0 if it was already stable).

        `firing_sequence` is the schedule tau of Riess and Ghrist (2022): an
        iterable (or a callable returning one) of node sets, one per step.
        `None` fires every node every step. `round_robin` and `random_firing`
        in this module supply live schedules.

        Termination needs care under a partial schedule: a step in which
        nothing changed proves only that the *firing* nodes had nothing to say.
        A fixed point is declared once every node has fired across an unbroken
        run of steps in which nothing changed, which is exactly what liveness
        (their Assumption 2) guarantees will happen. Under the default all-fire
        schedule this reduces to the obvious criterion.

        The run must be unbroken. Crediting the nodes that fired in a step that
        *did* change something is unsound, because a node can broadcast and then
        change within the same step; see the comment at the accumulator below.

        By Theorem 1 the result does not depend on the schedule: for any live
        tau the flow descends to the same fixed point, the greatest section
        below the initial assignment (Ghrist et al. 2026, Prop 6.15). The
        argument is schedule-independent -- if x[t] >= y for a section y then
        Ly >= y and monotonicity give x[t+1] >= y whatever subset was updated.

        Raises LivenessError if `max_sweeps` is exhausted with some node never
        having fired, and RuntimeError otherwise -- under the default
        `include_self` the flow is monotone and cannot cycle, so a live
        schedule exhausting its budget means a long or infinite descending
        chain; with `include_self=False` the iteration can genuinely cycle.

        With `verify`, asserts the theorem the flow is supposed to establish --
        a suffix point of the primal, or a prefix point of the dual -- against
        a frozen assignment once the flow settles.

        `on_step(step, unchanged)` is called after each step and before the
        termination test, which is the only moment at which an intermediate
        iterate exists: the flow overwrites each stalk in place, so a caller
        that wants to record the trajectory -- or to rewrite the assignment into
        a smaller equivalent form, which is what keeps an exact-contract flow
        from drowning in nested quantifiers -- has to do it here. Anything
        semantics-preserving is safe to do from the callback; the termination
        test that follows is decided by contract equality, which cannot see the
        difference. Anything else changes the flow, and the flow is what is
        being measured.

        `initial` is untouched throughout -- it is snapshotted on the first
        step only, so it still holds the assignment as given no matter how many
        steps run.
        """
        if callable(firing_sequence):
            firing_sequence = firing_sequence()
        schedule = iter(firing_sequence) if firing_sequence is not None else None

        everyone = set(self.nodes())
        fired_since_change = set()
        never_fired = set(everyone)

        for step in range(max_sweeps):
            firing = None if schedule is None else set(next(schedule))
            active = everyone if firing is None else firing
            never_fired -= active

            still = self.laplacian_update(dual=dual, legs=legs,
                                          include_self=include_self,
                                          firing=firing, in_place=in_place)
            if on_step is not None:
                on_step(step, still)
            # A change invalidates every firing so far, including the ones in THIS step: a node
            # that broadcast and then changed in the same step -- which happens whenever two
            # adjacent nodes fire together -- sent a value it no longer holds, so its neighbours
            # have not seen its current contract. Crediting `active` here (rather than clearing)
            # lets `converge` declare a fixed point that is not one, and the flow stops short of
            # a section. Round-robin never exposes it, since a lone firing node has no firing
            # neighbour and so cannot change in the step it fires; random firing does.
            fired_since_change = (fired_since_change | active) if still else set()

            if still and fired_since_change >= everyone:
                if verify:
                    ok = (self.is_prefix(legs=legs) if dual
                          else self.is_suffix(legs=legs))
                    if not ok:
                        raise RuntimeError(
                            "flow terminated at a point that is not a "
                            f"{'prefix' if dual else 'suffix'} point"
                        )
                return step

        if never_fired:
            raise LivenessError(
                f"{sorted(map(repr, never_fired))} never fired in {max_sweeps} steps, "
                f"so the flow says nothing about them. Either the firing sequence "
                f"starves them -- Riess and Ghrist (2022) Assumption 2 requires every "
                f"node to fire infinitely often, and it is the hypothesis of their "
                f"Theorem 1 -- or max_sweeps is too small to cover a full round. From "
                f"inside a truncated run the two are indistinguishable."
            )
        raise RuntimeError(
            f"no fixed point after {max_sweeps} steps (dual={dual}, legs={legs!r}, "
            f"include_self={include_self})"
        )
