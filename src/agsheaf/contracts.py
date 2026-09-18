import z3
from typing import List, Optional, Sequence


class Undecided(Exception):
    """Z3 returned `unknown`; the query has no truth value we can rely on."""


def _is_valid(expr: z3.ExprRef, on_unknown: str = "raise") -> bool:
    """
    Checks if a Z3 expression is universally valid.

    Z3 answers `unknown` on quantified formulas it cannot decide, which the Kan
    maps produce routinely. Reporting that as "not valid" silently turns a
    solver limitation into a false negative -- `refines` says "does not refine",
    `is_section` says "not a section", and `converge` burns through its sweep
    budget blaming divergence. Raise instead, so the caller can tell "false"
    from "don't know"; pass on_unknown="valid"/"invalid" to force an answer.
    """
    solver = z3.Solver()
    solver.add(z3.Not(expr))
    r = solver.check()
    if r == z3.unsat:
        return True
    if r == z3.sat:
        return False
    if on_unknown == "raise":
        raise Undecided(f"z3 returned unknown: {solver.reason_unknown()}")
    return on_unknown == "valid"


def _is_satisfiable(expr: z3.ExprRef, on_unknown: str = "raise") -> bool:
    """Helper to check if a Z3 expression is satisfiable."""
    solver = z3.Solver()
    solver.add(expr)
    r = solver.check()
    if r == z3.sat:
        return True
    if r == z3.unsat:
        return False
    if on_unknown == "raise":
        raise Undecided(f"z3 returned unknown: {solver.reason_unknown()}")
    return on_unknown == "valid"


def _bind(quantifier, vs: Sequence, keep: Sequence, body: z3.BoolRef) -> z3.BoolRef:
    """
    Quantifies `body` over `vs` minus `keep`, returning it unquantified when
    nothing is left to bind.

    The two subtractions matter. A variable appearing in *both* alphabets of a
    `Relation` is a coordinate the two sides share, and the relation implicitly
    equates the two copies -- there is only one z3 constant, so no other reading
    is available. Binding it would quantify away a variable the result is
    supposed to keep free, silently computing the Kan extension of a different
    relation. Subtracting `keep` first leaves the shared coordinates free, which
    is the Kan extension of `R` conjoined with the diagonal on them.

    The empty case is not hypothetical: z3 rejects `ForAll([], p)` outright, and
    every relation whose source alphabet is contained in its target reaches it.
    """
    free = [v for v in vs if not any(v.eq(k) for k in keep)]
    return body if not free else quantifier(free, body)


class Relation:
    """
    Represents a relation R(x, y) between two sets of variables X and Y.
    The relation is a Z3 boolean expression over the combined variable set.

    The alphabets may overlap. A variable in both is read as a coordinate the
    two sides hold in common, and the Kan maps leave it free rather than
    quantifying over it; see `_bind`.
    """
    def __init__(self, relation_expr: z3.BoolRef, source_vars: Sequence, target_vars: Sequence):
        self.rel = relation_expr
        self.source_vars = list(source_vars)
        self.target_vars = list(target_vars)

    def __repr__(self):
        rel = z3.simplify(self.rel)
        x = ", ".join(str(v) for v in self.source_vars)
        y = ", ".join(str(v) for v in self.target_vars)

        return (f"Relation(\n"
                f"  R: {rel}\n"
                f"  X: [{x}]\n"
                f"  Y: [{y}]\n"
                f")")


class Contract:
    """
    Assume-Guarantee contract evaluated using the Z3 SMT solver.
    A contract is a pair (a, g) of assumption and guarantee predicates over a
    shared variable alphabet; the operations implement the logical lattice.

    A contract denotes a saturation class, not a syntactic pair: (a, g) and
    (a, a => g) are the same contract. Every operation therefore goes through
    `sat_g` rather than the raw `g`. This is not cosmetic -- `ran` and
    `pullback` carry a universal quantifier on the guarantee leg, and `forall`
    does not distribute over disjunction, so feeding them an unsaturated
    representative yields a genuinely different result and breaks the
    adjunctions the Laplacian rests on.
    """
    def __init__(self, a: z3.BoolRef, g: z3.BoolRef,
                 vars: Optional[Sequence] = None, saturate=None):
        """
        Initializes a contract with assumption `a` and guarantee `g`, optionally
        over the alphabet `vars`.

        `saturate` is accepted for backwards compatibility and ignored:
        saturation is now applied on read via `sat_g`, so it is never possible
        to hold an unsaturated contract in the first place.
        """
        self.a = a
        self.g = g
        self.vars = None if vars is None else frozenset(str(v) for v in vars)
        self.saturate = saturate

    @property
    def sat_g(self) -> z3.BoolRef:
        """The saturated guarantee A => G. Every operation goes through this."""
        return z3.Implies(self.a, self.g)

    def _same_alphabet(self, other: 'Contract'):
        """
        Guards against combining contracts over different variable sets, which
        would otherwise produce a well-formed but meaningless result. `None`
        means "alphabet unknown" and is permissive.
        """
        if self.vars is not None and other.vars is not None and self.vars != other.vars:
            raise ValueError(
                f"alphabet mismatch: {sorted(self.vars)} vs {sorted(other.vars)}"
            )
        return self.vars if self.vars is not None else other.vars

    def __eq__(self, other) -> bool:
        if not isinstance(other, Contract):
            return NotImplemented
        return _is_valid(self.a == other.a) and _is_valid(self.sat_g == other.sat_g)

    # Equality is decided by the solver, so no hash can be consistent with it.
    __hash__ = None

    def __repr__(self):
        return (f"Contract(\n"
                f"  A: {z3.simplify(self.a)}\n"
                f"  G: {z3.simplify(self.sat_g)}\n"
                f")")

    def is_saturated(self) -> bool:
        """
        Checks if the contract was supplied in saturated form (a, a => g).
        Operations no longer depend on this -- it reports on user input only.
        """
        return _is_valid(self.g == self.sat_g)

    def is_simplified(self) -> bool:
        """
        Checks if the contract's assumptions and guarantees are simplified.
        """
        return (_is_valid(self.a == z3.simplify(self.a))
                and _is_valid(self.g == z3.simplify(self.g)))

    def is_consistent(self) -> bool:
        """
        Consistent iff the saturated guarantee is satisfiable: the contract
        promises something achievable. `bot()` is the inconsistent contract.
        """
        return _is_satisfiable(self.sat_g)

    def is_compatible(self) -> bool:
        """
        Compatible iff the assumption is satisfiable: some environment can meet
        it. `top()` is the incompatible contract.
        """
        return _is_satisfiable(self.a)

    def meet(self, other: 'Contract') -> 'Contract':
        """
        Computes the meet of contracts (intersection of viewpoints).
        """
        vs = self._same_alphabet(other)
        return Contract(z3.Or(self.a, other.a),
                        z3.And(self.sat_g, other.sat_g), vars=vs)

    def join(self, other: 'Contract') -> 'Contract':
        """
        Computes the join of contracts (dual of meet).
        """
        vs = self._same_alphabet(other)
        return Contract(z3.And(self.a, other.a),
                        z3.Or(self.sat_g, other.sat_g), vars=vs)

    def compose(self, other: 'Contract') -> 'Contract':
        """
        Composition (C || C'), the specification of the two components wired
        together. Incer et al. (2025), Equation (2):

            A = (a ^ a') v ~(satg ^ satg')        G = satg ^ satg'

        The guarantee is the same conjunction as `meet`, but the assumption is
        *discharged*: an environment need not satisfy both components'
        assumptions where the pair's joint guarantee already establishes them.
        That is what separates composition from the lattice meet, whose
        assumption is the plain disjunction.

        Note Naik et al. (2025) Section 3.1 states this assumption without the
        complement; Pacti's form is the one that satisfies the quotient
        adjunction, and is what this follows.
        """
        vs = self._same_alphabet(other)
        joint = z3.And(self.sat_g, other.sat_g)
        return Contract(z3.Or(z3.And(self.a, other.a), z3.Not(joint)),
                        joint, vars=vs)

    def quotient(self, other: 'Contract') -> 'Contract':
        """
        Quotient (C / C'), the largest contract whose composition with `other`
        still refines `self` -- the specification of the component missing from
        a design. Incer et al. (2025), Equation (3):

            A = a ^ satg'      G = (a' ^ satg) v ~a v ~satg'

        This is right adjoint to composition: `c2.compose(c3).refines(c1)` iff
        `c3.refines(c1.quotient(c2))`, which is its defining property rather
        than a consequence.
        """
        vs = self._same_alphabet(other)
        return Contract(z3.And(self.a, other.sat_g),
                        z3.Or(z3.And(other.a, self.sat_g),
                              z3.Not(self.a),
                              z3.Not(other.sat_g)), vars=vs)

    def refines(self, other: 'Contract') -> bool:
        """
        Checks if this contract refines 'other' (C_self <= C_other)
        Contravariant on assumptions, covariant on guarantees.
        """
        self._same_alphabet(other)
        return (_is_valid(z3.Implies(other.a, self.a))
                and _is_valid(z3.Implies(self.sat_g, other.sat_g)))

    # The four Kan-type operations form TWO adjunctions, not an adjoint triple:
    #
    #     lan  |-  pullback           (X -> Y  |-  Y -> X)
    #     dual_pullback  |-  ran      (Y -> X  |-  X -> Y)
    #
    # A triple Lan_R -| R^* -| Ran_R would require `pullback -| ran`, which
    # holds only when the relation is the graph of a total function. For a
    # general relation the two available adjunctions are exists_R -| forall_R'
    # and exists_R' -| forall_R (R' the converse); `pullback` and `ran` share
    # the (A: exists, G: forall) shape and so cannot be adjoint to one another.
    # The shapes are forced by the refinement order being contravariant on
    # assumptions and covariant on guarantees.

    def lan(self, relation: Relation) -> 'Contract':
        """
        Left Kan Extension (Lan_R). Pushes this contract from X to Y maximally/aggressively.
        Lan_R(A, G) = ( ∀ x. R(x,y) ⇒ A(x), ∃ x. R(x,y) ∧ G(x) )
        Left adjoint to `pullback`; the sheaf restriction map of a bisheaf.
        """
        src, tgt = relation.source_vars, relation.target_vars
        return Contract(
            _bind(z3.ForAll, src, tgt, z3.Implies(relation.rel, self.a)),
            _bind(z3.Exists, src, tgt, z3.And(relation.rel, self.sat_g)),
            vars=tgt)

    def ran(self, relation: Relation) -> 'Contract':
        """
        Right Kan Extension (Ran_R). Pushes this contract from X to Y safely/conservatively.
        Ran_R(A, G) = ( ∃ x. R(x,y) ∧ A(x), ∀ x. R(x,y) ⇒ G(x) )
        Right adjoint to `dual_pullback`, NOT to `pullback`.
        """
        src, tgt = relation.source_vars, relation.target_vars
        return Contract(
            _bind(z3.Exists, src, tgt, z3.And(relation.rel, self.a)),
            _bind(z3.ForAll, src, tgt, z3.Implies(relation.rel, self.sat_g)),
            vars=tgt)

    def pullback(self, relation: Relation) -> 'Contract':
        """
        Pullback (R^*). Pulls this contract from Y back to X.
        R^*(A, G) = ( ∃ y. R(x,y) ∧ A(y), ∀ y. R(x,y) ⇒ G(y) )
        Right adjoint to `lan`; the cosheaf coextension of a bisheaf, and so
        the second leg of primal parallel transport.
        """
        src, tgt = relation.source_vars, relation.target_vars
        return Contract(
            _bind(z3.Exists, tgt, src, z3.And(relation.rel, self.a)),
            _bind(z3.ForAll, tgt, src, z3.Implies(relation.rel, self.sat_g)),
            vars=src)

    def dual_pullback(self, relation: Relation) -> 'Contract':
        """
        Dual pullback: pulls this contract from Y back to X as the LEFT adjoint
        of `ran`, mirroring how `pullback` is the right adjoint of `lan`.
        dual_pullback(A, G) = ( ∀ y. R(x,y) ⇒ A(y), ∃ y. R(x,y) ∧ G(y) )

        Together with `ran` this forms the second adjoint bisheaf, the one
        `ContractSheaf` transports along under `legs="co"`: the primal pushes
        with `lan` and pulls with `pullback`, this one pushes with `ran` and
        pulls with `dual_pullback`.

        Note that reversing the legs like this is *not* what Ghrist et al.
        (2026) Def 7.3 does. Its prose speaks of reversing the two adjoint legs,
        but Eq. (13) keeps the primal's legs and dualises only the aggregation,
        meet -> par-weighted join. Both readings are implemented; see
        `ContractSheaf.transport` and `doc/correctness.md` for why the
        leg-reversed one is the one with content over a Boolean base.
        """
        src, tgt = relation.source_vars, relation.target_vars
        return Contract(
            _bind(z3.ForAll, tgt, src, z3.Implies(relation.rel, self.a)),
            _bind(z3.Exists, tgt, src, z3.And(relation.rel, self.sat_g)),
            vars=src)


def top(vars: Optional[Sequence] = None) -> Contract:
    """
    The top contract (⊤): assumes nothing (a = False, satisfied by no
    environment) and guarantees everything (g = True, holds under every
    behavior). It is the identity for `meet` and the greatest element
    under `refines` — every contract refines ⊤. It is the incompatible
    contract: a vertex that reaches it has localised an assumption failure.
    """
    return Contract(z3.BoolVal(False), z3.BoolVal(True), vars=vars)


def bot(vars: Optional[Sequence] = None) -> Contract:
    """
    The bottom contract (⊥): assumes everything (a = True, every
    environment is assumed) and guarantees nothing (g = False,
    unsatisfiable). It is the identity for `join` and the least element
    under `refines` — ⊥ refines every contract. It is the inconsistent
    contract: a vertex that reaches it has localised a guarantee conflict.
    """
    return Contract(z3.BoolVal(True), z3.BoolVal(False), vars=vars)
