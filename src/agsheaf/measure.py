"""
How far apart two contracts are.

The test suite decides whether the flow reached a section; it cannot say how far
from one it started, nor how much of the gap each sweep closed. That needs a
number, and a contract lattice does not come with one -- so this module puts a
measure on the alphabet and reads the lattice through it.

*The denotation.* Over a finite alphabet Sigma (every assignment to a stalk's
variables), a contract `C = (A, G)` denotes two subsets of Sigma:

    E(C) = [[A]]        the environments it admits
    M(C) = [[A => G]]   the behaviors it permits

Both are read through `Contract.sat_g`, so this is the denotation of the
saturation class rather than of the syntax -- `(a, g)` and `(a, a => g)` land on
the same pair, as they must. Refinement is inclusion, contravariant on the
first slot and covariant on the second:

    C <= C'   iff   E(C') subset E(C)   and   M(C) subset M(C')

which is `Contract.refines` read semantically.

*The two distances.* With mu the uniform measure on Sigma (`measure`, weightable
per state):

    distance(C, C')   = 1/2 [ mu(E delta E') + mu(M delta M') ]
    divergence(C, C') = 1/2 [ mu(E' \\ E)    + mu(M \\ M')     ]

`distance` is a genuine metric on semantic classes -- symmetric, and zero
exactly on contracts z3 calls equal. `divergence` is its one-sided half: it
vanishes exactly when `C` refines `C'`, so it is a Lawvere metric (a category
enriched in [0, infinity]) rather than a metric, and
`distance = divergence(C, C') + divergence(C', C)`. The directed reading is the
useful one along a flow. The primal harmonic flow descends, so its iterates
nest, so `distance(x_t, x_limit)` is not merely shrinking but *monotone* -- and
a violation is a bug rather than slow convergence.

`interval_volume` computes the same number a third way, as the measure of the
lattice interval `[C ^ C', C v C']` using the shipped `meet` and `join`. It
must agree with `distance`; `test_measure.py` checks that it does, which is what
keeps this module honest about measuring the lattice the rest of the library
implements rather than one of its own.

*Cost.* Everything here enumerates Sigma, so it is exact and it is exponential
in the alphabet. `Alphabet` refuses to build past `max_states` rather than
quietly taking a long time. This is a measuring instrument for experiment-scale
alphabets; `simplify.qf` is what handles the ones too big to tabulate.
"""

import itertools
from collections import namedtuple
from typing import Optional, Sequence, Tuple

import numpy as np
import z3

from .contracts import Contract, Undecided
from .simplify import has_quantifier, qf


class AlphabetTooLarge(ValueError):
    """
    Sigma exceeds `max_states`. Raised at construction rather than during a
    sweep, so a too-ambitious experiment fails while it is being set up instead
    of after an hour of enumeration.
    """


class _Unsupported(Exception):
    """The vectorised evaluator met a node it does not know; fall back to z3."""


#: How many states a factor may have before `formula` stops writing it flat and
#: starts splitting on a variable instead. Sixteen is two Booleans past the
#: point where a flat form is obviously readable, and small enough that the worst
#: flat case is a sixteen-term disjunction.
_FLAT_LIMIT = 16


def _domain(v, override=None) -> Tuple:
    """
    The finite set of z3 values `v` ranges over.

    Booleans and bit-vectors are finite and enumerable. Integers are not, so
    they need an explicit `override` naming the window to measure over -- see
    `Alphabet`'s `domains` for what that does and does not change.
    """
    if override is not None:
        return tuple(_as_value(v, raw) for raw in override)
    if z3.is_bool(v):
        return (z3.BoolVal(False), z3.BoolVal(True))
    if isinstance(v.sort(), z3.BitVecSortRef):
        width = v.sort().size()
        return tuple(z3.BitVecVal(k, width) for k in range(1 << width))
    raise TypeError(
        f"{v} has sort {v.sort()}, which is not finite; a measure needs a "
        f"finite alphabet. Booleans and bit-vectors are enumerated by default; "
        f"give an explicit window in `domains` for anything else."
    )


def _as_value(v, raw):
    """A python value as a z3 constant of `v`'s sort."""
    if isinstance(raw, z3.ExprRef):
        return raw
    if z3.is_bool(v):
        return z3.BoolVal(bool(raw))
    if isinstance(v.sort(), z3.BitVecSortRef):
        return z3.BitVecVal(int(raw), v.sort().size())
    if v.sort().kind() == z3.Z3_INT_SORT:
        return z3.IntVal(int(raw))
    raise TypeError(f"cannot read {raw!r} as a value of sort {v.sort()}")


def _constants(expr, found=None):
    """Every uninterpreted constant occurring in `expr`, by z3 expression."""
    if found is None:
        found = {}
    stack = [expr]
    seen = set()
    while stack:
        e = stack.pop()
        if e.get_id() in seen:
            continue
        seen.add(e.get_id())
        if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            found[str(e)] = e
        stack.extend(e.children())
    return found


class Alphabet:
    """
    A finite alphabet Sigma, together with the bookkeeping that turns a z3
    formula into the set of states satisfying it.

    States are indexed in `itertools.product` order -- the last variable varies
    fastest -- and a set of states is a numpy boolean array of length `size`.
    Set operations are then the numpy ones (`&`, `|`, `^`, `~`), which is why
    nothing downstream needs to know how a denotation is represented.

    `weights` gives a non-uniform measure, one non-negative weight per state in
    the same indexing. The default is uniform, which is the counting measure
    normalised to `mu(Sigma) = 1`. Nothing in the flow depends on the choice; it
    is the reader's judgement about which behaviors matter, kept separate from
    the lattice so that changing it cannot change what a section is.

    `domains` maps a variable's name to the values it should be measured over,
    which is how an alphabet containing an integer becomes finite. Note what
    this does *not* do: the formulas keep their own semantics, so a `ForAll`
    introduced by `lan` still ranges over all the integers, and only the ambient
    set the measure normalises against is the window. Distances over such an
    alphabet are therefore distances *relative to the window*, and two contracts
    differing only outside it read as equal.
    """

    def __init__(self, variables: Sequence, weights: Optional[Sequence] = None,
                 domains: Optional[dict] = None, max_states: int = 1 << 16):
        self.variables = tuple(variables)
        names = [str(v) for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate variables in alphabet: {sorted(names)}")
        self.names = tuple(names)
        given = dict(domains or {})
        self.domains = tuple(_domain(v, given.get(str(v))) for v in self.variables)

        sizes = [len(d) for d in self.domains]
        size = 1
        for s in sizes:
            size *= s
        if size > max_states:
            raise AlphabetTooLarge(
                f"alphabet {names} has {size} states, over the {max_states} "
                f"limit. Raise max_states deliberately, or measure a coarser "
                f"alphabet -- everything here enumerates Sigma."
            )
        self.size = size

        strides, acc = [], 1
        for s in reversed(sizes):
            strides.append(acc)
            acc *= s
        self.strides = tuple(reversed(strides))

        self.weights = None
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            if w.shape != (size,):
                raise ValueError(f"weights must have shape ({size},), got {w.shape}")
            if (w < 0).any() or not w.sum() > 0:
                raise ValueError("weights must be non-negative and not all zero")
            self.weights = w

        self._models_cache = {}
        self._expand_cache = {}

    def __repr__(self):
        return f"Alphabet({list(self.names)}, size={self.size})"

    # ------------------------------------------------------------------
    # the measure
    # ------------------------------------------------------------------

    def states(self):
        """Every state, as a tuple of z3 values, in index order."""
        return itertools.product(*self.domains)

    def measure(self, indicator: np.ndarray) -> float:
        """mu of a set of states, normalised so that mu(Sigma) = 1."""
        if self.weights is None:
            return float(np.count_nonzero(indicator)) / self.size
        return float(self.weights[indicator].sum() / self.weights.sum())

    def empty(self, indicator: np.ndarray) -> bool:
        """
        Whether a set is empty, decided on the array rather than on `measure`.

        A weighted measure can be zero on a non-empty set, and floating point
        can be zero on a set of a few states out of very many. Every predicate
        in this module goes through here rather than comparing a float to zero.
        """
        return not bool(indicator.any())

    # ------------------------------------------------------------------
    # formulas to sets, and back
    # ------------------------------------------------------------------

    def models(self, phi: z3.BoolRef) -> np.ndarray:
        """
        The set of states satisfying `phi`, as a read-only boolean array.

        Quantifiers are eliminated first where possible (`simplify.qf`), which
        is what the Kan maps produce and what makes the vectorised evaluation
        below apply. Anything the evaluator does not recognise falls back to
        substituting each state and asking z3, so the answer is the same either
        way and only the speed differs.

        Cached on the expression's id, holding a reference to the expression
        itself: z3 recycles ids of collected ASTs, so the cached expression is
        compared before its value is returned.
        """
        key = phi.get_id()
        hit = self._models_cache.get(key)
        if hit is not None and hit[0].eq(phi):
            return hit[1]
        value = self._compute_models(phi)
        value.flags.writeable = False
        self._models_cache[key] = (phi, value)
        return value

    def _compute_models(self, phi: z3.BoolRef) -> np.ndarray:
        # Both rewrites are equivalence-preserving -- z3's expression simplifier
        # by construction, `qf` by checking -- so this only ever changes the
        # speed of what follows. The per-state path below is correct on a
        # quantified formula too; it is just slower by a solver call per state.
        body = phi
        if has_quantifier(body):
            body = z3.simplify(body)
        if has_quantifier(body):
            body = qf(body)

        occurring = _constants(body)
        stray = sorted(set(occurring) - set(self.names))
        if stray:
            raise ValueError(
                f"{stray} occur free in the formula but not in the alphabet "
                f"{list(self.names)}. Measuring it would silently read those as "
                f"universally quantified, which is a different formula."
            )

        support = tuple(j for j, name in enumerate(self.names) if name in occurring)
        return self._expand(self._rows(body, support), support)

    def _rows(self, body: z3.BoolRef, support: Tuple[int, ...]) -> np.ndarray:
        """The truth table of `body` over the sub-alphabet indexed by `support`."""
        sizes = [len(self.domains[j]) for j in support]
        rows = 1
        for s in sizes:
            rows *= s

        # The vectorised path indexes a Boolean by one bit of the row number, so
        # it needs the full two-valued domain -- a windowed one would misindex.
        if all(z3.is_bool(self.variables[j]) and len(self.domains[j]) == 2
               for j in support):
            env, stride = {}, 1
            for p in reversed(range(len(support))):
                digit = (np.arange(rows) // stride) % 2
                env[self.names[support[p]]] = digit.astype(bool)
                stride *= 2
            try:
                return _vector_eval(body, env, rows)
            except _Unsupported:
                pass
        return self._rows_by_solver(body, support, rows)

    def _rows_by_solver(self, body, support, rows) -> np.ndarray:
        out = np.zeros(rows, dtype=bool)
        solver = z3.Solver()
        domains = [self.domains[j] for j in support]
        for k, state in enumerate(itertools.product(*domains)):
            grounded = z3.simplify(z3.substitute(
                body, *[(self.variables[j], value) for j, value in zip(support, state)]))
            if z3.is_true(grounded):
                out[k] = True
            elif z3.is_false(grounded):
                out[k] = False
            else:
                # Every variable of the alphabet has been substituted, so this is
                # a closed formula and "valid" and "true" coincide.
                solver.push()
                solver.add(z3.Not(grounded))
                verdict = solver.check()
                solver.pop()
                if verdict == z3.unknown:
                    raise Undecided(
                        f"z3 returned unknown evaluating a contract at one state: "
                        f"{solver.reason_unknown()}")
                out[k] = (verdict == z3.unsat)
        return out

    def _expand(self, rows: np.ndarray, support: Tuple[int, ...]) -> np.ndarray:
        """Lift a truth table over the support to one over the whole alphabet."""
        if not support:
            return np.full(self.size, bool(rows[0]), dtype=bool)
        index = self._expand_cache.get(support)
        if index is None:
            index = np.zeros(self.size, dtype=np.int64)
            positions = np.arange(self.size)
            stride = 1
            for p in reversed(range(len(support))):
                j = support[p]
                digit = (positions // self.strides[j]) % len(self.domains[j])
                index += digit * stride
                stride *= len(self.domains[j])
            self._expand_cache[support] = index
        return rows[index]

    def formula(self, indicator: np.ndarray, compact: bool = False) -> z3.BoolRef:
        """
        A formula denoting exactly `indicator` -- the inverse of `models`, up to
        logical equivalence.

        Naively this is a disjunction of the states in the set, which is
        exponential in the alphabet and useless for the thing it exists to do:
        writing a 4096-state stalk as 4096 minterms is a worse representation
        than the quantifier tower it was meant to replace.

        So the set is *factorised* first (`_factorize`). What a stalk denotes is
        almost always a product -- one constraint per fact, or per block of facts
        the abstraction keeps together -- and a product is written as the
        conjunction of its factors, each of which is small. A factor the set does
        not constrain contributes nothing and is dropped, which subsumes the
        question of which variables the set actually depends on.

        The decomposition is *verified* before it is used, and the whole alphabet
        is one factor when it fails, so the result is exact either way and only
        its length depends on the structure being there.

        `compact` additionally runs `ctx-solver-simplify`, which is a solver call
        per formula and rarely worth it inside a flow.
        """
        table = np.asarray(indicator).reshape([len(d) for d in self.domains])
        out = z3.simplify(self._cover(table, tuple(range(len(self.domains))), {}))
        return _ctx_simplify(out) if compact else out

    def _cover(self, table: np.ndarray, axes: Tuple[int, ...], memo) -> z3.BoolRef:
        """
        A formula for `table`, over the alphabet variables named by `axes`.

        Three moves, in order of how much structure they exploit:

          1. a set that is everything or nothing is a constant;
          2. a set that is a *product* is the conjunction of its factors, each
             handled recursively -- this is the common case, since a stalk
             usually says one thing per fact;
          3. otherwise split on one variable and recurse on the two cofactors.

        Step 3 is what a join needs. The dual flow aggregates by join, so its
        stalks denote *unions* of products, which factor over nothing: written
        flat, a 4096-state stalk came out as an 1363-node formula and grew from
        there. Split recursively and shared through the memo, the same set is
        written once per distinct cofactor -- and because z3 hash-conses, the
        repeated subterms are literally the same node rather than copies.
        """
        key = (axes, table.tobytes())
        cached = memo.get(key)
        if cached is not None:
            return cached

        if not table.any():
            value = z3.BoolVal(False)
        elif table.all():
            value = z3.BoolVal(True)
        else:
            parts = self._factorize(table)
            if len(parts) > 1:
                pieces = []
                for part in parts:
                    others = tuple(a for a in range(table.ndim) if a not in part)
                    projection = table.any(axis=others)
                    if projection.all():
                        continue        # this factor rules nothing out
                    pieces.append(self._cover(projection,
                                              tuple(axes[a] for a in part), memo))
                value = pieces[0] if len(pieces) == 1 else z3.And(*pieces)
            elif table.size <= _FLAT_LIMIT:
                value = self._normal_form(table, axes)
            else:
                variable = self.variables[axes[0]]
                branches = []
                for index, case in enumerate(self.domains[axes[0]]):
                    cofactor = table[index]
                    if not cofactor.any():
                        continue
                    rest = self._cover(cofactor, axes[1:], memo)
                    literal = _literal(variable, case, True)
                    branches.append(literal if z3.is_true(rest)
                                    else z3.And(literal, rest))
                value = branches[0] if len(branches) == 1 else z3.Or(*branches)

        memo[key] = value
        return value

    def _factorize(self, table: np.ndarray) -> Sequence[Tuple[int, ...]]:
        """
        Split the axes into groups the set is a product over, or return them all
        in one group if it is not a product.

        Two axes are linked when the set's shadow on that pair is not the product
        of its shadows on each -- that is, when knowing one coordinate tells you
        something about the other. The connected components of that relation are
        the only candidate factorisation, and it is checked against the set
        before being returned: pairwise independence does not imply joint
        independence, so the components are a hypothesis, not a conclusion.
        """
        axes = list(range(table.ndim))
        if table.ndim <= 1:
            return [tuple(axes)]

        shadows = {axis: table.any(axis=tuple(a for a in axes if a != axis)) for axis in axes}
        parent = {axis: axis for axis in axes}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i, a in enumerate(axes):
            for b in axes[i + 1:]:
                if find(a) == find(b):
                    continue
                pair = table.any(axis=tuple(c for c in axes if c not in (a, b)))
                if not np.array_equal(pair, shadows[a][:, None] & shadows[b][None, :]):
                    parent[find(a)] = find(b)

        groups = {}
        for axis in axes:
            groups.setdefault(find(axis), []).append(axis)
        parts = [tuple(group) for group in groups.values()]
        if len(parts) == 1:
            return parts

        rebuilt = np.ones_like(table)
        for part in parts:
            others = tuple(axis for axis in axes if axis not in part)
            projection = table.any(axis=others)
            for axis in sorted(others):
                projection = np.expand_dims(projection, axis)
            rebuilt &= np.broadcast_to(projection, table.shape)
        return parts if np.array_equal(rebuilt, table) else [tuple(axes)]

    def _normal_form(self, projection: np.ndarray, axes: Tuple[int, ...]) -> z3.BoolRef:
        """
        A small factor written flat, as whichever normal form is shorter: a
        disjunction of the states it allows, or a conjunction ruling out the
        states it forbids. Flat is the better shape when it fits, which is what
        `_FLAT_LIMIT` decides.
        """
        flat = projection.reshape(-1)
        domains = [self.domains[axis] for axis in axes]
        variables = [self.variables[axis] for axis in axes]
        states = list(itertools.product(*domains))

        if np.count_nonzero(flat) <= flat.size // 2:
            return z3.Or(*[z3.And(*[_literal(v, value, True)
                                    for v, value in zip(variables, state)])
                           for k, state in enumerate(states) if flat[k]])
        return z3.And(*[z3.Or(*[_literal(v, value, False)
                                for v, value in zip(variables, state)])
                        for k, state in enumerate(states) if not flat[k]])


def _literal(variable, value, positive: bool) -> z3.BoolRef:
    """`variable == value`, or its negation, spelled the short way for Booleans."""
    if z3.is_bool(variable):
        holds = z3.is_true(value)
        return variable if holds == positive else z3.Not(variable)
    return variable == value if positive else variable != value


def _ctx_simplify(expr: z3.BoolRef) -> z3.BoolRef:
    goal = z3.Goal()
    goal.add(expr)
    subgoals = z3.Tactic('ctx-solver-simplify')(goal)
    return z3.And(*[sg.as_expr() for sg in subgoals]) if len(subgoals) != 1 \
        else subgoals[0].as_expr()


def _vector_eval(e, env, rows: int) -> np.ndarray:
    """
    Evaluate a quantifier-free Boolean formula at every state at once.

    The formulas being measured are the same shape over and over -- a stalk's
    guarantee, evaluated at each of a few hundred states -- and doing that one
    `z3.substitute` at a time dominates everything else in a sweep. This walks
    the AST once with numpy arrays in place of truth values.

    Raises `_Unsupported` on any node it does not handle, which is the caller's
    signal to fall back; correctness never depends on the coverage here.
    """
    if z3.is_true(e):
        return np.ones(rows, dtype=bool)
    if z3.is_false(e):
        return np.zeros(rows, dtype=bool)
    if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
        try:
            return env[str(e)]
        except KeyError:
            raise _Unsupported(f"{e} is not a Boolean of the alphabet") from None
    if not z3.is_app(e):
        raise _Unsupported(str(e))

    kids = e.children()
    if z3.is_not(e):
        return ~_vector_eval(kids[0], env, rows)
    if z3.is_and(e):
        out = np.ones(rows, dtype=bool)
        for kid in kids:
            out &= _vector_eval(kid, env, rows)
        return out
    if z3.is_or(e):
        out = np.zeros(rows, dtype=bool)
        for kid in kids:
            out |= _vector_eval(kid, env, rows)
        return out
    if z3.is_implies(e):
        return ~_vector_eval(kids[0], env, rows) | _vector_eval(kids[1], env, rows)
    if z3.is_app_of(e, z3.Z3_OP_XOR):
        return _vector_eval(kids[0], env, rows) ^ _vector_eval(kids[1], env, rows)
    if z3.is_eq(e):
        return ~(_vector_eval(kids[0], env, rows) ^ _vector_eval(kids[1], env, rows))
    if z3.is_distinct(e):
        if len(kids) != 2:
            raise _Unsupported("distinct over more than two Booleans")
        return _vector_eval(kids[0], env, rows) ^ _vector_eval(kids[1], env, rows)
    if z3.is_app_of(e, z3.Z3_OP_ITE):
        return np.where(_vector_eval(kids[0], env, rows),
                        _vector_eval(kids[1], env, rows),
                        _vector_eval(kids[2], env, rows))
    raise _Unsupported(str(e.decl()))


# ---------------------------------------------------------------------------
# contracts
# ---------------------------------------------------------------------------

#: What a contract denotes: the environments it admits and the behaviors it
#: permits, as sets of states. Refinement is `admits` reversed and `permits`
#: forwards, which is the whole asymmetry of the contract order in one place.
Denotation = namedtuple("Denotation", "admits permits")


def denotation(contract: Contract, alphabet: Alphabet) -> Denotation:
    """`([[A]], [[A => G]])` -- the saturation class, never the syntax."""
    return Denotation(alphabet.models(contract.a), alphabet.models(contract.sat_g))


def distance(c1: Contract, c2: Contract, alphabet: Alphabet) -> float:
    """
    The symmetric distance, in [0, 1]. Zero exactly on contracts that are equal
    as contracts -- which `test_measure.py` checks against z3's own `==` rather
    than assuming.
    """
    d1, d2 = denotation(c1, alphabet), denotation(c2, alphabet)
    return 0.5 * (alphabet.measure(d1.admits ^ d2.admits)
                  + alphabet.measure(d1.permits ^ d2.permits))


def divergence(c1: Contract, c2: Contract, alphabet: Alphabet) -> float:
    """
    The directed distance from `c1` to `c2`: how much of `c2` is not yet
    accounted for by `c1`. Zero exactly when `c1` refines `c2`.

    This is the Lawvere metric of the refinement order -- asymmetric by
    construction, because refinement is. It is what to plot along a flow: the
    primal flow only ever descends, so `divergence(x_next, x_now)` is
    identically zero and `divergence(x_now, x_limit)` decreases monotonically.
    """
    d1, d2 = denotation(c1, alphabet), denotation(c2, alphabet)
    return 0.5 * (alphabet.measure(d2.admits & ~d1.admits)
                  + alphabet.measure(d1.permits & ~d2.permits))


def refines(c1: Contract, c2: Contract, alphabet: Alphabet) -> bool:
    """
    `c1 <= c2`, decided on the denotations rather than by the solver.

    Agrees with `Contract.refines` on every alphabet this module can measure,
    and is far cheaper once the tables are cached. Decided on set emptiness, not
    on `divergence` being zero, so a weighted measure cannot change the answer.
    """
    d1, d2 = denotation(c1, alphabet), denotation(c2, alphabet)
    return (alphabet.empty(d2.admits & ~d1.admits)
            and alphabet.empty(d1.permits & ~d2.permits))


def equivalent(c1: Contract, c2: Contract, alphabet: Alphabet) -> bool:
    """`c1 == c2` as contracts, decided on the denotations."""
    d1, d2 = denotation(c1, alphabet), denotation(c2, alphabet)
    return (np.array_equal(d1.admits, d2.admits)
            and np.array_equal(d1.permits, d2.permits))


def interval_volume(c1: Contract, c2: Contract, alphabet: Alphabet) -> float:
    """
    The measure of the lattice interval `[c1 ^ c2, c1 v c2]`, computed through
    the shipped `meet` and `join`.

    Equal to `distance` -- the meet unions the admitted environments and
    intersects the permitted behaviors while the join does the reverse, so the
    interval's width in each slot is exactly that slot's symmetric difference.
    Keeping both and checking they agree is what ties this module to the lattice
    `contracts.py` implements rather than to a paraphrase of it.
    """
    low, high = c1.meet(c2), c1.join(c2)
    dl, dh = denotation(low, alphabet), denotation(high, alphabet)
    return 0.5 * ((alphabet.measure(dl.admits) - alphabet.measure(dh.admits))
                  + (alphabet.measure(dh.permits) - alphabet.measure(dl.permits)))


def canonical(contract: Contract, alphabet: Alphabet,
              compact: bool = False) -> Contract:
    """
    The same contract with both slots rewritten from what they denote.

    Semantics-preserving by construction, and the reason an exact-contract flow
    finishes at all: transport nests a quantifier layer per sweep, so without
    this the formulas grow without bound while what they say stops changing (see
    `simplify`). The result is automatically saturated -- `[[A => G]]` always
    contains the complement of `[[A]]`, so rebuilding the second slot from it and
    re-implying gives the same set back.
    """
    d = denotation(contract, alphabet)
    return Contract(alphabet.formula(d.admits, compact=compact),
                    alphabet.formula(d.permits, compact=compact),
                    vars=contract.vars)
