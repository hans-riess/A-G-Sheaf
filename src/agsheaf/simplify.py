"""
Quantifier elimination for transported contracts.

Each transport nests an existential inside a universal, and the harmonic flow
composes transports across sweeps, so a vertex's formula grows a fresh
quantifier layer per sweep. Nothing is wrong with that logically -- z3 answers
correctly -- but every `refines`, `==` and `is_consistent` call re-solves the
whole accumulated tower, and the flow slows down superlinearly in the sweep
count rather than in the problem.

This is not a hypothetical cost. Measured on a 3-node path whose restrictions
are non-injective abstractions (2026-08-10): one sweep took a stalk's guarantee
from 5 AST nodes to 28 with *no* change in what it denotes, and the flow
produced no fixed point in four minutes. With the formulas canonicalised
between sweeps the same ensemble at 12 nodes settles in seconds. Eliminating
quantifiers eagerly is the difference between an experiment that sweeps N and
topology density and one that does not finish.

`qf` is semantics-preserving whenever it succeeds and returns its input
unchanged whenever it does not, so it is always safe to apply -- the flow it is
applied to is the same flow. `measure.canonical` is the exact alternative for
alphabets small enough to tabulate; this one has no size bound and no
completeness guarantee.
"""

import z3

from .contracts import Undecided, _is_valid

_qe = z3.Then(z3.Tactic('qe2'), z3.Tactic('simplify'))


def qf(expr: z3.BoolRef) -> z3.BoolRef:
    """
    Quantifier-free equivalent of `expr`, or `expr` unchanged if that cannot be
    obtained and *checked*.

    The check is not defensive padding, it is the contract of this function. A
    z3 `Goal` is a satisfiability problem, and a tactic applied to one is only
    required to preserve equisatisfiability -- it may legitimately *decide* the
    goal by choosing values for the free constants. On

        ForAll([p, q, r], Implies(e == Or(p, q), Or(p, q)))

    whose only quantifier-free equivalent is `e`, `Then(simplify, qe2)` returns
    the empty goal, which reads back as `True`: true for `e = False`, where the
    original is false. Every contract this is applied to has free variables --
    they are the stalk's alphabet -- so this is the common case, not a corner.

    Hence: eliminate, then ask z3 whether the result is equivalent to the input,
    and keep the input when the answer is not a clear yes. `expr` is always a
    sound return value, so a failed or undecided check costs speed and never
    correctness.
    """
    try:
        goal = z3.Goal()
        goal.add(expr)
        subgoals = _qe(goal)
        out = z3.And(*[sg.as_expr() for sg in subgoals]) if len(subgoals) != 1 \
            else subgoals[0].as_expr()
    except z3.Z3Exception:
        return expr

    if has_quantifier(out):
        return expr
    try:
        return out if _is_valid(expr == out) else expr
    except (Undecided, z3.Z3Exception):
        return expr


def has_quantifier(e) -> bool:
    """Whether `e` contains a quantifier anywhere below it."""
    if z3.is_quantifier(e):
        return True
    return any(has_quantifier(child) for child in e.children())


def ast_size(e) -> int:
    """
    The number of distinct nodes in `e`, counting shared subterms once -- z3
    keeps its ASTs hash-consed, so this is the size of what is actually stored
    rather than of what it would take to print.

    This is the quantity that runs away across sweeps, and the one worth
    plotting against `measure.distance`: a stalk whose formula grows while its
    distance to the limit does not has spent a sweep saying the same thing at
    greater length.
    """
    seen, stack, count = set(), [e], 0
    while stack:
        node = stack.pop()
        if node.get_id() in seen:
            continue
        seen.add(node.get_id())
        count += 1
        stack.extend(node.children())
    return count


def simplify_contract(c):
    """A contract with both slots put in quantifier-free form."""
    from .contracts import Contract
    return Contract(qf(c.a), qf(c.sat_g), vars=c.vars)
