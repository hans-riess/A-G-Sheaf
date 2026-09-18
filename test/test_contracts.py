import z3
import pytest

from agsheaf.contracts import (
    Contract,
    Relation,
    _is_valid,
    _is_satisfiable,
    top,
    bot
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def equiv(e1: z3.ExprRef, e2: z3.ExprRef) -> bool:
    """True iff e1 and e2 are logically equivalent (valid biconditional)."""
    return _is_valid(e1 == e2)


# ---------------------------------------------------------------------------
# Module-level SMT helpers
# ---------------------------------------------------------------------------

class TestSmtHelpers:
    def test_is_valid_tautology(self):
        x = z3.Bool("x")
        assert _is_valid(z3.Or(x, z3.Not(x)))

    def test_is_valid_non_tautology(self):
        x = z3.Bool("x")
        assert not _is_valid(x)

    def test_is_valid_contradiction(self):
        x = z3.Bool("x")
        assert not _is_valid(z3.And(x, z3.Not(x)))

    def test_is_satisfiable_true(self):
        x = z3.Bool("x")
        assert _is_satisfiable(x)

    def test_is_satisfiable_contradiction(self):
        x = z3.Bool("x")
        assert not _is_satisfiable(z3.And(x, z3.Not(x)))


# ---------------------------------------------------------------------------
# Construction / invariants
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_stores_assumption_and_guarantee(self):
        o = z3.Bool("o")
        c = Contract(z3.BoolVal(True), o)
        assert equiv(c.a, z3.BoolVal(True))
        assert equiv(c.g, o)

    def test_repr_contains_key_fields(self):
        i, o = z3.Bool("i"), z3.Bool("o")
        c = Contract(i, o)
        text = repr(c)
        assert "Contract" in text
        assert "i" in text and "o" in text


# ---------------------------------------------------------------------------
# refines
# ---------------------------------------------------------------------------

class TestRefines:
    def test_reflexive(self):
        i, o = z3.Bool("i"), z3.Bool("o")
        c = Contract(i, o)
        assert c.refines(c)

    def test_stronger_guarantee_refines(self):
        # self promises (o) which is stronger than other promising (True)
        o = z3.Bool("o")
        strong = Contract(z3.BoolVal(True), o)
        weak = Contract(z3.BoolVal(True), z3.BoolVal(True))
        assert strong.refines(weak)
        assert not weak.refines(strong)

    def test_weaker_assumption_refines(self):
        # self assumes True (accepts more environments) than other assuming i
        i, o = z3.Bool("i"), z3.Bool("o")
        broad = Contract(z3.BoolVal(True), o)
        narrow = Contract(i, o)
        assert broad.refines(narrow)
        assert not narrow.refines(broad)

    def test_unrelated_does_not_refine(self):
        i, o = z3.Bool("i"), z3.Bool("o")
        c1 = Contract(i, o)
        c2 = Contract(z3.Not(i), z3.Not(o))
        assert not c1.refines(c2)
        assert not c2.refines(c1)


# ---------------------------------------------------------------------------
# meet
# ---------------------------------------------------------------------------

class TestMeet:
    def test_assumption_is_disjunction(self):
        o = z3.Bool("o")
        p, q = z3.Bool("p"), z3.Bool("q")
        c1 = Contract(p, o)
        c2 = Contract(q, o)
        merged = c1.meet(c2)
        assert equiv(merged.a, z3.Or(p, q))

    def test_guarantee_is_conjunction_of_implications(self):
        p, q = z3.Bool("p"), z3.Bool("q")
        g1, g2 = z3.Bool("g1"), z3.Bool("g2")
        c1 = Contract(p, g1)
        c2 = Contract(q, g2)
        merged = c1.meet(c2)
        expected = z3.And(z3.Implies(p, g1), z3.Implies(q, g2))
        assert equiv(merged.g, expected)


# ---------------------------------------------------------------------------
# top / bottom
# ---------------------------------------------------------------------------

class TestTopBottom:
    def test_top_values(self):
        assert equiv(top().a, z3.BoolVal(False))
        assert equiv(top().g, z3.BoolVal(True))

    def test_bottom_values(self):
        assert equiv(bot().a, z3.BoolVal(True))
        assert equiv(bot().g, z3.BoolVal(False))

    def test_top_is_meet_identity(self):
        i, o = z3.Bool("i"), z3.Bool("o")
        c = Contract(i, o)
        assert c.meet(top()) == c
        assert top().meet(c) == c

    def test_bottom_is_join_identity(self):
        i, o = z3.Bool("i"), z3.Bool("o")
        c = Contract(i, o)
        assert c.join(bot()) == c
        assert bot().join(c) == c

    def test_everything_refines_top(self):
        i, o = z3.Bool("i"), z3.Bool("o")
        c = Contract(i, o)
        assert c.refines(top())

    def test_bottom_refines_everything(self):
        i, o = z3.Bool("i"), z3.Bool("o")
        c = Contract(i, o)
        assert bot().refines(c)


# ---------------------------------------------------------------------------
# join (dual of meet)
# ---------------------------------------------------------------------------

class TestJoin:
    def test_assumption_is_conjunction(self):
        o = z3.Bool("o")
        p, q = z3.Bool("p"), z3.Bool("q")
        c1 = Contract(p, o)
        c2 = Contract(q, o)
        joined = c1.join(c2)
        assert equiv(joined.a, z3.And(p, q))

    def test_guarantee_is_disjunction_of_implications(self):
        p, q = z3.Bool("p"), z3.Bool("q")
        g1, g2 = z3.Bool("g1"), z3.Bool("g2")
        c1 = Contract(p, g1)
        c2 = Contract(q, g2)
        joined = c1.join(c2)
        expected = z3.Or(z3.Implies(p, g1), z3.Implies(q, g2))
        assert equiv(joined.g, expected)

# ---------------------------------------------------------------------------
# Contract Extensions (lan, ran, pullback via Relations)
# ---------------------------------------------------------------------------

class TestContractExtensions:
    def _setup(self):
        # Relation R(x, y) := (y == x + 1)
        x = z3.Int("x")
        y = z3.Int("y")
        R = Relation((y == x + 1), [x], [y])
        return R, x, y

    def test_pullback_assumption_semantics(self):
        relation, x, y = self._setup()
        # Contract over target Y with A(y) = y > 0
        cy = Contract(y > 0, z3.BoolVal(True))
        px = cy.pullback(relation)
        # ∃y. (y == x+1) ∧ (y > 0)  <=>  x + 1 > 0
        assert equiv(px.a, x + 1 > 0)

    def test_pullback_guarantee_semantics(self):
        relation, x, y = self._setup()
        cy = Contract(z3.BoolVal(True), y > 5)
        px = cy.pullback(relation)
        # ∀y. (y == x+1) ⇒ (y > 5)  <=>  x + 1 > 5
        assert equiv(px.g, x + 1 > 5)

    def test_ran_guarantee_is_forall(self):
        relation, x, y = self._setup()
        cx = Contract(z3.BoolVal(True), x > 0)
        ry = cx.ran(relation)
        # ∀x. (y == x+1) ⇒ (x > 0)  <=>  y - 1 > 0
        assert equiv(ry.g, y - 1 > 0)

    def test_lan_assumption_is_forall(self):
        relation, x, y = self._setup()
        cx = Contract(x > 0, z3.BoolVal(True))
        ly = cx.lan(relation)
        # ∀x. (y == x+1) ⇒ (x > 0)  <=>  y - 1 > 0
        assert equiv(ly.a, y - 1 > 0)

    def test_lan_guarantee_is_exists(self):
        relation, x, y = self._setup()
        cx = Contract(z3.BoolVal(True), x > 0)
        ly = cx.lan(relation)
        # ∃x. (y == x+1) ∧ (x > 0)  <=>  y - 1 > 0
        assert equiv(ly.g, y - 1 > 0)

    def test_returns_contract(self):
        relation, x, y = self._setup()
        cx = Contract(z3.BoolVal(True), z3.BoolVal(True))
        ly = cx.lan(relation)
        assert isinstance(ly, Contract)

    def test_dual_pullback_assumption_is_forall(self):
        relation, x, y = self._setup()
        cy = Contract(y > 0, z3.BoolVal(True))
        dx = cy.dual_pullback(relation)
        # ∀y. (y == x+1) ⇒ (y > 0)  <=>  x + 1 > 0
        assert equiv(dx.a, x + 1 > 0)

    def test_dual_pullback_guarantee_is_exists(self):
        relation, x, y = self._setup()
        cy = Contract(z3.BoolVal(True), y > 5)
        dx = cy.dual_pullback(relation)
        # ∃y. (y == x+1) ∧ (y > 5)  <=>  x + 1 > 5
        assert equiv(dx.g, x + 1 > 5)

    def test_ran_assumption_is_exists(self):
        relation, x, y = self._setup()
        cx = Contract(x > 0, z3.BoolVal(True))
        ry = cx.ran(relation)
        # ∃x. (y == x+1) ∧ (x > 0)  <=>  y - 1 > 0
        assert equiv(ry.a, y - 1 > 0)

    # The two adjunctions that hold for *every* relation (lan ⊣ pullback and
    # dual_pullback ⊣ ran), the one that does not (pullback ⊣ ran, which needs a
    # total function), and which join/meet each map preserves are all shape
    # properties of the relation rather than of this one bijection. They are
    # parametrised over the relation zoo in test_theorems.py instead; asserting
    # them here against `y == x + 1` alone would pass for the wrong reason.

    def test_lan_preserves_join_universal(self):
        """
        Verify universally that Lan preserves joins:
        Lan_R(cx1 join cx2) == Lan_R(cx1) join Lan_R(cx2)
        """
        relation, x, y = self._setup()
        
        A_x1 = z3.Function("A_x1", z3.IntSort(), z3.BoolSort())
        G_x1 = z3.Function("G_x1", z3.IntSort(), z3.BoolSort())
        cx1 = Contract(A_x1(x), G_x1(x))
        
        A_x2 = z3.Function("A_x2", z3.IntSort(), z3.BoolSort())
        G_x2 = z3.Function("G_x2", z3.IntSort(), z3.BoolSort())
        cx2 = Contract(A_x2(x), G_x2(x))
        
        lhs = cx1.join(cx2).lan(relation)
        rhs = cx1.lan(relation).join(cx2.lan(relation))
        
        assert lhs == rhs, "Lan does not preserve joins"

    def test_ran_preserves_meet_universal(self):
        """
        Verify universally that Ran preserves meets:
        Ran_R(cx1 meet cx2) == Ran_R(cx1) meet Ran_R(cx2)
        """
        relation, x, y = self._setup()
        
        A_x1 = z3.Function("A_x1", z3.IntSort(), z3.BoolSort())
        G_x1 = z3.Function("G_x1", z3.IntSort(), z3.BoolSort())
        cx1 = Contract(A_x1(x), G_x1(x))
        
        A_x2 = z3.Function("A_x2", z3.IntSort(), z3.BoolSort())
        G_x2 = z3.Function("G_x2", z3.IntSort(), z3.BoolSort())
        cx2 = Contract(A_x2(x), G_x2(x))
        
        lhs = cx1.meet(cx2).ran(relation)
        rhs = cx1.ran(relation).meet(cx2.ran(relation))
        
        assert lhs == rhs, "Ran does not preserve meets"

    def test_pullback_preserves_meet_universal(self):
        """
        Verify universally that Pullback preserves meets:
        Pullback_R(cy1 meet cy2) == Pullback_R(cy1) meet Pullback_R(cy2)
        """
        relation, x, y = self._setup()
        
        A_y1 = z3.Function("A_y1", z3.IntSort(), z3.BoolSort())
        G_y1 = z3.Function("G_y1", z3.IntSort(), z3.BoolSort())
        cy1 = Contract(A_y1(y), G_y1(y))
        
        A_y2 = z3.Function("A_y2", z3.IntSort(), z3.BoolSort())
        G_y2 = z3.Function("G_y2", z3.IntSort(), z3.BoolSort())
        cy2 = Contract(A_y2(y), G_y2(y))
        
        lhs = cy1.meet(cy2).pullback(relation)
        rhs = cy1.pullback(relation).meet(cy2.pullback(relation))
        
        assert lhs == rhs, "Pullback does not preserve meets"

    # Pullback preserving *joins* is not universal -- its assumption leg is an
    # exists, which does not distribute over the conjunction `join` puts there,
    # so it holds only when the relation is functional. See
    # test_theorems.py::TestSupMorphism, which pins down the exact condition.


# ---------------------------------------------------------------------------
# Composition and quotient (Incer et al. 2025, Pacti Eqs. 2-3)
# ---------------------------------------------------------------------------

class TestCompositionAndQuotient:
    def test_composition_guarantee_is_the_joint_promise(self):
        p, q = z3.Bool("p"), z3.Bool("q")
        g1, g2 = z3.Bool("g1"), z3.Bool("g2")
        composed = Contract(p, g1).compose(Contract(q, g2))
        assert equiv(composed.g, z3.And(z3.Implies(p, g1), z3.Implies(q, g2)))

    def test_composition_discharges_assumptions(self):
        """
        The difference from `meet`: where the pair's joint guarantee already
        fails, no environment assumption is needed.
        """
        p, q = z3.Bool("p"), z3.Bool("q")
        g1, g2 = z3.Bool("g1"), z3.Bool("g2")
        joint = z3.And(z3.Implies(p, g1), z3.Implies(q, g2))
        composed = Contract(p, g1).compose(Contract(q, g2))
        assert equiv(composed.a, z3.Or(z3.And(p, q), z3.Not(joint)))

    def test_composition_unit_is_the_unconstrained_component(self):
        """
        Wiring in a component that assumes everything and promises everything
        changes nothing. Note the unit is (true, true), NOT `top` -- `top` is
        the identity for `meet`, and composing with it instead discharges the
        assumptions entirely.
        """
        i, o = z3.Bool("i"), z3.Bool("o")
        c = Contract(i, o)
        unit = Contract(z3.BoolVal(True), z3.BoolVal(True))
        assert c.compose(unit) == c
        assert c.compose(top()) != c

    def test_quotient_by_itself_admits_the_unit(self):
        """
        Nothing is left to build, so the unconstrained component suffices: it
        refines the quotient, which by the adjunction is just
        c.compose(unit) <= c restated.
        """
        i, o = z3.Bool("i"), z3.Bool("o")
        c = Contract(i, o)
        unit = Contract(z3.BoolVal(True), z3.BoolVal(True))
        assert unit.refines(c.quotient(c))

    def test_quotient_solves_the_missing_component_problem(self):
        """
        Composing the quotient back with the divisor recovers something that
        meets the top-level specification -- the whole point of the operation.
        """
        i, o = z3.Bool("i"), z3.Bool("o")
        top_level = Contract(i, o)
        have = Contract(z3.BoolVal(True), i)
        missing = top_level.quotient(have)
        assert have.compose(missing).refines(top_level)