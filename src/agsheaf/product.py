"""
Products of contract lattices, componentwise.

The region-abstracted belief sheaf (`gridsheaf.build_region_sheaf`) has no
coupling between regions anywhere in its construction: contracts, restriction
relations, meets and transports all factor over the region partition. This
module makes that factorization structural rather than hoped-for. A stalk is a
`ProductContract` -- one `Contract` per region -- and a restriction map is a
`ProductRelation` -- one `Relation` per region -- with every operation applied
componentwise.

The mathematics this rests on: a product of complete lattices is a complete
lattice with componentwise order, and a product of left adjoints is a left
adjoint. A cellular sheaf whose stalks are products and whose restriction maps
are componentwise is exactly a product of sheaves, its global sections are the
products of the factors' global sections, and the Tarski Laplacian of the
product is the product of the factors' Laplacians. Everything `ContractSheaf`
does -- transport, laplacian, agreement, the harmonic flow -- therefore works
on these objects unchanged, by duck typing; what the factorization buys is
that every z3 query stays inside one region's small alphabet, and that
agreement can be reported per (interface, region), which is the localisation
the demonstration renders.

What a product stalk is *not* is a single contract over the joined alphabet:
`Con(B1 x B2)` is strictly larger than `Con(B1) x Con(B2)`. The construction
never needs the extra room -- no contract in the demo couples two regions --
and staying inside the product is precisely what keeps the flow's formulas
from entangling regions as sweeps accumulate. `assemble` provides the joined
reading (conjunction of assumptions, conjunction of guarantees) for display
and for whole-stalk queries, as a view rather than as the state.
"""

import z3
from typing import Any, Dict, Iterator, List, Mapping, Optional

from .contracts import Contract, Relation


class ProductRelation:
    """
    One `Relation` per component, keyed like the `ProductContract`s it will
    transport. `source_vars` and `target_vars` are the flattened unions, which
    is what `ContractSheaf.add_interface` compares across an interface.
    """

    def __init__(self, components: Mapping[Any, Relation]):
        assert components, "a ProductRelation needs at least one component"
        self.components: Dict[Any, Relation] = dict(components)

    @property
    def source_vars(self) -> List:
        return [v for rel in self.components.values() for v in rel.source_vars]

    @property
    def target_vars(self) -> List:
        return [v for rel in self.components.values() for v in rel.target_vars]

    def __repr__(self):
        keys = ", ".join(str(k) for k in self.components)
        return f"ProductRelation({keys})"


class ProductContract:
    """
    One `Contract` per component, all operations componentwise.

    Binary operations require the same component keys on both sides -- a
    mismatch is a modelling error (two stalks built over different partitions),
    not something to reconcile silently. Component alphabets are guarded by
    `Contract._same_alphabet` exactly as plain contracts are.
    """

    def __init__(self, components: Mapping[Any, Contract]):
        assert components, "a ProductContract needs at least one component"
        self.components: Dict[Any, Contract] = dict(components)

    # -- plumbing ----------------------------------------------------------

    def _same_keys(self, other: 'ProductContract') -> List:
        if list(self.components.keys()) != list(other.components.keys()):
            raise ValueError(
                f"component mismatch: {list(self.components)} vs {list(other.components)}"
            )
        return list(self.components.keys())

    def _zip(self, relation: ProductRelation) -> Iterator:
        if list(self.components.keys()) != list(relation.components.keys()):
            raise ValueError(
                f"component mismatch between contract {list(self.components)} "
                f"and relation {list(relation.components)}"
            )
        for key, c in self.components.items():
            yield key, c, relation.components[key]

    def __getitem__(self, key: Any) -> Contract:
        return self.components[key]

    def keys(self) -> List:
        return list(self.components.keys())

    def __repr__(self):
        parts = "\n".join(f"  {key}: A={z3.simplify(c.a)}  G={z3.simplify(c.sat_g)}"
                          for key, c in self.components.items())
        return f"ProductContract(\n{parts}\n)"

    # -- the lattice, componentwise ---------------------------------------

    def meet(self, other: 'ProductContract') -> 'ProductContract':
        return ProductContract({k: self.components[k].meet(other.components[k])
                                for k in self._same_keys(other)})

    def join(self, other: 'ProductContract') -> 'ProductContract':
        return ProductContract({k: self.components[k].join(other.components[k])
                                for k in self._same_keys(other)})

    def refines(self, other: 'ProductContract') -> bool:
        return all(self.components[k].refines(other.components[k])
                   for k in self._same_keys(other))

    def __eq__(self, other) -> bool:
        if not isinstance(other, ProductContract):
            return NotImplemented
        return all(self.components[k] == other.components[k]
                   for k in self._same_keys(other))

    # Componentwise equality is decided by the solver; same rule as Contract.
    __hash__ = None

    def disagreeing(self, other: 'ProductContract') -> List:
        """
        The component keys on which the two products differ -- `__eq__`'s
        counterexamples rather than its verdict. This is what per-(interface,
        region) agreement reporting reads: the sheaf condition on a product is
        a conjunction over components, and naming the failing ones is the same
        localisation argument the Laplacian makes over a monolithic check.
        """
        return [k for k in self._same_keys(other)
                if not (self.components[k] == other.components[k])]

    def is_consistent(self) -> bool:
        return all(c.is_consistent() for c in self.components.values())

    def is_compatible(self) -> bool:
        return all(c.is_compatible() for c in self.components.values())

    # -- the four embedding maps, componentwise ---------------------------

    def lan(self, relation: ProductRelation) -> 'ProductContract':
        return ProductContract({k: c.lan(r) for k, c, r in self._zip(relation)})

    def ran(self, relation: ProductRelation) -> 'ProductContract':
        return ProductContract({k: c.ran(r) for k, c, r in self._zip(relation)})

    def pullback(self, relation: ProductRelation) -> 'ProductContract':
        return ProductContract({k: c.pullback(r) for k, c, r in self._zip(relation)})

    def dual_pullback(self, relation: ProductRelation) -> 'ProductContract':
        return ProductContract({k: c.dual_pullback(r) for k, c, r in self._zip(relation)})

    # -- the joined reading, as a view ------------------------------------

    def assemble(self, vars: Optional[List] = None) -> Contract:
        """
        The product read as one contract over the joined alphabet: assume every
        component's assumption, guarantee every component's guarantee. This is
        the mission contract as a person would state it, and the object
        whole-stalk queries (the planner's decode, the mission monitor) run
        against. It is a view: the componentwise conjunction is not the lattice
        meet (whose assumption would be the disjunction), and nothing writes
        it back into the product.
        """
        return Contract(z3.And([c.a for c in self.components.values()]),
                        z3.And([c.sat_g for c in self.components.values()]),
                        vars=vars)
