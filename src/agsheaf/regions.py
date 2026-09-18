"""
The region partition of a gridworld: the fibers of the abstraction map.

A region is a named set of tiles. The partition is what turns the belief sheaf's
restriction maps from bijections into genuine abstractions: agent stalks speak
per-tile, edge stalks speak per-region, and the restriction relation sends each
concrete state to its region summary (see `gridsheaf.abstraction_relation`).
Which tiles form a region is therefore a modelling decision, not an
implementation detail -- it decides exactly which disagreements the interfaces
tolerate -- so it is configured per experiment rather than derived.

This module deliberately imports neither `agsheaf.gridworld` (the Robotarium
stack) nor z3: it is shared by the sheaf construction and the renderer, and a
partition is just named sets of tiles.
"""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

Tile = Tuple[int, int]


class Regions:
    """
    A named partition of the grid's tiles.

    Immutable once built; `parse_regions` and `identity_regions` are the two
    constructors an experiment uses. Region names keep the order the
    configuration gave them, which is the order every per-region structure in
    the sheaf iterates in -- variable lists across an interface line up because
    both endpoints iterate the same `Regions`.
    """

    def __init__(self, tiles_by_name: Mapping[str, Iterable[Tile]],
                 grid_width_height: Tuple[int, int]):
        width, height = grid_width_height
        assert isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0, \
            "Failed to build the region partition. grid_width_height must be a pair of positive ints, received %r." % (grid_width_height,)
        self.grid_width_height = (width, height)

        self._tiles_by_name: Dict[str, Tuple[Tile, ...]] = {}
        self._region_of: Dict[Tile, str] = {}
        for name, tiles in tiles_by_name.items():
            name = str(name)
            ordered = []
            for tile in tiles:
                tile = (int(tile[0]), int(tile[1]))
                x, y = tile
                assert 0 <= x < width and 0 <= y < height, \
                    "Failed to build the region partition. Region %r contains tile %r, which is off the %dx%d grid." % (name, tile, width, height)
                assert tile not in self._region_of, \
                    "Failed to build the region partition. Tile %r appears in both region %r and region %r; regions must not overlap." % (tile, self._region_of[tile], name)
                self._region_of[tile] = name
                ordered.append(tile)
            assert ordered, \
                "Failed to build the region partition. Region %r contains no tiles." % (name,)
            self._tiles_by_name[name] = tuple(ordered)

        missing = [(x, y) for x in range(width) for y in range(height)
                   if (x, y) not in self._region_of]
        assert not missing, \
            "Failed to build the region partition. The regions do not cover the grid; uncovered tiles: %s." % (missing,)

    @property
    def names(self) -> List[str]:
        """Region names, in configuration order."""
        return list(self._tiles_by_name.keys())

    def tiles(self, name: str) -> Tuple[Tile, ...]:
        """The tiles of one region, in configuration order."""
        if name not in self._tiles_by_name:
            raise KeyError(f"no region named {name!r}; regions are {self.names}")
        return self._tiles_by_name[name]

    def region_of(self, tile: Tile) -> str:
        """The name of the region containing `tile`."""
        tile = (int(tile[0]), int(tile[1]))
        if tile not in self._region_of:
            raise KeyError(f"tile {tile} is off the {self.grid_width_height[0]}x{self.grid_width_height[1]} grid")
        return self._region_of[tile]

    def regions_of(self, tiles: Iterable[Tile]) -> List[str]:
        """The regions a set of tiles touches, in configuration order, each once."""
        touched = {self.region_of(t) for t in tiles}
        return [name for name in self._tiles_by_name if name in touched]

    def __len__(self) -> int:
        return len(self._tiles_by_name)

    def __iter__(self):
        return iter(self._tiles_by_name)

    def __contains__(self, name: object) -> bool:
        return name in self._tiles_by_name

    def __repr__(self):
        sizes = ", ".join(f"{name}:{len(tiles)}"
                          for name, tiles in self._tiles_by_name.items())
        return f"Regions({sizes} over {self.grid_width_height[0]}x{self.grid_width_height[1]})"

    def is_identity(self) -> bool:
        """
        Whether every region is a single tile -- the f = id partition under
        which the abstraction forgets nothing and the sheaf degenerates to the
        bijective-restriction case (see doc/DUALITY.md).
        """
        return all(len(tiles) == 1 for tiles in self._tiles_by_name.values())


def _rect_tiles(rect: Any, name: str) -> List[Tile]:
    """The tiles of an inclusive rectangle [x0, y0, x1, y1], column-major."""
    ok = (isinstance(rect, (list, tuple)) and len(rect) == 4
          and all(isinstance(v, int) and not isinstance(v, bool) for v in rect))
    assert ok, \
        "Failed to parse the regions configuration. Region %r has rect %r; expected [x0, y0, x1, y1] with integer corners." % (name, rect)
    x0, y0, x1, y1 = rect
    assert x0 <= x1 and y0 <= y1, \
        "Failed to parse the regions configuration. Region %r has rect %r; corners must satisfy x0 <= x1 and y0 <= y1." % (name, rect)
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def parse_regions(regions_cfg: Optional[Mapping[str, Any]],
                  grid_width_height: Tuple[int, int]) -> Regions:
    """
    The region partition an experiment configured, as a `Regions`.

    The `regions:` block maps each region name to either an inclusive
    rectangle -- ``{rect: [x0, y0, x1, y1]}`` -- or an explicit tile list --
    ``[[x, y], ...]``. The named regions must tile the grid exactly: every
    tile in exactly one region. A missing or empty block falls back to
    `identity_regions`, the partition under which the sheaf is the delivered
    f = id demo, so old configurations keep their old meaning.

    Args:
        regions_cfg: The `regions:` mapping from the configuration, or None.
        grid_width_height (Tuple[int, int]): The grid, for cover checking.
    """
    if not regions_cfg:
        return identity_regions(grid_width_height)
    assert isinstance(regions_cfg, Mapping), \
        "Failed to parse the regions configuration. Expected a mapping of region names to tile specs, received %r." % (regions_cfg,)

    tiles_by_name: Dict[str, List[Tile]] = {}
    for name, spec in regions_cfg.items():
        name = str(name)
        if isinstance(spec, Mapping):
            assert set(spec.keys()) == {"rect"}, \
                "Failed to parse the regions configuration. Region %r has keys %s; a mapping spec must be exactly {rect: [x0, y0, x1, y1]}." % (name, sorted(spec.keys()))
            tiles_by_name[name] = _rect_tiles(spec["rect"], name)
        else:
            assert isinstance(spec, (list, tuple)) and spec, \
                "Failed to parse the regions configuration. Region %r must be a non-empty tile list or {rect: ...}, received %r." % (name, spec)
            tiles_by_name[name] = [(int(t[0]), int(t[1])) for t in spec]
    return Regions(tiles_by_name, grid_width_height)


def identity_regions(grid_width_height: Tuple[int, int]) -> Regions:
    """
    One region per tile, named ``t{x}_{y}``: the identity abstraction. Under
    this partition the restriction relations are bijections and the sheaf
    reproduces the pre-region demonstration exactly -- the degenerate control
    arm the abstraction study is measured against.
    """
    width, height = grid_width_height
    return Regions({f"t{x}_{y}": [(x, y)]
                    for x in range(width) for y in range(height)},
                   grid_width_height)
