"""
Records everything one run of an experiment did, as a single JSON file.

Named experiment_log rather than logging so that it cannot shadow the standard library module
of that name for anything else importing from this package.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Optional, Union

from agsheaf.gridworld import GROUND_TRUTH_VIEW, TILE_LABELS


# One character per tile label, used to write a whole grid of labels as a few short strings.
# A label absent here would be written as "?", though every label in TILE_LABELS has one.
LABEL_CHARACTERS = {
    "target": "t",
    "safe": "s",
    "unsafe": "u",
    "unknown": ".",
}

assert set(LABEL_CHARACTERS) == set(TILE_LABELS), "Every tile label needs a character to be written as, missing %s." % str(sorted(set(TILE_LABELS) - set(LABEL_CHARACTERS)))
assert len(set(LABEL_CHARACTERS.values())) == len(LABEL_CHARACTERS), "Two tile labels share a character, so an encoded grid could not be read back."

# A belief sheaf's stalk holds, per tile, the set of labels an agent still considers possible,
# which is more than a labeling can say: a tile can have emptied under disagreement, or have
# several labels left. Those two get characters of their own, so a stalk writes as one string
# per row exactly like a labeling does.
CONTESTED_CHARACTER = "x"     # the possibility set has emptied: the network contradicted itself here
SEVERAL_CHARACTER = "+"       # more than one label still possible, but not all of them

assert not set(LABEL_CHARACTERS.values()) & {CONTESTED_CHARACTER, SEVERAL_CHARACTER}, "A possibility character collides with a label character, so an encoded stalk could not be read back."


def encode_label_grid(tile_labels: dict[tuple[int, int], str], grid_width_height: tuple[int, int]) -> list[str]:
    """
    Encodes a labeling of every tile as one string per grid row, each character a tile.

    Rows come highest row first, so that the strings read down the page the way the grid is
    drawn on the figure. A 9x5 grid becomes five 9-character strings, which keeps a labeling
    recorded at every step both small and legible.

    Args:
        tile_labels (dict[tuple[int,int], str]): The label of each (X,Y) tile.
        grid_width_height (tuple[int, int]): The width and height of the grid.
    """
    grid_width, grid_height = grid_width_height
    return [
        "".join(LABEL_CHARACTERS.get(tile_labels[(x, y)], "?") for x in range(grid_width))
        for y in range(grid_height - 1, -1, -1)
    ]


def decode_label_grid(encoded_rows: list[str]) -> dict[tuple[int, int], str]:
    """
    Decodes what encode_label_grid wrote, back into a label for each (X,Y) tile.

    Args:
        encoded_rows (list[str]): One string per grid row, highest row first.
    """
    characters_to_labels = {character: label for label, character in LABEL_CHARACTERS.items()}
    grid_height = len(encoded_rows)

    tile_labels = {}
    for row_index, row in enumerate(encoded_rows):
        y = grid_height - 1 - row_index
        for x, character in enumerate(row):
            assert character in characters_to_labels, "Failed to decode label grid. The character %r is not one of %s." % (character, str(sorted(characters_to_labels)))
            tile_labels[(x, y)] = characters_to_labels[character]
    return tile_labels


def encode_possibility_grid(
    possibilities: dict[tuple[int, int], tuple[str, ...]],
    grid_width_height: tuple[int, int]
) -> list[str]:
    """
    Encodes one belief-sheaf stalk as one string per grid row, the way
    encode_label_grid encodes a labeling: rows highest first, one character a tile.

    A stalk says, per tile, which labels an agent still considers possible
    (`gridsheaf.mask_labels`). A tile the agent leaves free is absent from the
    mapping and writes as "unknown"; a tile pinned to one label writes as that
    label; a tile whose set has emptied under disagreement writes as
    CONTESTED_CHARACTER, and one with several labels left as SEVERAL_CHARACTER.

    The last two are what a labeling cannot express, and so what makes the record
    of a sweep more than a record of what the agents would draw: the keep-own
    policy hides a contested tile behind the label the agent kept.

    Args:
        possibilities (dict[tuple[int,int], tuple[str, ...]]): The labels each
            constrained (X,Y) tile still admits. Tiles absent are unconstrained.
        grid_width_height (tuple[int, int]): The width and height of the grid.
    """
    grid_width, grid_height = grid_width_height

    def character(tile: tuple[int, int]) -> str:
        labels = possibilities.get(tile)
        if labels is None or set(labels) == set(LABEL_CHARACTERS) - {"unknown"}:
            return LABEL_CHARACTERS["unknown"]
        if not labels:
            return CONTESTED_CHARACTER
        if len(labels) == 1:
            return LABEL_CHARACTERS.get(labels[0], "?")
        return SEVERAL_CHARACTER

    return [
        "".join(character((x, y)) for x in range(grid_width))
        for y in range(grid_height - 1, -1, -1)
    ]


def encode_sheaf_state(state, grid_width_height: tuple[int, int],
                       contracts: Optional[dict] = None) -> dict:
    """
    Encodes one snapshot of a belief sheaf -- a `gridsheaf.SheafState` -- as the
    plain JSON-able dict the log carries.

    Takes the snapshot by its fields rather than by its type, so this module stays
    ignorant of the sheaf machinery the way it is of where conflicts come from.

    The stalks are written as grids, the sheaf-theoretic read-outs as flags: which
    interfaces have closed (both endpoints agreeing on their shared edge stalk),
    whether all of them have and the assignment is therefore a global section,
    whether each agent's knowledge still refines what it started with, and whether
    any stalk has collapsed to a lattice extreme -- which, disagreement being
    recorded in-lattice as an emptied possibility set, should never happen.

    Given contracts, each local section is recorded as the assume/guarantee pair it
    is as well as the possibility grid it means -- see `gridsheaf.render_contracts`,
    which is where the rendering and its two surprises live.

    Args:
        state (gridsheaf.SheafState): The snapshot to encode.
        grid_width_height (tuple[int, int]): The width and height of the grid.
        contracts (Optional[dict]): Per agent, the stalk contract as lines of text.
            Taken already rendered, so this module stays as ignorant of the contract
            algebra as it is of where conflicts come from.
    """
    encoded = {
        "stalks": {str(agent): encode_possibility_grid(possibilities, grid_width_height)
                   for agent, possibilities in state.possibilities.items()},
        "contested": {str(agent): [list(tile) for tile in tiles]
                      for agent, tiles in state.contested.items() if tiles},
        "sections": {"%s-%s" % interface: bool(agrees)
                     for interface, agrees in state.sections.items()},
        "is_section": bool(state.is_section),
        "refines_initial": {str(agent): bool(refines)
                            for agent, refines in state.refines_initial.items()},
        "collapsed": {str(agent): how for agent, how in state.collapsed.items()},
    }

    # The region sheaf's snapshot carries more: which regions agree on each interface
    # (agreement is measured in the interface alphabet, so this is the localisation the
    # abstraction buys), the claims picture, and the tile-level disagreement each
    # interface tolerates -- nonzero alongside is_section is the abstraction working.
    # Read by presence, so a legacy SheafState encodes exactly as before.
    if getattr(state, "section_regions", None) is not None:
        encoded["section_regions"] = {
            "%s-%s" % interface: {region: bool(agrees)
                                  for region, agrees in per_region.items()}
            for interface, per_region in state.section_regions.items()}
    if getattr(state, "contested_regions", None) is not None:
        encoded["contested_regions"] = {
            str(agent): {region: list(labels) for region, labels in per_region.items()}
            for agent, per_region in state.contested_regions.items() if per_region}
    if getattr(state, "claims", None) is not None:
        encoded["claims"] = {str(agent): entry
                             for agent, entry in state.claims.items()}
    if getattr(state, "concrete_disagreement", None) is not None:
        encoded["concrete_disagreement"] = {
            "%s-%s" % interface: int(count)
            for interface, count in state.concrete_disagreement.items()}

    if contracts:
        encoded["contracts"] = {str(agent): lines for agent, lines in contracts.items()}
    return encoded


class ExperimentLog:
    """
    Collects a run as it happens and writes it out as one JSON file.

    The file holds what was configured, what the world truly was, what each agent believed and
    where each agent went, so that a run can be reconstructed and compared against another
    without the video. Beliefs are recorded at every step, which is what makes a communicating
    run legible: they change as the agents share what they know along the topology.

    A run that communicates records the sheaf that carried it too -- its shape once, and then
    every Laplacian iterate as it is made, each with the sheaf-theoretic read-outs of where the
    flow has got to. What the agents believe is only the projection of that state a labeling
    can hold; the iterates are the state itself.

    One log covers the whole run rather than the view that was drawn: every agent's labeling is
    recorded alongside the ground truth, whichever single view the video happened to show.
    """

    def __init__(
        self,
        path: Union[str, Path],
        config: dict,
        grid_width_height: tuple[int, int],
        number_of_robots: int,
        record_poses: bool = True
    ):
        """
        Args:
            path (str | Path): Where the JSON file is written, on close().
            config (dict): The merged configuration the run was given, defaults included, so
                that the record says what actually ran rather than what one file asked for.
            grid_width_height (tuple[int, int]): The width and height of the grid.
            number_of_robots (int): The number of robots in the simulation.
            record_poses (bool): Whether to record the robots' continuous poses every frame.
                These outnumber the grid steps by about a hundred to one, so they are the bulk
                of the file; turning them off leaves a small step-by-step record.
        """
        self.path = Path(path)
        self.grid_width_height = grid_width_height
        self.number_of_robots = number_of_robots
        # Not named record_poses: that is the method below, and an attribute would shadow it
        self.recording_poses = record_poses

        self._record = {
            "experiment": {
                "name": config.get("experiment_name"),
                "seed": config.get("seed"),
                "grid_width": grid_width_height[0],
                "grid_height": grid_width_height[1],
                "num_agents": number_of_robots,
            },
            "config": config,
            "label_characters": LABEL_CHARACTERS,
            "row_order": "highest row first, matching the figure",
            "steps": [],
            "poses": [],
        }

    def record_setup(
        self,
        assigned_targets: np.ndarray,
        ground_truth: Optional[dict] = None,
        topology=None,
        videos: Optional[list[dict]] = None
    ) -> None:
        """
        Records what the run was set up with, before any stepping.

        A video shows one view, and a run may write several of them, but the log is of none of
        them: every side's labeling is recorded at every step. Which video shows which view is
        noted only to say what those files contain.

        Args:
            assigned_targets (np.ndarray): A 2xN array of the square assigned to each robot.
            ground_truth (Optional[dict]): The world's true labeling, if one was configured.
            topology (Optional[nx.Graph]): The communication topology, if one was configured.
            videos (Optional[list[dict]]): One {"filename", "view"} entry per video written,
                in the order they were written, so a video can be told from its siblings.
        """
        self._record["experiment"]["videos"] = videos or []
        self._record["assignments"] = _coordinates_to_pairs(assigned_targets)
        self._record["ground_truth"] = None if ground_truth is None else encode_label_grid(ground_truth, self.grid_width_height)
        self._record["communication"] = None if topology is None else {
            "edges": sorted(sorted(edge) for edge in topology.edges),
            "nodes": sorted(topology.nodes),
        }

        # Every side whose labeling this log carries, the drawn view notwithstanding
        self._record["sides"] = ([] if ground_truth is None else [GROUND_TRUTH_VIEW]) + list(range(self.number_of_robots))

    def record_sheaf(
        self,
        interfaces: list,
        stalk_alphabet: str,
        legs: str,
        sweeps_per_step: int | str,
        initial_state: Optional[dict] = None,
        regions: Optional[dict] = None
    ) -> None:
        """
        Records the belief sheaf a run communicated over: its shape, and the assignment it
        started from, before any sweep.

        The base graph is the communication topology already recorded by record_setup; what
        this adds is the sheaf structure over it -- what a stalk holds, which interfaces the
        restrictions run onto, which bisheaf transport carries data along -- and the initial
        0-cochain, so that the iterates recorded per step have a first term to be read
        against. A run with no sheaf never calls this and the log carries no "sheaf" key.

        Args:
            interfaces (list): The interfaces the restrictions run onto, as [u, v] pairs.
            stalk_alphabet (str): What one stalk is over, in words, so the record says what
                the grids under "stalks" are grids of.
            legs (str): Which adjoint bisheaf transport used, "kan" or "co".
            sweeps_per_step (int | str): How many Laplacian sweeps one grid step ran, as it
                was configured: a number of hops, or "converge" for however many the flow
                needed to reach its fixed point, which the per-step iterates then say.
            initial_state (Optional[dict]): The sheaf before the first sweep, as
                encode_sheaf_state writes it.
            regions (Optional[dict]): The region partition of an abstraction-level
                sheaf, as {name: [[x, y], ...]} -- the fibers of the restriction
                maps, without which the per-region read-outs in the iterates name
                regions the log does not define. A legacy run passes nothing and
                the record carries no "regions" key.
        """
        self._record["sheaf"] = {
            "interfaces": [list(interface) for interface in interfaces],
            "stalk_alphabet": stalk_alphabet,
            "legs": legs,
            "sweeps_per_step": sweeps_per_step,
            "initial": initial_state,
        }
        if regions is not None:
            self._record["sheaf"]["regions"] = {
                str(name): [list(tile) for tile in tiles]
                for name, tiles in regions.items()}

    def record_step(
        self,
        step_index: int,
        robot_coords: np.ndarray,
        step_scores: Optional[np.ndarray] = None,
        cumulative_scores: Optional[np.ndarray] = None,
        reached_target: Optional[np.ndarray] = None,
        tile_labels_by_agent: Optional[dict] = None,
        conflicts: Optional[list] = None,
        mission: Optional[dict] = None,
        sweeps: Optional[list] = None,
        observations: Optional[list] = None,
        settled: Optional[bool] = None,
        moved: Optional[list] = None,
        tick: Optional[int] = None,
        view: Optional[str] = None
    ) -> None:
        """
        Records one grid step: where every robot moved to, what that was worth, and what every
        robot believed at the time.

        Args:
            step_index (int): Which step this is, counting from 0.
            robot_coords (np.ndarray): A 2xN array of the robots' new (X,Y) coordinates.
            step_scores (Optional[np.ndarray]): What each robot scored on this step.
            cumulative_scores (Optional[np.ndarray]): What each robot has scored so far.
            reached_target (Optional[np.ndarray]): Whether each robot has reached a target yet.
            tile_labels_by_agent (Optional[dict]): Each robot's labels, keyed by robot index.
            conflicts (Optional[list]): Belief conflicts surfacing on this step, as plain
                dicts -- the caller converts whatever richer records it holds, so the log
                stays ignorant of where they came from.
            mission (Optional[dict]): Per-agent mission diagnostics for this step (e.g.
                corridor relied on, tiles the fused knowledge rules out), as plain dicts.
            sweeps (Optional[list]): The Laplacian iterates this step ran, in order, one entry
                per sweep. Each is a snapshot of the sheaf after that sweep as
                encode_sheaf_state writes it, plus which sweep of the run it was -- counted
                from 1, so that it names the same iterate a video's caption does -- and
                whether it changed anything. This is what lets the flow be replayed hop by
                hop, rather than only seen through the beliefs it left behind.
            observations (Optional[list]): What each agent saw for itself on the tile it
                stepped onto, as gridsheaf.observe_tiles returns it, with whether that was
                news to it. These are the entries into the run's knowledge that did not come
                from the flow, so a log that has them can tell what an agent was told from
                what it went and found out.
            settled (Optional[bool]): Whether the flow had reached its fixed point by the end
                of this step. The flow alone is monotone, so without observations this stays
                true once true and no later step sweeps; an observation is an outside input
                and reopens it.
            moved (Optional[list]): Which agents were planned and set moving on this step.
                Absent means all of them, which is what a synchronous run does and what every
                run recorded before agents could move independently. Present, the record is of
                a moment rather than of a round: the positions are where every agent was or was
                heading then, and only the listed agents had just been given a new tile.
            tick (Optional[int]): The control iteration this step was planned on, which is what
                orders the records of a run whose steps no longer happen in lockstep. 0.033 s
                each, and directly comparable with the `step` field of the pose samples.
            view (Optional[str]): Which view the projection was showing from here, as
                `gridworld.describe_view` names it. The same moment drawn for two different
                agents is two different pictures and nothing in the colours says which is on
                screen, so a run whose view moves is only readable against the footage with this.
        """
        step_record = {
            "step": step_index,
            "positions": _coordinates_to_pairs(robot_coords),
        }
        if step_scores is not None:
            step_record["step_scores"] = [float(score) for score in step_scores]
        if cumulative_scores is not None:
            step_record["cumulative_scores"] = [float(score) for score in cumulative_scores]
        if reached_target is not None:
            step_record["reached_target"] = [bool(flag) for flag in reached_target]
        if tile_labels_by_agent is not None:
            step_record["beliefs"] = {
                str(robot_index): encode_label_grid(tile_labels_by_agent[robot_index], self.grid_width_height)
                for robot_index in range(self.number_of_robots)
            }
        if conflicts:
            step_record["conflicts"] = conflicts
        if mission:
            step_record["mission"] = mission
        if sweeps:
            step_record["sweeps"] = sweeps
        if observations:
            step_record["observations"] = observations
        if settled is not None:
            step_record["settled"] = bool(settled)
        if moved is not None:
            step_record["moved"] = [int(robot_index) for robot_index in moved]
        if tick is not None:
            step_record["tick"] = int(tick)
        if view is not None:
            step_record["view"] = str(view)

        self._record["steps"].append(step_record)

    def record_poses(self, step_index: int, robot_poses: np.ndarray) -> None:
        """
        Records where the robots physically are on one simulation frame, which is what the grid
        steps above leave out: a step says which tile a robot ended on, this says how it drove
        there. Does nothing when record_poses was turned off.

        Args:
            step_index (int): The grid step this frame falls inside.
            robot_poses (np.ndarray): A 3xN array of the robots' (x, y, theta) poses.
        """
        if not self.recording_poses:
            return

        self._record["poses"].append({
            "step": step_index,
            "pose": [[round(float(value), 4) for value in robot_poses[:, i]] for i in range(robot_poses.shape[1])],
        })

    def close(
        self,
        final_scores: Optional[np.ndarray] = None,
        steps_taken: Optional[int] = None,
        all_arrived: Optional[bool] = None,
        projection_failures: Optional[dict] = None
    ) -> None:
        """
        Records how the run ended and writes the file.

        Args:
            final_scores (Optional[np.ndarray]): Each robot's score over the whole run.
            steps_taken (Optional[int]): How many grid steps the run took.
            all_arrived (Optional[bool]): Whether every robot reached its assigned square.
            projection_failures (Optional[dict]): Anything that went wrong in drawing the run,
                keyed by the piece that failed. A run whose drawing broke is still a run --
                nothing here is computed from what was drawn -- but it is one whose video shows
                less than its numbers do, and a record that did not say so would let the two be
                read as if they agreed. Absent, as it is on every run that drew cleanly, the
                record carries no such key.
        """
        self._record["result"] = {
            "steps_taken": steps_taken,
            "all_arrived": all_arrived,
            "final_scores": None if final_scores is None else [float(score) for score in final_scores],
            "total_score": None if final_scores is None else float(np.sum(final_scores)),
            "frames": len(self._record["poses"]),
        }
        if projection_failures:
            self._record["projection_failures"] = {
                what: dict(failure) for what, failure in projection_failures.items()
            }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as log_file:
            json.dump(self._record, log_file, indent=2)


def _coordinates_to_pairs(coordinates: np.ndarray) -> list[list[int]]:
    """
    Turns a 2xN array of grid coordinates into a list of [x, y] pairs, one per robot, which is
    how they are written in a configuration and so how they are easiest to read back.

    Args:
        coordinates (np.ndarray): A 2xN numpy array of (X,Y) coordinates.
    """
    return [[int(coordinates[0, i]), int(coordinates[1, i])] for i in range(coordinates.shape[1])]
