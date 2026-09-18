# `exp/worlds/`

One file per **world**: the grid, the region partition, who starts where, who is assigned where,
what is truly hazardous, who believes what, and who can talk to whom. Nothing else.

    agsheaf sims --world scaled
    agsheaf preview --world primary
    agsheaf robots --world scaled --dry-runs 1

`--world` takes the file name without the `.yaml`; a path still works via `--experiment` (or as
`agsheaf sims exp/worlds/scaled.yaml`), and `python exp/run.py exp/worlds/scaled.yaml`
does the same thing without the CLI.

| world | what it is |
|---|---|
| `primary.yaml` | the demonstration of the MS4 report, four agents. The default when no world is named. |
| `variation.yaml` | a second four-agent instance of the same design, on different geometry. Measured numbers are in its footer. |
| `scaled.yaml` | the same design scaled to eight agents, chain diameter 7. Measured numbers are in its footer. |

## What is deliberately not here

A world says *what is true and who believes it*. Three other things shape a run, and they live
apart from it on purpose, because each varies independently of the world:

- **`exp/defaults.yaml`** — how a run is conducted, for every world: tile costs, scoring, video,
  logging, rendering. A world overrides a key here only when the world is the reason.
- **control arms** — the ablations, as flags: `agsheaf sims --world primary --no-com`. They are
  defined once in `src/agsheaf/cli.py` (`ARMS`) rather than as files, so the arm and the world
  can be combined freely without a file per pair.
- **surfaces** — `exp/robotarium/config.yaml` is what the testbed's projector wants differently
  from a monitor, applied over any world by `--floor` or by `agsheaf robots`.

That split is why an arm is not a world and a world is not an arm: three worlds times three arms
times two surfaces is eighteen runs out of three files, three flags and one layer.

## Adding one

Copy the closest existing world, change what differs, and record what it measures in a footer
comment the way `variation.yaml` does — a world whose numbers are written down is one a later
edit can be told apart from a regression. Generated worlds
(`agsheaf sims --num-agents 8 --num-obstacles 2`) are not written here; see
`src/agsheaf/worldgen.py`, including its KNOWN LIMITATION.
