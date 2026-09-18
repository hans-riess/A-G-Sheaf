# Getting started

## Requirements

* **Python 3.11** — exactly. `pyproject.toml` declares `requires-python = ">=3.11,<3.12"`.
  The pin is not decoration: the Robotarium simulator submodule is developed
  against that interpreter.
* **git**, with SSH access to GitHub for the submodule
  (`git@github.com:robotarium/robotarium_python_simulator.git`).
* **ffmpeg on `PATH`**, only if you want videos. `agsheaf.utils.VideoSaver` pipes
  frames to it and raises a clear error if it is missing. Everything else runs
  without it.
* A working Qt or other Matplotlib backend if you want a live figure window.
  Headless runs (`gridworld.show_figure: false`) need none.

`z3-solver` is a hard dependency and is installed for you. The contract layer is
Z3 all the way down.

## Install

```bash
git clone https://github.com/hans-riess/agsheaf.git
cd agsheaf
python3.11 bootstrap.py
source .venv/bin/activate
```

`bootstrap.py` does four things, in order:

1. `git submodule update --init --recursive`, which fetches the Robotarium
   simulator into `libs/robotarium`.
2. creates `.venv` if it is not there,
3. upgrades `pip`,
4. installs **both** packages editable — this project, and
   `libs/robotarium` — with `--config-settings editable_mode=compat`.

Both installs are needed. `agsheaf.gridworld` imports `rps.robotarium` at module
import time, so without the submodule installed most of the demonstration half
of the package will not import at all. (The library half —
`contracts`, `sheaf`, `product`, `simplify`, `measure`, `diagnostics`,
`ensembles`, `regions` — deliberately does not import it, so it is usable on its
own.)

On Windows `bootstrap.py` picks `.venv\Scripts\python.exe` instead; activate with
`.venv\Scripts\activate`.

### Checking the install

```bash
pytest
```

The suite is the acceptance test for an install. It runs headless, needs no
ffmpeg, and exercises the Robotarium physics in `test/test_motion.py`, so a
green run means both halves imported and both work. CI runs exactly this on
Ubuntu with Python 3.11 (`.github/workflows/pytest.yml`).

## First run

```bash
agsheaf sims
```

runs `exp/worlds/primary.yaml`: four agents on a 12x8 grid partitioned into six
4x4 rooms, moving asynchronously on a random schedule, communicating over a
4-node path topology by sweeping the mission sheaf's Tarski Laplacian, and
replanning when the flow tells an agent its route assumption has been violated.
It opens a figure window, writes a video per view into `videos/` and a JSON log
into `logs/`.

Some useful variations:

```bash
agsheaf sims --world scaled              # the same design at eight agents
agsheaf sims --no-com                    # control arm: nobody communicates
agsheaf sims --no-abstraction            # control arm: tile-by-tile fusion
agsheaf sims --num-agents 8              # generate a world at that size
agsheaf sims --set gridworld.show_figure=false   # headless
```

To see what a run will look like without running it:

```bash
agsheaf preview                          # writes exp/previews/screen.png and floor.png
```

Full command reference: [cli.md](cli.md).

## What a run produces

| path | what |
|---|---|
| `logs/<timestamp>.json` | the whole run: config, ground truth, every step, every Laplacian iterate, poses. See [logs.md](logs.md). |
| `videos/<timestamp>.mp4` | the figure, encoded H.264 so it plays inline in a browser. `video.one_per_view: true` writes `<timestamp>-1.mp4`, `-2.mp4`, … one per configured view. |
| `exp/previews/screen.png`, `floor.png` | `agsheaf preview` output. |
| `exp/benchmarks/*.json` | the measurement campaigns (`converge`, `benchmark.py`, `validate_theory.py`, `scale_example.py`). |
| `exp/robotarium/builds/<arm>/` | a Robotarium submission bundle. |

All of these directories are gitignored.

## Using the library on its own

You do not need the Robotarium, the gridworld, or any configuration to use the
contract-sheaf machinery:

```python
import z3
from agsheaf.contracts import Contract, Relation, top, bot
from agsheaf.sheaf import ContractSheaf, round_robin, random_firing
from agsheaf.measure import Alphabet, distance, canonical
from agsheaf.diagnostics import Alphabets, dirichlet
```

None of those modules import `agsheaf.gridworld`, and therefore none of them
import `rps`. [library.md](library.md) is the reference.

## Troubleshooting

**`ModuleNotFoundError: No module named 'rps'`** — the submodule was not
initialised or not installed. Re-run `python3.11 bootstrap.py`, or
`pip install -e ./libs/robotarium` by hand.

**`Permission denied (publickey)` during bootstrap** — the submodule URL is
SSH. Either add a GitHub SSH key, or rewrite the URL:
`git config submodule."libs/robotarium".url https://github.com/robotarium/robotarium_python_simulator.git`
and re-run `git submodule update --init --recursive`.

**`ffmpeg not found on PATH`** — install ffmpeg, or turn video off. There is no
`video.enabled` key; a run writes a video when it draws a figure, so
`--set gridworld.show_figure=false` is the way to run without one.

**A run stalls and stops with "no agent has completed a move in N control
iterations"** — the arena is jammed. Most often too many obstacles in a
generated world (see `CROWDED_OBSTACLES` in `agsheaf/worldgen.py`: above three,
deadlock has been measured), or repulsion distances too large for the tile
width. `GridWorld._assert_grid_is_reachable` catches the second case at
construction when it can.

**`Failed to create GridWorld. A robot counts as arrived within … m of a tile
centre, but tiles are only … m across`** — the grid is finer than the arrival
radius allows. Lower `gridworld.arrival_distance`, or use a coarser grid. The
repulsion distances usually need scaling down with it; see the long note in
`GridWorld.__init__`.

**`z3 returned unknown`** (an `agsheaf.contracts.Undecided`) — the solver could
not decide a quantified query. This is reported rather than silently treated as
"false", because a solver limitation reported as a negative answer would make
`is_section` say "not a section" and `converge` burn its sweep budget blaming
divergence. If you hit it in your own sheaf, canonicalise between sweeps
(`measure.canonical`) or eliminate quantifiers (`simplify.qf`) — see
[correctness.md](correctness.md#formula-growth).

**`LivenessError: … never fired in N steps`** — the firing sequence starved a
node, or `max_sweeps` was too small to cover a full round. From inside a
truncated run the two are indistinguishable, which is why the flow refuses to
report a fixed point rather than guessing.
