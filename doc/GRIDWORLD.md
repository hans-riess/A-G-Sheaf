# Gridworld experiment configuration

Reference for the YAML that drives `exp/run.py`. **Every key is documented here**, plus the
`robotarium` block of `exp/robotarium/config.yaml`; the YAML itself carries no explanation, so
this file is the only account of what an option does.

## One runner, three arms

`exp/run.py` is the single experiment runner, and the configuration decides which of three arms
it runs:

| Arm | Selected by | What it is |
|---|---|---|
| **abstraction sheaf** | a `regions:` block | The demonstration: agent stalks speak per tile, interfaces speak per region, mission contracts carry assumptions and claims, and agreement is measured in the interface vocabulary. `src/agsheaf/regionsheaf.py`. |
| **legacy belief sheaf** | no `regions:` block | The f = id control arm: bijective restrictions, tile-by-tile fusion. The delivered pre-abstraction demonstration, kept verbatim. `src/agsheaf/gridsheaf.py`. |
| **no communication** | `sheaf.enabled: false` | Nobody talks; every agent plans on its own beliefs forever. Identical world, planner, scoring and rendering, so runs differ from the other arms in exactly one thing. |

## The two files

Configuration is split in two, loaded by `load_experiment_config` in `src/agsheaf/utils.py`:

| File | Holds |
|---|---|
| `exp/defaults.yaml` | Everything not specific to one experiment: the testbed geometry, colors, video settings, tile costs and scores, the contract policy. |
| `exp/worlds/primary.yaml` | One experiment: the world, the region partition, the agents' beliefs, the assignments, and what to draw. |

A Robotarium submission adds a third layer on top of these two, `exp/robotarium/config.yaml`,
holding only what the testbed needs differently — see [`robotarium`](#robotarium) below.

They are merged **section by section**, with the experiment file winning. Naming one tile cost
keeps the other three; lists are replaced outright rather than concatenated. A key that is present
but blank still wins — a bare `seed:` is an explicit request for a random seed, not a missing entry.

Run an experiment with:

```bash
.venv/bin/python exp/run.py
```

An alternative experiment file may be named on the command line — `exp/run.py my_world.yaml` —
which is how the control arms run without editing the main configuration; it replaces
`exp/worlds/primary.yaml` in the merge and the log records whichever file ran.

## Drawing a frame without a run

What the configuration draws, as a single PNG, with nothing driving behind it:

```bash
.venv/bin/python exp/preview_frame.py            # the screen figure, as exp/run.py draws it
.venv/bin/python exp/preview_frame.py --floor    # what the testbed projects, over exp/robotarium/config.yaml
```

The robots stand on their starting squares and the sheaf is swept `--sweeps` times before the
frame is drawn, so the tiles, the contested marks and the interface states are a moment the run
passes through rather than a mock-up of one. `--view` picks whose beliefs to draw. Files land in
`exp/previews/`.

It also reports where the caption ended up, in metres of arena — clear of the grid, over it, or
off the edge. That is the one piece of the floor layout that depends on the words in the caption
rather than on the settings alone: the margin beside a 12×8 grid is 0.4 m wide, and a comparison
view is three times the length of a single one.

## How a tile is specified

Every tile is an `[x, y]` pair, and anywhere tiles are expected you may give one pair or a list of
them. `(0, 0)` is the bottom-left tile.

```yaml
unsafe: [[3, 0], [3, 1], [3, 2]]
target: [[5, 3]]
```

There is no range or rectangle shorthand — a column of five tiles is written as five pairs. The
one exception is the `regions:` block, whose entries may be inclusive rectangles; see below.

## `config.yaml` — the experiment

### Top level

| Key | Meaning |
|---|---|
| `experiment_name` | Recorded in the log as `experiment.name`. Defaults to the name of the script that ran (`run`). Set it to name a variation — the control arms are the natural users. |
| `seed` | Seeds the RNG that places robots at the start, when `starts` does not name them. Leave blank for a different layout each run; set an integer to make runs comparable frame for frame. |

### `gridworld`

| Key | Meaning |
|---|---|
| `num_agents` | Number of robots. |

Grid dimensions and the potential-field tuning live in `defaults.yaml`, since they are properties
of the testbed rather than of an experiment.

### `goals`

| Key | Meaning |
|---|---|
| `max_steps` | Cap on grid steps. A run ends when every agent has reached its assigned square, or when this cap is hit — which is what catches an agent that cannot get through to its square. |

### `starts`

The square each agent **begins on**, one per agent index. Optional: leave the block out and the
layout is sampled from `seed` instead, which is how every run worked before the block existed.

```yaml
starts:
  0: [3, 1]
  1: [4, 2]
  2: [8, 3]
  3: [0, 3]
```

Same rules as `assignments` — every agent needs exactly one square, and no two may share, since
two robots cannot occupy one tile. Unlike an assignment, a start may be any square at all: one the
ground truth calls unsafe, or an agent's own assigned square, in which case that agent has simply
arrived already.

Naming them is what a demo wants. A seed hands out every square at once, so it can only be shopped
for — there is no way to ask a seed for one agent starting far from its target and another already
close. Nothing else in the experiments draws from the RNG, so a run that names its starts and one
that samples them differ in nothing but the starts, and the same `starts` block gives the same
layout under any seed.

### `assignments`

The square each agent is **told** to reach, one per agent index. This is what drives the movement.

```yaml
assignments:
  0: [6, 4]
  1: [5, 4]
  2: [6, 3]
  3: [5, 3]
```

Checked by `parse_assignments`:

- Every agent needs exactly one square.
- No two agents may share a square. Two robots cannot occupy one tile, so a shared assignment is a
  run that can never finish.
- Each square must be inside the grid.
- When a `ground_truth` is configured, each assigned square must genuinely be a target
  (`assert_assignments_are_targets`).

An assignment is fixed for the whole run. Agents drive to their squares once; there are no rounds.

### `communication`

Which agents can talk to each other, as an undirected edge list. The block is optional; leaving it
out means no topology and nothing drawn.

```yaml
communication:
  edges: [[0, 1], [1, 2], [2, 3], [3, 0]]
```

Parsed by `parse_communication_topology` into a `networkx.Graph`. Every agent becomes a node whether
or not it has an edge, so an agent that can talk to nobody is an *isolated node* rather than absent.
Being undirected, `[0, 1]` and `[1, 0]` are the same edge and repeating either is not an error.
Self-loops and out-of-range indices are rejected.

**What the topology does depends on `sheaf.enabled`.** With the sheaf off it is declared, drawn
and logged but nothing reads it: adding or removing edges changes only the picture and the log.
With it on, it is the base graph of the mission sheaf, so an edge is an interface along which
knowledge — and, under a `regions:` partition, route claims — actually moves, one hop per sweep,
and the topology's diameter is how many sweeps the flow takes to settle.

It is drawn as a line between each connected pair, running between where the robots actually are
rather than between tile centres, so the lines track the robots as they drive. In a communicating
run the line is colored by whether that interface agrees yet; otherwise it is a faint dashed line.
Styling lives under `rendering` in `defaults.yaml`.

### `beliefs`

#### Choosing the view

A **view** is what gets drawn, and the saved video is named after it. It is either one *side* or a
pair of sides to compare, where a side is an agent index or `ground_truth`.

```yaml
view: ground_truth        # the world as it really is
view: 0                   # agent 0's own labels
view: [0, 1]              # two agents side by side: where they agree, where they conflict
view: [ground_truth, 0]   # agent 0 held against the truth: where its beliefs are wrong
```

To flip through several views in one video, replace `view` with `views` and say how often to
advance. The two keys are mutually exclusive.

```yaml
views: [ground_truth, 0, [0, 1]]
switch_view_every: 20     # 'never', or a number of grid steps
```

To point the projection at whichever agent just set off instead of advancing on a count, set
`follow_moving_agent`. It picks the view by the moving agent's index, so `views` needs one entry
per agent; with several departing at once the first of them has it, which makes it worth
configuring only for an `asynchronous_moves` run, where they depart one at a time.

```yaml
views: [[ground_truth, 0], [ground_truth, 1], [ground_truth, 2]]
follow_moving_agent: true
```

While an agent drives into a square the floor then shows that agent's beliefs against the truth,
including about the square it is driving into. Each step record carries a `view` field naming what
was on the floor from there, since nothing in the colours themselves says whose beliefs they are.

`switch_view_every` accepts only `never` or an integer. (It once accepted `goal`, which no longer
means anything now that a run is a single pass.) Note the singular *view*, against the plural in
`tour_views_every` below — the two are each other's likeliest misspelling, so an entry the block
does not recognise is now rejected rather than silently ignored.

The view advances at the *end* of a step, once the robots have driven, so everything drawn for one
step is drawn in one view. Under `asynchronous_moves` a step is a single agent claiming a tile
rather than a round of them, so `switch_view_every: 1` advances the view **every time an agent
moves**. Paired with one view per agent it walks through the agents as they go:

```yaml
views: [[ground_truth, 0], [ground_truth, 1], [ground_truth, 2]]
switch_view_every: 1
```

Each of those is an agent held against the truth, so under the default comparison colours the
floor marks exactly the tiles that agent is wrong about, and the marks move from agent to agent
as the agents do.

#### Showing every view of the same moment — `tour_views_every`

`switch_view_every` shows one view at a time, so no single video ever holds two views of the same
moment of the run. A **tour** does: every so many grid steps it stops the robots and holds on each
configured view in turn, in the order `views` lists them, before anyone moves.

```yaml
views: [ground_truth, 0, 1, 2, 3]
tour_views_every: 1       # 'never', or a number of grid steps
```

That is what a demo filmed in one take needs — the ground truth followed by what each agent makes
of it, at every step. It is not free: a tour costs `len(views) × video.frames_per_view` frames, so
five views at the default 30 frames adds five seconds of video *per tour*. Raise `tour_views_every`
on a long run. On the Robotarium the dwell is `robotarium.hold_steps_per_view` instead, in control
iterations of 0.033 s, and the same budget applies against the 600 s cap.

A tour is drawn after that step's Laplacian sweeps and before its plan, so it shows the state the
step is about to be planned from. With `video.one_per_view` each video stays pinned to its own view
and simply holds there, so a tour leaves the set frame-for-frame aligned.

#### Learning from where you have been — `observe_on_arrival`

Off by default. With it on, each agent sees the ground truth of the square it has just stepped onto
and writes it down — that square only, nothing it merely routed past.

```yaml
observe_on_arrival: true
```

What an agent sees for itself **overrides** what it was told, so this is the one thing in a run that
is not a step of the flow, and it has two consequences worth knowing before reading a log:

- It can *weaken* a stalk — replacing one label with another, or giving a contested tile a label
  back — so `mission.<i>.refines_initial` may read `false` afterwards. That is the record of an
  observation having contradicted the network, not a broken invariant.
- A fixed point stops being final. An observation that changed something clears the settled latch,
  so `settled` can go back to `false` and later steps sweep again.

Where an observation contradicts a neighbour, both keep their own label and the tile reads as
contested (`×`) at both, until the neighbour goes and looks for itself. In a run with no sheaf
(`sheaf.enabled: false`) what an agent learns simply stays with it.

#### What an agent believes — `beliefs.agents`

Per agent, a list of tiles for each label. Labels are applied in the order written, so a later one
wins on an overlapping tile.

```yaml
agents:
  0:
    target: [[5, 4], [0, 2]]
    safe: [[0, 0], [3, 4]]
    unsafe: [[5, 2], [6, 2]]
```

The `target` list is the agent's **guesses about where the other agents were sent** — some right,
some wrong, which is the point. An agent's own assigned square is added automatically and does not
need restating here.

Tiles an agent is not given a belief about take `beliefs.default` (see `defaults.yaml`), normally
`unknown`. Note that `unknown` is a real label: a tile an agent does not believe is unsafe is *not*
thereby believed safe.

### `ground_truth`

The true labeling of the world, which the agents hold beliefs about. Unlike a belief, the truth is
total: every tile is `safe` or `unsafe` apart from the targets, so `unknown` is rejected here.

```yaml
ground_truth:
  default: safe
  unsafe: [[3, 0], [3, 1], [3, 2], [3, 3], [3, 4]]
  target: [[5, 3], [6, 3], [5, 4], [6, 4]]
```

Targets need not sit together or form any particular shape — scattered targets are fine. What each
robot must reach is settled by `assignments`, and each assignment is checked to be a true target, so
nothing downstream cares how the targets are arranged. The only requirement is that at least one
tile is a target.

### `regions`

The region partition: named sets of tiles that must tile the grid exactly — every tile in exactly
one region. Configuring it selects the **abstraction-level sheaf**; leaving it out runs the legacy
f = id arm.

```yaml
regions:
  NW: {rect: [0, 4, 3, 7]}      # inclusive corners [x0, y0, x1, y1]
  SW: {rect: [0, 0, 3, 3]}
  pass: [[4, 0], [4, 1], [4, 2]] # or an explicit tile list
```

The regions are the *fibers of the abstraction map*: agent stalks keep speaking per tile, but the
interfaces speak per region — per label, a *flagged* summary ("some tile here has narrowed to
within this label") and an *excluded* one ("no tile here still admits it"), plus one route-claim
bit per endpoint — and the sheaf condition compares what both endpoints push into that vocabulary.
Two agents pinning **different** tiles of one region push the same summary, so the section closes
over their disagreement: which tile carries the flag is exactly what the interface cannot see, and
the caption's `abstract: global section · concrete: agents differ on N tiles` is that fact on
screen. Choosing the partition is therefore a modelling decision, not a display one — it decides
which disagreements the mission tolerates. Walls interior to a region keep believers in
differently-placed walls in abstract agreement; a wall on a boundary splits them.

Three behavioral consequences of the coarser interface vocabulary, all deliberate:

- **Warnings arrive at region granularity.** "This region has a hazard somewhere" crosses the
  interface; *which tile* does not, unless the receiver's own knowledge narrows it. The planner
  handles this through the mission monitor's cost overlay (see `contracts` below), so a warned
  agent routes around the region — exactly the behavior the vocabulary licenses.
- **Partial masks stay private.** A belief that has not narrowed to one label entails neither the
  flagged nor the excluded summary, so it does not travel. First-hand pins and all-clear beliefs
  do, in both polarities.
- **Contradictions can be absorbed rather than contested.** Tile-level conflicts that used to
  empty a possibility set may now meet compatibly at region level; a genuine region-level
  contradiction shows as a label both flagged *and* excluded — `contested_regions` in the log —
  which only the emptied set satisfies, so it is the same conflict-as-a-value one floor up.

The partition is drawn on the floor (see the `rendering` keys) and recorded in the log under
`sheaf.regions`. Region names appear in captions, log read-outs and announcements, so keep them
short. `src/agsheaf/regions.py` parses and validates the block; `src/agsheaf/regionsheaf.py` is
the construction it selects.

## `defaults.yaml` — general settings

### `gridworld`

| Key | Meaning |
|---|---|
| `grid_width`, `grid_height` | Grid dimensions. See *Sizing the grid* below. |
| `grid_safety_gap` | Metres kept clear between the grid and each arena wall. |
| `min_distance_repulsion` | Distance within which repulsion is at full strength. |
| `max_distance_repulsion` | Distance beyond which nothing repels. |
| `show_arrows` | Draw the potential-field force arrows. |
| `show_figure` | Draw at all. `false` runs headless — no window and no video, since a video is a recording of the figure — which is the path for batch runs of the physics. It is **not** the path for a Robotarium submission: the testbed scales the figure to the arena and projects it onto the surface the robots drive on, so anything plotted inside the boundaries appears on the floor (Robotarium Python Guide §2.7.8). See `exp/robotarium/`. |
| `sim_in_real_time` | Throttle the simulation to wall-clock time. |
| `asynchronous_moves` | Whether an agent departs for its next tile as soon as it arrives, rather than waiting for every other agent. See *Moving without a barrier* below. |
| `move_schedule` | Whose turn it is to claim the next tile: `null`, `round_robin` or `random`. The schedule over *moves* — the granularity the grid is played on. See *Taking turns* below. |
| `arrival_distance` | Metres from a tile centre at which an agent counts as arrived. The main lever on how long a move takes; see *What a move costs* below. |

### `goals`

| Key | Meaning |
|---|---|
| `max_steps` | Cap on grid steps, so a run always ends. With `asynchronous_moves` there are no rounds to count, and the two keys below bound the run instead. |
| `max_ticks` | Cap on control iterations. 18000 is the Robotarium's own 600 s limit. |
| `stall_ticks` | Stop once no agent has completed a move in this many iterations — the agents left cannot reach their squares. |

### `beliefs`

| Key | Meaning |
|---|---|
| `default` | Label given to tiles an agent holds no belief about. Normally `unknown`. |
| `mark_other_agents_unsafe` | Whether the other agents' current tiles count as unsafe in an agent's labeling. |
| `observe_on_arrival` | Whether an agent learns the true label of the square it steps onto. Off by default; see above for what it costs the flow's invariants. |
| `tour_views_every` | `never`, or how often in grid steps to hold on every configured view in turn. See above. |

### `contracts`

The mission-contract policy of the abstraction-level sheaf. Read only when the experiment
configures a `regions:` partition; inert otherwise.

Each agent holds, per region, a two-slot assume/guarantee contract. The **guarantee** is what the
flow circulates: the agent's tile knowledge and its route commitment — which regions its planned
corridor does and does not cross, published as claim bits on every interface. The **assumption**
is what the mission *relies on*, and it is monitored rather than broadcast: every step, the
mission monitor checks it against the agent's fused stalk, and what it finds — a hazard flagged in
a corridor region, a neighbour known to claim a region the agent's route needs — becomes a cost
overlay on that region's tiles. The agent replans against the overlaid costs; if the route that
comes out crosses a different *set of regions*, it **recommits**: the sheaf's contract is
rewritten, the claims propagated under the old plan are retracted network-wide, and the flow
re-closes the sections the new commitment opened. An arrived agent recommits to the empty route,
freeing its regions.

| Key | Meaning |
|---|---|
| `reliance` | Put the corridor into the assumption: every tile the route crosses keeps a possibility outside hazard. Violated both by a tile pinned hazardous and by one contested down to the empty set. |
| `exclusive_claims` | Also assume no neighbour routes through the regions the agent's own route crosses. A neighbour's claim then violates the assumption crisply, and the monitor reports who. |
| `region_penalty` | Extra cost of entering one tile of a region the network has flagged hazardous or a lower-indexed neighbour has claimed. Finite, like the `unsafe` tile cost: a warned-off region is expensive, not forbidden, so an agent with no way round still gets through. Claim conflicts yield by index — an agent gives way only to lower-indexed claimants — which breaks the symmetric mutual-yield oscillation two equally polite agents would otherwise fall into. |

### `planning`

| Key | Meaning |
|---|---|
| `route_around_after_ticks` | Control iterations an agent may sit unable to move before the others plan around it instead of behind it. `null` lets them wait indefinitely. See *Agents that wait on each other* below. |

### `planning.tile_costs`

Cost of entering a tile, **by what the agent believes about it**. Every one of the four labels needs
a cost; a missing one is rejected rather than failing later inside the planner.

```yaml
tile_costs:
  target: 1.0
  safe: 1.0
  unknown: 3.0     # above safe: prefer a known route to an unexplored one
  unsafe: 30.0     # large but finite: detour if possible, cross if not
```

Agents rank a move by the cost of entering the next tile plus the cheapest cost onward to the goal,
computed by Dijkstra from the goal outwards. Scoring by cost-to-go rather than by the next tile
alone is what stops an agent stalling beside a region it believes is unsafe when a cheap detour
exists.

### `scoring.tile_scores`

Points for entering a tile, **by what that tile truly is**. Only the three ground-truth labels take
a score; `unknown` is rejected, since the world is never unknown to itself.

```yaml
tile_scores:
  target: 5.0
  safe: 1.0
  unsafe: -2.0
```

Moving onto a truly unsafe tile always costs points. Safe and target tiles earn points only until
that agent first reaches a target; afterwards it is free to move on but no longer scores for them.

### `video`

| Key | Meaning |
|---|---|
| `output_dir` | Relative to the experiment file, or an absolute path. |
| `fps` | Frames per second of the saved video. |
| `one_per_view` | Write one video per configured view instead of a single cycling one. |
| `frames_per_sweep` | Hold this many frames on each Laplacian iterate, robots stationary, so the sweeps are watchable rather than only their effect. `0` writes no extra frames. Worth raising under `sheaf.sweeps_per_step: converge`, which lands several iterates in the time the robots take one step. Communicating synchronous runs only — an asynchronous run never holds the floor. |
| `frames_per_view` | How long a tour dwells on each view, when `beliefs.tour_views_every` asks for one. A second apiece at 30 fps. |

The timestamp names the run: a video is `videos/20260802_171657.mp4` and its log is
`logs/20260802_171657.json`. Nothing about what was configured appears in the filename — the log
carries the whole configuration, the views included, so a run describes itself rather than leaning
on its name to do it.

#### One video per view

Set `one_per_view: true` alongside a `views` list to get a separate video of each:

```yaml
beliefs:
  views: [ground_truth, 0, [ground_truth, 0]]
video:
  one_per_view: true
```

```
videos/20260802_172737-1.mp4    ground truth
videos/20260802_172737-2.mp4    agent 0
videos/20260802_172737-3.mp4    ground truth vs agent 0
```

A run writing one video leaves it unnumbered; a run writing several numbers them from 1, in the
order the views are configured. `experiment.videos` in the log says which file shows which view.

**The videos are frame-aligned.** The simulation is stepped once and each frame is drawn once per
view, so the files come out with identical frame counts and dimensions, showing the same robots in
the same places at the same instant, differing only in the belief layer painted over them. They can
be played side by side — in slides, or in a grid — and stay in step throughout.

`switch_view_every` is ignored under `one_per_view`, since no video needs to switch. Leave
`one_per_view` false to get the old behaviour: one video cycling through the views.

### `rendering`

A screen and a projected floor are two surfaces, and several of these keys are set differently
for each — the testbed layer is where that is written down. `exp/preview_frame.py --floor` draws
a frame with those overrides applied, which is how to see what one of them does without building
a submission and waiting for the overhead footage.

| Key | Meaning |
|---|---|
| `show_view_in_status` | Name the view on the caption line, e.g. `agent 0 ◣ vs agent 1 ◥ · sweep 3 · 2/3 interfaces agree · not a section`. The same moment drawn for two agents is two different pictures and the colors do not say which one is on screen, so without this the legend is the only thing that names it — and the legend is what a projected run cannot afford. |
| `show_legend` | Draw a legend naming whichever view is rendered. It sits inside the arena and, with the sheaf marks, runs seven rows deep — enough to cover the corner tiles — so it is off by default. With `show_view_in_status` on, what it adds is the key to the colors rather than the name of the view. |
| `legend_fontsize` | Size of the legend, which is most of what its footprint over the arena comes down to. `x-small`, `xx-small`, or a number of points. |
| `belief_alpha` | Tile fill opacity, kept low enough to see the robots. |
| `show_agent_badges` | Numbered white disc under each robot. On the Robotarium it is projected onto the floor beneath a robot that is physically there, so it names it for the overhead footage rather than standing in for it; turn it off to leave the robots unmarked. |
| `agent_badge_radius`, `robot_length` | Geometry of the badge each robot carries. |
| `badge_edge_color` | Badge outline; also the numeral on an assigned square. |
| `assigned_number_fontsize` | Size of the agent number drawn on its assigned square. |
| `show_assignment_numbers` | Number the assigned squares at all. `false` leaves the target fills unlabelled, which is what the projected floor wants: a 7-point numeral in the corner of a tile does not survive the projector, and a numeral nobody can resolve is a smudge over a tile rather than an answer to which target is whose. |
| `static_assignment_numbers` | Keep every assigned square numbered whatever the view. The assignments never move, so this is stable; `false` numbers only the squares the current view is entitled to know about, which flickers on a run whose view changes with every move. |
| `agent_label_offset` | Agents are numbered from one where a person reads them — badges, assigned squares, captions — and from zero everywhere they are reasoned about: configuration, logs, the sheaf's vertices. This is only the display. |
| `label_colors` | Tile fill per label. `none` leaves the tile unfilled. |
| `fill_ground_truth_safe` | Whether a view of the ground truth **alone** fills the tiles it labels safe. The truth labels every tile, and away from the hazards and the targets that label is `safe`, so filling them washes the whole arena in one colour that says only that nothing is wrong there — and the hazards and the targets, which are what the view is looked at for, have to be picked out of it. A view holding the truth against an agent — `[ground_truth, 0]` — fills them whatever this says: there `safe` is a claim the agent is making, which the truth can be short of or contradict. |
| `comparison_colors` | Outline per comparison, when two sides are drawn against each other. |
| `show_communication` | Draw the communication topology. Set false to keep the topology without the lines. |
| `communication_color`, `communication_alpha`, `communication_linewidth`, `communication_linestyle` | Styling of those lines. |
| `show_idle_communication` | Draw an interface even when nothing is crossing it. `false` shows the topology only as it is used — a line on the floor means a message is passing — at the price that the topology is no longer legible from a still frame. |
| `show_sections` | Color an interface by whether it agrees. Communicating runs only; the `communication_*` styling above is then unused, since an edge always has one of the two states. |
| `section_color` | An interface that agrees: the sheaf condition holds there. |
| `no_section_color` | An interface that does not agree yet. Both endpoints are drawn in this until the flow closes it. |
| `section_alpha`, `section_linewidth`, `section_linestyle` | Weight of both, so only the color carries the meaning. |
| `no_section_linestyle` | Line style of an interface that has not closed yet. `null` draws it like one that has, leaving the color to carry the state. A surface that cannot hold the color sets this instead and draws every interface in one color: on the arena floor the green and the magenta both come back as one washed-out grey, so the testbed layer draws them all black and separates them solid against dashed. |
| `contested_marker` | Glyph marking a tile whose possibility set the network emptied. Blank draws no marks. |
| `contested_color`, `contested_fontsize` | Styling of that glyph. |
| `show_sheaf_status` | Add the iterate on screen and how much of the sheaf condition holds to that caption — under a `regions:` partition, the two levels at once: `abstract: global section (3/3 interfaces) · concrete: agents differ on 70 tiles`. Communicating runs only. |
| `sheaf_status_fontsize` | Size of that caption. |
| `sheaf_status_color` | Color of the caption. `null` takes `badge_edge_color`. |
| `sheaf_status_position`, `sheaf_status_ha`, `sheaf_status_va`, `sheaf_status_rotation` | Where the caption sits, in axes coordinates over the whole arena rather than over the grid: `[0, 0]` is the arena's bottom left, `[1, 1]` its top right. The grid is square-tiled inside a 3.2×2.0 m rectangle and so never fills it; the strip that leaves is the only place a caption can be drawn large without covering ground the robots drive over. `sheaf_status_rotation: 90` runs it up the side, which is what fits a large one beside a wide grid. |
| `sheaf_status_weight`, `sheaf_status_background` | Font weight, and whether to back the text with a white box. The box keeps the caption readable over the tiles; drawn clear of them, on a floor that is already white, it is a rectangle of glare and nothing else. |
| `show_regions` | Draw the region partition on the floor: a heavier boundary along every edge between two regions. Tile colors are what an agent believes; the boundaries are the resolution at which the interfaces measure agreement. Nothing is drawn for the identity partition, which would just repaint the grid. |
| `region_line_color`, `region_line_alpha`, `region_line_width` | Styling of those boundaries, heavier than the tile grid. |
| `show_region_names` | Write each region's name at its centroid. |
| `region_name_color`, `region_name_alpha`, `region_name_fontsize` | Styling of the names. |

### `sheaf`

The communication layer. Inert under `sheaf.enabled: false`, which is the no-communication arm.
See [the belief sheaf](correctness.md#the-belief-sheaf-over-the-gridworld) for the legacy
construction and the `regionsheaf` module docstring for the abstraction-level one.

| Key | Meaning |
|---|---|
| `enabled` | Communicate along the topology at all. Turning it off, or leaving `communication` out of the experiment file, runs the no-communication baseline on an otherwise identical configuration. |
| `sweeps_per_step` | Laplacian sweeps per grid step, i.e. how many communication hops knowledge travels while the robots cross one tile. Or `converge`, which runs the flow to its fixed point before the agents move, so they step on everything the network between them knows rather than on what has reached them so far. |
| `sweep_every_ticks` | How often the flow advances while the robots drive, in control iterations. Only read under `asynchronous_moves`, where there are no grid steps to hang a round on; `null` leaves the arrivals as the only occasion to talk. See *Moving without a barrier*. |
| `max_sweeps` | The bound `converge` stops at. A safety net rather than a parameter: the flow closes in at most diameter + 1 sweeps, so this only ever bites pathologically. |
| `share_assignments` | Seed each agent's assigned square as a shareable target belief, so the network can learn where everyone was sent and contradict a wrong guess. |
| `mission_monitor` | Check each agent's mission assumption against its fused stalk every step. Under a `regions:` partition this is the contract layer proper — violations, hazard regions, claim conflicts, the cost overlay and recommitment (see `contracts`); on the legacy arm it is the original reliance read-out over the corridor's tiles. |
| `legs` | Which adjoint bisheaf transports a broadcast, `kan` or `co`. On the legacy arm the restrictions are bijections and the two coincide; under a partition they genuinely differ. |
| `record_iterates` | Log every Laplacian iterate, not only what it left behind. |
| `record_contracts` | Also record the contracts as the assume/guarantee pairs they *are*, alongside the possibility grids they mean. See below. Costs a z3 simplify and a few KB per agent per sweep, which is the bulk of a communicating run's log once on. |

### `robotarium`

Read only by `exp/robotarium/experiment_template.py`, and set only in
`exp/robotarium/config.yaml` — the testbed layer, applied over `exp/worlds/primary.yaml`. Absent from
`defaults.yaml`, so every key here also has a fallback in the template.

The testbed writes no video: it projects the figure onto the arena floor and films the arena from
overhead, in one continuous take. So the `video.frames_per_*` knobs do not apply, and these stand
in for them. All three are counted in **control iterations of 0.033 s**, and while holding, the
robots are commanded zero velocity rather than left alone — a robot that hears nothing for half a
second pauses itself, which would end the experiment's control of it.

| Key | Meaning |
|---|---|
| `hold_steps_per_sweep` | Dwell on each Laplacian iterate. A sweep takes no time on the floor, so without this the projection jumps from one iterate to the next while the robots keep driving and the flow is never actually seen. 60 ≈ 2.0 s. |
| `hold_steps_per_view` | Dwell on each view of a tour, when `beliefs.tour_views_every` asks for one. The footage is a single take, so this is the only way it shows more than one view of a given moment. 45 ≈ 1.5 s. |
| `hold_steps_at_end` | Dwell on the settled state after the last motion, so the footage closes on whatever the flow reached rather than cutting. 90 ≈ 3.0 s. |

Budget these against the portal's 600 s cap: a tour costs `len(views) × hold_steps_per_view` per
step it runs. `build_submission.py --dry-runs N` measures the real length and fills the duration
field from it.

### `logging`

| Key | Meaning |
|---|---|
| `enabled` | Write a JSON record of every run. |
| `output_dir` | Relative to the experiment file, or an absolute path. |
| `record_poses` | Whether to record the continuous per-frame poses. These outnumber the grid steps roughly a hundred to one and dominate the file — a 14-step run is about 480 KB with them and 20 KB without. |

Each run writes one self-contained JSON, `logs/20260802_171657.json`, sharing its name with the
video of the same run. Run logs are gitignored.

**The log is not of a view.** A video shows one view and a run may write several, but the log
records every agent's labeling at every step alongside the ground truth, whichever views were drawn.
Which views were drawn is recorded twice: `config.beliefs.view` (or `views`) exactly as configured,
and `experiment.videos` as a `{filename, view}` entry per video written. The `sides` field lists
whose labelings are present.

```
experiment      name, timestamp, seed, grid size, agent count,
                videos[] ({filename, view} per video written)
config          the full merged config, defaults included: what actually ran
sides           every side whose labeling is recorded: ground_truth and each agent
assignments     each agent's assigned square
ground_truth    encoded label grid
communication   the edge list and node list, or null
sheaf           the mission sheaf, if the run communicated: interfaces, what a stalk is
                over, which bisheaf transported, sweeps per step, the initial state, and
                regions ({name: tile list}) under a partition
steps[]         step index, positions, step_scores, cumulative_scores,
                reached_target, beliefs (one encoded grid per agent), observations[]
                under beliefs.observe_on_arrival, and for a communicating run:
                conflicts, mission, sweeps[], settled
poses[]         owning grid step, and (x, y, theta) per agent, one record per frame
result          steps_taken, all_arrived, final_scores, total_score, frames
label_characters, row_order   how to read the encoded grids
```

Label grids — the ground truth and every belief snapshot — are written as **one string per grid row,
highest row first**, so they read the way the figure looks, one character per tile: `t` target,
`s` safe, `u` unsafe, `.` unknown. A 9×5 grid is five 9-character strings, which keeps a labeling
recorded at every step both small and legible. `encode_label_grid` and `decode_label_grid` in
`src/agsheaf/log.py` are the pair, and the character legend is written into each file so it is
self-describing.

Initial beliefs are `steps[0].beliefs`; there is no separate section for them. With the sheaf off
the belief series is constant (apart from observations), since nothing else changes what an agent
believes. In a communicating run it is the record of the diffusion: diff it step to step and you
have watched knowledge travel.

A grid step and the frames spent driving it carry the **same index**, so `poses` can be joined to
`steps` on `step` directly.

### What a communicating run records

A labeling is less than a stalk holds. An agent's stalk says which labels it still considers
*possible* per tile, and the two places that differ are exactly the interesting ones: a tile whose
set has emptied under disagreement, where the agent falls back on its own label, and a tile with
several labels left. So the flow is recorded as itself, not only through the beliefs it leaves
behind.

`sheaf.initial` and each entry of `steps[].sweeps` are the same shape — one snapshot of the whole
sheaf, `sheaf.initial` being the 0-cochain before any sweep:

```
stalks          one grid per agent, of possibility sets rather than labels:
                t/s/u pinned, . unconstrained, x emptied, + several left
contracts       under sheaf.record_contracts, the contracts as the assume/guarantee
                pairs they are, one list of lines per agent
contested       per agent, the tiles whose possibility set emptied
sections        per interface, whether both endpoints agree on their shared edge stalk
is_section      whether all of them do: whether the assignment is a global section
refines_initial per agent, whether its knowledge still refines what it started with
collapsed       any stalk that reached a lattice extreme; expected to stay empty,
                since disagreement is recorded in-lattice as an emptied set
```

Under a `regions:` partition each snapshot carries four more fields, which are the abstraction
made legible:

```
section_regions       per interface, per region, whether that region's summaries agree --
                      the sheaf condition localised to (interface, region)
concrete_disagreement per interface, how many tiles the two endpoints' belief dicts
                      differ on. Nonzero alongside is_section: true is the abstraction
                      working -- agreement in the shared vocabulary over tile-level
                      disagreement inside the fibers
claims                per agent: committed (the regions its route crosses) and expected
                      (per region, the neighbours it knows to be coming through)
contested_regions     per agent, per region, the labels both flagged and excluded at
                      once -- the region-level contested marker. The per-tile contested
                      list often stays empty here precisely because the interface
                      vocabulary absorbs tile-level conflicts
```

A `sweeps[]` entry adds `sweep` (which sweep of the run this is, counting from 1, the same number
the video's caption shows), `sweep_in_step`, `changed`, and the `conflicts` that sweep surfaced.
`steps[].settled` says whether the flow had reached its fixed point by the end of that step; the
flow alone being monotone, once it is true no later step sweeps and no later step carries `sweeps`
— until an observation or a recommitment reopens it.

#### The mission record

`steps[].mission` is the monitor's read-out, one entry per agent still short of its square. On the
legacy arm it is the original reliance check: `corridor`, and the relied tiles split into
`contested` (the network disagrees about them) and `broken` (the network agrees they are unsafe
and the plan crosses anyway). Under a partition it is the contract layer:

```
corridor        the tile path the plan relies on, from here to the goal
route_regions   the regions that corridor crosses: the agent's current claims
violated        the regions where the fused guarantee refutes the mission assumption
hazards         per region, corridor tiles the fused knowledge has flagged hazardous --
                named only when something actually pins them; region-level hearsay
                cannot (see hazard_regions)
claimed         per route region, the neighbours known to route through it
hazard_regions  the regions the fused guarantee entails contain a hazard somewhere --
                the region-granular warning the cost overlay acts on
yielded         the claimed regions this agent routed around this step (it yields only
                to lower-indexed claimants)
recommitted     whether this step's replanning changed the agent's route regions, hence
                its published claims -- the out-of-flow update that reopens the flow
refines_initial as in the snapshot
```

`beliefs.observe_on_arrival` is the exception to that, and `steps[].observations` is what records
it: one `{agent, tile, label, changed}` per agent per step, saying what it saw on the square it
stepped onto and whether that was news to it. These are the entries into a run's knowledge that did
*not* come from the flow, so a log carrying them can tell what an agent was told from what it went
and found out — and an observation with `changed: true` is what sends `settled` back to `false`.

The interfaces close one at a time, which is what `sections` is for: an assignment can have half its
interfaces agreeing and not be a global section, and knowing *which* half is the practical argument
for a Laplacian formulation over a monolithic consistency check.

#### The contracts themselves

`stalks` records what a stalk *means* — the possibility set per tile. `contracts`, under
`sheaf.record_contracts`, records the assume/guarantee propositions, split into lines so the JSON
stays readable. What is rendered differs by arm, and the difference is the point.

On the **legacy arm** it is the local section as printing the `Contract` prints it:

```
"contracts": {
  "0": ["Contract(",
        "  A: True",
        "  G: And(S0_0_0 == 0,",
        "    ...",
        ")"]
}
```

The assumption is `True` at every legacy belief stalk, always, by construction — a broadcast is an
assertion, not a promise — and the guarantee is printed saturated and simplified, each mask bound
`S & ~mask == 0` spelled out as one `Extract` equality per excluded bit.

Under a **`regions:` partition** it is each agent's per-region *mission contracts* — the two-slot
pairs the monitor checks — with the regions an agent has nothing to say about omitted:

```
"contracts": {
  "0": ["[SC]",
        "  A: And(Extract(2,0,S0_5_1) != 4, Not(O0_1_SC))",
        "  G: And(Extract(1,0,S0_4_2) == 0, U0_SC == True)"]
}
```

The assumption here is genuinely non-trivial — the corridor reliance and the exclusivity
expectation — and it is deliberately *not* saturated into what the flow circulates: contract
semantics voids every promise on assumption-violating states, so a guarantee conditioned on a
non-trivial assumption entails almost nothing under the existential push leg, and the network
would go mute exactly where coordination is needed. The stalk circulates the guarantee as an
assertion; the assumption is what `mission` monitors against it. The `regionsheaf` module
docstring is the full account of that division.

Nothing here is information the log lacked — the state determines its contracts exactly. What it
buys is that a run states its assume/guarantee propositions rather than leaving them to be
reconstructed.

## How the drawing works

Each agent's labeling is assembled in four layers, later layers overriding earlier ones:

1. Every tile starts at `beliefs.default`.
2. The agent's static beliefs from `beliefs.agents` are applied.
3. If `mark_other_agents_unsafe`, the other agents' current tiles become `unsafe`.
4. The square the agent was assigned becomes `target`.

Layers 2 and 4 are the two different things an agent can know about a target. Layer 4 is the square
it was told to reach, held with certainty; layer 2 carries its guesses about the others. Both come
out as `target` and are drawn in the same color, since the agent routes towards them the same way.
**The assigned square carries the agent's number in its corner** — a numbered square is one an agent
was told to reach, an unnumbered one is a guess it holds about somebody else. A view numbers only
what it is entitled to know: an agent's own view numbers its one square, the ground-truth view
numbers every one.

The renderer and the planner both read `resolve_tile_labels`, so agents move by exactly the
labeling drawn for them.

Every tile is drawn as two triangular halves split along the anti-diagonal. Given a single side,
both halves carry that side's label, so the tile reads as one flat color. Given a pair, the
**lower-left half is the first side and the upper-right the second**:

| | Appearance | Meaning |
|---|---|---|
| `both_unknown` | empty tile | neither side has a belief |
| `one_sided` | half-filled tile | exactly one side has a belief |
| `agree` | uniform tile | same label on both sides |
| `conflict` | split tile, outlined | both have beliefs, and they differ |

A half-filled tile is one side knowing more, not disagreement — only a genuine contradiction is
outlined.

Belief colors change once per grid step; many video frames are captured per grid step while the
robots physically drive between tiles.

## Moving without a barrier

`gridworld.asynchronous_moves` decides whether a grid step exists.

With it `false`, it does: nobody plans until every agent is standing on its tile, and the run
advances in rounds. That barrier is what lets the planner keep two agents off a tile by
observation — it plans them in sequence against one array of positions, every agent at rest.

With it `true` there are no rounds, only agents arriving and leaving. An agent that reaches its
tile replans and departs immediately, whoever else is still driving. Three things change with it:

- **Tiles are booked rather than observed.** An agent in transit holds the tile it is heading
  for *and* the one it is leaving, so nobody is driven onto ground another has not cleared. A
  booked tile is worth waiting behind rather than routing around — unlike an agent parked on its
  own goal, it clears in a moment — so agents hold position for a tick instead of detouring.
- **Communication runs on its own clock.** A sweep takes no time on the testbed and a run is
  about three seconds of driving for every decision, so there is nothing to be gained by only
  talking when somebody is about to move. `sheaf.sweep_every_ticks` advances the flow a hop
  every so many control iterations — 15 by default, twice a second — with everyone broadcasting,
  so an agent that arrives has been listening the whole way there. `null` switches the clock
  off, leaving the arrivals as the only occasion to talk.

  On top of that, an agent about to plan has its own neighbours broadcast, which is the firing
  set τ_t of Riess and Ghrist (2022), Definition 5. Their Theorem 1 is the licence for a partial
  set: the sections are the fixed points of the flow under any schedule that starves nobody, so
  who broadcasts when changes the route and not the answer.

  Gating the broadcast on which agents are *standing still* is the tempting alternative and does
  almost nothing — an agent departs on the tick it arrives, so it is never idle long enough for
  any usable clock to catch it. Measured with one hop per round, the arrivals alone got the flow
  3 sweeps and it never reached a section; with the clock running it settled on a global section
  by the second event.
- **Nothing holds the floor.** The 2 s pause per Laplacian iterate stops every agent, including
  the ones that sweep has nothing to do with, so it is skipped. The iterate still reaches the
  floor — the projection is redrawn every tick and the caption names the sweep — it is just not
  held still for the camera. The view tour goes for the same reason: it wants a moment at which
  every agent is stationary, and there is no longer one.

The log records a moment rather than a round: each entry carries `tick` and a `moved` list, and
the positions are where every agent was or was heading then.

Scoring differs, and the difference is not cosmetic. Synchronously every agent is scored every
round for the tile it now sits on, whether or not it moved onto it, so an agent short of a target
and unable to move keeps earning for where it stands. Asynchronously only the agents that just
arrived somewhere are scored. Runs of the two are therefore comparable on a clean run and not on
a stalled one.

## Taking turns

`gridworld.move_schedule` decides who is allowed to claim their next tile.

| Value | Meaning |
|---|---|
| `null` | Every agent moves as soon as it can. The fastest the run goes. |
| `round_robin` | One agent at a time, cycling. Agents claim tiles in strict rotation. |
| `random` | Each agent with probability 1/2 per turn, drawn from the top-level `seed`. |

A turn is over once every agent in it has had its chance — taken a tile, been found with nowhere
to go, or been found already home. An agent whose turn comes up while it is still driving is
waited for; that is the serialising the schedule is for. An agent that is merely blocked is not
waited for, or a boxed-in agent would hold the rotation up indefinitely.

Rotation orders who *departs*, not who is moving: agents are still in transit together, so it
costs little. On the 3-agent configuration, `round_robin` came in at 79.6 s against 78.2 s
unrestricted, reaching the same tiles with the same scores.

These are the firing sequences of `agsheaf.sheaf` — the same `round_robin` and `random_firing`
that drive the asynchronous Tarski Laplacian (Riess and Ghrist 2022, Definition 5), over agent
indices rather than sheaf vertices. Liveness, their Assumption 2, is what both layers ask of a
schedule. `GridWorld.update` takes a firing set of its own, a level below this one — who receives
a velocity on a given control tick — but nothing reads that from configuration; it is there for
callers modelling a controller that misses updates.

## Messages on the wire

An interface lights up while a message is crossing it: **green** where the two ends now agree,
**red** where they still do not. It goes back to its resting styling a few frames later.

```yaml
rendering:
  show_messages: true
  message_blink_ticks: 10     # ~0.33 s lit
  message_alpha: 1.0
  message_width_scale: 2.5
```

An edge carries a message when the agent at one end of it broadcasts, so this is the firing set
τ_t of the sweep, read as edges. A sweep is instantaneous — it happens between one frame and the
next — so without holding it lit there is nothing to film; `message_blink_ticks` is how long it
is held.

The colour is about the message, not about the run: it says what the interface made of what it
just heard. That is deliberately separate from `show_sections`, which colours the *settled* state
of an interface. With sections on you see both, the settled green accumulating as interfaces
close and a brighter pulse each time one carries something; with sections off the lines rest
faint and dashed and only the pulses show, which is the cleaner read of the flow as a process.

## Agents that wait on each other

The planner treats other agents two different ways. One still going somewhere is worth *waiting
behind*, since it will clear. One that has settled on its own goal is terrain to *route around*,
since it never will. An agent that is blocked is neither, and two of them facing each other in a
corridor wait on each other for good — a deadlock that no amount of communication resolves,
because nobody is wrong about anything.

`planning.route_around_after_ticks` is the tie-break: after that many control iterations of
getting nowhere, an agent joins the terrain everybody else routes around, and the deadlock breaks.
It counts only iterations spent wanting to move and finding every move blocked — driving, or
sitting on its goal, is not being kept from anything. Keep it longer than a single move takes, or
agents will detour around each other's ordinary right of way; 150 iterations is 5 s, against the
2–3 s a move takes.

Where there is genuinely no way round — a one-wide corridor — the planner falls back to routing as
though the blocking agent were not there, which leaves the agent waiting rather than wandering.
That case needs one agent to give way, which nothing here does.

`goals.max_steps` counts rounds in which something moved, not planning attempts. A round where
every agent found itself blocked is not a grid step spent; counting it would end a run on the very
thing `route_around_after_ticks` exists to get it out of, and long before that had its say. What
stops a run that genuinely cannot go on is `goals.stall_ticks` and `goals.max_ticks`.

## What a move costs

Measured on the 3-agent 9×5 configuration, in 0.033 s control iterations:

| | events | driving ticks | ticks per move | total |
|---|---|---|---|---|
| synchronous | 12 rounds | 1169 | 97 | 94.9 s |
| asynchronous | 34 events | 1150 | 96 | 88.3 s |
| asynchronous, `arrival_distance: 0.10` | 34 events | 843 | 70 | 78.2 s |

Two things worth reading off it. Removing the barrier is worth less than it sounds when the
agents are well matched — they arrive at similar times anyway, so the wait for the slowest is
short. And about half the run is the Robotarium driving the robots to their initial conditions
before the experiment begins, which nothing here can touch.

What governs the *rate* is `arrival_distance`. The attraction pulling an agent onto its tile
decays as it closes, so the approach is exponential and the last few centimetres cost as much as
the first twenty: crossing a 0.31 m tile takes ln(0.31 / `arrival_distance`) / 0.67 seconds, 2.7 s
at the 0.05 default and 1.7 s at 0.10. Raising it also raises the attraction an agent still has as
it arrives, which is the quantity *Sizing the grid* below weighs against the repulsion, so a wider
radius makes that check easier to satisfy rather than harder.

What it costs is precision, and the binding limit is what the projection looks like rather than
anything geometric. Agents stop *at* the radius, not near the centre — measured, all three came to
rest within a centimetre of it — so `arrival_distance` is directly how far off-centre a robot
parks. As a fraction of a 0.311 m tile:

| `arrival_distance` | off-centre | time per tile |
|---|---|---|
| 0.05 | 16% of a tile | 2.7 s |
| 0.07 | 22% | 2.2 s |
| 0.10 | 32% — reads as straddling the boundary | 1.7 s |

The assertion only catches half a tile, which is where a robot could be nearer the wrong tile than
the right one. Legibility goes long before that: at 0.10 a robot sitting still looks stuck between
two squares. For a projected run keep it around 0.07.

## Sizing the grid

The arena is 3.2 × 2.0 m, and the grid keeps `grid_safety_gap` clear of each wall. Tiles are square,
sized to fill whichever dimension binds.

The 15 cm minimum tile size is **not** usually the binding constraint — the potential-field
repulsion distances are. Two things repel a robot: the robot on the neighbouring tile, one tile
width away, and the arena wall, which the outermost tile centres sit `grid_safety_gap + tile/2`
away from. Either one, if it still acts when a robot is closing the final 5 cm onto its tile,
beats the attraction the robot has left, so the robot never registers as arrived and any caller
waiting on `robots_done_moving()` waits forever.

`GridWorld._assert_grid_is_reachable` checks this pairing at construction, **before** the simulator
is built, so an unworkable grid fails immediately with the numbers and both remedies rather than
hanging.

Practical pairings:

| Grid | Tile | Repulsion (`min`/`max`) | Notes |
|---|---|---|---|
| 9×5 | 31 cm | 0.2 / 0.35 | Stock tuning. Adjacent robots repel gently; real collision avoidance. |
| 18×10 | 15.6 cm | 0.13 / 0.16 | The finest grid at ≥15 cm. Repulsion must be scaled down to below one tile width, which leaves only ~2 cm of margin over the 11 cm robot wheelbase — tight for hardware. |

Widening the tiles (fewer of them) or lowering `max_distance_repulsion` below the tile width are the
two ways out of a failed check.
