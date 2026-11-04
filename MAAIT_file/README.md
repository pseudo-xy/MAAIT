# RESCHU-MAAIT — USV Mission-Control Simulation

A research prototype simulating an unmanned surface vehicle (USV) mission-control
console on an abstract grid. This repository is the **pure-Python simulation
core** — the logic layer that the web interface will later sit on top of.

## Design principle: logic first, interface second

The mockup deck already specifies *what the interface does*. The hard, testable
part is the simulation underneath, so it is built and verified headless — no
server, no browser — before any UI is added.

## What's here so far

| Module | Responsibility |
| --- | --- |
| `types.py` | Vocabulary (classifications, behaviors, threat tiers, comms states, USV modes) + 2-D geometry |
| `models.py` | Entities: `Grid`, `Boat`, `USV`, `CommsSchedule`, `World` |
| `behaviors.py` | Autonomous boat movement, one strategy per behavior |
| `identification.py` | Proximity + mode rules for identifying boats |
| `engine.py` | `tick()` (state advance) and `run()` (asyncio loop) |
| `scenario.py` | A demo world matching the mockup |
| `demo.py` | Runnable headless demo |

## Core rules encoded

**Boat behaviors (end-goal driven):** every boat heads toward an end goal set by
its type and picks a new one on arrival — aggressive enemies attack a USV (aim at
the nearest USV, re-aimed each tick); standoff enemies head to a far corner of
the map; civilians travel in a straight line from their start; friendlies head to
random grids. Speeds: civilian 5 kn, aggressive 10 kn, standoff 4 or 8 kn,
friendly 8 kn.

**Truth vs. perception:** every boat has a *true* type known to the engine from
t=0 and a *perceived* state shown to the operator. All boats start as UNKNOWN.

**Identification:** agent-mode USVs identify nearby boats automatically; manual
USVs require the operator to bring the USV close and explicitly identify.

**Threat tier:** derived from behavior and speed on correct identification —
aggressive → High, standoff at 8 kn → Medium, standoff at 4 kn → Low. A
*misidentification* (labelling a boat as the wrong class) records the label for
display but does not change the boat's true behavior, and shows no threat tier;
the model counts identification attempts so we can study how many it takes to get
the label right. Revisable at any time.

**Comms:** each USV's per-state schedule (start + duration) drives its comms
signal, matching the admin-screen configuration and the Health & Status panel.

## Setup (uv)

```bash
uv venv
uv pip install -e ".[dev]"
```

## Run the demo

```bash
uv run python -m reschu.demo
```

This prints the mission log and writes **`results/demo_run.csv`** (overwritten
each run) — one row per boat per timestep with position, nearest agent USV,
distance, identification status, and perceived class/threat.

## Run the tests

```bash
uv run pytest
```

This runs the suite and writes **`results/pytest_results.csv`** (overwritten each
run) — one row per test with its name, outcome, and duration.

## USV movement (in progress, built in stages)

- **Stage 1 (done):** shared base speed for all USVs; manual USVs accept an
  operator speed override; a USV can travel toward a target point and stop on
  arrival (`usv_movement.py`).
- **Stage 2 (done):** agent-mode USVs seek the nearest unidentified boat each
  tick and drive to it; auto-identification fires when within range. Manual USVs
  hold position until the operator acts.
- **Stage 3 (done):** post-identification behavior — circle enemies, escort
  friendlies/civilians (both modes).
- **Engagement lifecycle (done):** after identifying its assigned boat, an agent
  circles a (perceived) enemy for a fixed 2 minutes then releases it and
  re-assigns by priority; for a (perceived) friendly/civilian it escorts the
  boat off the grid. An escorted boat abandons its normal behavior and routes to
  the nearest edge while staying clear of enemy/unidentified boats (a danger
  radius), then exits. Freed USVs pick the next boat that still needs work, so
  the agents work through the whole field instead of getting stuck on one boat.
- **Assignment (done):** at start each agent USV is linked to one boat (closest
  wins, aggressiveness breaks ties; links are exclusive — no two agents on one
  boat). The operator can double-click an agent USV and pick its new target from
  a list of every boat on the map (`retarget_usv`). Leftover USVs idle. Manual USVs travel in a straight
  line from their start until the operator acts. USVs and boats carry explicit
  `start_pos`, configurable per run like end goals.
- **Stage 4 (done):** manual interception — the operator points a manual USV at
  a boat at a chosen speed (`commit_intercept`); the sim reports whether the
  intercept is feasible (`evaluate_intercept`) and can advise the minimum speed
  needed (`minimum_intercept_speed_knots`). The USV steers to the intercept and
  stops at identification range for the operator to identify; `cancel_intercept`
  returns it to straight-line travel.

## Next steps

1. Web layer: wrap the engine, stream the world over WebSockets (FastAPI/NiceGUI).
2. Render the grid + symbols in the browser.
3. Add operator interactions one at a time, each mapping to a mockup slide.
4. Build the admin/config screen (mode timeline + comms schedule).

## Misidentification (implemented)

Each AGENT USV has a `misid_rate` (e.g. 0.30 = wrong 30% of the time), set in
config per run. On an identification it may apply a random *wrong* class; the
boat keeps its true behavior, so a misidentified enemy still acts hostile while
displaying the wrong label. A USV won't re-roll a boat it already tried, but a
*different* USV coming into range can correct it — so correction is possible, not
forced. Every attempt is counted (`Boat.id_attempts`), and
`Boat.is_correctly_identified()` reports whether the label finally matches truth
— the "how many tries to get it right" measure. Operator (manual) IDs are correct
for now. Engaged USVs act on the PERCEIVED class (circle a perceived enemy,
escort a perceived friendly/civilian), so a wrong label changes behavior too.

## Noted for later

- **Operator click logging.** Every operator click should be logged (what was
  clicked, when). This will live in the interaction layer once it exists; the
  intent is a per-session click log for later analysis.

## Web UI (simplified scenario)

A first operator-console UI for the simplified 6-boat scenario, matching the
mockup's tactical map. It runs the simulation server-side and streams each tick
to the browser over a WebSocket; the browser draws the map and sends operator
classify actions back.

Run it:

```
uv pip install -e ".[web]"      # fastapi + uvicorn (quote the brackets in zsh)
uv run uvicorn reschu.web.server:app --reload
# then open http://127.0.0.1:8000
```

What it shows: the two stationary centre USVs with their identification-range
rings, the six boats moving (civilian random walk, friendly zigzag, enemies
toward centre), each drawn by *perceived* class (unknown until identified). The
agent USV auto-identifies in range (Bernoulli draw); for the manual USV,
right-click a contact within its ring to classify it (Civilian / Friendly /
Enemy → Low / Med / High) — matching the mockup's classify menu. A researcher-only
side panel shows ground truth next to the operator view so you can compare
perceived vs. true. The operator map itself is fed truth-free data only.

## Known simplifications / deferred items

These are deliberate, documented gaps — things simplified on purpose for now,
to be revisited when the scenario grows.

- **Standoff boats don't yet avoid the USVs.** Conceptually a standoff enemy
  should keep its distance from the USVs while retreating to the grid edge. This
  is NOT implemented yet, on purpose: the USVs are currently stationary in the
  centre and the standoff boats already head *outward* to the edges (away from
  the centre), so avoidance would rarely trigger and would be hard to verify.
  Add it when the USVs become mobile — that's when a USV could get between a
  standoff boat and its retreat, so the avoidance would do observable work.

- **Boats are point contacts (no boat-to-boat collision).** Boats can currently
  overlap / pass through each other. This doesn't affect the identification /
  classification measurements the study is about. Revisit if physical separation
  becomes important for realism (see the notes on lightweight repulsion vs. full
  avoidance).

- **Identification and danger ranges are larger than real life.** The map is at
  true nautical scale (3 grid cells = 1 nautical mile, so the grid is 33 x 21 nm,
  and 1 knot = 1 nm/hour). At that scale the identification range (`IDENTIFY_RANGE
  = 7.5` units = 2.5 nm) and the aggressive-boat danger radius (`DANGER_ZONE_RADIUS
  = 6.0` units = 2 nm) are more generous than realistic small-boat sensor and
  engagement ranges, which are typically well under a nautical mile. They are kept
  deliberately large **for on-screen visibility and gameplay** — a physically
  realistic range would render as only a few pixels on a 33 nm map and be almost
  invisible to the operator. This is an intentional abstraction, not a scale
  error: the space and speed scales are physically exact; only these two
  interaction radii are inflated. To make them realistic later, set them to a real
  distance and convert with `grid_units = nautical_miles x 3`.

## Putting it online

To give participants one link that works in any browser, see **DEPLOY.md** (Docker + Render).
