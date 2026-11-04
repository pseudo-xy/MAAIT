# RESCHU-MAAIT — USV Mission-Control Simulation

A research prototype simulating an unmanned surface vehicle (USV) mission-control
console on an abstract grid. 

# file descriptions
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
random grids. Speeds: civilian 3 kn, aggressive 10 kn, standoff 4 (low standoff and  8 kn (medium standoff),
friendly 8 kn.

**Truth vs. perception:** every boat has a *true* type known to the engine from
t=0 and a *perceived* state shown to the operator. some boats start as KNOWN and others, UNKNOWN

**Identification:** agent-mode USVs identify nearby boats automatically; manual
USVs require the operator to bring the USV close and explicitly identify.

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
