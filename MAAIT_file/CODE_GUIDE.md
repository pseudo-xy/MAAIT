# RESCHU-MAAIT — Source Code Guide

A map of every file in `src/reschu/`, what lives inside it, and **where to go to
change specific things**. Read this when you want to adjust a rule, a number, or
a behavior and aren't sure which file to open.

The mental model: the files form layers, from vocabulary at the bottom to the
runnable demo at the top. Each layer only depends on the ones below it.

```
demo.py            <- runs everything, writes the CSVs (top layer)
web/server.py      <- FastAPI: runs the sim, streams state, takes classify actions
web/index.html     <- browser: draws the tactical map, right-click to classify
  engine.py        <- the tick loop that drives one step of the whole sim
    scenario.py    <- builds the starting world (USVs, boats, settings)
    policies.py    <- the USV decision agents (llm / mdp; scripted baseline)
    observation.py <- the truth-free view agents get (hides ground truth)
    sensing.py     <- what agents can MEASURE: range-dependent tracks + camera looks
    assignment.py  <- which agent USV is linked to which boat
    usv_movement.py<- how USVs move (seek, circle, escort, intercept, manual)
    interception.py<- the manual-interception "will I make it?" math
    behaviors.py   <- how BOATS move (their end goals + escort-out routing)
    identification.py <- when/how a boat gets identified (LLM/Greedy auto-ID; MDP declarations)
    mdp/              <- the MDP teammate: cue features, fitted cue model, engagement MDP
  models.py        <- the data objects: Grid, Boat, USV, World, CommsSchedule
  types.py         <- the vocabulary: enums + Vec2 geometry (bottom layer)
```

---

## types.py — the vocabulary and geometry

The words the whole system speaks, plus 2-D math. Nothing here "does" anything;
it defines the categories and the point/vector type.

**Contains:**
- `Classification` — the THREE real boat types: civilian, friendly, enemy.
- `Perception` — what the operator SEES: unknown, or one of the three once
  identified. (Kept separate so a boat can never *be* "unknown".)
- `Behavior` — how a boat moves: civilian, aggressive, standoff, friendly.
- `ThreatTier` — L / M / H.
- `threat_for_behavior(behavior, speed)` — the rule mapping behavior+speed to a
  tier (aggressive→H, standoff@8kn→M, standoff@4kn→L).
- `STANDOFF_SPEED_LOW = 4.0`, `STANDOFF_SPEED_MEDIUM = 8.0` — the two standoff
  speeds and which tier each implies.
- `USVMode` — the three control modes: manual / llm / mdp (one per USV per run; a run may mix them). `.is_agent()` = llm or mdp.
- `CommsState` — in_range / near_edge / out_of_range.
- `Vec2` — an (x, y) point/vector with add, subtract, multiply, `length()`,
  `distance_to()`, `normalized()`.

**Go here to change:**
- The threat-tier rule, or the standoff speeds that define Low vs Medium →
  `threat_for_behavior` and the `STANDOFF_SPEED_*` constants.
- Add/rename a classification, behavior, comms state, or mode → the enums.

---

## models.py — the data objects

The "nouns" of the simulation. These are Pydantic models (data + a few helper
methods). This is where most **per-thing settings and state fields** live.

**Contains:**
- `Grid` — width/height of the world; `contains()` and `clamp()` keep points in
  bounds.
- `CommsWindow` / `CommsSchedule` — a USV's comms timeline; `state_at(t)` returns
  the comms state at a given mission time (most-degraded active window wins).
- `Boat` — a contact. Holds:
  - ground truth: `true_class`, `behavior`, `standoff_speed`
  - movement: `position`, `heading`, `start_pos`, `end_goal`, `straight_heading`
  - what the operator sees: `identified`, `perceived_class`, `threat_tier`
  - ID accounting: `id_attempts`, `is_correctly_identified()`
  - escort state: `being_escorted`, `left_grid`, `escort_start_min`,
    `escort_end_min`
  - methods: `speed_knots()` (per-behavior speed), `identify(as_class=None)`
    (records a correct or wrong ID; wrong keeps true behavior).
- `USV_BASE_SPEED_KNOTS = 12.0` — the one base speed all USVs share.
- `USV` — a vehicle. Holds:
  - `position`, `heading`, `start_pos`, `mode`, `manual_heading`
  - `operator_speed_knots` (manual override), `pending_declaration` (an MDP's class commitment)
  - `attempted_boat_ids`, `assigned_target_id`, `intercept_target_id`
  - `circling_since_min`, `circle_events`
  - methods: `effective_speed_knots()`, `set_operator_speed()`, `update_comms()`
- `World` — everything at once: grid, usvs, boats, `mission_time_min`; `usv(id)`
  and `boat(id)` lookups.

**Go here to change:**
- The USV base speed → `USV_BASE_SPEED_KNOTS`.
- Boat speeds by behavior → `Boat.speed_knots()`.
- What a correct vs wrong identification does → `Boat.identify()`.
- Add a new field to a boat or USV (e.g. a new setting to record) → the `Boat`
  or `USV` class.
- Comms "most-degraded wins" logic or the in-range default → `CommsSchedule.state_at`.

---

## behaviors.py — how BOATS move

One place for all autonomous boat motion. Every boat heads toward an **end goal**
chosen by its type, and picks a new one on arrival. Also holds the special
"escort me off the grid" routing.

**Contains:**
- `UNITS_PER_KNOT_MIN = 1.0` — the knots→grid-units conversion (shared scale for
  boats AND usvs).
- `_step_distance`, `_move_towards` — movement helpers.
- Goal rules, one per behavior:
  - `_goal_aggressive` — attack the nearest USV OR friendly boat, but only once
    it comes within `AGGRESSIVE_DETECTION_RANGE` (6.0) — it must "see" a target
    first. Beyond that range it SEEKS (random wander). Gives up and re-seeks if a
    locked target moves back out of range.
  - `_goal_standoff` — a random point on a random grid edge, weighted toward
    edges away from the boat (heads outward; varied approach angles, not only
    diagonal). NOTE: standoff boats don't yet avoid the USVs while retreating —
    deliberately deferred until the USVs become mobile (they're stationary in the
    centre now and the boats already head outward, so avoidance would be dormant).
  - `_goal_civilian` — random walk: a random direction + distance each leg; wide
    reach + commit-to-goal so it roams across much of the grid.
  - `_goal_friendly` — zigzag TRANSIT: a stable base course with a small
    non-accumulating heading wobble; travels across the grid and EXITS at the far
    edge (no wall-bounce). On exit the engine despawns it and spawns a new
    friendly (B7, B8, ...) from a random edge, keeping the friendly count constant.
- `choose_goal` — dispatches to the right rule.
- `ESCORT_DANGER_RADIUS = 12.0` — how far an escorted boat stays from threats.
- `_nearest_edge_point`, `_threat_positions`, `escort_exit_step` — the escort-out
  routing: head to the nearest edge while veering away from enemy/unknown boats.
- `step_boat` — the ONE function the engine calls per boat per tick. Handles: left
  the grid already? being escorted? otherwise normal goal-seeking.
- `_on_boundary` — detects when a boat reaches the edge (used to mark it exited).

**Go here to change:**
- What "end goal" each boat type heads for → the `_goal_*` functions.
- How far escorted boats keep from threats → `ESCORT_DANGER_RADIUS`.
- How escorted boats route out (e.g. always exit a specific side) →
  `escort_exit_step` / `_nearest_edge_point`.
- The knots→units scale (physical calibration) → `UNITS_PER_KNOT_MIN`.

---

## identification.py — when a boat gets identified

The bridge from a boat's truth to what the operator sees. Wrong labels come from
a decision (an MDP declaration or an operator's call), except for the **Greedy**
mode, whose number of wrong IDs per mission is set on purpose in the settings.

**Contains:**
- `IDENTIFY_RANGE = 2.0` — base ID range (grid units on the Small map; scaled
  x1/x2/x3 with map size via `identify_range(grid)`).
- `_in_range` — the proximity check.
- `_needs_identification` — a boat is eligible unless it's already CORRECTLY
  identified (so a wrong label can still be corrected).
- `auto_identify(world, rng, greedy_rng)` — LLM- and Greedy-mode USVs identify
  boats in range immediately. LLM: always correct. Greedy: correct except for
  the planned wrong ones. A label a Greedy/LLM USV gave sticks (other agent USVs
  don't re-check it; only the operator can relabel). MDP-mode USVs are skipped.
- `plan_greedy_errors(world, rng)` — at mission start, picks which greedy IDs
  (1st, 2nd, ... over all greedy USVs together) will be wrong, so the mission
  gets exactly `World.greedy_wrong_ids` wrong greedy IDs (settings page:
  "Greedy: wrong IDs per mission"). If contacts run short, the remaining wrong
  IDs are made on the next greedy IDs. A wrong ID gets one of the other two
  classes at random; the boat keeps its true behavior (a friendly labelled
  enemy is circled and eliminated, an enemy labelled civilian is left alone).
  Logged: contacts CSV `identified_by`, `greedy_planned_wrong`; summary
  `greedy_wrong_ids_set`, `greedy_ids_made`, `greedy_wrong_ids_made`.
- `apply_declarations(world)` — applies each MDP USV's
  `pending_declaration = (boat_id, class)`: only inside ID range, may be WRONG
  (the MDP's own call), always consumed.
- `manual_identify(world, usv_id, boat_id, as_class)` — the operator action;
  the operator can pick a wrong class.

**Go here to change:**
- The identification range → `IDENTIFY_RANGE`.
- The rules for applying an MDP declaration (range, what counts) →
  `apply_declarations`.
- Whether a USV can re-attempt its own boat → the `attempted_boat_ids` check in
  `auto_identify`.

---

## sensing.py — what an agent can measure (agents only)

The team's sensor picture of each boat. Only agent Observations use it; physics,
identification range checks, elimination and the operator's map use TRUE values.
Strength is one knob: `World.sensor_noise` (config key `sensor_noise`; 0 =
perfect sensing, 1 = default). Test worlds default to 0.

- **Track errors** (correlated, AR(1)) on position, heading and speed, growing
  with r = distance to the nearest USV / ID range. Position is noisy too, so an
  agent can't recover exact motion by differencing positions.
- **Track smoothing** (the main effect at range): reported heading/speed are
  filtered more heavily with range (`smoothing_alpha`), so fine maneuvers (a
  zigzag weave, a civilian's turns) vanish far away; speed and "moving away from
  us" survive.
- **Sensor range** (`SENSOR_RANGE_RATIO` = 5 ID ranges): beyond it, position only.
- **Camera looks** (`World.visual_cue`, config `visual_cue`): within 1.5 ID ranges
  (the camera range), a fresh look every 10 s names a contact type, correct with
  probability `visual_accuracy(r)` (95% point-blank, falling with range); wrong
  looks prefer look-alikes. Each look has an id so it is counted once.
- Uses its own RNG stream, so switching noise on/off never changes true motion.

**Go here to change:** noise levels / ranges / camera accuracy → the constants at
the top of `sensing.py` (then refit: `python -m reschu.mdp.fit`).

---

## manual_engagement.py — the operator finishes what the manual USV identifies

After the operator identifies a contact with the manual USV, the operator handles it
with the normal manual controls (double-click the USV, click where to go):
- **Civilian** → left alone.
- **Friendly** → it carries on to its exit edge (its normal transit course) and the
  operator follows it out manually. `escort_start_min` = the ID time,
  `escort_end_min` = when it leaves the map.
- **Enemy** → the operator circles it manually. When the manual USV has spent
  `CIRCLE_DURATION_MIN` (1 min, the same time an AI USV circles) within
  identification range of it, it is eliminated as if an AI had circled it, and the
  circle is logged in the USV's `circle_events`. Time out of range pauses the clock
  (progress is kept).
- The contact is claimed (`Boat.manual_handler`, logged in `claimed_by`): AI USVs
  release it and every assignment layer skips it (`needs_work()`,
  `BoatView.operator_handling`). Only if the manual USV is destroyed or returns to
  base before finishing is the claim released, so the AI team may take over.
- If the USV's settings timeline switches it Manual → MDP/LLM mid-job,
  `keep_on_mode_switch()` gives the contact to that same USV's AI (pinned), which
  finishes it: an enemy's circling time carries over, a friendly is escorted out.
- Called each tick from `engine.tick()` (after the danger check). The map shows a
  ring that fills over the minute ("circling 35s / 60s"); the timestep log shows
  `manual_circling_Bx` / `manual_escorting_Bx`; the contacts CSV gains
  `claimed_by_operator_usv`.

---

## Administrator access (web console)

Participants use the console; only the administrator may change settings, pause,
change the simulation speed, record, or open results / recordings / replay.
- Console: one **🔒 ADMIN** button in the header opens a passcode box; the
  correct passcode opens a panel with SPEED, PAUSE, REC and SETTINGS. The panel
  locks again whenever it is closed (Lock & close, or a click outside), and every
  page load starts locked. Everyone sees a **⏸ PAUSED** badge while paused.
- Server (`web/server.py`): the passcode is checked on the server, never in the
  page. `POST /admin/unlock` issues a session token (30 min idle expiry), kept in
  page memory for websocket commands and as an HttpOnly cookie for admin pages;
  `POST /admin/lock` revokes it. The `_admin_guard` middleware blocks
  `ADMIN_PATHS` (/settings, /config, /gridref, /results, /recordings, /replay)
  without it, and the websocket ignores `set_paused` / `set_sim_speed` without a
  valid token.
- **Set the passcode before trials:** `MAAIT_ADMIN_PASSCODE=yourcode` in the
  environment when starting the server. Without it the default `maait-admin`
  is used and the server prints a warning.

---

## mdp/ — the MDP teammate's building blocks

Everything the MDP needs that is not the policy itself. Ground truth is used only
OFFLINE (to label training data); at runtime the MDP sees Observations + the
fitted tables.

- `features.py` — `ContactTracker`: fed one `Observation` per tick, emits a
  10-second `WindowFeatures` per undecided contact (mean speed, closing toward the
  nearest USV, turn rate, straightness, distance band near/mid/far vs ID range,
  and the newest fresh camera look). Works on MEASURED values (sensing.py).
- `collect.py` — OFFLINE labelled data from randomized missions (identification
  and USV elimination switched off in this tool only; USVs run a scripted
  "investigator" pattern).
- `cue_model.py` — `CueModel`: P(speed | type) used once, P(behavior | type, band)
  and P(camera look | type, band) per window, each tempered for correlation; `posterior()` gives the 5-type belief,
  `declaration_probs()` collapses it to enemy / friendly / civilian.
- `fit.py` — one command to rebuild everything:
  `python -m reschu.mdp.fit` → writes `cue_model.json` + `cue_model_report.md`.

**Refit when** boat behaviors, class speeds, `speed_spread` / `speed_jitter`, or
anything in sensing.py changes — the tables describe the current simulator.

### The engagement MDP (one USV working one contact)

- `engagement.py` — the model. State = (distance band, belief cell): 6 bands by
  SENSED distance in ID ranges (danger ≤0.75, id ≤1, cam_near ≤1.25, cam_far ≤1.5,
  mid ≤2.5, far) × 45 belief cells (most likely declaration × its probability ×
  P(aggressor)) = 270 states. Actions: approach / hold / back_off /
  declare_enemy|friendly|civilian / abandon. A declaration from outside ID range
  is a COMMITTED close-in (it applies on the first tick in range — an aggressor
  locks on at ID range, so a USV must make up its mind before arriving).
  Decisions every 10 s or on a band change (semi-MDP; time charged per second).
  `DEFAULT_REWARDS` holds the reward values (research choices). `Counts` holds
  simulated transitions; `EngagementMDP` estimates the model (sparse states borrow
  P(type | state) × per-type band physics) and solves it by value iteration;
  `PolicyTable` is the solved policy saved as `engagement_<size>.json`.
- `agent.py` — the runtime: `EngagementAgent.act(usv, obs, dt)` (truth-free)
  keeps the team's cue track, computes the belief, asks a chooser, and moves the
  USV (station-keeping at a distance from the contact). Choosers: `TableChooser`
  (the solved MDP), `RushChooser` and `ThresholdChooser` (baselines),
  `ExploreChooser` (offline exploration).
- `episodes.py` — single engagements in the real simulator (one USV, one contact
  of random type at 1–5 ID ranges): `collect` (transition counts; truth only
  labels data) and `evaluate` / `summarize` (truth only scores).
- `build.py` — one command: `python -m reschu.mdp.build` explores, solves,
  refines (ε-greedy around the solved policy), solves again, evaluates against the
  baselines on fresh seeds, and writes `engagement_<size>.json` +
  `engagement_report.md`.

**Rebuild the MDP when** you refit the cue model or change rewards, sensing,
speeds or behaviors.

### The team assignment (which contact each MDP USV works)

- `assignment.py` — `TeamAssigner`: every 10 s (at once if an MDP USV is free)
  matches the free MDP USVs one-to-one to contacts, maximising total GAIN
  (`scipy.optimize.linear_sum_assignment`, with an idle option). Gain of USV u on
  an undeclared contact c = V(s) − Q(s, abandon) from the solved MDP, s = (band
  of c from u, belief cell of c), minus extra travel beyond 3.75 ID ranges;
  for a declared enemy / friendly still needing work = handling reward −
  leaving reward − travel. The current pairing gets `SWITCH_MARGIN` (hysteresis).
  Locked (keep their contact): circling / escorting, committed to a declaration,
  or operator-picked (`USV.assignment_pinned`, set by the double-click reassign).
- Called from `MDPPolicy._coordinate()` once per tick. Uses only own-side state
  from the world (our USVs) plus the truth-free Observation for boats.
- `World.mdp_assignment` (config key `mdp_assignment`): `"rate"` (default:
  gain PER MINUTE of USV time — expected engagement time from the MDP's
  `time_to_go_s` plus travel), `"value"` (total gain), or `"nearest"` (the
  engine's old closest-boat rule, kept for comparison).
### Frozen for human trials

- `frozen.py` — the teammate is FROZEN as **mdp-v1.0** (`mdp/FROZEN.json`: version,
  date, SHA-256 of the cue model + three engagement tables, the rewards and
  assignment settings, and one 12-character fingerprint). `MDPPolicy` checks it
  on first use and refuses to run if anything changed; build / fit refuse to
  overwrite frozen files. `python -m reschu.mdp.frozen --check | --unfreeze |
  --freeze NAME`. Unfreezing archives the old manifest (`FROZEN-<version>.json`).
- `decision_log.py` — `DecisionLog`: one row per MDP decision (band, distance,
  full belief, action, its value, the next-best action and the MARGIN, a
  plain-language reason) and per assignment change; stamped with version +
  fingerprint. Truth (true type, whether a declaration was right) is joined
  ONLY at export. The console writes `results-<ts>-mdp_decisions.csv`, and the
  summary CSV gains `mdp_policy_version` / `mdp_policy_fingerprint`.
- Results CSVs: contacts gain `spawned_min`, `known_at_start`, `time_to_id_min`
  (from when the contact APPEARED; blank for pre-known contacts). The summary's
  `identified / correct / incorrect / pct_correct` and mean / median time to ID
  now count only contacts that NEEDED an identification decision, with new
  `known_at_start` and `needed_id` columns. The decision log has
  `action_changed` (1 = the USV changed what it was doing about that contact).
- Hovering at a band edge (re-deciding every 1–3 s while keeping station near
  1.25 ID ranges) is intended: decisions are event-driven on band changes. A
  dead zone at band edges was tested (0.05 and 0.10 ID ranges) and REJECTED: it
  cut decisions by ~15% but cost −11 to −16 points per mission on the small map
  and raised friendly/civilian misfires (paired test, 600 missions).
- Reproducibility: same seed + same config → identical decisions (tested).
  `SimulationEngine` resets the movement module's sway state on creation, so a
  mission never depends on missions run before it in the same process.

**To change the teammate after freezing:** `--unfreeze`, rebuild / refit, check
the reports, then `--freeze mdp-v1.1` (the previous manifest stays archived).

- `sensitivity.py` — `python -m reschu.mdp.sensitivity`: halves / doubles each
  key reward, re-solves on the same simulator counts, and reports how the MDP's
  behavior changes → `sensitivity_report.md`.
- `missions.py --tune 200 --missions 1500` — grid-searches the assignment's
  `SWITCH_MARGIN` / `RATE_FLOOR_MIN` on tuning missions, then confirms on fresh
  missions → `assignment_tuning_report.md`.
- `missions.py` — `python -m reschu.mdp.missions --missions 300`: random
  multi-contact missions run under both rules on the same seeds, scored from
  ground truth offline → `assignment_report.md`.

---

## usv_movement.py — how USVs move

The vehicle-side counterpart to behaviors.py. All USV motion and the engagement
lifecycle live here.

**Contains:**
- `usv_step_distance`, `move_towards`, `has_arrived` — basic travel toward a point.
- `nearest_unknown_boat`, `agent_seek_step` — go to the nearest unidentified boat.
- Engagement constants:
  - `CIRCLE_RADIUS = 4.0` — how far a USV holds while circling an enemy.
  - `CIRCLE_RATE = 1.2` — how fast it orbits.
  - `ESCORT_OFFSET = 3.0` — how far behind an escorted boat it keeps station.
  - `CIRCLE_DURATION_MIN = 2.0` — how long it circles before moving on.
- `circle_step`, `escort_step` — the two engagement movements.
- `assigned_agent_step(usv, world, dt, now_min)` — **the main agent brain.** Per
  tick: approach the assigned boat → once identified, circle (timed) a perceived
  enemy or escort a perceived friendly/civilian → release when done. Records
  circle/escort timestamps.
- `_release` — ends an engagement, logs the circle window, clears the link.
- `manual_straight_step`, `manual_step` — manual USVs: straight-line travel, or an
  operator interception course that stops at ID range.
- (`agent_step` is an older non-assignment version, kept for reference/tests.)

**Go here to change:**
- Circle radius / speed / **2-minute duration** → `CIRCLE_RADIUS`, `CIRCLE_RATE`,
  `CIRCLE_DURATION_MIN`.
- Escort standoff distance → `ESCORT_OFFSET`.
- The agent decision logic (what a USV does after identifying) →
  `assigned_agent_step`.
- What manual USVs do by default → `manual_straight_step` / `manual_step`.

---

## policies.py — the USV decision agents (scripted / LLM / MDP)

The "brain" seam. An AGENT-mode USV is driven by a *policy* that makes the
choices (which boat, speed, direction, when to identify, circle vs escort); the
*mechanics* stay in usv_movement.py. This is where trained agents plug in.

**Contains:**
- `Policy` — the base class. Real agents override `act(usv, obs, dt)`, which is
  handed ONLY the truth-free `Observation` — never the `World`. The base
  `decide()` builds the observation and calls `act()`. Because `act()` has no
  `world` parameter, an agent's decision code *cannot* reach ground truth (true
  class/behavior/tier) even by accident — the world object is not in scope.
- `ScriptedPolicy` — the rule-based baseline (works today). As an INTERNAL
  mechanics component (not an agent under study), it overrides `decide()`
  directly so it can use the movement helpers (which take `world` to carry out
  motion). It still reads no ground truth — the movement layer uses perceived
  class only.
- `LLMPolicy` — LLM-driven agent, **stub, not trained/connected yet**; currently
  falls back to scripted, with a TODO marking where the real LLM call goes
  (implement `act(usv, obs, dt)`).
- `MDPPolicy` — the solved engagement MDP (mdp/). Each tick it updates the team
  cue track; while its assigned contact is undeclared, `act(usv, obs, dt)` runs
  the MDP (approach / hold / back off / declare / abandon); after a declaration
  the shared mechanics circle / escort / release. Which contact it works is
  chosen by the MDP team assignment (mdp/assignment.py), run once per tick
  from `MDPPolicy._coordinate()`.
- `policy_for(usv)` — returns the policy for a USV based on its `mode` (llm/mdp).

**Ground-truth hardening:** the guarantee is structural, not just conventional.
`act()` never receives `world`, so a future LLM/MDP agent physically cannot see a
boat's true identity — it can only build its decision from the perceived
`Observation`. Enforced by tests (`test_agent_act_signature_excludes_world`,
`test_observation_object_has_no_world_or_truth`).

**Go here to change:**
- Implement a real LLM or MDP agent → override `act(usv, obs, dt)` (NOT `decide`)
  so it reads only the truth-free observation; drop the fallback `decide`.
- Add a new kind of agent → subclass `Policy`, add a mode to `USVMode` (types.py)
  and to the `_POLICIES` map.
- Which mode a USV uses → set `mode=` (USVMode.MANUAL / LLM / MDP) on the USV in scenario.py.

---

## web/ — the operator-console UI (simplified scenario)

The first browser UI. `server.py` (FastAPI) runs the simulation server-side and
streams each tick's state to the browser over a WebSocket; `index.html` draws the
tactical map on a canvas and sends operator classify actions back. Truth-free by
construction: the operator map is fed perceived data only; ground truth is sent
separately (nested under each boat's `truth`) purely for a researcher side panel.

**Contains:**
- `web/server.py` — the FastAPI app: `/` serves the page, `/ws` runs the engine
  (via its `on_tick` hook) and streams state; handles the `classify` action by
  calling `manual_identify`.
- `web/index.html` — the canvas map, contact markers by perceived class, the
  identification rings, the right-click classify menu, and the operator/researcher
  side panels.

**Run:** `uv pip install -e ".[web]"` then
`uv run uvicorn reschu.web.server:app --reload` → http://127.0.0.1:8000

**Go here to change:**
- What the browser receives → `_state_payload` / `_boat_payload` in server.py.
- The map look, marker shapes, or the classify menu → index.html.
- A new operator action → add a branch in the `/ws` receive loop + a handler.

---

## observation.py — the truth-free view agents get

Enforces the rule that agents (LLM, MDP, and the scripted baseline) decide from
**perceived information only** — never a boat's ground truth. Ground truth stays
engine-side, used only to reveal identity through identification and to score
results.

**Contains:**
- `BoatView` — a boat as an agent perceives it: id, position, distance,
  identified?, perceived_class, threat_tier, escort/left-grid flags. It
  deliberately has NO `true_class`, `behavior`, or `standoff_speed` fields.
- `SelfView` — the USV's own state (position, speed, comms, assignment, etc.).
- `Observation` — the whole truth-free snapshot a USV's agent sees, with helpers
  like `unidentified()`, `perceived_enemies()`, `perceived_nonenemies()`.
- `observe(usv, world, now_min)` — the ONLY bridge from the full world (which
  holds truth) to an agent; builds the filtered view.

**Go here to change:**
- What an agent is allowed to perceive (add/remove an observed field) → `BoatView`
  / `SelfView` / `observe`. (Adding a ground-truth field here would break the
  boundary — don't.)

---

## assignment.py — which USV is linked to which boat

Decides the USV→boat links (the line drawn in the mockup). Selection is closest
first, aggressiveness as the tie-break, and links are exclusive.

**Contains:**
- `_is_aggressive`, `_assigned_boat_ids`, `_pick_boat_for` — helpers.
- `assign_all(world)` — the START-of-run assignment (each agent takes its best
  available boat).
- `reassign_free_agents(world)` — gives a new boat to any agent that just finished
  an engagement (prefers boats still needing work). Called every tick by the
  engine.
- `_pick_unworked_boat` — the "next boat that still needs work" picker.
- `retarget_usv(world, usv_id, boat_id)` — the OPERATOR double-click action:
  "Change target" lists every boat on the map (free or engaged, with who is on
  it); the operator picks one. This USV drops its current boat; another agent USV
  on the picked boat lets it go (and is re-assigned by its algorithm); a boat the
  manual USV was following up is handed over. The pick is pinned
  (`assignment_pinned`): the team assignment keeps it, and the MDP never abandons
  a pinned contact (it takes its best other action instead, `MDPAgent._best_kept`).
- `reassign_usv(world, usv_id)` — the old "algorithm picks another boat" action,
  still used if a change_target message carries no boat_id.

**Go here to change:**
- The selection rule (closest vs aggressiveness weighting) → `_pick_boat_for`
  and `_pick_unworked_boat` (the `key=` sort).
- Whether/how USVs re-assign after finishing a boat → `reassign_free_agents`.
- What double-click "Change target" does → `retarget_usv` (server) and
  `openTargetList()` in `web/index.html` (the list).

---

## interception.py — the manual "will I make it?" math

The operator-driven interception feature. Pure geometry + a couple of operator
actions.

**Contains:**
- `InterceptResult` — the answer object: feasible?, point, time, speeds, reason.
- `_boat_velocity`, `_solve_intercept_time` — the moving-target interception
  quadratic.
- `evaluate_intercept(usv, boat, world, speed)` — can this USV catch this boat at
  this speed? Returns an `InterceptResult`.
- `minimum_intercept_speed_knots(...)` — the lowest speed that works (advisory).
- `commit_intercept(world, usv_id, boat_id, speed)` — operator action: point a
  manual USV at a boat at a chosen speed; it will steer there and stop at ID
  range. Returns the feasibility.
- `cancel_intercept(world, usv_id)` — clear it, back to straight-line.

**Go here to change:**
- The feasibility rules (e.g. allow intercept off-grid) → `evaluate_intercept`.
- What committing an interception does → `commit_intercept`.

---

## engine.py — the tick loop

The heart that advances the whole simulation one step. Everything above is
orchestrated here.

**Contains:**
- `SimulationEngine.__init__(world, dt_min, seed, assign_at_start)` — sets up the
  run, captures USV start positions, does the initial assignment.
- `SimulationEngine.tick()` — ONE simulated step, in order:
  1. advance mission time
  2. move every boat (and record escort-exit times)
  3. move every USV (agents work assignments; manuals go straight/intercept)
  4. re-assign any freed agents
  5. update comms states
  6. run auto-identification (with misID)
- `run(...)` — the async loop that calls `tick()` repeatedly (with the `on_tick`
  hook the demo/CSV and a future web UI use).
- `stop()`.

**Go here to change:**
- The ORDER of operations each tick, or to add a new per-tick step → `tick()`.
- The default timestep, seed, or whether to assign at start → `__init__`.
- Real-time pacing / the broadcast hook → `run()`.

---

## scenario.py — the starting world (the "settings")

Builds the initial `World`: the USVs, the boats, and all their per-run settings.
**This is the file you edit to change a run's setup.**

**Contains:**
- `_schedule(...)` — a shorthand for building a comms schedule.
- `simple_world()` — the simplified "bit by bit" scenario: 6 boats (known/unknown
  enemies at low/high tier, a friendly, a civilian) and 2 STATIONARY centre USVs
  (one manual, one agent). It's the default; `RESCHU_SCENARIO=demo` runs the full
  4-USV world instead.
- `demo_world()` — creates the grid (100×60), the 4 USVs (modes, start positions,
  manual headings, comms schedules, misID rates), and the 5 boats (types,
  behaviors, standoff speeds, start positions).

**Go here to change:**
- Grid size; how many USVs/boats; their start positions, modes, speeds,
  behaviors, comms schedules, misID rates → all inside `demo_world()`.
- This is the "settings screen" in code form until a real config/UI exists.

---

## demo.py — runs it and writes the CSVs

The top-level runnable. Builds the demo world, runs the engine, prints the
mission log, and writes the two result files.

**Contains:**
- `CSV_PATH`, `USV_CSV_PATH`, `COMBINED_CSV_PATH` — output file locations. The
  combined file merges boats and USVs into one long file with a `row_type`
  column ("boat"/"usv") for easy tracking/filtering.
- `CSV_COLUMNS`, `USV_CSV_COLUMNS` — the columns in each result file.
- `_rows_for_tick` — builds one boat-CSV row per boat per tick (with the
  blank-until-applicable rule for ID and escort cells).
- `_usv_activity`, `_usv_rows_for_tick` — builds the USV-CSV rows (activity label,
  target, distances, circling time).
- `_true_class_label` — formats `enemy (H)` etc.
- `_summary`, `_boat_lines` — the console printout.
- `main()` — orchestrates the run, prints the circle/escort timing summaries, and
  writes both CSVs.

**Go here to change:**
- What columns the CSVs have, or how a cell is formatted (e.g. blank vs value) →
  `CSV_COLUMNS` / `USV_CSV_COLUMNS` and the `_rows_for_tick` / `_usv_rows_for_tick`
  builders.
- The mission length or timestep for the demo run → the `SimulationEngine(...)`
  and `engine.run(...)` calls in `main()`.
- The console output → `_summary`, `_boat_lines`, and the print blocks in `main()`.

---

## Quick "I want to change X" index

| I want to change... | File | What to edit |
| --- | --- | --- |
| Identification range (6 units) | identification.py | `IDENTIFY_RANGE` |
| Circle duration (2 min) | usv_movement.py | `CIRCLE_DURATION_MIN` |
| Circle radius / orbit speed | usv_movement.py | `CIRCLE_RADIUS`, `CIRCLE_RATE` |
| Escort standoff distance | usv_movement.py | `ESCORT_OFFSET` |
| Escort threat-avoidance distance | behaviors.py | `ESCORT_DANGER_RADIUS` |
| USV base speed (12 kn) | models.py | `USV_BASE_SPEED_KNOTS` |
| Boat speeds by type | models.py | `Boat.speed_knots()` |
| Threat-tier rule / standoff speeds | types.py | `threat_for_behavior`, `STANDOFF_SPEED_*` |
| A boat's end-goal behavior | behaviors.py | the `_goal_*` functions |
| How MDP declarations are applied | identification.py | `apply_declarations` |
| MDP rewards / state space | mdp/engagement.py | `DEFAULT_REWARDS`, band and cell edges (then `python -m reschu.mdp.build`) |
| Boat speed variation (per run) | scenario config | `speed_spread`, `speed_jitter` (then refit: `python -m reschu.mdp.fit`) |
| How well agents can sense contacts | scenario config / sensing.py | `sensor_noise`, `visual_cue`; constants in `sensing.py` (then refit) |
| Grid size / who starts where | scenario.py | `demo_world()` |
| Assignment selection rule (LLM USVs; MDP with "nearest") | assignment.py | `_pick_boat_for` / `_pick_unworked_boat` |
| MDP team assignment rule | mdp/assignment.py | `TeamAssigner.gain`, `SWITCH_MARGIN`; switch off with `mdp_assignment: "nearest"` |
| What each tick does, in order | engine.py | `tick()` |
| Timestep / mission length | scenario.py + demo.py | engine calls |
| CSV columns / blank-cell rules | demo.py | `*_COLUMNS`, `_rows_for_tick` |
| Which mode a USV runs (manual/llm/mdp) | scenario.py | `mode=` on the USV |
| Implement the real LLM agent | policies.py | override `LLMPolicy.act(usv, obs, dt)` |
| What the MDP agent does per tick | policies.py + mdp/agent.py | `MDPPolicy.decide`, `EngagementAgent.act` |
| Which scenario the demo runs | env var | `RESCHU_SCENARIO=simple` or `demo` |
| Make a USV stationary | scenario.py | `stationary=True` on the USV |
| Civilian random-walk / friendly zigzag | behaviors.py | `_goal_civilian` / `_goal_friendly` |
| Interception feasibility rules | interception.py | `evaluate_intercept` |

## Elimination mechanics (added)

- **USV in an aggressive boat's danger zone** (`DANGER_ZONE_RADIUS = 6.0`, engine.py)
  → the USV is eliminated and removed from the world (disappears). Applies to all
  USVs (manual + agents). Checked each tick in `_eliminate_usvs_in_danger`.
- **Enemy after circling** → once a USV circles a perceived enemy for
  `CIRCLE_DURATION_MIN` (defined as `CIRCLE_DURATION_REAL_SEC = 10.0` real seconds,
  converted via the playback ratio → 2.0 sim-min at 12x), the enemy is eliminated
  (`left_grid = True`, `eliminated = True`). The circle TRACKS the moving boat
  (circle centre = boat's current position each tick), so the boat can't escape.
- **Escorted civilian/friendly at the edge** → disappears (leaves grid) as before.
- **Distance chain:** ID range 7.5 > circle radius 7.0 > danger zone 6.0. The USV
  must ID (within 7.5) before it can circle (at 7.0), and 7.0 keeps it outside the
  6.0 danger zone so circling a (non-aggressive-adjacent) enemy is safe.
- **Simple scenario now has 4 USVs:** USV-1 manual, USV-2 LLM, USV-3 MDP, USV-4 LLM.

## Recording & replay (added)

Two independent recording features in the web layer:

- **Video recording** (`index.html`, marked `VIDEO RECORDING`): a `REC` button records the MAP CANVAS directly (canvas.captureStream) to a downloadable `.webm` — NO screen-share prompt or permission dialog. Arm-then-auto-start: click once to arm, recording begins automatically when the sim starts. Captures the map only; side panels live in the DVR replay. Remove by deleting the marked block.
- **Data-replay DVR** (`server.py`, marked `DATA-REPLAY DVR`; `replay.html`): every
  run auto-logs each tick's state to `recordings/reschu-recording-*.json` on the
  server (no user action). The `/replay` page lists recordings and plays them back
  with play/pause, step, scrub, and 0.25x-8x speed — reconstructing the full UI
  from the data. Endpoints: `/recordings` (list), `/recordings/{name}` (fetch or
  download JSON), `/replay` (viewer). The DVR is the durable, study-oriented
  record; the video is for sharing.

## Scenario configuration / settings page (added)

Before starting a run, the scenario can be configured on a settings page instead
of being hardcoded.

- `scenario.py` → `world_from_config(cfg)` builds a `World` from a settings dict
  (grid, per-boat type/behavior/tier/start/goal/known, per-USV mode+start).
  `default_config()` returns the dict that reproduces `simple_world()`.
- `web/settings.html` → the "ADMINISTRATOR — Scenario Configuration" page
  (Scenario Setup, USV Control, Boats Panel), reached from the `⚙ SETTINGS` link
  on the main screen. It POSTs the config and returns to the sim.
- `web/server.py` → holds the current config in `_CURRENT_CONFIG`; endpoints
  `/settings` (page), `GET /config` (current), `POST /config` (save). The `/ws`
  handler builds the world via `world_from_config(_CURRENT_CONFIG)`.
- Scope: only settings the sim supports (counts, mission duration, boat setup,
  USV mode+start). The mockup's timed mode-sequence and comms schedule are NOT
  built (they need engine features that don't exist yet).
