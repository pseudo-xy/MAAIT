"""A demo scenario mirroring the mockup: four USVs, a mix of boats (civilian,
aggressive enemy, standoff enemy, friendly), and per-USV comms schedules that
match the Health & Status signals from the deck.

This is the seam where, later, a JSON scenario file would be loaded instead of
being built in code.
"""

from __future__ import annotations

import random

from .models import Boat, CommsSchedule, CommsWindow, Grid, USV, World
from .types import Behavior, Classification, CommsState, USVMode, Vec2

# --------------------------------------------------------------------------- #
# Cell labels. Admins always address positions in the SMALL-map coordinate
# system: columns 1..40 (x) and rows A..Y (A at the TOP, down to Y), so "A1" is
# the top-left cell. On bigger maps each label covers a block of real cells:
#   small  (40x25):  1 real cell  per label (1x1)
#   medium (80x50):  4 real cells per label (2x2)
#   large  (120x75): 9 real cells per label (3x3)
# A label is expanded to a block and ONE real cell is chosen at random within it
# (once, at world build time).
# --------------------------------------------------------------------------- #
_ROWS = "ABCDEFGHIJKLMNOPQRSTUVWXY"          # 25 rows, A(top) .. Y(bottom)
_REF_W, _REF_H = 40, 25                       # the small-map reference grid


def _cell_label_to_xy(label: str, grid: Grid, rng: random.Random):
    """Convert an 'A1'-style label to a real (x, y) on `grid`. Returns a Vec2, or
    None if the label can't be parsed. y is flipped so row A (top of the label
    grid) maps to the TOP of the map (high y), since the sim's y=0 is the bottom.
    """
    if not isinstance(label, str):
        return None
    s = label.strip().upper()
    if len(s) < 2 or s[0] not in _ROWS or not s[1:].isdigit():
        return None
    row = _ROWS.index(s[0])            # 0 = row A (top)
    col = int(s[1:]) - 1               # 0 = column 1 (left)
    if not (0 <= col < _REF_W and 0 <= row < _REF_H):
        return None
    # scale factor from the reference grid to this map (1 / 2 / 3)
    k = max(1, round(grid.width / _REF_W))
    # block of real cells this label covers
    cx0 = col * k
    ry0 = row * k
    # pick a random cell within the block
    cx = cx0 + rng.randrange(k)
    ry = ry0 + rng.randrange(k)
    # centre of that cell; flip y so row A (ry=0) is at the TOP (high y)
    x = cx + 0.5
    y = grid.height - (ry + 0.5)
    return grid.clamp(Vec2(x=x, y=y))


def _edge_point_for_cell(label: str, grid: Grid):
    """For a friendly's EXIT: map an edge cell label to the exact boundary point
    the boat should exit through. Returns a Vec2 on the grid boundary, or None.
    The label must be an edge cell (column 1/40 or row A/Y)."""
    if not isinstance(label, str):
        return None
    s = label.strip().upper()
    if len(s) < 2 or s[0] not in _ROWS or not s[1:].isdigit():
        return None
    row = _ROWS.index(s[0]); col = int(s[1:]) - 1
    if not (0 <= col < _REF_W and 0 <= row < _REF_H):
        return None
    k = max(1, round(grid.width / _REF_W))
    # cell centre in real coords (no randomness for the exit target)
    cx = col * k + k / 2.0
    ry = row * k + k / 2.0
    x = cx
    y = grid.height - ry
    # snap to the nearest boundary (left/right/top/bottom)
    dl, dr = x, grid.width - x
    db, dt = y, grid.height - y
    m = min(dl, dr, db, dt)
    if m == dl:   x = 0.0
    elif m == dr: x = grid.width
    elif m == db: y = 0.0
    else:         y = grid.height
    return Vec2(x=x, y=y)


def _schedule(*windows: tuple[CommsState, float, float]) -> CommsSchedule:
    return CommsSchedule(
        windows=[
            CommsWindow(state=s, start_min=start, duration_min=dur)
            for (s, start, dur) in windows
        ]
    )


def demo_world() -> World:
    grid = Grid(width=99.0, height=63.0)

    usvs = [
        USV(
            id="USV-1",
            position=Vec2(x=20.0, y=30.0),
            start_pos=Vec2(x=20.0, y=30.0),
            mode=USVMode.MANUAL,
            manual_heading=Vec2(x=1.0, y=0.0),  # heads east in a straight line
            comms_schedule=_schedule((CommsState.IN_RANGE, 0, 12)),
        ),
        USV(
            id="USV-2",
            position=Vec2(x=45.0, y=20.0),
            start_pos=Vec2(x=45.0, y=20.0),
            mode=USVMode.LLM,  # driven by the LLM agent (stub for now)
            comms_schedule=_schedule(
                (CommsState.IN_RANGE, 0, 6),
                (CommsState.NEAR_EDGE, 6, 6),
            ),
        ),
        USV(
            id="USV-3",
            position=Vec2(x=60.0, y=40.0),
            start_pos=Vec2(x=60.0, y=40.0),
            mode=USVMode.MANUAL,
            manual_heading=Vec2(x=0.0, y=1.0),  # heads north in a straight line
            comms_schedule=_schedule((CommsState.IN_RANGE, 0, 12)),
        ),
        USV(
            id="USV-4",
            position=Vec2(x=80.0, y=25.0),
            start_pos=Vec2(x=80.0, y=25.0),
            mode=USVMode.MDP,  # driven by the MDP agent (stub for now)
            comms_schedule=_schedule(
                (CommsState.IN_RANGE, 0, 4),
                (CommsState.NEAR_EDGE, 4, 4),
                (CommsState.OUT_OF_RANGE, 8, 4),
            ),
        ),
    ]

    # All boats exist from t=0, shown as UNKNOWN until a USV identifies them.
    boats = [
        Boat(
            id="B1",
            position=Vec2(x=55.0, y=12.0),
            start_pos=Vec2(x=55.0, y=12.0),
            true_class=Classification.ENEMY,
            behavior=Behavior.AGGRESSIVE,
        ),
        Boat(
            id="B2",
            position=Vec2(x=85.0, y=32.0),
            start_pos=Vec2(x=85.0, y=32.0),
            true_class=Classification.ENEMY,
            behavior=Behavior.STANDOFF,
            standoff_speed=9.0,  # 9 kn standoff -> MEDIUM threat
        ),
        Boat(
            id="B3",
            position=Vec2(x=40.0, y=45.0),
            start_pos=Vec2(x=40.0, y=45.0),
            true_class=Classification.CIVILIAN,
            behavior=Behavior.CIVILIAN,
        ),
        Boat(
            id="B4",
            position=Vec2(x=5.0, y=50.0),
            start_pos=Vec2(x=5.0, y=50.0),
            true_class=Classification.FRIENDLY,
            behavior=Behavior.FRIENDLY,
        ),
        Boat(
            id="B5",
            position=Vec2(x=70.0, y=48.0),
            start_pos=Vec2(x=70.0, y=48.0),
            true_class=Classification.ENEMY,
            behavior=Behavior.STANDOFF,
            standoff_speed=4.0,  # 4 kn standoff -> LOW threat
        ),
    ]

    return World(grid=grid, usvs=usvs, boats=boats)


def simple_world() -> World:
    """Simplified 'bit by bit' scenario (professor's step 1):

      6 boats:
        - known enemy, LOW threat   (starts already identified)
        - unknown enemy, LOW threat  (needs identifying)
        - known enemy, HIGH threat  (starts already identified)
        - unknown enemy, HIGH threat (needs identifying)
        - friendly                   (needs identifying)
        - civilian                   (needs identifying)
      2 USVs in the centre: one MANUAL (operator-driven), one AGENT (policy-driven);
      they identify boats that drift within range (manual by operator action,
      agent automatically via a per-tick Bernoulli coin-flip).

    Threat tiers come from behavior+speed: aggressive -> HIGH, standoff@4kn -> LOW.
    "Known" boats are pre-identified here by calling .identify() on them.
    """
    grid = Grid(width=99.0, height=63.0)
    cx, cy = grid.width / 2, grid.height / 2  # centre

    # two USVs, spaced apart on the centre line (manual left, agent right)
    usvs = [
        USV(
            id="USV-1",
            position=Vec2(x=35.0, y=cy),
            start_pos=Vec2(x=35.0, y=cy),
            mode=USVMode.MANUAL,   # moves along manual_heading from t=0; the
            manual_heading=Vec2(x=1.0, y=0.0),  # operator redirects by drawing a course
        ),
        USV(
            id="USV-2",
            position=Vec2(x=65.0, y=cy),
            start_pos=Vec2(x=65.0, y=cy),
            mode=USVMode.LLM,          # an agent (stub) — moves via its policy
        ),
        USV(
            id="USV-3",
            position=Vec2(x=30.0, y=45.0),
            start_pos=Vec2(x=30.0, y=45.0),
            mode=USVMode.MDP,          # MDP agent (stub) — moves via its policy
        ),
        USV(
            id="USV-4",
            position=Vec2(x=70.0, y=15.0),
            start_pos=Vec2(x=70.0, y=15.0),
            mode=USVMode.LLM,          # LLM agent (stub) — moves via its policy
        ),
    ]

    boats = [
        # known enemy, LOW threat (standoff @ 4 kn) -> pre-identified
        Boat(id="B1", position=Vec2(x=20, y=15), start_pos=Vec2(x=20, y=15),
             true_class=Classification.ENEMY, behavior=Behavior.STANDOFF,
             standoff_speed=4.0),
        # unknown enemy, MEDIUM threat (standoff @ 9 kn)
        Boat(id="B2", position=Vec2(x=80, y=15), start_pos=Vec2(x=80, y=15),
             true_class=Classification.ENEMY, behavior=Behavior.STANDOFF,
             standoff_speed=9.0),
        # known enemy, HIGH threat (aggressive) -> pre-identified
        Boat(id="B3", position=Vec2(x=20, y=45), start_pos=Vec2(x=20, y=45),
             true_class=Classification.ENEMY, behavior=Behavior.AGGRESSIVE),
        # unknown enemy, HIGH threat (aggressive)
        Boat(id="B4", position=Vec2(x=80, y=45), start_pos=Vec2(x=80, y=45),
             true_class=Classification.ENEMY, behavior=Behavior.AGGRESSIVE),
        # friendly (zigzag movement) — starts at the left edge heading east so
        # its mostly-straight path sweeps through the centre where the USVs sit
        Boat(id="B5", position=Vec2(x=5, y=30), start_pos=Vec2(x=5, y=30),
             true_class=Classification.FRIENDLY, behavior=Behavior.FRIENDLY,
             straight_heading=Vec2(x=1.0, y=0.0)),
        # civilian (random walk) — starts just outside the MANUAL USV's ID range
        # (7 units, range is 6), so it's not identified at start; its wandering
        # keeps it nearby, and the operator can classify it via the link that
        # appears when it drifts into range
        Boat(id="B6", position=Vec2(x=35, y=37), start_pos=Vec2(x=35, y=37),
             true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN),
    ]

    # pre-identify the two "known" enemies (B1 low, B3 high)
    boats[0].identify()  # B1
    boats[2].identify()  # B3
    boats[0].known_at_start = boats[2].known_at_start = True

    return World(grid=grid, usvs=usvs, boats=boats)


# --------------------------------------------------------------------------- #
# Configurable scenario built from the settings page (Administrator config).
# The settings screen produces a plain dict; this builds a World from it. Only
# the fields the simulation actually supports are honored (counts, mission
# duration, per-boat type/behavior/tier/start/goal, per-USV mode + start).
# --------------------------------------------------------------------------- #
_BEHAVIOR_BY_NAME = {
    "aggressive": Behavior.AGGRESSIVE,
    "standoff": Behavior.STANDOFF,
    "civilian": Behavior.CIVILIAN,
    "friendly": Behavior.FRIENDLY,
}
_CLASS_BY_NAME = {
    "enemy": Classification.ENEMY,
    "friendly": Classification.FRIENDLY,
    "civilian": Classification.CIVILIAN,
}
_MODE_BY_NAME = {
    "manual": USVMode.MANUAL,
    "llm": USVMode.LLM,
    "mdp": USVMode.MDP,
}
# tier -> standoff speed (only meaningful for standoff boats)
# Default boat speed variation for config-built scenarios (see World.speed_spread
# / speed_jitter). Override per run with "speed_spread" / "speed_jitter" in the
# config; set both to 0 for exact class speeds.
DEFAULT_SPEED_SPREAD = 0.15   # each boat's persistent speed: +/-15% (1 std-dev)
DEFAULT_SPEED_JITTER = 0.05   # slow drift on top: +/-5% (1 std-dev)
DEFAULT_SENSOR_NOISE = 1.0    # agent sensing: range-dependent track noise (0 = perfect)

_STANDOFF_SPEED_BY_TIER = {"low": 4.0, "med": 9.0, "medium": 9.0, "high": 9.0}


_REF_COLS = 40   # the A1 label space is always the small map's 40 x 25 grid
_REF_ROWS = 25
_ROW_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXY"   # A (top) .. Y (bottom)


def _parse_cell_label(label: str):
    """Parse an 'A1'-style label into (col0, row0), 0-indexed, in the 40x25 small-
    map reference space. Column is 1..40 (x), row is a letter A..Y (A at top).
    Returns None if it doesn't look like a cell label."""
    import re
    if not isinstance(label, str):
        return None
    m = re.fullmatch(r"\s*([A-Ya-y])\s*(\d{1,2})\s*", label)
    if not m:
        return None
    row0 = _ROW_LETTERS.index(m.group(1).upper())
    col1 = int(m.group(2))
    if not (1 <= col1 <= _REF_COLS) or not (0 <= row0 < _REF_ROWS):
        return None
    return col1 - 1, row0   # col 0-indexed, row 0-indexed (0 = row A = top)


def _cell_label_to_pos(label: str, grid: Grid, rng: random.Random) -> Vec2 | None:
    """Convert an 'A1' label to a real (x, y) on `grid`. The label lives in the
    40x25 small-map space; on bigger maps each label covers a scale x scale block
    of real cells (Small 1x1, Medium 2x2, Large 3x3) and we pick ONE random cell
    inside that block (randomize-once at build time). None if not a valid label."""
    parsed = _parse_cell_label(label)
    if parsed is None:
        return None
    col0, row0 = parsed
    scale = max(1, round(grid.width / _REF_COLS))   # 1 / 2 / 3
    # block origin in real cells
    bx = col0 * scale
    by_top = row0 * scale          # measured from the TOP (row A = top)
    # random cell within the scale x scale block
    rx = bx + rng.randrange(scale)
    ry_top = by_top + rng.randrange(scale)
    # centre of that cell; flip y so row A (top) -> high y (grid top)
    x = rx + 0.5
    y = grid.height - (ry_top + 0.5)
    return grid.clamp(Vec2(x=x, y=y))


def _snap_to_edge(p: Vec2, grid: Grid) -> Vec2:
    """Move a point to the nearest edge of the grid (so a friendly actually exits
    there). Picks whichever of the 4 edges the point is closest to."""
    dl, dr = p.x, grid.width - p.x
    db, dt = p.y, grid.height - p.y
    m = min(dl, dr, db, dt)
    if m == dl:
        return Vec2(x=0.0, y=p.y)
    if m == dr:
        return Vec2(x=grid.width, y=p.y)
    if m == db:
        return Vec2(x=p.x, y=0.0)
    return Vec2(x=p.x, y=grid.height)


def _resolve_start(start, grid: Grid, rng: random.Random) -> Vec2:
    """Resolve a config 'start' value to a position. Accepts an 'A1' label (new)
    or a raw [x, y] pair (old configs, for backward compatibility)."""
    if isinstance(start, str):
        pos = _cell_label_to_pos(start, grid, rng)
        if pos is not None:
            return pos
    if isinstance(start, (list, tuple)) and len(start) >= 2:
        try:
            return grid.clamp(Vec2(x=float(start[0]), y=float(start[1])))
        except (TypeError, ValueError):
            pass
    # fallback: centre of the map
    return Vec2(x=grid.width / 2, y=grid.height / 2)


def _mdp_assignment(v) -> str:
    v = str(v).lower()
    return v if v in ("value", "rate", "nearest") else "rate"


def world_from_config(cfg: dict) -> World:
    """Build a World from a settings dict produced by the config page.

    Expected shape (all optional; sensible defaults applied):
      {
        "grid": {"width": 99, "height": 63},
        "boats": [
          {"true_class": "enemy", "behavior": "standoff", "tier": "low",
           "start": [x, y], "goal": [x, y] | null, "known": false}, ...
        ],
        "usvs": [
          {"mode": "manual"|"llm"|"mdp", "start": [x, y]}, ...
        ],
      }
    """
    g = cfg.get("grid", {}) or {}
    grid = Grid(width=float(g.get("width", 120.0)), height=float(g.get("height", 75.0)))
    try:
        mission_minutes = float(cfg.get("mission_minutes", 20.0))
    except (TypeError, ValueError):
        mission_minutes = 20.0

    # --- USVs ---
    # RNG for the randomize-once placement of labels within their map blocks.
    place_rng = random.Random(cfg.get("placement_seed", 20259))

    usvs = []
    for i, u in enumerate(cfg.get("usvs", []) or [], start=1):
        mode = _MODE_BY_NAME.get(str(u.get("mode", "llm")).lower(), USVMode.LLM)
        pos = _resolve_start(u.get("start"), grid, place_rng)
        kwargs = dict(id=f"USV-{i}", position=pos, start_pos=pos, mode=mode)
        if mode is USVMode.MANUAL:
            # manual USV patrols east from t=0 until the operator redirects it
            kwargs["manual_heading"] = Vec2(x=1.0, y=0.0)
        # fuel the USV starts the mission with (percent, default full)
        try:
            kwargs["fuel_pct"] = min(100.0, max(0.0, float(u.get("fuel_start", 100.0))))
        except (TypeError, ValueError):
            kwargs["fuel_pct"] = 100.0
        # per-USV comms schedule: list of {state, start, dur}. Windows with a
        # blank/None start (or dur 0) are skipped. Defaults to in-range for the
        # whole mission when nothing is given.
        cw = u.get("comms") or []
        windows = []
        _cs = {"in_range": CommsState.IN_RANGE, "near_edge": CommsState.NEAR_EDGE,
               "out_of_range": CommsState.OUT_OF_RANGE}
        for w in cw:
            st = w.get("start")
            dur = w.get("dur")
            if st in (None, "", "-") or not dur:
                continue
            try:
                windows.append(CommsWindow(
                    state=_cs.get(str(w.get("state")), CommsState.IN_RANGE),
                    start_min=float(st), duration_min=float(dur)))
            except (TypeError, ValueError):
                continue
        if windows:
            kwargs["comms_schedule"] = CommsSchedule(windows=windows)
        else:
            kwargs["comms_schedule"] = _schedule((CommsState.IN_RANGE, 0, 9999))

        # per-USV control-mode timeline: list of blocks with mode + minute span.
        # Accepts either explicit start_min/end_min, or from/to fractions (0..1)
        # that we scale by the mission duration.
        tl = u.get("timeline") or []
        mode_sched = []
        for blk in tl:
            m = _MODE_BY_NAME.get(str(blk.get("mode", "")).lower())
            if m is None:
                continue
            if blk.get("start_min") is not None and blk.get("end_min") is not None:
                s0, s1 = float(blk["start_min"]), float(blk["end_min"])
            elif blk.get("from") is not None and blk.get("to") is not None:
                s0 = float(blk["from"]) * mission_minutes
                s1 = float(blk["to"]) * mission_minutes
            else:
                continue
            if s1 > s0:
                mode_sched.append({"mode": m, "start_min": s0, "end_min": s1})
        if mode_sched:
            # the USV's starting mode is the first block's
            kwargs["mode"] = mode_sched[0]["mode"]
            kwargs["mode_schedule"] = mode_sched
            if kwargs["mode"] is USVMode.MANUAL and "manual_heading" not in kwargs:
                kwargs["manual_heading"] = Vec2(x=1.0, y=0.0)

        usvs.append(USV(**kwargs))

    # --- boats ---
    boats = []
    for i, b in enumerate(cfg.get("boats", []) or [], start=1):
        behavior = _BEHAVIOR_BY_NAME.get(str(b.get("behavior", "civilian")).lower(),
                                         Behavior.CIVILIAN)
        true_class = _CLASS_BY_NAME.get(str(b.get("true_class", "civilian")).lower(),
                                        Classification.CIVILIAN)
        pos = _resolve_start(b.get("start"), grid, place_rng)
        bk = dict(
            id=f"B{i}", position=pos, start_pos=pos,
            true_class=true_class, behavior=behavior,
        )
        # friendly exit cell (A1 label of the edge cell it should exit through)
        exit_cell = b.get("exit")
        if exit_cell:
            ep = _cell_label_to_pos(exit_cell, grid, place_rng)
            if ep is not None:
                # snap to the nearest edge so it actually leaves the map
                bk["exit_point"] = _snap_to_edge(ep, grid)
        if behavior is Behavior.STANDOFF:
            tier = str(b.get("tier", "low")).lower()
            bk["standoff_speed"] = _STANDOFF_SPEED_BY_TIER.get(tier, 4.0)
        goal = b.get("goal")
        if goal:
            bk["end_goal"] = grid.clamp(Vec2(x=float(goal[0]), y=float(goal[1])))
        boat = Boat(**bk)
        # "known" boats start pre-identified (like B1/B3 in the simple scenario)
        if b.get("known"):
            boat.identify()
            boat.known_at_start = True
        boats.append(boat)

    def _frac(key, default):
        try:
            return max(0.0, float(cfg.get(key, default)))
        except (TypeError, ValueError):
            return default
    return World(grid=grid, usvs=usvs, boats=boats,
                 speed_spread=_frac("speed_spread", DEFAULT_SPEED_SPREAD),
                 speed_jitter=_frac("speed_jitter", DEFAULT_SPEED_JITTER),
                 sensor_noise=_frac("sensor_noise", DEFAULT_SENSOR_NOISE),
                 visual_cue=bool(cfg.get("visual_cue", True)),
                 mdp_assignment=_mdp_assignment(cfg.get("mdp_assignment", "rate")))


def default_config() -> dict:
    """The settings dict that reproduces the current simple_world() scenario, so
    the config page can open pre-filled with the known-good default."""
    return {
        "grid": {"width": 120.0, "height": 75.0},
        "mission_minutes": 20.0,
        "speed_spread": DEFAULT_SPEED_SPREAD,
        "speed_jitter": DEFAULT_SPEED_JITTER,
        "sensor_noise": DEFAULT_SENSOR_NOISE,
        "visual_cue": True,
        "mdp_assignment": "rate",
        "usvs": [
            {"mode": "manual", "start": "M14"},
            {"mode": "llm", "start": "M27"},
            {"mode": "mdp", "start": "I12"},
            {"mode": "llm", "start": "S28"},
        ],
        "boats": [
            {"true_class": "enemy", "behavior": "standoff", "tier": "low",
             "start": "Q8", "known": True},
            {"true_class": "enemy", "behavior": "standoff", "tier": "med",
             "start": "Q32", "known": False},
            {"true_class": "enemy", "behavior": "aggressive", "tier": "high",
             "start": "H8", "known": True},
            {"true_class": "enemy", "behavior": "aggressive", "tier": "high",
             "start": "H32", "known": False},
            {"true_class": "friendly", "behavior": "friendly", "tier": "n/a",
             "start": "M3", "exit": "M40", "known": False},
            {"true_class": "civilian", "behavior": "civilian", "tier": "n/a",
             "start": "K15", "known": False},
        ],
    }
