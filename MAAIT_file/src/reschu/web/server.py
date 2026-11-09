"""FastAPI web layer for the simplified RESCHU-MAAIT scenario.

Runs the simulation server-side and streams each tick's state to the browser
over a WebSocket; the browser draws the tactical map and sends operator actions
(classify a contact via the manual USV) back.

Truth boundary: the browser is sent BOTH a truth-free operator view (what the
console shows) and, separately flagged, the ground truth — the latter only so the
researcher can see it in a side panel. The operator-facing map uses perceived
data only. The engine stays authoritative on the server.

Run:  uvicorn reschu.web.server:app --reload
      then open http://127.0.0.1:8000
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from ..assignment import reassign_usv, retarget_usv
from ..behaviors import UNITS_PER_KNOT_MIN
from ..engine import SimulationEngine, danger_zone_radius
from ..identification import identify_range, manual_identify
from ..scenario import simple_world, world_from_config, default_config
from ..types import Perception, Vec2

app = FastAPI(title="RESCHU-MAAIT")

# Report which MDP teammate this server will run (and fail loudly if the frozen
# policy files were changed — trials must not silently use a different teammate).
try:
    from ..mdp.frozen import check as _mdp_check
    _v = _mdp_check()
    print(f"[MAAIT] MDP teammate: {_v['version']} (fingerprint {_v['fingerprint']})")
except Exception as _e:  # noqa: BLE001
    print(f"[MAAIT] WARNING — MDP teammate NOT usable: {_e}")

# How long a web mission runs, in SIMULATION minutes. The run ends (clock freezes
# and a "mission complete" notice is sent) once the sim clock reaches this. At the
# How long a web mission runs, in SIMULATION minutes. The run ends (clock freezes
# and a "mission complete" notice is sent) once the sim clock reaches this. The
# slider runs the sim in real time at 1x, so at 1x a 20-minute mission is 20
# minutes of real watching (the operator can speed up with the slider).
MISSION_LENGTH_MIN = 20.0

_HTML = (Path(__file__).parent / "index.html").read_text()
_REPLAY_HTML = (Path(__file__).parent / "replay.html").read_text()
_SETTINGS_HTML = (Path(__file__).parent / "settings.html").read_text()
_GRIDREF_HTML = (Path(__file__).parent / "gridref.html").read_text()

# ---- DATA-REPLAY DVR: where session recordings are stored on the server ---- #
import json as _json
import random as _random
from . import messages as _messages
from datetime import datetime as _dt

# All files the server writes (results CSVs, recordings, the saved scenario)
# live under MAAIT_DATA_DIR (default: the folder the server is started from).
# On a hosting service point it at a persistent disk, e.g. MAAIT_DATA_DIR=/data.
import os as _os_data
DATA_DIR = Path(_os_data.environ.get("MAAIT_DATA_DIR", "."))
RECORDINGS_DIR = DATA_DIR / "recordings"
RESULTS_DIR = DATA_DIR / "results"
CONFIG_FILE = DATA_DIR / "scenario_config.json"


def _clear_previous_results() -> int:
    """Delete every results CSV already in RESULTS_DIR, so the results folder
    (and ⬇ DOWNLOAD RESULTS) only ever holds the latest run. Called when a
    simulation is started. Returns how many files were removed."""
    n = 0
    if RESULTS_DIR.exists():
        for f in RESULTS_DIR.glob("results-*.csv"):
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
    return n


def _write_results_csv(world, config: dict,
                       timestep_log: list | None = None,
                       operator_log: list | None = None) -> dict | None:
    """Write the results CSVs for a finished run (regardless of the REC button):
      * results-{ts}-contacts.csv  : one row per boat.
      * results-{ts}-summary.csv   : one row for the whole mission.
      * results-{ts}-setup.csv     : the experiment set-up (all settings).
      * results-{ts}-timesteps.csv : one row per USV per tick (mode + activity).
      * results-{ts}-operator.csv  : one row per operator action.
      * results-{ts}-mdp_decisions.csv : one row per MDP decision / assignment
        change (belief, action, margin, reason; truth joined at export only).
    Returns a dict of the file names written, or None."""
    import csv
    from ..types import threat_for_behavior

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    contacts_name = f"results-{ts}-contacts.csv"
    summary_name = f"results-{ts}-summary.csv"

    boats = world.boats
    # ---- per-contact rows ----
    contact_fields = [
        "boat_id", "true_class", "behavior", "true_tier",
        "nominal_speed_knots", "speed_scale",
        "identified", "perceived_class", "correct", "id_attempts",
        "first_identified_min", "outcome", "escort_start_min", "escort_end_min",
        "spawned_min", "known_at_start", "time_to_id_min",
        "claimed_by_operator_usv",
    ]
    rows = []
    for b in boats:
        from ..types import Classification
        if b.true_class is Classification.ENEMY:
            try:
                tt = threat_for_behavior(b.behavior, b.nominal_speed_knots())
                true_tier = tt.value if tt else ""
            except Exception:
                true_tier = ""
        else:
            true_tier = ""   # tier only meaningful for enemies
        outcome = ("eliminated" if b.eliminated
                   else "exited" if b.left_grid
                   else "on_grid")
        rows.append({
            "boat_id": b.id,
            "true_class": b.true_class.value,
            "behavior": b.behavior.value,
            "true_tier": true_tier,
            "nominal_speed_knots": b.nominal_speed_knots(),
            "speed_scale": round(b.speed_scale, 4) if b.speed_scale is not None else "",
            "identified": int(b.identified),
            "perceived_class": b.perceived_class.value if b.identified else "",
            "correct": int(b.is_correctly_identified()) if b.identified else "",
            "id_attempts": b.id_attempts,
            "first_identified_min": b.first_identified_min if b.first_identified_min is not None else "",
            "outcome": outcome,
            "escort_start_min": b.escort_start_min if b.escort_start_min is not None else "",
            "escort_end_min": b.escort_end_min if b.escort_end_min is not None else "",
            "spawned_min": b.spawned_min,
            "known_at_start": int(b.known_at_start),
            # from when the contact APPEARED; blank for pre-known contacts
            "time_to_id_min": _time_to_id(b),
            "claimed_by_operator_usv": b.claimed_by or "",
        })
    with (RESULTS_DIR / contacts_name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=contact_fields)
        w.writeheader()
        w.writerows(rows)

    # ---- per-run summary ----
    n_boats = len(boats)
    # ID metrics count only contacts that NEEDED an identification decision:
    # pre-known contacts are reported separately; times run from when each
    # contact appeared (replacement friendlies spawn mid-mission).
    known = [b for b in boats if b.known_at_start]
    identified = [b for b in boats if b.identified and not b.known_at_start]
    correct = [b for b in identified if b.is_correctly_identified()]
    incorrect = [b for b in identified if not b.is_correctly_identified()]
    id_times = [t for t in (_time_to_id(b) for b in identified) if t != ""]
    usvs = world.usvs
    usvs_eliminated = sum(1 for u in usvs if getattr(u, "eliminated", False) or u.left_grid)
    # The distinct control modes each USV was CONFIGURED to use over the mission
    # (from its timeline), not a meaningless end-of-mission snapshot. Per-USV,
    # per-tick modes are in the timesteps CSV.
    cfg_modes = set()
    for uc in (config.get("usvs") or []):
        tl = uc.get("timeline") or []
        if tl:
            cfg_modes.update(str(blk.get("mode", "")).lower() for blk in tl)
        elif uc.get("mode"):
            cfg_modes.add(str(uc["mode"]).lower())
    modes_used = sorted(m for m in cfg_modes if m)

    import statistics
    mean_id = round(statistics.mean(id_times), 3) if id_times else ""
    median_id = round(statistics.median(id_times), 3) if id_times else ""

    summary = {
        "timestamp": ts,
        "num_usvs": len(usvs),
        "num_boats": n_boats,
        "grid_w": int(world.grid.width),
        "grid_h": int(world.grid.height),
        "mission_minutes": config.get("mission_minutes", ""),
        "usv_modes_configured": "|".join(modes_used),
        "contacts": n_boats,
        "known_at_start": len(known),
        "needed_id": n_boats - len(known),
        "identified": len(identified),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "pct_correct": round(100 * len(correct) / len(identified), 1) if identified else "",
        "total_id_attempts": sum(b.id_attempts for b in boats),
        "mean_time_to_id_min": mean_id,
        "median_time_to_id_min": median_id,
        "usvs_lost": usvs_eliminated,
        **_mdp_version_fields(world),
    }
    with (RESULTS_DIR / summary_name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    out_names = {"contacts": contacts_name, "summary": summary_name}

    # ---- experiment set-up: everything set on the settings page for this run ----
    setup_name = f"results-{ts}-setup.csv"
    _write_setup_csv(RESULTS_DIR / setup_name, world, config, ts)
    out_names["setup"] = setup_name

    # ---- per-USV per-tick timestep log ----
    if timestep_log:
        ts_name = f"results-{ts}-timesteps.csv"
        ts_fields = ["mission_min", "usv_id", "mode", "x", "y",
                     "speed_knots", "comms_state", "fuel_pct",
                     "assigned_target", "activity"]
        with (RESULTS_DIR / ts_name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ts_fields)
            w.writeheader()
            w.writerows(timestep_log)
        out_names["timesteps"] = ts_name

    # ---- operator action log ----
    if operator_log:
        op_name = f"results-{ts}-operator.csv"
        op_fields = ["mission_min", "usv_id", "usv_mode", "action", "detail"]
        with (RESULTS_DIR / op_name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=op_fields)
            w.writeheader()
            w.writerows(operator_log)
        out_names["operator"] = op_name

    # ---- MDP teammate decision log (only if MDP USVs ran this mission) ----
    pol = _mdp_policy_for(world)
    if pol is not None and len(pol.log):
        dec_name = f"results-{ts}-mdp_decisions.csv"
        pol.log.write_csv(RESULTS_DIR / dec_name, world)
        out_names["mdp_decisions"] = dec_name

    return out_names


def _flatten_setting(prefix: str, value, out: list) -> None:
    """Flatten nested settings into (name, value) pairs:
    {"a": {"b": 1}} -> a.b ; lists -> name[1], name[2] ... (1-based)."""
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten_setting(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        for i, v in enumerate(value, 1):
            _flatten_setting(f"{prefix}[{i}]", v, out)
    elif isinstance(value, list):
        out.append((prefix, "|".join(str(v) for v in value)))
    else:
        out.append((prefix, "" if value is None else value))


def _write_setup_csv(path, world, config: dict, ts: str) -> None:
    """One file with the whole experiment set-up of a run, one setting per row:
    section (run / mission / usv / boat), item (USV or boat id), setting, value.
    Written from the scenario config the run was built from, so any setting
    added to the settings page later is included automatically."""
    import csv
    rows: list[dict] = []

    def add(section, item, pairs):
        for k, v in pairs:
            rows.append({"section": section, "item": item, "setting": k, "value": v})

    add("run", "", [("timestamp", ts), *_mdp_version_fields(world).items()])
    general = {k: v for k, v in config.items() if k not in ("usvs", "boats")}
    pairs: list = []
    _flatten_setting("", general, pairs)
    add("mission", "", pairs)
    usv_ids = [u.id for u in world.usvs]
    for i, uc in enumerate(config.get("usvs", []) or []):
        pairs = []
        _flatten_setting("", uc, pairs)
        uid = usv_ids[i] if i < len(usv_ids) else f"USV-{i + 1}"
        # what the run actually used, defaults filled in (a setting left blank
        # on the settings page still shows its value here)
        u = world.usvs[i] if i < len(world.usvs) else None
        if u is not None:
            pairs.append(("used.fuel_start_pct", uc.get("fuel_start", 100.0)))
            pairs.append(("used.mode_timeline", " ; ".join(
                f"{b['start_min']:g}-{b['end_min']:g} min {getattr(b['mode'], 'value', b['mode'])}"
                for b in u.mode_schedule) or f"whole mission {uc.get('mode', '')}"))
            pairs.append(("used.comms_windows", " ; ".join(
                " ".join(f"{k}={getattr(v, 'value', v)}" for k, v in w.model_dump().items())
                for w in u.comms_schedule.windows) or "in_range whole mission"))
        add("usv", uid, pairs)
    for i, bc in enumerate(config.get("boats", []) or []):
        pairs = []
        _flatten_setting("", bc, pairs)
        add("boat", f"B{i + 1}", pairs)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "item", "setting", "value"])
        w.writeheader()
        w.writerows(rows)


def _time_to_id(b):
    """Minutes from the contact appearing to its first identification ('' if
    never identified or known at the start)."""
    if b.known_at_start or b.first_identified_min is None:
        return ""
    return round(b.first_identified_min - b.spawned_min, 3)


def _mdp_policy_for(world):
    """The MDP policy instance, if it served THIS world (else None)."""
    from ..policies import _POLICIES
    from ..types import USVMode
    pol = _POLICIES.get(USVMode.MDP)
    return pol if pol is not None and getattr(pol, "_world", None) is world else None


def _mdp_version_fields(world) -> dict:
    pol = _mdp_policy_for(world)
    v = pol.version if pol is not None else {}
    return {"mdp_policy_version": v.get("version", ""),
            "mdp_policy_fingerprint": v.get("fingerprint", "")}


def _save_recording(frames: list) -> str | None:
    """Write a session's per-tick frames to a JSON file on the server and return
    its filename. Each frame is a truth-free-plus-researcher state payload (the
    same data the browser was sent), so the run can be replayed exactly."""
    if not frames:
        return None
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"reschu-recording-{ts}.json"
    payload = {
        "version": 1,
        "created": ts,
        "dt_min": 0.1,
        "frame_count": len(frames),
        "frames": frames,
    }
    (RECORDINGS_DIR / name).write_text(_json.dumps(payload))
    return name
# --------------------------------------------------------------------------- #


# ---- ADMINISTRATOR ACCESS ------------------------------------------------- #
# Participants use the console; only the administrator may change settings,
# pause, change the simulation speed, record, or open results / recordings /
# replay. The passcode is checked HERE on the server (never in the page, where a
# participant could read it), and a successful unlock issues a short-lived
# session token: kept in page memory for websocket commands, and as an HttpOnly
# cookie for the admin pages. Set the passcode before trials:
#     MAAIT_ADMIN_PASSCODE=yourcode uvicorn ...
import hmac as _hmac
import os as _os
import secrets as _secrets
import time as _time

ADMIN_PASSCODE = _os.environ.get("MAAIT_ADMIN_PASSCODE", "maait-admin")
ADMIN_COOKIE = "maait_admin"
ADMIN_TTL_S = 30 * 60                 # an unlocked session lapses after 30 min idle
_ADMIN_TOKENS: dict[str, float] = {}  # token -> expiry (epoch seconds)
# pages / APIs that only the administrator may open (prefix match)
ADMIN_PATHS = ("/settings", "/config", "/gridref", "/results", "/recordings", "/replay")

if ADMIN_PASSCODE == "maait-admin":
    print("[MAAIT] WARNING — using the default admin passcode. Set MAAIT_ADMIN_PASSCODE "
          "before running trials.")


def _admin_token_ok(token: str | None) -> bool:
    now = _time.time()
    for t in [t for t, exp in _ADMIN_TOKENS.items() if exp < now]:
        _ADMIN_TOKENS.pop(t, None)
    if not token or token not in _ADMIN_TOKENS:
        return False
    _ADMIN_TOKENS[token] = now + ADMIN_TTL_S        # activity keeps it alive
    return True


def _is_admin_path(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") or path.startswith(p + "?")
               for p in ADMIN_PATHS)


_LOCKED_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>MAAIT</title>
<style>body{background:#0b1622;color:#d9e4f0;font-family:system-ui,sans-serif;display:flex;
align-items:center;justify-content:center;height:100vh;margin:0}
.box{background:#0f1c2b;border:1px solid #23364d;border-radius:12px;padding:28px 36px;text-align:center}
a{color:#9bc7ff}</style></head><body><div class="box"><h2>&#128274; Administrator access required</h2>
<p>Unlock the admin menu on the console first.</p><p><a href="/">Back to the console</a></p>
</div></body></html>"""


@app.middleware("http")
async def _admin_guard(request: Request, call_next):
    if _is_admin_path(request.url.path) and not _admin_token_ok(
            request.cookies.get(ADMIN_COOKIE)):
        if request.method == "GET" and "text/html" in request.headers.get("accept", ""):
            return HTMLResponse(_LOCKED_HTML, status_code=403)
        return JSONResponse({"error": "administrator access required"}, status_code=403)
    return await call_next(request)


@app.post("/admin/unlock")
async def admin_unlock(body: dict) -> JSONResponse:
    code = str(body.get("passcode", ""))
    if not _hmac.compare_digest(code.encode(), ADMIN_PASSCODE.encode()):
        await asyncio.sleep(0.8)                    # slow down guessing
        return JSONResponse({"ok": False}, status_code=401)
    token = _secrets.token_urlsafe(24)
    _ADMIN_TOKENS[token] = _time.time() + ADMIN_TTL_S
    resp = JSONResponse({"ok": True, "token": token})
    resp.set_cookie(ADMIN_COOKIE, token, max_age=ADMIN_TTL_S, httponly=True,
                    samesite="strict")
    return resp


@app.post("/admin/lock")
async def admin_lock(request: Request, body: dict | None = None) -> JSONResponse:
    for t in (request.cookies.get(ADMIN_COOKIE), (body or {}).get("token")):
        if t:
            _ADMIN_TOKENS.pop(t, None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(ADMIN_COOKIE)
    return resp
# --------------------------------------------------------------------------- #


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(_HTML)


# ---- SCENARIO CONFIG (settings page) -------------------------------------- #
# The current scenario config. Starts as the default (reproduces simple_world);
# the settings page overwrites it, and the next simulation run uses it.
_CURRENT_CONFIG: dict = default_config()
# the last scenario saved from the settings page survives a server restart
try:
    if CONFIG_FILE.exists():
        _CURRENT_CONFIG = _json.loads(CONFIG_FILE.read_text())
except (OSError, ValueError):
    pass
if "mission_minutes" in _CURRENT_CONFIG:
    MISSION_LENGTH_MIN = float(_CURRENT_CONFIG["mission_minutes"])


@app.get("/settings")
async def settings_page() -> HTMLResponse:
    return HTMLResponse(_SETTINGS_HTML)


@app.get("/static/video/{name}")
async def video_file(name: str):
    from fastapi.responses import Response, FileResponse
    # only serve known camera clips
    types = ["civilian","friendly","low_enemy","medium_enemy","high_enemy"]
    scenarios = ["front","crossing","away","behind","behind_away","side_crossing","behind_crossing","side_approaching","side_away"]
    safe = {f"{t}_{s}" for t in types for s in scenarios}
    if name not in safe:
        return Response(status_code=404)
    path = Path(__file__).parent / "static" / "videos" / f"{name}.mp4"
    if not path.exists():
        return Response(status_code=404)
    # FileResponse streams the file and supports HTTP Range requests, which the
    # browser's <video> element needs — a plain full-body Response is rejected
    # with MEDIA_ERR_SRC_NOT_SUPPORTED for streaming playback.
    return FileResponse(path, media_type="video/mp4")


@app.get("/static/grid/{size}")
async def grid_image(size: str):
    from fastapi.responses import Response
    if size not in ("small", "medium", "large"):
        return Response(status_code=404)
    path = Path(__file__).parent / "static" / f"grid_{size}.png"
    if not path.exists():
        return Response(status_code=404)
    return Response(path.read_bytes(), media_type="image/png")


@app.get("/gridref")
async def gridref_page() -> HTMLResponse:
    """Standalone labelled grid, opened from the settings page in a new window so
    admins can see where (x, y) start positions land."""
    return HTMLResponse(_GRIDREF_HTML)


@app.get("/config")
async def get_config() -> JSONResponse:
    return JSONResponse(_CURRENT_CONFIG)


@app.post("/config")
async def set_config(cfg: dict) -> JSONResponse:
    """Save a new scenario config from the settings page.

    Enforces the hard caps server-side (not just in the browser): at most
    MAX_USVS USVs and MAX_BOATS boats. Mission duration is not capped (any
    positive length is allowed). The grid is fixed and cannot be changed here.
    """
    global _CURRENT_CONFIG, MISSION_LENGTH_MIN

    # ---- hard caps (mirror the settings-page limits) ----
    MAX_USVS, MAX_BOATS = 8, 15
    if isinstance(cfg.get("usvs"), list):
        cfg["usvs"] = cfg["usvs"][:MAX_USVS]
    if isinstance(cfg.get("boats"), list):
        cfg["boats"] = cfg["boats"][:MAX_BOATS]
    try:
        mm = float(cfg.get("mission_minutes", 60.0))
        cfg["mission_minutes"] = max(1.0, mm)   # floor at 1 min, no upper cap
    except (TypeError, ValueError):
        cfg["mission_minutes"] = 60.0
    # grid is limited to the three offered map sizes; anything else snaps to the
    # nearest offered size so start positions stay meaningful. Map sizes are
    # abstract grid sizes; boat/USV speeds are tuned for pace, not real distance.
    _MAP_SIZES = {(40, 25), (80, 50), (120, 75)}
    g = cfg.get("grid") or {}
    try:
        gw, gh = int(g.get("width", 120)), int(g.get("height", 75))
    except (TypeError, ValueError):
        gw, gh = 120, 75
    if (gw, gh) not in _MAP_SIZES:
        # pick the offered size whose width is closest
        gw, gh = min(_MAP_SIZES, key=lambda s: abs(s[0] - gw))
    cfg["grid"] = {"width": float(gw), "height": float(gh)}

    _CURRENT_CONFIG = cfg
    try:
        MISSION_LENGTH_MIN = float(cfg.get("mission_minutes", MISSION_LENGTH_MIN))
    except (TypeError, ValueError):
        pass
    # validate by attempting to build the world; report any error
    try:
        world_from_config(cfg)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    try:                                   # keep it across restarts
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(_json.dumps(cfg))
    except OSError:
        pass
    return JSONResponse({"ok": True})
# --------------------------------------------------------------------------- #


# ---- DATA-REPLAY DVR endpoints -------------------------------------------- #
@app.get("/results")
async def list_results() -> JSONResponse:
    """List saved results CSV files (newest first)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted((p.name for p in RESULTS_DIR.glob("results-*.csv")), reverse=True)
    return JSONResponse({"results": files})


@app.get("/results/all.zip")
async def download_all_results():
    """Every results CSV in one zip (administrator only)."""
    import io, zipfile
    from fastapi.responses import Response
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(RESULTS_DIR.glob("results-*.csv")):
            z.write(f, f.name)
    stamp = _dt.now().strftime("%Y-%m-%d_%H-%M")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="maait-results-{stamp}.zip"'})


@app.get("/results/{name}")
async def get_result(name: str):
    """Download one results CSV."""
    from fastapi.responses import Response
    safe = Path(name).name
    path = RESULTS_DIR / safe
    if not path.exists() or not safe.startswith("results-") or not safe.endswith(".csv"):
        return Response(status_code=404)
    return Response(
        path.read_bytes(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{safe}"'},
    )


@app.get("/recordings")
async def list_recordings() -> JSONResponse:
    """List saved session recordings (newest first)."""
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        (p.name for p in RECORDINGS_DIR.glob("reschu-recording-*.json")),
        reverse=True,
    )
    return JSONResponse({"recordings": files})


@app.get("/recordings/{name}")
async def get_recording(name: str) -> JSONResponse:
    """Return a single recording's JSON (the frames), for replay/download."""
    # basic guard against path traversal
    safe = Path(name).name
    path = RECORDINGS_DIR / safe
    if not path.exists() or not safe.startswith("reschu-recording-"):
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(_json.loads(path.read_text()))


@app.get("/replay")
async def replay_page() -> HTMLResponse:
    """The replay viewer page (reuses the main UI in replay mode)."""
    return HTMLResponse(_REPLAY_HTML)
# --------------------------------------------------------------------------- #


def _circle_progress(boat) -> float:
    from ..manual_engagement import circle_progress
    return circle_progress(boat)


def _boat_payload(boat, world) -> dict:
    """State for one boat. Perceived fields drive the map; true_* are flagged as
    ground truth for the researcher panel only."""
    return {
        "id": boat.id,
        "x": boat.position.x,
        "y": boat.position.y,
        "hx": boat.heading.x,
        "hy": boat.heading.y,
        "identified": boat.identified,
        "preview_video": boat.preview_video,
        "perceived_class": boat.perceived_class.value,   # operator view
        "threat_tier": boat.threat_tier.value if boat.threat_tier else None,
        "id_attempts": boat.id_attempts,
        "correctly_identified": boat.is_correctly_identified(),
        "left_grid": boat.left_grid,
        # operator follow-up after a manual ID (manual_engagement.py)
        "operator_task": (None if boat.manual_handler is None else
                          "circle" if boat.perceived_class.value == "enemy" else "escort"),
        "circle_progress": round(_circle_progress(boat), 3),
        # --- ground truth: researcher panel only, never drawn on the map ---
        "truth": {
            "true_class": boat.true_class.value,
            "behavior": boat.behavior.value,
        },
    }


def _usv_target_point(usv, world):
    """The point this USV is heading toward, for ETA:
      - manual USV: its operator-drawn destination, else the boat it's intercepting
      - agent USV : the boat its policy is currently working (assigned_target_id)
    Returns a Vec2-like (has .x/.y) or None if it isn't heading anywhere."""
    if usv.mode.value == "manual":
        if usv.manual_destination is not None:
            return usv.manual_destination
        if usv.intercept_target_id is not None:
            b = world.boat(usv.intercept_target_id)
            if b is not None and not b.left_grid:
                return b.position
        return None
    # agent USV: its assigned boat
    if usv.assigned_target_id is not None:
        b = world.boat(usv.assigned_target_id)
        if b is not None and not b.left_grid:
            return b.position
    return None


def _usv_eta_min(usv, world):
    """Estimated minutes for this USV to reach its target, at its current speed.
    None if it has no target or isn't moving. Uses the same knots->units/min
    scale as movement (UNITS_PER_KNOT_MIN)."""
    target = _usv_target_point(usv, world)
    if target is None:
        return None
    speed = usv.effective_speed_knots() * UNITS_PER_KNOT_MIN  # grid units per min
    if speed <= 0:
        return None
    dist = usv.position.distance_to(target)
    return dist / speed


def _boat_video_key(boat, scenario: str = "front") -> str:
    """The camera-clip key for a boat: '{type}_{scenario}'. `scenario` is 'front'
    (boat approaching head-on) or 'crossing' (boat crossing in front). Enemies
    split by threat tier (low/med/high)."""
    from ..types import Classification, Behavior, threat_for_behavior, ThreatTier
    cls = boat.true_class
    if cls is Classification.CIVILIAN:
        base = "civilian"
    elif cls is Classification.FRIENDLY:
        base = "friendly"
    else:
        tier = threat_for_behavior(boat.behavior, boat.nominal_speed_knots())
        base = ("high_enemy" if tier is ThreatTier.HIGH
                else "medium_enemy" if tier is ThreatTier.MEDIUM
                else "low_enemy")
    return f"{base}_{scenario}"


def _manual_usv_id(world) -> str:
    """The id of the current MANUAL USV (classify / move_to / set_speed target)."""
    for u in world.usvs:
        if u.mode.value == "manual" and not u.left_grid:
            return u.id
    return ""


def _usv_activity(usv, world) -> str:
    """A short label of what a USV is doing right now, for the timestep log."""
    if getattr(usv, "eliminated", False):
        return "eliminated"
    if usv.left_grid:
        return "off_grid"
    if usv.rtb:
        return "returning_to_base"
    if usv.mode.value == "manual":
        # operator follow-up on a contact this USV identified
        for b in world.boats:
            if b.manual_handler == usv.id and not b.left_grid:
                near = usv.position.distance_to(b.position) <= identify_range(world.grid)
                if near and b.perceived_class.value == "enemy":
                    return f"manual_circling_{b.id}"
                if near and b.perceived_class.value == "friendly":
                    return f"manual_escorting_{b.id}"   # following it out
        if usv.manual_destination is not None:
            return "manual_en_route_to_point"
        if usv.intercept_target_id:
            return f"manual_intercepting_{usv.intercept_target_id}"
        return "manual_holding"
    # agent modes
    if usv.circling_since_min is not None and usv.assigned_target_id:
        return f"circling_{usv.assigned_target_id}"
    if usv.assigned_target_id:
        tgt = world.boat(usv.assigned_target_id)
        if tgt is not None and tgt.identified and tgt.perceived_class.value == "friendly":
            return f"escorting_{usv.assigned_target_id}"
        return f"seeking_{usv.assigned_target_id}"
    return "patrolling"


def _usv_seen_boat(usv, world):
    """The boat this USV currently 'sees' on camera. A video plays for the
    scenarios we have footage for, chosen by the boat's motion relative to the USV.

    For the nearest boat within the camera detection distance (1.5x the ID range),
    the boat must be IN FRONT of the USV (within +/-45 deg of its heading). Then:
      * APPROACHING head-on (heading toward the USV)  -> '{type}_front' clip.
      * CROSSING (heading roughly perpendicular)      -> '{type}_crossing' clip.
      * RECEDING (moving away from the USV)            -> '{type}_away' clip.
      * BEHIND the USV, facing/approaching it          -> '{type}_behind' clip.
      * BEHIND the USV, moving away from it             -> '{type}_behind_away' clip.
      * off to the side, or behind+crossing            -> no clip.
    """
    import math
    if usv.left_grid:
        return None
    det = identify_range(world.grid) * 1.5
    FRONT_COS = math.cos(math.radians(45))   # ~0.707 : +/-45 deg forward cone
    APPROACH_MIN = 0.35                       # closing dot above this = approaching
    CROSS_MAX = 0.5                           # |closing dot| below this = crossing

    best, bestd = None, det
    for b in world.boats:
        if b.left_grid:
            continue
        d = usv.position.distance_to(b.position)
        if d <= bestd:
            best, bestd = b, d
    if best is None:
        return None

    # geometry gate: which sector is the boat in relative to the USV heading?
    #   FRONT cone (+/-45 deg of heading): classify motion -> front/crossing/away
    #   REAR  cone (+/-45 deg of the OPPOSITE of heading): we only have the boat-
    #         facing/approaching clip -> 'behind' (other rear motions: no clip yet)
    #   sides: no clip.
    to_boat = best.position - usv.position
    dist = to_boat.length()
    if dist < 1e-6:
        return None
    uh = usv.heading.normalized()
    dir_to_boat = Vec2(x=to_boat.x / dist, y=to_boat.y / dist)
    fwd_dot = uh.x * dir_to_boat.x + uh.y * dir_to_boat.y   # +1 ahead, -1 behind

    bh = best.heading.normalized()
    dir_boat_to_usv = Vec2(x=-dir_to_boat.x, y=-dir_to_boat.y)
    # closing component: +1 = boat heading straight at USV, 0 = perpendicular, -1 = away
    closing = bh.x * dir_boat_to_usv.x + bh.y * dir_boat_to_usv.y

    if fwd_dot >= FRONT_COS:
        # boat is IN FRONT
        if closing > APPROACH_MIN:
            scenario = "front"        # coming toward the USV
        elif abs(closing) <= CROSS_MAX:
            scenario = "crossing"     # moving across the field of view
        else:
            scenario = "away"         # moving away from the USV (receding)
    elif fwd_dot <= -FRONT_COS:
        # boat is BEHIND. We have facing/approaching and moving-away rear clips.
        if closing > APPROACH_MIN:
            scenario = "behind"        # behind the USV, facing/approaching it
        elif closing < -CROSS_MAX:
            scenario = "behind_away"   # behind the USV, moving away from it
        else:
            scenario = "behind_crossing"  # behind the USV, moving across the view
    else:
        # boat is OFF TO THE SIDE. All three side motions now have footage.
        if closing > APPROACH_MIN:
            scenario = "side_approaching"  # off to the side, heading toward the USV
        elif abs(closing) <= CROSS_MAX:
            scenario = "side_crossing"     # moving across the field of view, off to the side
        else:
            scenario = "side_away"         # off to the side, moving away from the USV

    return {"boat_id": best.id, "video": _boat_video_key(best, scenario)}


def _usv_payload(usv, world) -> dict:
    dest = usv.manual_destination
    eta = _usv_eta_min(usv, world)
    return {
        "id": usv.id,
        "target_id": usv.assigned_target_id,     # current target (agent USVs)
        "x": usv.position.x,
        "y": usv.position.y,
        "hx": usv.heading.x,
        "hy": usv.heading.y,
        "mode": usv.mode.value,
        "stationary": usv.stationary,
        "speed_knots": round(usv.effective_speed_knots(), 1),
        # active operator-drawn course (manual USV heading to this point), or None
        "destination": ({"x": dest.x, "y": dest.y} if dest is not None else None),
        # estimated minutes to reach its target (None if not heading anywhere)
        "eta_min": (round(eta, 2) if eta is not None else None),
        # Health & Status fields
        "fuel_pct": round(usv.fuel_pct, 1),
        "camera": _usv_seen_boat(usv, world),
        "comms_state": usv.comms_state.value,
        "rtb_blocked": usv.rtb_blocked_reason(),     # None, or why RTB is not possible
        # engage-agent toggle is ON for agentic (LLM/MDP) USVs, OFF for manual.
        "agent_on": usv.mode.is_agent(),
        "rtb": usv.rtb,
        "left_grid": usv.left_grid,
    }


def _manual_links(world) -> list[dict]:
    """Links from the MANUAL USV to every boat currently within its ID range that
    is not yet correctly identified. Each link persists (recomputed each tick)
    until the boat is correctly identified or leaves the range. The operator uses
    these to see which contacts are identifiable right now, and picks which to
    classify first — several links can be active at once."""
    manual = next((u for u in world.usvs if u.mode.value == "manual"), None)
    if manual is None or manual.rtb or manual.left_grid:
        return []
    links = []
    for b in world.boats:
        if b.left_grid or b.is_correctly_identified():
            continue
        d = manual.position.distance_to(b.position)
        if d <= identify_range(world.grid):
            links.append({"usv_id": manual.id, "boat_id": b.id, "distance": round(d, 2)})
    return links


def _agent_links(world) -> list[dict]:
    """Links from each AGENT USV to the boat it is currently working (its
    assigned target). The link persists until the agent finishes its whole
    routine on that boat and releases it (assigned_target_id clears)."""
    links = []
    for u in world.usvs:
        if not u.mode.is_agent() or u.rtb or u.left_grid:
            continue
        tid = u.assigned_target_id
        if tid is None:
            continue
        b = world.boat(tid)
        if b is None or b.left_grid:
            continue
        d = u.position.distance_to(b.position)
        links.append({"usv_id": u.id, "boat_id": b.id, "distance": round(d, 2)})
    return links


def _state_payload(world) -> dict:
    # capture the pre-identification camera clip for each boat: for every USV,
    # what it currently sees; if that boat is still unidentified, remember the
    # clip so the operator can replay it after ID (from the Contacts panel).
    for u in world.usvs:
        seen = _usv_seen_boat(u, world)
        if seen is None:
            continue
        b = world.boat(seen["boat_id"])
        if b is not None and not b.identified:
            b.preview_video = seen["video"]
    return {
        "type": "state",
        "now_min": round(world.mission_time_min, 2),
        "mission_length_min": round(MISSION_LENGTH_MIN, 2),
        "grid": {"width": world.grid.width, "height": world.grid.height},
        "identify_range": identify_range(world.grid),
        "danger_zone": danger_zone_radius(world.grid),
        "usvs": [_usv_payload(u, world) for u in world.usvs],
        "boats": [_boat_payload(b, world) for b in world.boats],
        "manual_links": _manual_links(world),
        "agent_links": _agent_links(world),
    }


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()

    # build the world from the saved scenario config (falls back to the default
    # simple scenario if the config is somehow invalid)
    try:
        world = world_from_config(_CURRENT_CONFIG)
    except Exception:  # noqa: BLE001
        world = simple_world()

    # Playback pacing. The slider is the ONLY speed control: at 1.0 the sim runs
    # in REAL TIME (1 real second = 1 simulation second); >1 speeds up, <1 slows
    # down. We advance 1 simulation second per tick and wait 1 real second per
    # tick (divided by the slider multiplier), so slider=1 is true real-time.
    DT_MIN = 1.0 / 60.0        # 1 simulation SECOND advanced per tick
    TICK_SECONDS = 1.0         # 1 real second per tick at slider 1x -> real time

    engine = SimulationEngine(world, dt_min=DT_MIN, seed=42, assign_at_start=True)

    # a lock so operator actions and the tick loop don't race on the world
    lock = asyncio.Lock()

    # The live speed multiplier driven by the UI slider. 1.0 = real time; >1 runs
    # faster (shorter delay per tick), <1 runs slower. Stored in a one-element
    # list so the receive loop can mutate it while the tick loop reads it.
    speed_mult = [1.0]
    # pause flag driven by the PAUSE button; while True the engine holds.
    paused = [False]

    # DATA-REPLAY DVR: accumulate every tick's state so the run can be replayed.
    recorded_frames: list = []
    timestep_log: list = []      # one row per USV per tick
    operator_log: list = []      # one row per operator action
    saved = {"done": False}

    def _log_op(usv_id: str, action: str, detail: str = "") -> None:
        """Record one operator action with the mission time and the USV's mode."""
        w = engine.world
        u = w.usv(usv_id) if usv_id else None
        operator_log.append({
            "mission_min": round(w.mission_time_min, 4),
            "usv_id": usv_id or "",
            "usv_mode": (u.mode.value if u is not None else ""),
            "action": action,
            "detail": detail,
        })

    # message box: a queue of messages to push to the client, a periodic-status
    # counter (which USV reports next, and how long since the last report), and a
    # dedicated RNG so replies/status read naturally but reproducibly.
    msg_queue: list = []
    msg_rng = _random.Random(1234)
    msg_state = {"next_usv": 0, "ticks_since": 0, "was_oor": set()}
    MSG_EVERY_TICKS = 8    # a status report roughly every ~8 ticks (feed flows continuously)

    def _finalize_recording(extra: dict | None = None) -> str | None:
        if saved["done"]:
            return None
        saved["done"] = True
        # Always write the results CSVs for this run (independent of REC).
        try:
            saved["csv"] = _write_results_csv(
                world, _CURRENT_CONFIG, timestep_log, operator_log)
        except Exception:  # noqa: BLE001 — logging must never break the run
            saved["csv"] = None
        # The frame-by-frame recording only if REC captured frames.
        return _save_recording(recorded_frames)

    async def on_tick(w) -> None:
        # per-USV per-tick log row (mode + what it's doing right now)
        t = round(w.mission_time_min, 4)
        for u in w.usvs:
            timestep_log.append({
                "mission_min": t,
                "usv_id": u.id,
                "mode": u.mode.value,
                "x": round(u.position.x, 3),
                "y": round(u.position.y, 3),
                "speed_knots": round(u.effective_speed_knots(), 2),
                "comms_state": u.comms_state.value,
                "fuel_pct": round(u.fuel_pct, 1),
                "assigned_target": u.assigned_target_id or "",
                "activity": _usv_activity(u, w),
            })

        # periodic incoming status report from the USVs (round-robin), flowing
        # continuously in the incoming feed.
        #
        # Comms gating: the moment a USV goes OUT OF RANGE, it sends one final
        # "comms lost" message, then stays silent until it recovers. A USV that is
        # out of range is skipped for the ordinary round-robin status reports.
        oor_now = {u.id for u in w.usvs
                   if not u.left_grid and u.comms_state.value == "out_of_range"}
        # newly out of range this tick -> emit one comms-lost notice each
        for u in w.usvs:
            if u.id in oor_now and u.id not in msg_state["was_oor"]:
                msg_queue.append(_messages.make_incoming(u, w))  # says "comms lost"
        # any USV that recovered drops out of the set so it can report again
        msg_state["was_oor"] = oor_now

        live = [u for u in w.usvs
                if not u.left_grid and u.comms_state.value != "out_of_range"]
        if live:
            msg_state["ticks_since"] += 1
            if msg_state["ticks_since"] >= MSG_EVERY_TICKS:
                msg_state["ticks_since"] = 0
                idx = msg_state["next_usv"] % len(live)
                msg_state["next_usv"] = idx + 1
                msg_queue.append(_messages.make_incoming(live[idx], w))

        payload = _state_payload(w)
        payload["paused"] = paused[0]
        if msg_queue:
            payload["messages"] = list(msg_queue)
            msg_queue.clear()
        recorded_frames.append(payload)   # log this frame for replay
        async with lock:
            await sock.send_json(payload)
        # wait one real second per tick, divided by the slider multiplier
        # (slider 1.0 = real time)
        await asyncio.sleep(TICK_SECONDS / max(0.01, speed_mult[0]))
        # while paused, hold here (re-sending the current frame) so the engine
        # doesn't advance until the operator resumes.
        while paused[0]:
            async with lock:
                await sock.send_json({**_state_payload(w), "paused": True})
            await asyncio.sleep(0.1)

    # receive operator actions concurrently with the sim loop
    async def receive_actions() -> None:
        try:
            while True:
                msg = await sock.receive_json()
                action = msg.get("action")
                if action == "classify":
                    await _handle_classify(world, lock, msg, sock)
                    _log_op(_manual_usv_id(world),
                            "classify",
                            f"{msg.get('boat_id','')} -> {msg.get('perceived','')}")
                elif action == "move_to":
                    await _handle_move_to(world, lock, msg)
                    _log_op(_manual_usv_id(world), "move_to",
                            f"({msg.get('x','')},{msg.get('y','')})")
                elif action == "set_speed":
                    await _handle_set_speed(world, lock, msg)
                    _log_op(_manual_usv_id(world), "set_speed",
                            f"{msg.get('knots','')} knt")
                elif action == "change_target":
                    await _handle_change_target(world, lock, msg)
                    _log_op(msg.get("usv_id",""), "change_target",
                            str(msg.get("boat_id","")))
                # ---- DEV SPEED CONTROL (remove before deployment) ---------- #
                elif action == "set_sim_speed":
                    if not _admin_token_ok(msg.get("admin_token")):
                        continue                    # administrator only
                    try:
                        speed_mult[0] = float(msg.get("mult", 1.0))
                    except (TypeError, ValueError):
                        pass
                elif action == "set_paused":
                    if not _admin_token_ok(msg.get("admin_token")):
                        continue                    # administrator only
                    paused[0] = bool(msg.get("paused", False))
                elif action == "set_rtb":
                    uid = msg.get("usv_id")
                    async with lock:
                        u = engine.world.usv(uid)
                        # an RTB order needs a comms link and >= 20% fuel
                        # (the button is greyed out too); see USV.rtb_blocked_reason
                        blocked = u.rtb_blocked_reason() if u is not None else None
                        if u is not None and not u.left_grid and not blocked:
                            u.rtb = True
                            # disengage immediately: drop any assignment / circle
                            # / escort / intercept so it heads straight for the
                            # nearest edge instead of finishing its current job.
                            u.assigned_target_id = None
                            u.circling_since_min = None
                            u.intercept_target_id = None
                            u.manual_destination = None
                    _log_op(uid or "", "rtb", f"blocked: {blocked}" if blocked
                            else "return to base")
                # ------------------------------------------------------------ #
                elif action == "send_message":
                    text = (msg.get("text") or "").strip()
                    to = msg.get("to") or "ALL USV"
                    if text:
                        async with lock:
                            w = engine.world
                            # echo the operator's message, then the USV reply/ies
                            out = [_messages.operator_message(text, to, w)]
                            out += _messages.replies_to(text, to, w, msg_rng)
                        await sock.send_json({"type": "messages", "messages": out})
                        _log_op("", "message", f"to {to}: {text}")
                # ------------------------------------------------------------ #
                elif action == "start":
                    # a new run replaces the previous run's results
                    if not start_event.is_set():
                        _clear_previous_results()
                    start_event.set()   # operator pressed Start Simulation
        except WebSocketDisconnect:
            engine.stop()
            start_event.set()   # unblock the waiter so the handler can clean up

    # Send the initial scene (frozen at t=0) so the operator can see the setup
    # and arm recording, then wait for the Start action before running the sim.
    start_event = asyncio.Event()
    async with lock:
        intro = _state_payload(world)
        intro["awaiting_start"] = True
        await sock.send_json(intro)

    recv_task = asyncio.create_task(receive_actions())
    await start_event.wait()          # block here until "start" (or disconnect)

    # if the socket disconnected before Start was pressed, clean up and exit
    if recv_task.done():
        engine.stop()
        recv_task.cancel()
        return

    sim_task = asyncio.create_task(
        engine.run(
            duration_min=MISSION_LENGTH_MIN, realtime_factor=0.0, on_tick=on_tick
        )
    )
    try:
        # when the sim finishes its mission length, tell the browser it's over
        await sim_task
        async with lock:
            final = _state_payload(world)
            final["type"] = "state"
            final["mission_complete"] = True
            # DVR: save the recording and tell the browser its filename
            rec_name = _finalize_recording()
            if rec_name:
                final["recording"] = rec_name
            if saved.get("csv"):
                final["results_csv"] = saved["csv"]
            await sock.send_json(final)
        # keep the socket open so the operator can still read the final state /
        # send late classify actions; end when they disconnect
        await recv_task
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        engine.stop()
        _finalize_recording()   # save if not already (early disconnect / cancel)
        for t in (sim_task, recv_task):
            t.cancel()


async def _handle_set_speed(world, lock, msg) -> None:
    """Operator set the manual USV's speed via the stepper. `knots` is the new
    absolute speed (the stepper already stepped it by +/-0.5); the model clamps
    it to a sensible range."""
    knots = msg.get("knots")
    if knots is None:
        return
    manual = next((u for u in world.usvs if u.mode.value == "manual"), None)
    if manual is None:
        return
    async with lock:
        manual.set_operator_speed(float(knots))


async def _handle_change_target(world, lock, msg) -> None:
    """Operator picked a new target for an AGENT USV from the change-target list
    (`usv_id` = the agent USV, `boat_id` = the boat chosen). Any boat on the map
    may be chosen, engaged by another USV or not (see retarget_usv). Without a
    boat_id (older clients) the algorithm picks, as before."""
    usv_id = msg.get("usv_id")
    if usv_id is None:
        return
    async with lock:
        if msg.get("boat_id"):
            retarget_usv(world, usv_id, msg["boat_id"], world.mission_time_min)
        else:
            reassign_usv(world, usv_id)


async def _handle_move_to(world, lock, msg) -> None:
    """Operator drew a move for the manual USV: set its destination point. The
    USV then travels toward it each tick (see manual_step)."""
    x, y = msg.get("x"), msg.get("y")
    if x is None or y is None:
        return
    manual = next((u for u in world.usvs if u.mode.value == "manual"), None)
    if manual is None:
        return
    async with lock:
        manual.manual_destination = world.grid.clamp(Vec2(x=float(x), y=float(y)))


async def _handle_classify(world, lock, msg, sock) -> None:
    """Operator classifies a contact using the manual USV (must be in range).
    `perceived` is one of unknown/friendly/civilian/enemy; enemies also carry a
    tier for display but tier doesn't change the class recorded."""
    boat_id = msg.get("boat_id")
    perceived = msg.get("perceived", "")
    # find the manual USV to attribute the action to
    manual = next((u for u in world.usvs if u.mode.value == "manual"), None)
    if manual is None:
        return
    try:
        as_class = Perception(perceived)
    except ValueError:
        return
    async with lock:
        ok = manual_identify(world, manual.id, boat_id, as_class=as_class)
        result = _state_payload(world)
        result["type"] = "state"
        result["last_action"] = {
            "boat_id": boat_id, "perceived": perceived, "ok": ok,
        }
        await sock.send_json(result)
