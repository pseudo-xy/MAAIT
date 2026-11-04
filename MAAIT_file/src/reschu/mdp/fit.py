"""Fit, calibrate and evaluate the MDP's cue model — one reproducible command.

    python -m reschu.mdp.fit                      # collect fresh data, fit, save
    python -m reschu.mdp.fit --data DIR           # reuse DIR/{train,val,test}.csv
    python -m reschu.mdp.fit --episodes 160 40 60 --minutes 10

Writes src/reschu/mdp/cue_model.json (the fitted model the MDP loads) and
src/reschu/mdp/cue_model_report.md (accuracy/calibration on held-out missions).
Refit whenever boat behaviors, speeds, or speed_spread / speed_jitter change.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
from pathlib import Path

from ..scenario import DEFAULT_SENSOR_NOISE, DEFAULT_SPEED_JITTER, DEFAULT_SPEED_SPREAD
from .cue_model import CLASSES, DEFAULT_PATH, CueModel, evaluate, tune_tempering
from .features import WINDOW_TICKS

SEED_BASE = {"train": 1000, "val": 2000, "test": 3000}


def _read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def report_md(model: CueModel, res: dict) -> str:
    m = model.meta
    lines = [
        "# MDP cue model — fit report", "",
        f"Fitted {m.get('fitted_at', '')}. Window = {WINDOW_TICKS} ticks (10 s). "
        f"Speed variation: spread {m.get('speed_spread')}, jitter {m.get('speed_jitter')}. "
        f"Sensor noise: {m.get('sensor_noise')}. Camera looks in training: {m.get('n_camera_looks')}.",
        f"Training windows: {m.get('n_train_windows')} · tempering: "
        f"tau_speed = {model.tau_speed}, tau_beh = {model.tau_beh}, tau_cam = {model.tau_cam} "
        f"(validation log-loss {m.get('val_log_loss')}).", "",
        "## Held-out accuracy as evidence accumulates", "",
        "| Observed for | Contacts | 5-type accuracy | Declaration accuracy (enemy/friendly/civilian) | Calibration error |",
        "|---|---|---|---|---|",
    ]
    for k, r in res.items():
        lines.append(f"| {int(k) * 10} s | {r['n']} | {r['acc5']:.1%} | {r['acc_declare']:.1%} | "
                     f"{r['ece_declare']:.3f} |")
    k6 = res.get(6) or res.get("6")
    if k6:
        lines += ["", "## Declaration confusion after 60 s (rows = true type)", "",
                  "| True type | → enemy | → friendly | → civilian |", "|---|---|---|---|"]
        for c in CLASSES:
            tot = sum(v for (t, _), v in k6["confusion"].items() if t == c) or 1
            cells = [k6["confusion"].get((c, d), 0) / tot for d in ("enemy", "friendly", "civilian")]
            lines.append(f"| {c} | " + " | ".join(f"{x:.0%}" for x in cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, help="dir with train/val/test.csv (skip collection)")
    ap.add_argument("--save-data", type=Path, help="also write collected CSVs here")
    ap.add_argument("--episodes", type=int, nargs=3, default=[160, 40, 60])
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    a = ap.parse_args(argv)
    from .frozen import guard_write
    guard_write(a.out.parent)

    if a.data:
        splits = {k: _read(a.data / f"{k}.csv") for k in ("train", "val", "test")}
    else:
        from .collect import collect
        splits = {}
        for (name, base), n in zip(SEED_BASE.items(), a.episodes):
            splits[name] = collect(range(base, base + n), a.minutes, a.processes)
            print(f"collected {name}: {len(splits[name])} windows")
            if a.save_data:
                a.save_data.mkdir(parents=True, exist_ok=True)
                _write(a.save_data / f"{name}.csv", splits[name])

    model = CueModel.fit(splits["train"])
    tune_tempering(model, splits["val"])
    res = evaluate(model, splits["test"])
    model.meta.update({
        "fitted_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "speed_spread": DEFAULT_SPEED_SPREAD, "speed_jitter": DEFAULT_SPEED_JITTER,
        "sensor_noise": DEFAULT_SENSOR_NOISE,
        "window_ticks": WINDOW_TICKS,
        "test": {str(k): {kk: v for kk, v in r.items() if kk != "confusion"}
                 for k, r in res.items()},
    })
    model.save(a.out)
    rep = report_md(model, res)
    a.out.with_name("cue_model_report.md").write_text(rep)
    print(rep)


if __name__ == "__main__":
    main()
