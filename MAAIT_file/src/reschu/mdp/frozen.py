"""Freezing and versioning the MDP teammate for human trials.

Every participant must face the SAME teammate. Freezing writes mdp/FROZEN.json:
a version name, the date, a SHA-256 hash of every file that defines the policy
(cue model + the three engagement tables) and the settings that shape its
behavior (rewards, team-assignment settings), plus one short FINGERPRINT over
all of it.

While frozen:
  * the MDP refuses to run if anything changed since the freeze
    (FrozenPolicyError) — trials can't silently use a different teammate;
  * `python -m reschu.mdp.build` / `python -m reschu.mdp.fit` refuse to write
    into the package (pass --out elsewhere, or unfreeze first);
  * every results file names the version and fingerprint that ran.

    python -m reschu.mdp.frozen --freeze mdp-v1.0 --note "pilot study"
    python -m reschu.mdp.frozen --check
    python -m reschu.mdp.frozen --unfreeze        # deliberate: start a new version
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path

MODEL_DIR = Path(__file__).parent
MANIFEST = MODEL_DIR / "FROZEN.json"
POLICY_FILES = ("cue_model.json", "engagement_small.json", "engagement_medium.json",
                "engagement_large.json")


class FrozenPolicyError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_settings() -> dict:
    """Settings outside the policy files that shape the teammate's behavior."""
    from . import assignment as A
    from .engagement import DEFAULT_REWARDS
    return {
        "rewards": DEFAULT_REWARDS,
        "assignment": {k: getattr(A, k) for k in ("ASSIGN_TICKS", "SWITCH_MARGIN",
                                                   "RATE_FLOOR_MIN", "ESCORT_MIN",
                                                   "FAR_REF_RATIO")},
    }


def current_state(directory: Path | None = None) -> dict:
    directory = Path(directory) if directory is not None else MODEL_DIR
    files = {f: _sha(directory / f) for f in POLICY_FILES if (directory / f).exists()}
    settings = current_settings()
    blob = json.dumps({"files": files, "settings": settings}, sort_keys=True).encode()
    return {"files": files, "settings": settings,
            "fingerprint": hashlib.sha256(blob).hexdigest()[:12]}


def is_frozen() -> bool:
    return MANIFEST.exists()


def check() -> dict:
    """{'version', 'fingerprint', 'frozen'} for the policy that would run now.
    Raises FrozenPolicyError if frozen and anything differs from the freeze."""
    st = current_state()
    if not MANIFEST.exists():
        return {"version": "unfrozen", "fingerprint": st["fingerprint"], "frozen": False}
    man = json.loads(MANIFEST.read_text())
    problems = []
    for f, h in man["files"].items():
        if st["files"].get(f) != h:
            problems.append(f"{f} changed")
    if json.dumps(st["settings"], sort_keys=True) != json.dumps(man["settings"], sort_keys=True):
        problems.append("rewards / assignment settings changed")
    if problems:
        raise FrozenPolicyError(
            f"MDP policy {man['version']} is frozen but " + "; ".join(problems) +
            ". Restore the frozen files, or unfreeze deliberately "
            "(python -m reschu.mdp.frozen --unfreeze) and freeze a new version.")
    return {"version": man["version"], "fingerprint": man["fingerprint"], "frozen": True}


def freeze(version: str, note: str = "") -> dict:
    if MANIFEST.exists():
        raise FrozenPolicyError(f"already frozen as {json.loads(MANIFEST.read_text())['version']}"
                                " — unfreeze first")
    missing = [f for f in POLICY_FILES if not (MODEL_DIR / f).exists()]
    if missing:
        raise FrozenPolicyError(f"cannot freeze, missing: {missing}")
    st = current_state()
    man = {"version": version, "frozen_at": _dt.datetime.now().isoformat(timespec="seconds"),
           "note": note, "fingerprint": st["fingerprint"], "files": st["files"],
           "settings": st["settings"]}
    MANIFEST.write_text(json.dumps(man, indent=1))
    return man


def unfreeze() -> None:
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        archive = MODEL_DIR / f"FROZEN-{old['version']}.json"
        archive.write_text(MANIFEST.read_text())      # keep the record of what ran
        MANIFEST.unlink()


def guard_write(out_dir: Path) -> None:
    """Called by build / fit before writing policy files."""
    if is_frozen() and Path(out_dir).resolve() == MODEL_DIR.resolve():
        raise FrozenPolicyError(
            "the MDP policy is frozen for trials; refusing to overwrite it. Write to "
            "another directory with --out, or unfreeze first (python -m reschu.mdp.frozen "
            "--unfreeze).")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--freeze", metavar="VERSION")
    g.add_argument("--check", action="store_true")
    g.add_argument("--unfreeze", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    if a.freeze:
        m = freeze(a.freeze, a.note)
        print(f"frozen {m['version']} fingerprint {m['fingerprint']}")
    elif a.unfreeze:
        unfreeze()
        print("unfrozen (previous manifest archived)")
    else:
        print(check())


if __name__ == "__main__":
    main()
