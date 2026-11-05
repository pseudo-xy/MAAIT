"""The experiment set-up (everything from the settings page) is saved with the results."""
import csv

from reschu.scenario import default_config, world_from_config
from reschu.web import server


def test_setup_csv_has_every_setting(tmp_path):
    cfg = default_config()
    cfg["usvs"][0]["fuel_start"] = 55
    cfg["usvs"][1]["timeline"] = [{"mode": "llm", "from": 0, "to": 0.5},
                                  {"mode": "manual", "from": 0.5, "to": 1}]
    w = world_from_config(cfg)
    path = tmp_path / "setup.csv"
    server._write_setup_csv(path, w, cfg, "2026-11-05_12-00-00")
    rows = list(csv.DictReader(path.open()))
    get = {(r["section"], r["item"], r["setting"]): r["value"] for r in rows}
    assert get[("mission", "", "grid.width")] == "120.0"
    assert get[("mission", "", "mission_minutes")] == "20.0"
    assert get[("usv", "USV-1", "fuel_start")] == "55"
    assert get[("usv", "USV-2", "timeline[2].mode")] == "manual"
    assert get[("boat", "B5", "exit")] == "M40"
    assert get[("run", "", "timestamp")] == "2026-11-05_12-00-00"
    assert len([r for r in rows if r["section"] == "boat" and r["setting"] == "true_class"]) == 6
