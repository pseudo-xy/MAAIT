"""Per-USV starting fuel, and the RTB rules (comms link and >= 20% fuel)."""

from reschu.models import RTB_MIN_FUEL_PCT, USV
from reschu.scenario import world_from_config
from reschu.types import CommsState, Vec2


def test_fuel_start_from_settings():
    w = world_from_config({"grid": {"width": 40, "height": 25},
                           "usvs": [{"mode": "manual", "start": "A1", "fuel_start": 45},
                                    {"mode": "mdp", "start": "B2"},
                                    {"mode": "llm", "start": "C3", "fuel_start": 250}],
                           "boats": []})
    assert [u.fuel_pct for u in w.usvs] == [45.0, 100.0, 100.0]


def test_rtb_blocked_below_20_percent_fuel():
    u = USV(id="U", position=Vec2(x=1, y=1))
    u.fuel_pct = 20.0
    assert u.rtb_blocked_reason() is None
    u.fuel_pct = 19.9
    assert "fuel" in u.rtb_blocked_reason()
    assert RTB_MIN_FUEL_PCT == 20.0


def test_rtb_blocked_out_of_comms():
    u = USV(id="U", position=Vec2(x=1, y=1))
    u.comms_state = CommsState.OUT_OF_RANGE
    assert "comms" in u.rtb_blocked_reason()
