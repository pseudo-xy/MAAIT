"""Tests for range-dependent agent sensing (sensing.py) and its use by agents."""

import math
import statistics as st

import pytest

from reschu.engine import SimulationEngine
from reschu.models import USV, Boat, Grid, World
from reschu.observation import observe
from reschu.sensing import (CONTACT_TYPES, SENSOR_RANGE_RATIO, SMOOTH_MIN,
                            VISUAL_RANGE_RATIO, VISUAL_REFRESH_TICKS, smoothing_alpha,
                            visual_accuracy)
from reschu.types import Behavior, Classification, USVMode, Vec2

DT = 1 / 60


def _world(boat_x, noise=1.0, behavior=Behavior.CIVILIAN, visual=True, n_boats=1):
    """Small map (ID range 2.0); one stationary USV at x=5."""
    boats = [Boat(id=f"B{i}", position=Vec2(x=boat_x, y=12 + 0.01 * i),
                  true_class=Classification.CIVILIAN if behavior is Behavior.CIVILIAN
                  else Classification.ENEMY, behavior=behavior,
                  start_pos=Vec2(x=boat_x, y=12)) for i in range(n_boats)]
    return World(grid=Grid(width=40, height=25),
                 usvs=[USV(id="M", position=Vec2(x=5, y=12), mode=USVMode.MANUAL,
                           stationary=True)],
                 boats=boats, sensor_noise=noise, visual_cue=visual)


def test_perfect_sensing_is_exact_and_has_no_camera():
    w = _world(boat_x=20, noise=0.0)
    e = SimulationEngine(w, dt_min=DT, seed=0, assign_at_start=False)
    for _ in range(30):
        e.tick()
    b = w.boats[0]
    ob = observe(w.usvs[0], w, w.mission_time_min).boat(b.id)
    assert ob.position == b.position
    assert ob.speed_knots == b.speed_knots()
    assert ob.camera_report is None


def test_noise_does_not_change_true_motion():
    """Experimental control: sensing noise must not reshuffle the world's RNG."""
    runs = []
    for noise in (0.0, 1.0):
        w = _world(boat_x=20, noise=noise, n_boats=4)
        e = SimulationEngine(w, dt_min=DT, seed=5, assign_at_start=False)
        for _ in range(200):
            e.tick()
        runs.append([(b.position.x, b.position.y) for b in w.boats])
    assert runs[0] == runs[1]


def test_beyond_sensor_range_no_kinematics():
    far_x = 5 + 2.0 * (SENSOR_RANGE_RATIO + 2)        # well beyond sensor range
    w = _world(boat_x=far_x)
    e = SimulationEngine(w, dt_min=DT, seed=1, assign_at_start=False)
    e.tick()
    ob = observe(w.usvs[0], w, w.mission_time_min).boat("B0")
    assert ob.heading is None and ob.speed_knots is None
    assert ob.position is not None


def _pos_err_sd(boat_x, ticks=600):
    w = _world(boat_x=boat_x, behavior=Behavior.STANDOFF)  # keeps roughly its range
    w.boats[0].standoff_speed = 0.0001
    e = SimulationEngine(w, dt_min=DT, seed=2, assign_at_start=False)
    errs = []
    for _ in range(ticks):
        e.tick()
        errs.append(math.hypot(w.boats[0].track_pos_err.x, w.boats[0].track_pos_err.y))
    return st.mean(errs[200:])


def test_errors_grow_with_range():
    assert _pos_err_sd(boat_x=6.0) < _pos_err_sd(boat_x=13.0)


def test_smoothing_and_camera_accuracy_curves():
    assert smoothing_alpha(0.5) == 1.0 and smoothing_alpha(1.0) == 1.0
    assert smoothing_alpha(2.0) < 1.0
    assert smoothing_alpha(10.0) == pytest.approx(SMOOTH_MIN)
    assert smoothing_alpha(3.0) == pytest.approx(1.0 - (1.0 - SMOOTH_MIN) * 2 / 3)
    assert visual_accuracy(0.0) > visual_accuracy(1.0) > visual_accuracy(1.5)


def test_camera_report_iff_in_range_and_accuracy_matches_model():
    hits = n = 0
    expected = []
    for seed in range(15):                          # pool seeds for statistical power
        w = _world(boat_x=6.0)                      # close: in camera range
        b = w.boats[0]
        b.behavior, b.standoff_speed = Behavior.STANDOFF, 0.0001   # near-stationary
        e = SimulationEngine(w, dt_min=DT, seed=seed, assign_at_start=False)
        ids = set()
        for _ in range(VISUAL_REFRESH_TICKS * 100):
            e.tick()
            ob = observe(w.usvs[0], w, w.mission_time_min).boat("B0")
            in_range = b.track_range_ratio <= VISUAL_RANGE_RATIO
            assert (ob.camera_report is not None) == in_range   # a report iff in range
            if in_range and ob.camera_look_id not in ids:
                assert ob.camera_report in CONTACT_TYPES
                ids.add(ob.camera_look_id)
                n += 1
                hits += ob.camera_report == "standoff_low"
                expected.append(visual_accuracy(b.track_range_ratio))
    assert n > 500
    assert abs(hits / n - st.mean(expected)) < 0.05   # matches the camera model


def test_no_camera_out_of_range():
    far = _world(boat_x=5 + 2.0 * 3)                 # 3 ID ranges: out of camera range
    e = SimulationEngine(far, dt_min=DT, seed=3, assign_at_start=False)
    for _ in range(20):
        e.tick()
    assert observe(far.usvs[0], far, far.mission_time_min).boat("B0").camera_report is None


def test_visual_cue_switch():
    w = _world(boat_x=6.0, visual=False)
    e = SimulationEngine(w, dt_min=DT, seed=3, assign_at_start=False)
    for _ in range(20):
        e.tick()
    assert observe(w.usvs[0], w, w.mission_time_min).boat("B0").camera_report is None


def test_tracker_counts_each_camera_look_once():
    from reschu.mdp.features import ContactTracker, WINDOW_TICKS
    w = _world(boat_x=6.0)
    w.boats[0].behavior = Behavior.STANDOFF        # near-stationary: stays in camera range
    w.boats[0].standoff_speed = 0.0001
    e = SimulationEngine(w, dt_min=DT, seed=4, assign_at_start=False)
    tr = ContactTracker()
    wins = []
    for _ in range(WINDOW_TICKS * 20):
        e.tick()
        wins += tr.update(observe(w.usvs[0], w, w.mission_time_min))
    with_look = [x for x in wins if x.camera]
    # never more camera evidence than there were fresh looks
    assert len(with_look) <= w.boats[0].visual_look_id
    assert len(with_look) >= len(wins) - 2
