import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board" / "campus-nav" / "python"))
import nav_core as nc  # noqa: E402


def rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def simulate_foot(total_yaw_deg, turn_t0, turn_t1, seconds=12, rate=100):
    """A walking foot: pitches up to 40 deg each step (swing), heading ramps total_yaw_deg
    between turn_t0 and turn_t1. Returns IMU samples (g, deg/s) as a body-frame sensor reads."""
    dt = 1 / rate
    R_prev = None
    out = []
    for i in range(int(seconds * rate)):
        t = i * dt
        frac = min(max((t - turn_t0) / (turn_t1 - turn_t0), 0), 1)
        yaw = math.radians(total_yaw_deg) * frac
        pitch = math.radians(40) * math.sin(math.pi * 0.9 * t) ** 2
        R = rot([0, 0, 1], yaw) @ rot([0, 1, 0], pitch)          # body -> world
        if R_prev is not None:
            Wb = R.T @ (R - R_prev) / dt                          # body-frame angular velocity (skew)
            w = np.array([Wb[2, 1], Wb[0, 2], Wb[1, 0]])
        else:
            w = np.zeros(3)
        R_prev = R
        up_body = R.T @ np.array([0, 0, 1.0])                     # accelerometer reads +1 g up
        out.append(list(up_body) + list(np.degrees(w)))
    return out


def test_yaw_survives_foot_swing():
    for target in (90, -90, 45):
        yt = nc.YawTracker()
        for s in simulate_foot(target, 4, 6):
            yt.update(s)
        assert abs(yt.yaw - target) < 15, (target, yt.yaw)


def test_turn_tracker_finds_one_left_turn():
    yaws = [0.0] * 6 + [10, 30, 55, 80, 92, 91] + [90.0] * 8
    states, events = nc.replay_turns(yaws)
    assert len(events) == 1 and events[0][1] == 90
    assert states[-1]["heading"] == 90 and states[-1]["turns"] == 1


def test_turn_tracker_ignores_drift_and_wobble():
    yaws = list(np.linspace(0, 12, 40) + np.random.default_rng(0).normal(0, 3, 40))
    assert nc.replay_turns(yaws)[1] == []


def test_labels_and_policy_roundtrip():
    def walk(shift):
        yaws = [0.0] * (10 + shift) + [30, 60, 90, 91] + [90.0] * 10
        n = len(yaws)
        return {"id": f"w{shift}", "steps": [{"yaw": y} for y in yaws],
                "markers": [{"step": n - 3, "kind": "arrive"}], "corrections": []}
    walks = [walk(s) for s in (0, 1, 2, 3)]
    X, lab = nc.walk_rows(walks[0])
    assert "left" in lab and lab[-1] == "arrive"
    pol = nc.fit_policy(walks[:3])
    ev = nc.evaluate_loo(walks)
    assert all(e["acc"] > 0.7 for e in ev), ev
    assert pol.predict(X[0])[0] == "straight"


def _stairs_walk(wid, seed):
    rng = np.random.default_rng(seed)
    base = {"flat": [0.55, 1.0, 0.6, 0.8, 0.5, 0.6, 1.0, 0.5],
            "up": [0.6, 1.5, 0.9, 0.7, 0.5, 1.0, 1.5, 0.8],
            "down": [0.45, 1.1, 0.7, 1.6, 0.9, 0.6, 1.1, 0.5]}
    seq = ["flat"] * 12 + ["down"] * 10 + ["flat"] * 8 + ["up"] * 6 + ["flat"] * 6
    steps = [{"yaw": 0.0, "f": list(np.array(base[c]) + rng.normal(0, 0.06, 8))} for c in seq]
    n = len(seq)
    return {"id": wid, "steps": steps, "corrections": [],
            "markers": [{"step": 12, "kind": "stairs_down"}, {"step": 21, "kind": "stairs_end"},
                        {"step": 30, "kind": "stairs_up"}, {"step": 35, "kind": "stairs_end"},
                        {"step": n - 2, "kind": "arrive"}]}


def test_stairs_detector_learns_and_tracks():
    walks = [_stairs_walk(f"s{i}", i) for i in range(3)]
    ev = nc.evaluate_stairs_loo(walks)
    assert all(e["acc"] > 0.85 and e["stairs_recall"] > 0.8 for e in ev), ev
    clf = nc.fit_stairs(walks[:2])
    states = nc.replay_stairs(walks[2], clf)
    assert [s for s, _ in states[14:19]] == ["down"] * 5          # sensed within a few steps of entering
    assert states[0][0] == "flat" and states[-1][0] == "flat"
    X, lab = nc.walk_rows(walks[2], clf)
    assert X.shape[1] == 6 and X[16, 4] == -2.0                    # on-stairs-down flag reaches the policy
    assert "stairs_down" in lab and "stairs_up" in lab
