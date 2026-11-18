"""Replay a recorded IMU CSV through the Campus Nav engine on the laptop (no board needed).

  python replay.py ../"Gait Wearable"/data_ankle/walking_20261102-214308.csv
  python replay.py FILE.csv --policy-walks data/walks      # also print the cue the policy gives

Prints the step count, the heading and detected turns. Handy for checking the pipeline and for
testing against walks pulled off the board (their .csv files are in the board's data/walks/).
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "board" / "campus-nav" / "python"))
from engine import Engine  # noqa: E402
from stepnet_np import LiveStepper, StepNetNP  # noqa: E402


def replay(csv, data_dir=None, chunk=10):
    data_dir = data_dir or tempfile.mkdtemp()
    eng = Engine(LiveStepper(StepNetNP(HERE / "board" / "campus-nav" / "python" / "stepnet.npz")), data_dir)
    w = pd.read_csv(csv, comment="#")[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy(float)
    eng.start_walk("replay", Path(csv).name, "test", navigate=False)
    for i in range(0, len(w), chunk):
        eng.add_samples(w[i:i + chunk].tolist())
    eng.add_samples([w[-1].tolist()] * 150)          # flush: steps are confirmed 1 s late
    return eng


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    a = ap.parse_args()
    e = replay(a.csv)
    st = e.state()
    print(f"steps {st['steps']}  final heading {st['heading']:.0f} deg  turns {st['turns']}")
    print("turn events (step, snapped deg):", e.turns.events)
    print("step headings:", np.round([y for y in e.turns.ys], 0).astype(int).tolist())
