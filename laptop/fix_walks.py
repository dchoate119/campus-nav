"""One-off: recompute the per-step heading ("yaw") in walks recorded before the gyro-bias fix.

  python fix_walks.py ../data/walks          # dry run: prints old vs new turns
  python fix_walks.py ../data/walks --write  # rewrites the wNN.json files

Each walk's raw CSV is replayed through the engine (which now measures the gyro bias over the
first second). Each saved step gets the yaw of the replayed step closest in time; everything
else in the walk (markers, corrections, stairs features) is left alone. Walks run in order, so a
walk that started moving reuses the previous walk's bias. Then copy data/walks back to the board.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
import nav_core as nc  # noqa: E402
from engine import Engine  # noqa: E402
from stepnet_np import LiveStepper, StepNetNP  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("walk_dir")
ap.add_argument("--write", action="store_true")
a = ap.parse_args()

eng = Engine(LiveStepper(StepNetNP(HERE.parent / "python" / "stepnet.npz")), tempfile.mkdtemp())
for jp in sorted(Path(a.walk_dir).glob("w*.json")):
    walk = json.loads(jp.read_text())
    csv = jp.with_suffix(".csv")
    if not csv.exists() or not walk["steps"]:
        print(f"{jp.stem}: skipped (no csv or no steps)")
        continue
    raw = pd.read_csv(csv, comment="#")[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy(float)
    eng.start_walk("fix", jp.stem, "test", navigate=False)
    for i in range(0, len(raw), 10):
        eng.add_samples(raw[i:i + 10].tolist())
    eng.add_samples([raw[-1].tolist()] * 150)            # flush: steps are confirmed 1 s late
    new = eng.walk["steps"]
    eng.stop_walk()
    nt = np.array([s["t"] for s in new])
    old_yaws = [s["yaw"] for s in walk["steps"]]
    for s in walk["steps"]:
        s["yaw"] = new[int(np.argmin(abs(nt - s["t"])))]["yaw"]
    bias = "moved, reused last bias" if not eng.calib_ok else "bias " + str(np.round(eng.yaw.bias, 1).tolist())
    print(f"{jp.stem} ({walk['kind']}, {bias})")
    print("   old turns:", nc.replay_turns(old_yaws)[1])
    print("   new turns:", nc.replay_turns([s["yaw"] for s in walk["steps"]])[1])
    if a.write:
        jp.write_text(json.dumps(walk))
if not a.write:
    print("\ndry run: add --write to save")
