"""Smoke test of the board app's web API with a fake Bridge (no hardware)."""
import json
import sys
import types
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PY = HERE.parent / "python"
sys.path.insert(0, str(PY))

stub = types.ModuleType("arduino.app_utils")
stub.Bridge = types.SimpleNamespace(call=lambda *a, **k: "ok" if a[0] == "imu_status" else "0|")
stub.App = types.SimpleNamespace(run=lambda: None)
pkg = types.ModuleType("arduino")
sys.modules.update({"arduino": pkg, "arduino.app_utils": stub})


def call(path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:8000{path}", data=None if body is None else json.dumps(body).encode())
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def test_full_flow(tmp_path, monkeypatch):
    import main
    main.engine.dir = tmp_path
    main.engine.walk_dir = tmp_path / "walks"
    main.engine.walk_dir.mkdir()
    csv = HERE.parent / "Gait Wearable" / "data_ankle" / "walking_20261102-214308.csv"
    if not csv.exists():
        import pytest
        pytest.skip("needs a recorded ankle walk CSV (Gait Wearable/data_ankle)")
    w = pd.read_csv(csv, comment="#")[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy(float).tolist()
    assert b"Campus Nav" in urllib.request.urlopen("http://127.0.0.1:8000/").read()

    def walk(kind="demo", navigate=False):
        assert call("/api/walk/start", {"who": "Jude", "name": "t", "kind": kind, "navigate": navigate})["ok"]
        for i in range(0, len(w), 20):
            main.engine.add_samples(w[i:i + 20])
            if i == 3000:
                call("/api/walk/mark", {"kind": "stairs_down"})
            if i == 4000:
                call("/api/walk/mark", {"kind": "stairs_end"})
                if kind == "dagger":
                    call("/api/walk/correct", {"action": "straight"})
        main.engine.add_samples([w[-1]] * 150)
        call("/api/walk/mark", {"kind": "arrive"})
        return call("/api/walk/stop", {})["id"]

    assert "no policy" in call("/api/walk/start", {"kind": "test", "navigate": True})["error"]
    ids = [walk(), walk()]
    out = call("/api/train", {})
    assert out["ok"] and out["policy"]["rows"] > 50 and len(out["policy"]["loo"]) == 2
    assert len(call("/api/walks")) == 2
    r = call(f"/api/walk/{ids[0]}")
    assert len(r["labels"]) == len(r["walk"]["steps"]) > 80
    assert call(f"/api/walk/{ids[0]}/edit", {"markers": [{"step": 5, "kind": "arrive"}]})["ok"]
    d = walk("dagger", True)
    s = call("/api/state")
    assert s["has_policy"] and s["walk_count"] == 3
    assert call(f"/api/walk/{d}")["walk"]["corrections"]
    assert any("pred" in st for st in call(f"/api/walk/{d}")["walk"]["steps"])
