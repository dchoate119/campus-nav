"""Campus Nav engine: ties the step counter, heading, turn detector, walk recorder and policy
together. No hardware here: feed it IMU samples (add_samples) and it works the same on the
UNO Q and in a laptop replay (see Campus Nav/replay.py)."""
import json
import threading
from collections import deque
import time
from pathlib import Path

import nav_core as nc

RATE_HZ = 100


class Engine:
    def __init__(self, stepper, data_dir):
        self.stepper = stepper
        self.dir = Path(data_dir)
        self.walk_dir = self.dir / "walks"
        self.walk_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.yaw = nc.YawTracker(RATE_HZ)
        self.turns = nc.TurnTracker()
        self.steps = 0
        self.mode = "idle"              # idle | record | navigate
        self.walk = None
        self.csv = None
        self.policy = None
        self.policy_info = None
        self.smoother = nc.CueSmoother()
        self.cue = {"action": "straight", "shown": "straight", "conf": 0.0, "votes": {}}
        self.last_autosave = 0.0
        self.samples_seen = 0
        self.raw = deque(maxlen=600)        # last 6 s of IMU, for the per-step stairs features
        self.prev_step = None
        self.stairs_clf = None
        self.stairs_tracker = nc.StairsTracker()
        self.stairs = ("flat", 0)
        self.last_step_t = None
        self._load_policy()

    # ---------------------------------------------------------------- data flow

    def add_samples(self, samples):
        with self.lock:
            for s in samples:
                self.yaw.update(s)
            self.raw.extend(samples)
            if self.csv:
                for s in samples:
                    self.csv.write(f"{self.samples_seen / RATE_HZ:.2f}," + ",".join(f"{v:g}" for v in s) + "\n")
            self.samples_seen += len(samples)
            for a in self.stepper.add(samples):
                self._on_step(a)
            now = time.monotonic()
            if self.walk and now - self.last_autosave > 15:
                self._save_walk()

    def _on_step(self, sample_index):
        y = self.yaw.at(sample_index)
        self.turns.update(y)
        self.steps += 1
        # how the foot moved since the last step -> flat / up / down
        start = self.samples_seen - len(self.raw)
        lo = (self.prev_step if self.prev_step is not None else sample_index - 60) - start
        hi = sample_index + 10 - start
        f = nc.step_features(list(self.raw)[max(0, lo):max(0, hi)])
        self.prev_step, self.last_step_t = sample_index, time.monotonic()
        guess = self.stairs_clf.predict(f) if self.stairs_clf else "flat"
        self.stairs = self.stairs_tracker.update(guess)
        st = {**self.turns.state(), "stairs": self.stairs[0], "stairs_steps": self.stairs[1]}
        pred = None
        if self.policy is not None:
            action, conf, votes = self.policy.predict(nc.features(st))
            shown = self.smoother.update(action)
            self.cue = {"action": action, "shown": shown, "conf": round(conf, 2), "votes": votes}
            pred = action
        if self.walk is not None:
            self.walk["steps"].append({"k": st["step"] - 1, "t": round(self.samples_seen / RATE_HZ, 2),
                                       "yaw": round(y, 1),
                                       **({"f": [round(v, 3) for v in f]} if f else {}),
                                       "sensed": self.stairs[0], **({"pred": pred} if pred else {})})

    def zero(self):
        """Forget heading and step count (start of a new route)."""
        with self.lock:
            self.yaw.reset()
            self.turns = nc.TurnTracker()
            self.stepper.reset()
            self.steps = 0
            self.samples_seen = 0
            self.raw.clear()
            self.prev_step = self.last_step_t = None
            self.stairs_tracker, self.stairs = nc.StairsTracker(), ("flat", 0)
            self.smoother = nc.CueSmoother()
            self.cue = {"action": "straight", "shown": "straight", "conf": 0.0, "votes": {}}

    # ---------------------------------------------------------------- walks

    def _next_id(self):
        nums = [int(p.stem[1:]) for p in self.walk_dir.glob("w*.json") if p.stem[1:].isdigit()]
        return f"w{(max(nums) + 1 if nums else 1):02d}"

    def start_walk(self, who, name, kind, navigate):
        """kind: 'demo' (you are the expert), 'dagger' (policy guides, you correct it), 'test' (just
        guide, not saved) or 'stairs' (short stairwell practice for the stairs detector only; not
        a navigation walk, so it does not count toward the 10)."""
        with self.lock:
            if self.mode != "idle":
                self.stop_walk()
            if navigate and self.policy is None:
                raise ValueError("no policy yet: record demos and press Train first")
            self.zero()
            self.walk = {"id": self._next_id(), "name": name or "", "who": who or "?", "kind": kind,
                         "created": time.strftime("%Y-%m-%d %H:%M:%S"), "include": kind != "test",
                         "steps": [], "markers": [], "corrections": []}
            self.mode = "navigate" if navigate else "record"
            if kind != "test":
                self.csv = open(self.walk_dir / f"{self.walk['id']}.csv", "w")
                self.csv.write("t,ax,ay,az,gx,gy,gz\n")
            self._save_walk()
            return self.walk["id"]

    def mark(self, kind):
        with self.lock:
            if self.walk is None:
                return
            self.walk["markers"].append({"step": self.steps, "kind": kind, "t": round(self.samples_seen / RATE_HZ, 1)})

    def correct(self, action):
        """DAgger: 'the right answer here is <action>'."""
        with self.lock:
            if self.walk is None or action not in nc.ACTIONS:
                return
            self.walk["corrections"].append({"step": max(0, self.steps - 1), "action": action,
                                             "policy_said": self.cue["shown"]})

    def stop_walk(self):
        with self.lock:
            if self.walk is None:
                return None
            wid = self.walk["id"]
            self.walk["seconds"] = round(self.samples_seen / RATE_HZ)
            if self.walk["kind"] != "test":
                self._save_walk()
            if self.csv:
                self.csv.close()
                self.csv = None
            self.walk, self.mode = None, "idle"
            return wid

    def _save_walk(self):
        self.last_autosave = time.monotonic()
        if self.walk["kind"] == "test":
            return
        tmp = self.walk_dir / f"{self.walk['id']}.json.tmp"
        tmp.write_text(json.dumps(self.walk))
        tmp.replace(self.walk_dir / f"{self.walk['id']}.json")

    def list_walks(self):
        out = []
        for p in sorted(self.walk_dir.glob("w*.json")):
            try:
                out.append(json.loads(p.read_text()))
            except ValueError:
                pass
        return out

    def get_walk(self, wid):
        p = self.walk_dir / f"{wid}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def save_edit(self, wid, markers=None, include=None, name=None, corrections=None):
        with self.lock:
            if self.walk and self.walk["id"] == wid:
                raise ValueError("stop the walk before editing it")
            w = self.get_walk(wid)
            if w is None:
                raise ValueError("no such walk")
            if markers is not None:
                w["markers"] = [{"step": int(m["step"]), "kind": m["kind"]} for m in markers]
            if corrections is not None:
                w["corrections"] = corrections
            if include is not None:
                w["include"] = bool(include)
            if name is not None:
                w["name"] = name
            (self.walk_dir / f"{wid}.json").write_text(json.dumps(w))
            return w

    def delete_walk(self, wid):
        with self.lock:
            for ext in ("json", "csv"):
                p = self.walk_dir / f"{wid}.{ext}"
                if p.exists():
                    p.unlink()

    def review(self, wid):
        """What the labeller made of a walk: per-step yaw, detected turns, labels."""
        w = self.get_walk(wid)
        if w is None:
            return None
        yaws = [s["yaw"] for s in w["steps"]]
        _, turns = nc.replay_turns(yaws)
        lab = nc.label_walk(len(yaws), turns, w["markers"], w.get("corrections", []))
        return {"walk": w, "turns": [{"step": s, "deg": d} for s, d in turns], "labels": lab}

    # ---------------------------------------------------------------- policy

    def train(self):
        """Fit the policy on every included walk and report leave-one-walk-out scores."""
        with self.lock:
            allw = [w for w in self.list_walks() if w.get("include") and w["kind"] != "test" and w["steps"]]
            walks = [w for w in allw if w["kind"] in ("demo", "dagger")]
            if not walks:
                raise ValueError("no walks to train on")
            self.stairs_clf = nc.fit_stairs(allw)                 # stairs practice walks help here too
            self.policy = nc.fit_policy(walks, allw)
            loo = nc.evaluate_loo(walks) if len(walks) > 1 else []
            sloo = nc.evaluate_stairs_loo(allw) if len(allw) > 1 else []
            self.policy_info = {"walks": [w["id"] for w in walks], "rows": len(self.policy.y),
                                "loo": loo, "stairs_loo": sloo, "stairs_rows": len(self.stairs_clf.y),
                                "stairs_counts": {c: self.stairs_clf.y.count(c) for c in nc.STAIRS},
                                "trained": time.strftime("%H:%M:%S")}
            self.smoother = nc.CueSmoother()
            (self.dir / "policy_info.json").write_text(json.dumps(self.policy_info))
            return self.policy_info

    def _load_policy(self):
        """After a reboot: refit from the saved walks (cheap, kNN has no weights)."""
        try:
            if any(w.get("include") for w in self.list_walks()):
                self.train()
        except Exception as e:  # never stop the app over this
            print("couldn't rebuild the policy:", repr(e))

    # ---------------------------------------------------------------- state for the page

    def state(self):
        with self.lock:
            st = self.turns.state()
            total = len([w for w in self.list_walks() if w["kind"] in ("demo", "dagger")])
            idle = self.last_step_t is None or time.monotonic() - self.last_step_t > 3.0
            activity = ("standing" if idle else "turning" if st["turning"]
                        else {"up": "stairs up", "down": "stairs down"}.get(self.stairs[0], "walking"))
            return {"mode": self.mode, "steps": self.steps, "heading": round(self.yaw.yaw, 1),
                    "snapped_heading": st["heading"], "turns": st["turns"], "turning": st["turning"],
                    "since_turn": st["since_turn"], "activity": activity, "cue": self.cue, "has_policy": self.policy is not None,
                    "walk": None if self.walk is None else {k: self.walk[k] for k in ("id", "name", "who", "kind")}
                    | {"markers": len(self.walk["markers"]), "corrections": len(self.walk["corrections"])},
                    "walk_count": total, "max_walks": nc.MAX_WALKS, "policy": self.policy_info}
