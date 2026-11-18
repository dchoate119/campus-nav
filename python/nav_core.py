"""Navigation brain: heading from the ankle gyro, turn detection, labels, and the cloned policy.

Pure Python + NumPy, no board code, so the same file runs on the UNO Q and in the laptop tests.

Pipeline (everything is per STEP, because the step counter is the base layer, not raw IMU):
  1. YawTracker      gyro + accel -> heading (deg) at every sample; we read it at each step.
  2. TurnTracker     step headings -> detected turns, a snapped heading, steps since the last turn.
  3. label_walk      a recorded walk (steps + markers + corrections) -> one action per step.
  4. KNNPolicy       behavior cloning: state features -> action (k-nearest-neighbours over all
                     the labelled steps of the training walks).
"""
import math

import numpy as np

ACTIONS = ["straight", "left", "right", "stairs_down", "stairs_up", "arrive"]
LEAD = 4              # a cue is given this many steps BEFORE the event it announces
TURN_AFTER = 3        # ... and stays on for this many steps after the turn starts
MAX_WALKS = 10        # the assignment's budget: demos + DAgger walks, both partners together


# ------------------------------------------------------------------ heading

class YawTracker:
    """Heading about the vertical, from a foot-mounted 6-axis IMU.

    A foot swings and pitches a lot, so just integrating one gyro axis is useless. Instead we
    track which way 'up' is in the sensor frame (u: a complementary filter between the gyro,
    which rotates it, and the accelerometer, which pulls it toward the measured gravity when the
    foot is quiet) and integrate only the gyro component along u: yaw_rate = omega . u.
    Left turns are positive (counter-clockwise seen from above).
    """

    def __init__(self, rate_hz=100):
        self.dt = 1.0 / rate_hz
        self.u = np.array([0.0, 0.0, 1.0])
        self.primed = False
        self.yaw = 0.0
        self.history = []             # yaw after every sample since reset (index = sample number)

    def reset(self):
        """Zero the heading and the sample counter (keeps the 'up' estimate)."""
        self.yaw = 0.0
        self.history = []

    def update(self, s):
        a = np.array(s[:3], float)
        w = np.radians(np.array(s[3:], float))
        an = np.linalg.norm(a)
        if not self.primed and an > 0.5:
            self.u, self.primed = a / an, True
        # rotate u by the gyro: du/dt = -omega x u
        self.u = self.u - np.cross(w, self.u) * self.dt
        self.u /= np.linalg.norm(self.u)
        if 0.8 < an < 1.25:                              # accelerometer ~ pure gravity: trust it a bit
            quiet = np.linalg.norm(w) < 1.0              # (rad/s) foot nearly still: trust it a lot
            k = 0.2 if quiet else 0.01
            self.u = (1 - k) * self.u + k * a / an
            self.u /= np.linalg.norm(self.u)
        self.yaw += math.degrees(float(np.dot(w, self.u))) * self.dt
        self.history.append(self.yaw)

    def at(self, sample_index):
        if not self.history:
            return 0.0
        i = min(max(int(sample_index), 0), len(self.history) - 1)
        return self.history[i]


# ------------------------------------------------------------------ turns

def wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


class TurnTracker:
    """Causal turn detector over per-step headings. Used identically on demo data (replayed)
    and live, so the policy sees the same features in training and when navigating.

    A turn starts when the heading leaves the current straight-line heading by > START_DEG and
    ends when it stops changing. Its size is snapped to a multiple of 45 degrees (corridors are
    mostly square), which also throws away gyro drift inside each segment.
    """
    START_DEG, SETTLE_DEG, MIN_TURN = 30.0, 8.0, 45.0

    def __init__(self):
        self.ys, self.events = [], []   # events: (start_step, snapped_deg), appended when the turn ENDS
        self.h0 = None
        self.turning = False
        self.start = 0
        self.heading = 0.0              # sum of snapped turns so far (deg, left positive)
        self.last_turn = None           # step index where the last turn started

    def update(self, y):
        k = len(self.ys)
        self.ys.append(y)
        if self.h0 is None:
            self.h0 = y
            return None
        ev = None
        if not self.turning:
            if abs(y - self.h0) < 15:
                self.h0 += 0.1 * (y - self.h0)                  # follow slow gyro drift
            elif abs(y - self.h0) >= self.START_DEG:
                self.turning = True
                lo = max(0, k - 8)
                self.start = next((j for j in range(lo, k + 1) if abs(self.ys[j] - self.h0) > 12), k)
        else:
            settled = k - self.start >= 3 and abs(y - self.ys[k - 2]) < self.SETTLE_DEG
            if settled or k - self.start > 14:
                turn = y - self.h0
                if abs(turn) >= self.MIN_TURN:
                    snapped = round(turn / 45.0) * 45.0
                    self.heading = wrap(self.heading + snapped)
                    self.last_turn = self.start
                    ev = (self.start, snapped)
                    self.events.append(ev)
                self.turning = False
                self.h0 = y
        return ev

    def state(self):
        k = len(self.ys) - 1
        since = k - self.last_turn if self.last_turn is not None else k + 1
        return {"step": k + 1, "heading": self.heading, "since_turn": since,
                "turns": len(self.events), "turning": self.turning}


def replay_turns(yaws):
    """Run a TurnTracker over a whole walk: -> (per-step state dicts, turn events)."""
    t = TurnTracker()
    states = []
    for y in yaws:
        t.update(y)
        states.append(t.state())
    return states, t.events


def features(st):
    """State dict -> the 6 numbers the policy looks at, scaled to be comparable:
    steps, heading, steps since the last turn, turns so far, on-stairs flag (+1 up, -1 down), steps on them."""
    s = {"flat": 0.0, "up": 1.0, "down": -1.0}[st.get("stairs", "flat")]
    return [st["step"] / 10.0, st["heading"] / 45.0, min(st["since_turn"], 40) / 10.0, float(st["turns"]),
            s * 2.0, min(st.get("stairs_steps", 0), 30) / 10.0]


# ------------------------------------------------------------------ stairs

STAIRS = ["flat", "up", "down"]


def step_features(w):
    """IMU samples (n, 6; g, deg/s) between the previous step and this one -> 8 numbers that
    describe how the foot moved. Stairs change the stride: the foot lifts higher going up (bigger
    swing), lands harder and shorter going down. Fixed scales, like the step net's inputs."""
    w = np.asarray(w, float)
    if len(w) < 5:
        return None
    a, g = w[:, :3], w[:, 3:]
    amag = np.linalg.norm(a, axis=1)
    gmag = np.linalg.norm(g, axis=1)
    rng = g.max(0) - g.min(0)
    return [len(w) / 100.0, gmag.max() / 300.0, gmag.mean() / 100.0, (amag.max() - 1.0) / 2.0,
            amag.std() / 0.5, rng[0] / 300.0, rng[1] / 300.0, rng[2] / 300.0]


class StairsClassifier:
    """kNN over the per-step features: flat / up / down. Votes are weighted so the (far more
    common) flat steps do not drown out the stairs steps."""
    K = 5

    def __init__(self):
        self.F = np.zeros((0, 8))
        self.y = []
        self.w = {}

    def fit(self, F, y):
        self.F, self.y = np.asarray(F, float).reshape(-1, 8), list(y)
        self.w = {c: 1.0 / max(1, self.y.count(c)) ** 0.5 for c in STAIRS}
        return self

    def predict(self, f):
        if len(self.y) < self.K or f is None or len(set(self.y)) < 2:
            return "flat"
        d = np.linalg.norm(self.F - np.asarray(f, float), axis=1)
        votes = {}
        for i in np.argsort(d)[: self.K]:
            votes[self.y[i]] = votes.get(self.y[i], 0.0) + self.w[self.y[i]] / (d[i] + 0.1)
        return max(votes, key=votes.get)


class StairsTracker:
    """Smooths per-step flat/up/down guesses into a state. Enter stairs when 3 of the last 5
    steps agree; leave when 4 of the last 5 are flat."""

    def __init__(self):
        self.recent, self.state, self.steps_in = [], "flat", 0

    def update(self, guess):
        self.recent = (self.recent + [guess])[-5:]
        if self.state == "flat":
            for c in ("up", "down"):
                if self.recent.count(c) >= 3:
                    self.state, self.steps_in = c, 0
        elif self.recent.count("flat") >= 4:
            self.state, self.steps_in = "flat", 0
        if self.state != "flat":
            self.steps_in += 1
        return self.state, self.steps_in


def stairs_labels(n_steps, markers):
    """Per-step truth ('flat'/'up'/'down') from the stairs start/end markers."""
    lab = ["flat"] * n_steps
    kind, at = None, 0
    for m in sorted(markers, key=lambda m: m["step"]):
        if m["kind"] in ("stairs_down", "stairs_up"):
            kind, at = ("down" if m["kind"] == "stairs_down" else "up"), m["step"]
        elif m["kind"] == "stairs_end" and kind:
            for j in range(max(0, at), min(n_steps, m["step"] + 1)):
                lab[j] = kind
            kind = None
    if kind:
        for j in range(max(0, at), n_steps):
            lab[j] = kind
    return lab


def stairs_data(walks):
    """-> (F, y) of per-step features and marker-derived truth from saved walks."""
    F, y = [], []
    for w in walks:
        lab = stairs_labels(len(w["steps"]), w.get("markers", []))
        for s, l in zip(w["steps"], lab):
            if s.get("f"):
                F.append(s["f"])
                y.append(l)
    return F, y


def fit_stairs(walks):
    return StairsClassifier().fit(*stairs_data(walks))


def replay_stairs(walk, clf):
    """Causal (state, steps_in) per step for a saved walk, as the board would have sensed it."""
    tr = StairsTracker()
    return [tr.update(clf.predict(s.get("f")) if clf else "flat") for s in walk["steps"]]


# ------------------------------------------------------------------ labels

def label_walk(n_steps, turns, markers, corrections=()):
    """One action per step.

    turns: [(start_step, snapped_deg)] from the TurnTracker (not tapped: taps come too late).
    markers: [{"step": k, "kind": "stairs_down"|"stairs_up"|"stairs_end"|"arrive"}], editable in the UI.
    corrections: [{"step": k, "action": a}] DAgger: expert label for LEAD steps starting at k.
    """
    lab = ["straight"] * n_steps

    def put(lo, hi, action):
        for j in range(max(0, lo), min(n_steps, hi + 1)):
            lab[j] = action

    for start, deg in turns:
        put(start - LEAD, start + TURN_AFTER, "left" if deg > 0 else "right")
    open_kind, open_at = None, 0
    for m in sorted(markers, key=lambda m: m["step"]):
        if m["kind"] in ("stairs_down", "stairs_up"):
            open_kind, open_at = m["kind"], m["step"]
        elif m["kind"] == "stairs_end" and open_kind:
            put(open_at - LEAD, m["step"], open_kind)
            open_kind = None
        elif m["kind"] == "arrive":
            put(m["step"] - 2, n_steps, "arrive")
    if open_kind:                                   # forgot to tap "end": stairs run to the last step
        put(open_at - LEAD, n_steps, open_kind)
    for c in corrections:
        put(c["step"], c["step"] + LEAD - 1, c["action"])
    return lab


def walk_rows(walk, stairs_clf=None):
    """A saved walk dict -> (X (m, 6), labels list) with one row per labelled step.
    Steps after the arrive marker are dropped. The on-stairs inputs are what the stairs
    detector would have sensed (stairs_clf), not the tapped markers, so training matches navigating."""
    yaws = [s["yaw"] for s in walk["steps"]]
    if not yaws:
        return np.zeros((0, 6)), []
    states, turns = replay_turns(yaws)
    for st, (name, n_in) in zip(states, replay_stairs(walk, stairs_clf)):
        st["stairs"], st["stairs_steps"] = name, n_in
    lab = label_walk(len(yaws), turns, walk.get("markers", []), walk.get("corrections", []))
    end = len(yaws)
    for m in walk.get("markers", []):
        if m["kind"] == "arrive":
            end = min(end, m["step"] + 2)
    X = np.array([features(s) for s in states[:end]])
    return X, lab[:end]


# ------------------------------------------------------------------ the cloned policy

class KNNPolicy:
    """Behavior cloning with k-nearest-neighbours: the action is the (distance-weighted) vote of
    the K most similar steps in the demonstrations. No training loop: 'fit' = remember the data,
    which is why it can run on the board right after a DAgger walk."""
    K = 7

    def __init__(self):
        self.X = np.zeros((0, 6))
        self.y = []

    def fit(self, X, y):
        self.X, self.y = np.asarray(X, float), list(y)
        return self

    def predict(self, f):
        """-> (action, confidence 0..1, {action: vote share})"""
        if len(self.y) == 0:
            return "straight", 0.0, {}
        d = np.linalg.norm(self.X - np.asarray(f, float), axis=1)
        near = np.argsort(d)[: self.K]
        votes = {}
        for i in near:
            votes[self.y[i]] = votes.get(self.y[i], 0.0) + 1.0 / (d[i] + 0.3)
        tot = sum(votes.values())
        best = max(votes, key=votes.get)
        return best, votes[best] / tot, {a: round(v / tot, 2) for a, v in votes.items()}


def fit_policy(walks, stairs_walks=None):
    """Policy from the navigation walks. The stairs detector it relies on is trained with
    leave-one-walk-out predictions (when there are other walks), so the stairs inputs carry
    realistic detector errors, as they will on the real route."""
    Xs, ys = [], []
    pool = stairs_walks if stairs_walks is not None else walks
    for w in walks:
        others = [x for x in pool if x.get("id") != w.get("id")]
        clf = fit_stairs(others) if others else fit_stairs(pool)
        X, lab = walk_rows(w, clf)
        if len(lab):
            Xs.append(X)
            ys += lab
    return KNNPolicy().fit(np.vstack(Xs) if Xs else np.zeros((0, 6)), ys)


def evaluate_loo(walks):
    """Leave-one-walk-out: train on all other walks, run the policy over the held-out one.
    Returns per-walk {id, steps, acc, turn_hit, turn_total, false_turn_cues}."""
    out = []
    for i, w in enumerate(walks):
        rest = [x for j, x in enumerate(walks) if j != i]
        if not rest:
            continue
        pol = fit_policy(rest)
        X, lab = walk_rows(w, fit_stairs(rest))
        if not len(lab):
            continue
        pred = [pol.predict(f)[0] for f in X]
        acc = float(np.mean([p == t for p, t in zip(pred, lab)]))
        true_turns = [(k, t) for k, t in enumerate(lab) if t in ("left", "right") and (k == 0 or lab[k - 1] != t)]
        hit = sum(1 for k, t in true_turns if t in pred[max(0, k - 2): k + LEAD + TURN_AFTER + 2])
        false = sum(1 for k, p in enumerate(pred)
                    if p in ("left", "right") and (k == 0 or pred[k - 1] != p)
                    and not any(t == p for t in lab[max(0, k - 4): k + 5]))
        out.append({"id": w.get("id"), "steps": len(lab), "acc": round(acc, 3), "turn_hit": hit,
                    "turn_total": len(true_turns), "false_turn_cues": false})
    return out


class CueSmoother:
    """Show a cue only once it has been the policy's answer on 2 of the last 3 steps (stops flicker)."""

    def __init__(self):
        self.recent, self.shown = [], "straight"

    def update(self, action):
        self.recent = (self.recent + [action])[-3:]
        for a in set(self.recent):
            if self.recent.count(a) >= 2:
                self.shown = a
        return self.shown


def evaluate_stairs_loo(walks):
    """Leave-one-walk-out accuracy of the stairs detector: {id, steps, stairs_steps, acc, stairs_recall}."""
    out = []
    for i, w in enumerate(walks):
        rest = [x for j, x in enumerate(walks) if j != i]
        F, y = stairs_data([w])
        if not rest or not F:
            continue
        clf = fit_stairs(rest)
        pred = [clf.predict(f) for f in F]
        st = [k for k, t in enumerate(y) if t != "flat"]
        out.append({"id": w.get("id"), "steps": len(y), "stairs_steps": len(st),
                    "acc": round(float(np.mean([p == t for p, t in zip(pred, y)])), 3),
                    "stairs_recall": round(float(np.mean([pred[k] == y[k] for k in st])), 3) if st else None})
    return out
