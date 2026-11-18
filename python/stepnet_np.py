"""The ankle step counter (from Gait Wearable/StepCounter.py) in plain NumPy, plus the live
wrapper that turns the 100 Hz IMU stream into step events. No torch needed on the board.
Weights come from stepnet.npz, written by laptop/export_stepnet.py."""
from collections import deque
from pathlib import Path

import numpy as np

RATE_HZ = 100
MIN_EVENT_GAP_S = 0.22
PEAK_THRESHOLD = 0.5          # step curve cutoff. The held-out sweep in step_scores.md: 0.35 gave +10% too many
                              # steps; 0.5-0.6 is closer to true counts and has fewer false steps when standing.
GAITS = ["none", "walking", "running"]


def moving_average(x, n):
    k = np.ones(n) / n
    return np.convolve(np.pad(x, n // 2, mode="edge"), k, mode="valid")[: len(x)]


def features(w, rate_hz=RATE_HZ):
    """(n, 6) raw IMU (g, deg/s) -> (9, n) net input. Same as StepCounter.features."""
    a, g = w[:, :3], w[:, 3:]
    amag = np.linalg.norm(a, axis=1)
    gmag = np.linalg.norm(g, axis=1)
    dyn = amag - moving_average(amag, rate_hz)
    return np.concatenate([a / 2, g / 300, (amag - 1)[:, None], (gmag / 300)[:, None],
                           dyn[:, None]], axis=1).T.astype(np.float32)


def find_peaks(p, height, distance):
    """Local maxima of p above height, at least `distance` samples apart (highest wins)."""
    if len(p) < 3:
        return np.array([], int)
    cand = np.flatnonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]) & (p[1:-1] >= height)) + 1
    cand = cand[np.argsort(-p[cand])]
    taken = np.zeros(len(p), bool)
    keep = []
    for i in cand:
        if not taken[max(0, i - distance + 1): i + distance].any():
            keep.append(i)
            taken[i] = True
    return np.sort(np.array(keep, int))


class StepNetNP:
    def __init__(self, path):
        z = np.load(path)
        self.z = {k: z[k] for k in z.files}
        self.n_blocks = int(self.z["n_blocks"])

    @staticmethod
    def _conv(x, w, b, dil):
        """x (cin, n), w (cout, cin, k): 'same' conv with zero padding, like torch."""
        k = w.shape[2]
        pad = (k // 2) * dil
        xp = np.pad(x, ((0, 0), (pad, pad)))
        n = x.shape[1]
        out = np.zeros((w.shape[0], n), np.float32)
        for j in range(k):
            out += w[:, :, j] @ xp[:, j * dil: j * dil + n]
        return out + b[:, None]

    def forward(self, x):
        z = self.z
        h = np.maximum(self._conv(x, z["stem_w"], z["stem_b"], 1), 0)
        for i in range(self.n_blocks):
            y = np.maximum(self._conv(h, z[f"b{i}_w1"], z[f"b{i}_b1"], int(z[f"b{i}_dil"])), 0)
            y = self._conv(y, z[f"b{i}_w2"], z[f"b{i}_b2"], 1)
            h = np.maximum(h + y, 0)
        return self._conv(h, z["head_w"], z["head_b"], 1)

    def steps(self, w):
        """Raw IMU window (n, 6) -> (sample indices of steps, gait name per step)."""
        out = self.forward(features(w))
        p = 1 / (1 + np.exp(-out[0]))
        g = np.exp(out[1:] - out[1:].max(0))
        g /= g.sum(0)
        idx = find_peaks(p, PEAK_THRESHOLD, int(MIN_EVENT_GAP_S * RATE_HZ))
        half = int(0.3 * RATE_HZ)
        return idx, [GAITS[int(g[:, max(0, i - half): i + half].mean(axis=1).argmax())] for i in idx]


class LiveStepper:
    """Feeds on IMU samples, returns the absolute sample index of each newly confirmed step.
    The net needs ~0.7 s of context after a step, so a step is reported once it is 1 s old."""
    KEEP_S, LOOKAHEAD_S, EVERY_S = 8, 1.0, 0.5

    def __init__(self, net):
        self.net = net
        self.buf = deque(maxlen=self.KEEP_S * RATE_HZ)
        self.reset()

    def reset(self):
        self.total = 0
        self.last_run = 0
        self.last_step = None
        self.buf.clear()

    def add(self, samples):
        """-> list of absolute sample indices (since reset) of new steps (walking/running only)."""
        self.buf.extend(samples)
        self.total += len(samples)
        if self.total - self.last_run < self.EVERY_S * RATE_HZ or len(self.buf) < 3 * RATE_HZ:
            return []
        self.last_run = self.total
        w = np.asarray(self.buf, np.float32)
        idx, gaits = self.net.steps(w)
        base = self.total - len(w)
        new = []
        for i, g in zip(idx, gaits):
            a = base + int(i)
            if a > self.total - self.LOOKAHEAD_S * RATE_HZ:
                break
            if self.last_step is not None and a - self.last_step < MIN_EVENT_GAP_S * RATE_HZ:
                continue
            self.last_step = a
            if g in ("walking", "running"):
                new.append(a)
        return new
