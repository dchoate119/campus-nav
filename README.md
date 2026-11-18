# 🧭 campus-nav

### Description

GPS-free, wifi-free walking guide to Nolop, learned by behavior cloning. Jude + Dan.
One route: classroom → down the hall → stairs down → some turns → a few stairs up → Nolop.

Everything runs on the UNO Q with an ankle-worn MPU-6050. The phone only shows the page served by the board (`http://<board-ip>:8000`).

```
MPU-6050 (ankle, 100 Hz) -> StepNet in NumPy (trained step counter) -> step events
                         -> YawTracker (gravity-aware gyro heading), read at each step
                         -> TurnTracker (turns snapped to 45°, steps since last turn)
                         -> stairs classifier (flat / up / down per step)
                         -> KNNPolicy (cloned from demos) -> cue: straight / left / right / stairs / arrive
```

Budget: ≤ 10 full walks (demos + DAgger, both partners). Stairs practice walks don't count.

### Code layout

- `app.yaml`: app config (port 8000, no bricks)
- `sketch/`: `sketch.ino` + `MPU6050.cpp/.h`, 100 Hz IMU FIFO → `read_samples()` over Bridge
- `python/`
  - `main.py`: sampler thread + HTTP server (port 8000)
  - `engine.py`: ties step counter, heading, turns, stairs, walk recorder, policy
  - `nav_core.py`: `YawTracker`, `TurnTracker`, stairs classifier, labeling, `KNNPolicy`
  - `stepnet_np.py` + `stepnet.npz`: step counter in NumPy
  - `page.html`: phone UI (Walk / Review / Train / DAgger)
- `data/walks/`: recorded walks `wNN.json` + `wNN.csv` (live on the board, pulled down for laptop work)
- `laptop/`
  - `replay.py`: run a walk CSV through the engine without the board
  - `export_stepnet.py` + `stepcounter_source/`: PyTorch step counter → `stepnet.npz`
  - `tests/`: pytest for `nav_core` and the web API

### Execution plan

1. [x] **Base pipeline**: step counter, heading, turns, stairs, kNN policy, phone page (tested on laptop only).
2. [ ] **Board check**: deploy, IMU rate ~100 Hz, step count correct on Dan's ankle.
3. [ ] **Turn/stairs check**: "stairs" practice walk with known turns; Review page should show the right turns. Tune `TurnTracker` thresholds only if not.
4. [ ] **Demos**: ~3 walks each partner. Tap Stairs start/end and Arrived.
5. [ ] **Review**: fix marker steps (taps lag 1–2 s), untick bad walks.
6. [ ] **Train**: on the board; check leave-one-walk-out scores.
7. [ ] **DAgger**: policy guides, tap the right answer when a cue is wrong, retrain (~2–3 walks).
8. [ ] **Final test**: guide someone who doesn't know the route; Notion page + video.

### Walk budget

| # | who | type | notes |
|---|-----|------|-------|
