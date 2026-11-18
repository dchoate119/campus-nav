# Campus Nav

GPS-free, wifi-free walking guide to Nolop, learned by behavior cloning. Jude + Dan.
Route: classroom -> down the hall -> stairs down -> some turns -> a few stairs up -> Nolop.
Task and plan: PLAN.md.

## How it works (all on the UNO Q)
```
MPU-6050 (ankle) -> StepNet in NumPy (your trained step counter) -> step events
                 -> YawTracker (gravity-aware gyro heading) read at each step
                 -> TurnTracker (turns snapped to 45 deg, steps since last turn)
                 -> features [steps, heading, steps since turn, turns so far]
                 -> KNNPolicy (cloned from demos) -> cue: straight / left / right / stairs / arrive
```
Code: `board/campus-nav/python/` (`stepnet_np.py`, `nav_core.py`, `engine.py`, `main.py`, `page.html`).
`export_stepnet.py` converts `Gait Wearable/models/step_counter.pt` to `stepnet.npz` (torch and numpy
agree to 4e-6). `replay.py` runs any recorded CSV through the engine on the laptop.

## Use it
1. `python deploy.py --run` (syncs to ArduinoApps/campus-nav and deploys; stops the running app).
2. Board and phone on the same phone hotspot. Open `http://<board-ip>:8000`.
3. **Walk > Demo**: strap on the ankle IMU, stand still, Start. Walk the route; tap Stairs start/end and Arrived.
4. **Review**: look at the heading plot, fix marker steps (taps lag a second or two), untick bad walks.
5. **Train** (instant, on the board), then **Walk > Test** to check the cues.
6. **Walk > DAgger**: the policy guides, you tap the right answer when a cue is wrong. Retrain.
Walk budget: 10 total (demos + DAgger, both people). The header shows the count.

## Tests
`python -m pytest tests` (from this folder, using the gaitVenv python): heading survives foot swing,
turn detector, labels/policy, and a full web-API flow against a fake board.

## Field setup (battery, no wifi): set up 2026-11-18
- Campus Nav is the board's **default app**, so it starts at boot. (Was Finger Lock Cam Test.
  Switch back: `arduino-app-cli properties set default /home/arduino/ArduinoApps/finger-lock-cam-test`.)
- The board hosts its own wifi: network **CampusNav**, password **campusnav193**, page at
  **http://10.42.0.1:8000**. Profile `CampusNavAP`, autoconnect priority 100 (beats Tufts wifi).
- While it hosts the hotspot it can't join Tufts/home wifi, so develop over USB (adb): `python deploy.py --run`.
- Turn the hotspot off (back to normal wifi): `python ../ArduinoApps/tools/deploy.py --cmd "nmcli connection modify CampusNavAP autoconnect no; nmcli connection down CampusNavAP"`
- Wait ~1 minute after power-on for the app to start; the page header shows the IMU rate when ready.
