# Campus Nav: GPS-free navigation to Nolop by behavior cloning

Assignment: step counter + IMU + behavior cloning -> a device that guides you across campus
with no GPS or wifi. At most 10 full walks total (demos + DAgger corrections), pair project,
train on both partners' data. Grading: Notion page, video + description + reflection answers,
guides us to Nolop, guides someone who doesn't know the way.

## Decisions (2026-11-18)
- Folder: `Campus Nav/` (laptop code) + `board/campus-nav/` (UNO Q app; copy to ArduinoApps/campus-nav to deploy).
- Base layer: ankle step counter (PyTorch, from branch `sumo`, `Gait Wearable/StepCounter.py`), copied to `base_stepcounter/`.
- Sensor: MPU-6050 on the ankle (same mount as the step counter's data_ankle recordings).
- Navigator: small policy cloned from walks. State = (steps since last waypoint, gyro-yaw heading
  relative to start, stairs flag, total steps). Action = straight / left / right / stairs / arrive.
- Heading: gyro-z integration, reset at start, snapped to discrete turns to limit drift. No magnetometer.
- UI: mobile web page served by the UNO Q. Modes: Record demo, Navigate, DAgger correct.
- Phone link: phone hotspot (board joins it; page on its local IP, no internet needed).

## Open issues
1. Step counter is a PyTorch net that runs on the laptop (via MQTT). Navigation must run on the board
   with no wifi/laptop, so either export the net to numpy (like gait_models.py did) or fall back to
   the peak detector on the board. Decide after checking board resources.
2. Ankle sensor sees both feet; STEPS_PER_EVENT=1 in StepCounter.py. Heading from an ankle gyro is
   noisy mid-stride; take yaw during stance/at step events, not continuously.
3. Labeling during a demo is hard: record only sensors + steps, then label turns afterwards
   (auto turn detection from yaw, editable on the web page) before training.
4. Partner: Dan. Route: classroom, hall, stairs down, turns, a few stairs up, into Nolop.

## Status 2026-11-18
Built and tested on the laptop (not yet on the board): numpy step counter, heading, turn detector, labels, kNN policy, web page, DAgger.
Not yet tested: on real turns/stairs data, on the board.

## Build order
1. Board app: IMU -> step events + yaw, logs per-walk CSV, serves web page.
2. Walk recorder + post-hoc cleaning/labeling UI (turn markers, stairs markers).
3. Train policy on partners' combined walks (leave-one-walk-out check).
4. Navigate mode: live state -> policy -> big cue on phone.
5. DAgger: run policy, user taps correction, corrected walk added (counts toward the 10).
6. Test with someone who doesn't know the route; Notion page + video.

## Walk budget (max 10 total)
| # | who | type | route | notes |
|---|-----|------|-------|-------|
