# Copying Campus Nav into another repo

1. Copy the whole `Campus Nav/` folder into the new repo (keep it as one folder).
2. Board app: `Campus Nav/board/campus-nav/` is the complete UNO Q App Lab app
   (app.yaml, python/, sketch/). Put it in the repo's ArduinoApps folder as `campus-nav/`
   (or run `python deploy.py` in Campus Nav/, which does that copy: edit APPS in deploy.py if the layout differs).
3. Deploy with your own tools/deploy.py (`python deploy.py campus-nav`). Your board_secrets.py / IPs are NOT included.
4. Laptop tests (need numpy, pandas, torch, scipy, pytest): `python -m pytest tests` inside Campus Nav/.
   `python export_stepnet.py` regenerates stepnet.npz from stepcounter_source/ (needs torch).
5. Not included: recorded walks (they live on the board in python/data/walks) and the Jude-only wifi hotspot setup,
   which is documented at the bottom of README.md (nmcli profile CampusNavAP) and must be run on each board.

Both partners use the same step counter weights (stepnet.npz). Each board keeps its own walks; to train on both
people's walks, copy walk JSON files (data/walks/wNN.json) from one board into the other's data/walks folder, rename
ids so they don't clash, then press Train.
