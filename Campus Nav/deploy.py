"""Copy board/campus-nav into ArduinoApps/campus-nav (the folder tools/deploy.py ships to the
UNO Q) and, with --run, deploy and start it.

  python deploy.py          # just sync the files
  python deploy.py --run    # sync, then deploy to the board (stops whatever app is running)
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "board" / "campus-nav"
APPS = HERE.parent / "ArduinoApps"
DST = APPS / "campus-nav"

if not APPS.exists():
    sys.exit(f"Expected the App Lab apps folder at {APPS}. Edit APPS in this file, or copy board/campus-nav to the board yourself.")
if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("__pycache__", "data", "*.pyc"))
print("synced", DST)
if "--run" in sys.argv:
    sys.exit(subprocess.call([sys.executable, str(APPS / "tools" / "deploy.py"), "campus-nav"], cwd=APPS / "tools"))
