"""Campus Nav on the UNO Q: GPS-free, wifi-free walking guide, learned by behavior cloning.

Runs entirely on the board: IMU (via the sketch) -> NumPy step counter -> heading -> policy.
Your phone just needs to reach the board's web page (hotspot, no internet required):
    http://<board-ip>:8000

Pages: Record demos, Review/clean them, Train, Navigate (+ DAgger corrections).
"""
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from arduino.app_utils import *

from engine import Engine
from stepnet_np import LiveStepper, StepNetNP

HERE = Path(__file__).resolve().parent
PORT = 8000
POLL_S = 0.05
ACCEL_COUNTS_PER_G = 4096     # sketch uses ACCEL_8G
GYRO_COUNTS_PER_DPS = 32.8    # sketch uses GYRO_1000DPS

engine = Engine(LiveStepper(StepNetNP(HERE / "stepnet.npz")), HERE / "data")
imu_info = {"text": "starting", "rate_hz": 0}


def parse(reply):
    """'0|ax,ay,az,gx,gy,gz;...' raw counts -> (overflow, [[g,g,g,dps,dps,dps], ...])"""
    flag, _, body = reply.partition("|")
    out = []
    for s in body.split(";"):
        if s:
            ax, ay, az, gx, gy, gz = (int(v) for v in s.split(","))
            out.append([ax / ACCEL_COUNTS_PER_G, ay / ACCEL_COUNTS_PER_G, az / ACCEL_COUNTS_PER_G,
                        gx / GYRO_COUNTS_PER_DPS, gy / GYRO_COUNTS_PER_DPS, gz / GYRO_COUNTS_PER_DPS])
    return flag == "1", out


def sampler_loop():
    while True:                      # the sketch may still be starting: keep asking
        try:
            imu_info["text"] = Bridge.call("imu_status")
            print("IMU:", imu_info["text"])
            break
        except Exception as e:
            imu_info["text"] = f"sketch not answering yet ({e!r})"
            time.sleep(2)
    count, report = 0, time.monotonic() + 1
    while True:
        try:
            overflow, samples = parse(Bridge.call("read_samples"))
        except Exception as e:
            print(f"read error: {e!r}")
            time.sleep(1)
            continue
        try:
            engine.add_samples(samples)
        except Exception as e:
            print(f"engine error: {e!r}")
        count += len(samples)
        now = time.monotonic()
        if now >= report:
            imu_info["rate_hz"] = count
            count, report = 0, now + 1
        time.sleep(POLL_S)


def lan_page():
    return (HERE / "page.html").read_bytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json", code=200):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path == "/":
                self._send(lan_page(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send({**engine.state(), "imu": imu_info})
            elif path == "/api/walks":
                self._send([{k: w.get(k) for k in ("id", "name", "who", "kind", "created", "include", "seconds")}
                            | {"steps": len(w["steps"]), "markers": len(w["markers"]),
                               "corrections": len(w.get("corrections", []))} for w in engine.list_walks()])
            elif m := re.fullmatch(r"/api/walk/(w\d+)", path):
                r = engine.review(m.group(1))
                self._send(r if r else {"error": "no such walk"}, code=200 if r else 404)
            else:
                self._send({"error": "not found"}, code=404)
        except Exception as e:
            self._send({"error": repr(e)}, code=500)

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if path == "/api/zero":
                engine.zero()
                out = {"ok": True}
            elif path == "/api/walk/start":
                out = {"ok": True, "id": engine.start_walk(body.get("who"), body.get("name"), body.get("kind", "demo"),
                                                           bool(body.get("navigate")))}
            elif path == "/api/walk/mark":
                engine.mark(body["kind"])
                out = {"ok": True}
            elif path == "/api/walk/correct":
                engine.correct(body["action"])
                out = {"ok": True}
            elif path == "/api/walk/stop":
                out = {"ok": True, "id": engine.stop_walk()}
            elif path == "/api/train":
                out = {"ok": True, "policy": engine.train()}
            elif m := re.fullmatch(r"/api/walk/(w\d+)/edit", path):
                engine.save_edit(m.group(1), body.get("markers"), body.get("include"), body.get("name"))
                out = {"ok": True}
            elif m := re.fullmatch(r"/api/walk/(w\d+)/delete", path):
                engine.delete_walk(m.group(1))
                out = {"ok": True}
            else:
                self._send({"error": "not found"}, code=404)
                return
            self._send(out)
        except Exception as e:
            self._send({"error": str(e)}, code=400)


threading.Thread(target=sampler_loop, daemon=True).start()
srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print(f"Campus Nav: open http://<board-ip>:{PORT} on your phone")

App.run()
