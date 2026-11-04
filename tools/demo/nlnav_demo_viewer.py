"""NL navigation demo viewer (Windows): the Godot chase view with the semantic map drawn over it.

    python tools\\demo\\nlnav_demo_viewer.py "go to the ladder" [--record NAME] [--no-send] [--fps 15]

One window, three things on it:
  * the chase view of the robot (the Godot window, captured with PrintWindow while it stays parked off-screen),
    with the object map's 3D boxes projected into it (the target in orange), Nav2's plan on the floor,
    the instruction and the live status in a banner;
  * the semantic map on the right: the site map, every object the navigator knows (label, box), the robot where
    the STACK believes it is (AMCL, from TF), its trail, the plan, the goal and the target;
  * optionally the same frames to an mp4 (Videos\\nlnav_demo\\NAME.mp4) and a mid-run still.

The viewer drives the spectator camera itself ({"cmd":"viewpoint"} each tick, the sim's own chase offsets) so it
knows the camera pose exactly and can project the boxes; nothing here blocks the sim's main thread (the sim's
{"cmd":"screenshot"} does, and corrupts odometry: see tools/demo/README.md). Needs the stack up (demo_up.sh).
Keys: q quit (cancels a running goal), t toggle the true pose on the map, b toggle boxes in the chase view.
"""
import argparse, ctypes, ctypes.wintypes as w, glob, json, math, os, socket, subprocess, sys, threading, time
from collections import deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO_WSL = "/mnt/" + REPO[0].lower() + REPO[2:].replace("\\", "/")
WSL = ["wsl.exe", "-d", "Ubuntu-24.04", "--"]
SRC = ("source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; "
       "export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp PYTHONUNBUFFERED=1")
MAP_YAML = os.path.join(REPO, "tools", "sim", "godot_circuit4_map.yaml")
OBJECT_MAP_WSL = "~/m20_sim/semmap/circuit4/object_map.json"
MIN_LABEL_SCORE = 0.4            # the navigator's own cut (nlnav_up.sh --min-label-score)
VFOV_DEG = 75.0                  # the spectator is a stock Camera3D: Godot's default vertical FOV
CHASE_OFFSET = (0.0, 1.0, 2.2)   # spectator.gd GROUND_CHASE_OFFSET (Godot: x right, y up, z back), in the heading frame
CHASE_LOOK_UP = 0.35             # spectator.gd GROUND_CHASE_LOOK_UP
CHASE_LAG, CHASE_YAW_LAG = 3.0, 2.2
BOX_RANGE_M = 9.0
VIEW_W, VIEW_H = 1280, 720
MAP_PX_PER_M = 27.0

ORANGE, CYAN, GREEN, WHITE, RED, YELLOW = (0, 140, 255), (220, 200, 60), (90, 220, 90), (255, 255, 255), (60, 60, 230), (60, 220, 240)


# ---------------------------------------------------------------- the sim's API (persistent connection)
class SimApi:
    def __init__(self, host="127.0.0.1", port=7799):
        self.addr, self.s = (host, port), None

    def call(self, d, timeout=3.0):
        for attempt in (0, 1):
            try:
                if self.s is None:
                    self.s = socket.create_connection(self.addr, timeout=timeout)
                self.s.sendall((json.dumps(d) + "\n").encode())
                b = b""
                while not b.endswith(b"\n"):
                    chunk = self.s.recv(1 << 16)
                    if not chunk:
                        raise ConnectionError("closed")
                    b += chunk
                return json.loads(b)
            except (OSError, ValueError):
                try:
                    self.s.close()
                except Exception:
                    pass
                self.s = None
                if attempt:
                    raise
        return {}


def enu_to_godot(p):
    return (p[0], p[2], -p[1])


def yaw_of(q):
    x, y, z, w_ = q
    return math.atan2(2.0 * (w_ * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ---------------------------------------------------------------- window capture (PrintWindow, works off-screen)
class Grabber:
    def __init__(self):
        self.u32, self.g32 = ctypes.windll.user32, ctypes.windll.gdi32
        self.u32.SetProcessDPIAware()
        self.hwnd = None

        def enum(h, _):
            n = self.u32.GetWindowTextLengthW(h)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                self.u32.GetWindowTextW(h, buf, n + 1)
                if buf.value.startswith("Bright Robotics Simulator") and self.u32.IsWindowVisible(h):
                    self.hwnd = h
            return True
        self.u32.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, w.HWND, w.LPARAM)(enum), 0)
        if not self.hwnd:
            sys.exit("no Godot window (bring the stack up: bash tools/demo/demo_up.sh)")
        r = w.RECT()
        self.u32.GetClientRect(self.hwnd, ctypes.byref(r))
        self.W, self.H = r.right, r.bottom
        hdc = self.u32.GetDC(0)
        self.mdc = self.g32.CreateCompatibleDC(hdc)
        self.bmp = self.g32.CreateCompatibleBitmap(hdc, self.W, self.H)
        self.g32.SelectObject(self.mdc, self.bmp)

        class BMI(ctypes.Structure):
            _fields_ = [("biSize", w.DWORD), ("biWidth", w.LONG), ("biHeight", w.LONG), ("biPlanes", w.WORD),
                        ("biBitCount", w.WORD), ("biCompression", w.DWORD), ("biSizeImage", w.DWORD),
                        ("biXPelsPerMeter", w.LONG), ("biYPelsPerMeter", w.LONG), ("biClrUsed", w.DWORD),
                        ("biClrImportant", w.DWORD)]
        self.bmi = BMI(ctypes.sizeof(BMI), self.W, -self.H, 1, 32, 0, 0, 0, 0, 0, 0)
        self.buf = ctypes.create_string_buffer(self.W * self.H * 4)
        self.box = self._content_box()

    def grab_raw(self):
        self.u32.PrintWindow(self.hwnd, self.mdc, 2)
        self.g32.GetDIBits(self.mdc, self.bmp, 0, self.H, self.buf, ctypes.byref(self.bmi), 0)
        return np.frombuffer(self.buf.raw, np.uint8).reshape(self.H, self.W, 4)[:, :, :3]

    def _content_box(self):
        """The rendered area: the capture carries the title bar and a black margin around the viewport."""
        g = cv2.cvtColor(self.grab_raw(), cv2.COLOR_BGR2GRAY)
        top = 40 if self.H > 200 else 0
        ys, xs = np.where(g[top:] > 8)
        if len(xs) == 0:
            return (0, top, self.W, self.H)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min() + top, ys.max() + 1 + top
        return (int(x0), int(y0), int(x1 - ((x1 - x0) % 2)), int(y1 - ((y1 - y0) % 2)))

    def grab(self):
        x0, y0, x1, y1 = self.box
        return np.ascontiguousarray(self.grab_raw()[y0:y1, x0:x1])


# ---------------------------------------------------------------- the stack (relay inside docker-commander-1)
class Relay:
    """Runs tools/demo/nlnav_demo_relay.py inside the commander container; keeps the newest state."""

    def __init__(self):
        script = open(os.path.join(HERE, "nlnav_demo_relay.py"), encoding="utf-8").read()
        # copy the script into the container, then run it with stdin as the command channel
        subprocess.run(WSL + ["docker", "exec", "-i", "docker-commander-1", "bash", "-c", "cat > /tmp/nlnav_demo_relay.py"],
                       input=script.encode(), check=True)
        cmd = WSL + ["docker", "exec", "-i", "docker-commander-1", "bash", "-lc",
                     f"{SRC}; python3 -u /tmp/nlnav_demo_relay.py"]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", bufsize=1)
        self.lock = threading.Lock()
        self.pose, self.plan, self.result, self.feedback, self.sent_at = None, [], None, None, None
        self.markers = None
        self.trail = deque(maxlen=4000)
        self.log = deque(maxlen=50)
        self.alive = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self.p.stdout:
            line = line.strip()
            if not line.startswith("{"):
                self.log.append(line)
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            with self.lock:
                t = d.get("type")
                if t == "pose":
                    self.pose = (d["x"], d["y"], d["yaw"])
                    if not self.trail or math.hypot(d["x"] - self.trail[-1][0], d["y"] - self.trail[-1][1]) > 0.05:
                        self.trail.append((d["x"], d["y"]))
                elif t == "plan":
                    self.plan = d["pts"]
                elif t == "feedback":
                    self.feedback = d
                elif t == "result":
                    self.result = d
                elif t == "markers":
                    self.markers = d
                elif t == "sent":
                    self.sent_at = time.monotonic()
                    self.markers, self.plan, self.result, self.feedback = None, [], None, None
                self.log.append(line[:160])
        self.alive = False

    def send(self, text):
        self.p.stdin.write(f"send {text}\n")
        self.p.stdin.flush()

    def cancel(self):
        try:
            self.p.stdin.write("cancel\n")
            self.p.stdin.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
            self.p.wait(5)
        except Exception:
            self.p.kill()


# ---------------------------------------------------------------- data: object map + site map
def load_object_map():
    raw = subprocess.run(WSL + ["bash", "-c", f"cat {OBJECT_MAP_WSL}"], capture_output=True, check=True).stdout
    d = json.loads(raw)
    objs = [o for o in d["objects"] if o.get("label_score", 0) >= MIN_LABEL_SCORE]
    return objs, len(d["objects"])


class SiteMap:
    def __init__(self, yaml_path):
        meta = {}
        for line in open(yaml_path, encoding="utf-8"):
            if ":" in line and not line.startswith("#"):
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        self.res = float(meta["resolution"])
        self.ox, self.oy = [float(v) for v in meta["origin"].strip("[]").split(",")[:2]]
        img = cv2.imread(os.path.join(os.path.dirname(yaml_path), meta["image"]), cv2.IMREAD_GRAYSCALE)
        self.rows, self.cols = img.shape
        self.scale = MAP_PX_PER_M * self.res          # panel px per map px
        self.W, self.H = int(self.cols * self.scale), int(self.rows * self.scale)
        # free = light, occupied = dark, unknown = mid grey: a quieter palette than the raw PGM
        base = np.full((self.rows, self.cols, 3), 46, np.uint8)
        base[img > 230] = (62, 62, 62)
        base[img < 60] = (200, 200, 200)
        self.base = cv2.resize(base, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

    def px(self, x, y):
        return (int((x - self.ox) / self.res * self.scale), int(self.H - (y - self.oy) / self.res * self.scale))


# ---------------------------------------------------------------- projection into the chase view
class ChaseCam:
    """Where the spectator is: the viewer places it (same offsets and lags as spectator.gd), so it knows."""

    def __init__(self, api, W, H):
        self.api, self.W, self.H = api, W, H
        self.f = (H / 2.0) / math.tan(math.radians(VFOV_DEG) / 2.0)
        self.pos, self.yaw, self.aim, self.last_t = None, None, None, None
        self.basis = None

    def update(self, robot_godot, heading):
        now = time.monotonic()
        dt = 0.0 if self.last_t is None else min(now - self.last_t, 0.2)
        self.last_t = now
        if self.yaw is None:
            self.yaw = heading
        else:
            d = (heading - self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw += d * min(1.0, dt * CHASE_YAW_LAG)
        ox, oy, oz = CHASE_OFFSET
        want = (robot_godot[0] + oz * math.sin(self.yaw) + ox * math.cos(self.yaw), robot_godot[1] + oy,
                robot_godot[2] + oz * math.cos(self.yaw) - ox * math.sin(self.yaw))
        if self.pos is None:
            self.pos = want
        else:
            a = min(1.0, dt * CHASE_LAG)
            self.pos = tuple(p + (q - p) * a for p, q in zip(self.pos, want))
        self.aim = (robot_godot[0], robot_godot[1] + CHASE_LOOK_UP, robot_godot[2])
        self.api.call({"cmd": "viewpoint", "pos": list(self.pos), "look_at": list(self.aim)})
        c, a = np.array(self.pos), np.array(self.aim)
        fwd = a - c
        fwd /= np.linalg.norm(fwd)
        z = -fwd
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        self.basis = (c, x, y, z)

    def project(self, p_godot):
        """Pixel of a Godot-frame point, or None when behind the camera."""
        if self.basis is None:
            return None
        c, x, y, z = self.basis
        v = np.asarray(p_godot, float) - c
        d = -float(v @ z)
        if d < 0.05:
            return None
        return (int(self.W / 2 + self.f * float(v @ x) / d), int(self.H / 2 - self.f * float(v @ y) / d))

    def project_enu(self, p_enu):
        return self.project(enu_to_godot(p_enu))


def box_corners(o):
    (cx, cy, cz), (sx, sy, sz) = o["xyz"], o["size"]
    return [(cx + dx * sx / 2, cy + dy * sy / 2, cz + dz * sz / 2) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]


BOX_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


# ---------------------------------------------------------------- drawing
def text(img, s, org, scale=0.6, color=WHITE, thick=1, shadow=True):
    if shadow:
        cv2.putText(img, s, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def shade(img, x0, y0, x1, y1, alpha=0.55):
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha)).astype(np.uint8)


class Status:
    """What the banner says, from the relay's events. Plain words, with the numbers that matter."""

    def __init__(self, instruction):
        self.instruction = instruction
        self.t_sent, self.t_done = None, None

    def line(self, relay, target):
        r, fb = relay.result, relay.feedback
        if relay.sent_at is None:
            return "Listening...", WHITE
        if r is not None:
            if self.t_done is None:
                self.t_done = time.monotonic()
            took = self.t_done - relay.sent_at
            if r["status"] == "done":
                return f"Arrived at the {target['cls'] if target else 'goal'}  ({took:.0f} s)", GREEN
            if r["status"] in ("not_found", "not_understood"):
                return f"{r['status'].replace('_', ' ')}: {r.get('detail') or r.get('interpretation')}", RED
            return f"{r['status']}: {r.get('interpretation', '')} {r.get('detail', '')}".strip(), YELLOW
        if fb is None or fb["stage"] == "resolving":
            return "Understanding the instruction (LLM + object map)...", YELLOW
        if fb["stage"] == "navigating":
            dist = ""
            if relay.pose and target and target.get("goal"):
                gx, gy = target["goal"]
                dist = f"  {math.hypot(gx - relay.pose[0], gy - relay.pose[1]):.1f} m to go"
            lbl = fb.get("label") or (target["cls"] if target else "")
            return f"Driving to the {lbl} (Nav2){dist}", CYAN
        return f"{fb['stage']} {fb.get('status', '')}", WHITE


def draw_view(view, cam, objs, target, relay, show_boxes, status, instruction):
    H, W = view.shape[:2]
    # Nav2's plan and the goal, on the floor
    if relay.plan:
        pts = [cam.project_enu((x, y, 0.02)) for x, y in relay.plan]
        pts = [p for p in pts if p is not None]
        if len(pts) > 1:
            cv2.polylines(view, [np.array(pts, np.int32)], False, GREEN, 2, cv2.LINE_AA)
    if target and target.get("goal"):
        gx, gy = target["goal"]
        g = cam.project_enu((gx, gy, 0.02))
        if g:
            cv2.circle(view, g, 9, GREEN, 2, cv2.LINE_AA)
    # the object map's boxes
    if show_boxes and cam.basis is not None:
        c = cam.basis[0]
        for o in objs:
            p = enu_to_godot(o["xyz"])
            d = math.dist(p, c)
            is_t = target is not None and o["id"] == target["id"]
            if d > BOX_RANGE_M and not is_t:          # the target is always drawn, however far
                continue
            pts = [cam.project_enu(k) for k in box_corners(o)]
            if any(p is None for p in pts):
                continue
            if not any(0 <= p[0] < W and 0 <= p[1] < H for p in pts):
                continue
            col, th = (ORANGE, 3) if is_t else (CYAN, 1)
            for a, b in BOX_EDGES:
                cv2.line(view, pts[a], pts[b], col, th, cv2.LINE_AA)
            top = min(pts, key=lambda p: p[1])
            label = f"{o['label']} #{o['id']}" + (f"  {o['label_score']:.2f}" if is_t else "")
            text(view, label, (top[0], max(14, top[1] - 6)), 0.6 if is_t else 0.42, col, 2 if is_t else 1)
    # the banner: the instruction and what is happening now (bottom: the sim's camera PIP is top-left)
    shade(view, 0, H - 96, W, H)
    text(view, f'"{instruction}"', (16, H - 54), 1.0, WHITE, 2)
    line, col = status.line(relay, target)
    text(view, line, (16, H - 18), 0.7, col, 2)
    if target:
        t = f"target: {target['cls']} #{target['id']}  score {target['score']:.2f}"
        text(view, t, (W - 16 - len(t) * 11, H - 18), 0.6, ORANGE, 1)
    text(view, "cyan = object map, orange = target, green = Nav2 plan", (W - 470, 20), 0.45, (200, 200, 200), 1)


def draw_map(sm, static, objs, target, relay, show_true, true_pose):
    m = static.copy()
    # trail, plan, goal
    if len(relay.trail) > 1:
        pts = np.array([sm.px(x, y) for x, y in relay.trail], np.int32)
        cv2.polylines(m, [pts], False, (120, 120, 255), 2, cv2.LINE_AA)
    if relay.plan:
        pts = np.array([sm.px(x, y) for x, y in relay.plan], np.int32)
        cv2.polylines(m, [pts], False, GREEN, 2, cv2.LINE_AA)
    if target:
        x, y = target["xyz"][:2]
        cx, cy = sm.px(x, y)
        r = int(10 + 4 * math.sin(time.monotonic() * 4))
        cv2.circle(m, (cx, cy), r, ORANGE, 2, cv2.LINE_AA)
        lx, ly = (cx + 14, cy + 5) if cy < sm.H - 30 else (cx - 40, cy - 16)   # keep the label inside the panel
        text(m, f"TARGET {target['cls']} #{target['id']}", (lx, ly), 0.5, ORANGE, 2)
        if target.get("goal"):
            gx, gy = target["goal"]
            g = sm.px(gx, gy)
            cv2.circle(m, g, 6, GREEN, -1, cv2.LINE_AA)
            yaw = target.get("goal_yaw", 0.0)
            cv2.arrowedLine(m, g, sm.px(gx + 0.6 * math.cos(yaw), gy + 0.6 * math.sin(yaw)), GREEN, 2, cv2.LINE_AA, tipLength=0.4)
    # the robot (believed) and, optionally, the truth
    if show_true and true_pose:
        x, y, yaw = true_pose
        cv2.circle(m, sm.px(x, y), 9, (160, 160, 160), 1, cv2.LINE_AA)
    if relay.pose:
        x, y, yaw = relay.pose
        tri = [(x + 0.45 * math.cos(yaw), y + 0.45 * math.sin(yaw)),
               (x + 0.3 * math.cos(yaw + 2.4), y + 0.3 * math.sin(yaw + 2.4)),
               (x + 0.3 * math.cos(yaw - 2.4), y + 0.3 * math.sin(yaw - 2.4))]
        cv2.fillPoly(m, [np.array([sm.px(*p) for p in tri], np.int32)], RED, cv2.LINE_AA)
        text(m, f"robot ({x:.1f}, {y:.1f})", (sm.px(x, y)[0] + 12, sm.px(x, y)[1] - 8), 0.45, RED, 1)
    return m


def static_map(sm, objs):
    m = sm.base.copy()
    for o in objs:
        (cx, cy, _), (sx, sy, _) = o["xyz"], o["size"]
        p0, p1 = sm.px(cx - sx / 2, cy + sy / 2), sm.px(cx + sx / 2, cy - sy / 2)
        cv2.rectangle(m, p0, p1, CYAN, 1, cv2.LINE_AA)
    # labels after the boxes so they stay readable; one label per object, small
    for o in objs:
        cx, cy = o["xyz"][:2]
        p = sm.px(cx, cy)
        text(m, o["label"], (p[0] + 3, p[1] - 3), 0.33, (240, 240, 200), 1)
    text(m, "semantic map: %d objects (score >= %.1f)" % (len(objs), MIN_LABEL_SCORE), (8, 16), 0.45, (200, 200, 200), 1)
    text(m, "red = robot (AMCL)  green = plan/goal  orange = target", (8, sm.H - 8), 0.4, (200, 200, 200), 1)
    return m


# ---------------------------------------------------------------- main
def pick_target(relay, objs):
    """The object the navigator chose: from its markers (published when it resolves, before the drive), matched
    to the object map by position; the result (at the end) adds the id, class and score when it arrives."""
    with relay.lock:
        mk, r = relay.markers, relay.result
    if not mk or not mk["targets"]:
        return None
    tm = mk["targets"][0]
    o = min(objs, key=lambda o: math.dist(o["xyz"][:2], (tm["x"], tm["y"])), default=None)
    if o is None or math.dist(o["xyz"][:2], (tm["x"], tm["y"])) > 0.5:
        o = {"id": -1, "label": "target", "label_score": 0.0, "xyz": [tm["x"], tm["y"], tm["z"]],
             "size": [tm["sx"], tm["sy"], tm["sz"]]}
    t = {"id": o["id"], "cls": o["label"], "score": o["label_score"], "xyz": o["xyz"]}
    if r and r["candidates"]:
        c = r["candidates"][0]
        if c["id"] == o["id"]:
            t["cls"] = c["cls"]                     # the score stays the map's label score (one number throughout)
    if mk["goals"]:
        t["goal"] = (mk["goals"][0]["x"], mk["goals"][0]["y"])
        t["goal_yaw"] = mk["goals"][0]["yaw"]
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instruction", nargs="?", default="go to the ladder")
    ap.add_argument("--record", metavar="NAME", help="also write Videos\\nlnav_demo\\NAME.mp4 and NAME.jpg")
    ap.add_argument("--no-send", action="store_true", help="only watch: do not send the instruction")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--hold", type=float, default=6.0, help="seconds to keep showing the result before exiting (--record)")
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"), "Videos", "nlnav_demo"))
    ap.add_argument("--duration", type=float, default=0.0, help="stop after S seconds regardless (calibration runs)")
    ap.add_argument("--debug", action="store_true", help="draw the robot's projected body (projection check)")
    a = ap.parse_args()

    api = SimApi()
    st = api.call({"cmd": "state"})
    print(f"sim: {st.get('scene')} {st.get('fps', 0):.0f} fps", flush=True)
    api.call({"cmd": "camview", "on": False})
    api.call({"cmd": "follow", "on": False})          # the viewer places the spectator itself
    objs, n_all = load_object_map()
    print(f"object map: {len(objs)} of {n_all} objects kept (label score >= {MIN_LABEL_SCORE})", flush=True)
    sm = SiteMap(MAP_YAML)
    smap = static_map(sm, objs)
    grab = Grabber()
    x0, y0, x1, y1 = grab.box
    print(f"Godot window {grab.W}x{grab.H}, viewport {x1 - x0}x{y1 - y0} at {x0},{y0}", flush=True)
    # the shown view keeps the viewport's aspect (Godot keeps the VERTICAL fov, so the projection only needs H)
    view_w = int(round(VIEW_H * (x1 - x0) / (y1 - y0))) // 2 * 2
    cam = ChaseCam(api, view_w, VIEW_H)
    relay = Relay()
    for _ in range(100):
        if relay.pose is not None or not relay.alive:
            break
        time.sleep(0.1)
    if not relay.alive:
        sys.exit("relay died:\n" + "\n".join(relay.log))
    print("relay: " + ("pose ok" if relay.pose else "no TF yet"), flush=True)

    map_h = VIEW_H
    map_w = int(sm.W * map_h / sm.H)
    out_w = view_w + map_w + (view_w + map_w) % 2
    ff = None
    if a.record:
        os.makedirs(a.out, exist_ok=True)
        FF = glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\BtbN.FFmpeg*\ffmpeg-*\bin\ffmpeg.exe"))
        if not FF:
            sys.exit("ffmpeg not found (winget install BtbN.FFmpeg.GPL.Shared)")
        mp4 = os.path.join(a.out, a.record + ".mp4")
        ff = subprocess.Popen([FF[0], "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                               "-s", f"{out_w}x{VIEW_H}", "-framerate", str(a.fps), "-i", "-", "-c:v", "libx264",
                               "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", mp4], stdin=subprocess.PIPE)
        print(f"recording -> {mp4}", flush=True)

    status = Status(a.instruction)
    show_true, show_boxes = False, True
    cv2.namedWindow("M20 NL navigation demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("M20 NL navigation demo", out_w, VIEW_H)
    t0, n, sent, still_done, t_result = time.monotonic(), 0, a.no_send, False, None
    period = 1.0 / a.fps
    try:
        while True:
            tick = time.monotonic()
            tel = api.call({"cmd": "telemetry"})
            true_pose = None
            if tel.get("base_enu"):
                t, q = tel["base_enu"]["t"], tel["base_enu"]["q"]
                true_pose = (t[0], t[1], yaw_of(q))
                cam.update(enu_to_godot(t), float(tel.get("heading_rad", 0.0)))
            view = cv2.resize(grab.grab(), (view_w, VIEW_H), interpolation=cv2.INTER_AREA)
            target = pick_target(relay, objs)
            if a.debug and true_pose:
                p = cam.project_enu((true_pose[0], true_pose[1], 0.3))   # the robot's body: must sit ON the robot
                if p:
                    cv2.drawMarker(view, p, (255, 0, 255), cv2.MARKER_CROSS, 40, 2)
            draw_view(view, cam, objs, target, relay, show_boxes, status, a.instruction)
            mpanel = cv2.resize(draw_map(sm, smap, objs, target, relay, show_true, true_pose), (map_w, map_h),
                                interpolation=cv2.INTER_AREA)
            frame = np.zeros((VIEW_H, out_w, 3), np.uint8)
            frame[:, :view_w] = view
            frame[:, view_w:view_w + map_w] = mpanel
            cv2.imshow("M20 NL navigation demo", frame)
            if ff is not None:
                ff.stdin.write(frame.tobytes())
            n += 1
            if not sent and time.monotonic() - t0 > 2.0:
                relay.send(a.instruction)
                sent = True
                print(f"sent: {a.instruction}", flush=True)
            if relay.result is not None and t_result is None:
                t_result = time.monotonic()
                r = relay.result
                print(f"result: {r['status']}: {r['interpretation']} {r.get('detail', '')} ({r['latency_ms']} ms)", flush=True)
                for it in r["items"]:
                    print(f"  goal {it['label']}: ({it['x']:.2f}, {it['y']:.2f})", flush=True)
                for c in r["candidates"][1:4]:
                    print(f"  other match #{c['id']} {c['cls']} ({c['x']:.1f}, {c['y']:.1f}) {c['why']}", flush=True)
            if ff is not None and t_result is not None and time.monotonic() - t_result > a.hold:
                break
            if a.duration > 0 and time.monotonic() - t0 > a.duration:
                break
            if a.record and relay.feedback and relay.feedback["stage"] == "navigating" and not still_done:
                cv2.imwrite(os.path.join(a.out, a.record + ".jpg"), frame)
                still_done = True
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                if relay.result is None and sent:
                    relay.cancel()
                break
            if k == ord("t"):
                show_true = not show_true
            if k == ord("b"):
                show_boxes = not show_boxes
            if cv2.getWindowProperty("M20 NL navigation demo", cv2.WND_PROP_VISIBLE) < 1:
                break
            d = period - (time.monotonic() - tick)
            if d > 0:
                time.sleep(d)
    finally:
        relay.close()
        if ff is not None:
            ff.stdin.close()
            ff.wait()
            print(f"saved {mp4} ({n} frames, {os.path.getsize(mp4) / 1e6:.1f} MB)", flush=True)
        api.call({"cmd": "follow", "on": True})
        cv2.destroyAllWindows()
    print(f"sim fps at the end: {api.call({'cmd': 'state'}).get('fps', 0):.0f}", flush=True)


if __name__ == "__main__":
    main()
