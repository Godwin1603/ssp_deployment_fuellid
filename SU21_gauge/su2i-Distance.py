"""
SU21 Fuel Filler Door Inner Panel - real-time distance gauge
Catch plug -> trimline (cutout edge) distance, horizontal / X as in CAD.
Spec (SU21 CAD): 140.769 mm +/- 1.0 mm. Outside spec -> NG with measured dimension and difference.

STEP 1 - one-time calibration on a known-GOOD SU21 part:
    python fuel_door_gauge_su21.py calibrate --image golden_su21.jpg
    python fuel_door_gauge_su21.py calibrate --image golden_su21.jpg --scale-mm 100
        (--scale-mm: click both ends of a ruler/gauge block of that length lying on the
         part plane. Without it, scale is derived from the golden part = nominal.)

    Point A = catch plug face (the side facing the cutout)
    Point B = trimline: edge of the large cutout on the hinge side, at the catch-plug height

STEP 2 - run:
    python fuel_door_gauge_su21.py run --source 0          (camera index)
    python fuel_door_gauge_su21.py run --source test.jpg   (still image)
    python fuel_door_gauge_su21.py run --source line.mp4   (video)
    Keys: q = quit, s = save snapshot
"""

import argparse
import csv
import json
import os
import sys
import threading
from collections import deque
from datetime import datetime

import cv2
import numpy as np

# ------------------------------------------------------------------ SU21 spec
# From SU21 CAD: catch plug -> trimline (cutout edge), horizontal (X) = 140.769 mm
MODEL = "SU21"
NOMINAL_MM = 140.769        # SU21 CAD standard distance
TOLERANCE_MM = 1.0          # +/- allowed deviation (SU21 client spec)
CFG_DIR = os.path.join("gauge_cfg", MODEL)
CFG_FILE = os.path.join(CFG_DIR, "config.json")

MEASURE_AXIS = "x"          # "x" = horizontal distance (as in CAD), "xy" = straight line
MIN_MATCH_SCORE = 0.80      # template match confidence; below = feature missing (tune on 20+ good parts)
SEARCH_MARGIN_PX = 80       # how far a feature may move from its taught position
DEBOUNCE_FRAMES = 5         # consecutive NG frames before alarm
DISPLAY_W = 1280            # display width (aspect ratio preserved)
LOG_DIR = "ng_logs"

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ------------------------------------------------------------------ helpers
def preprocess(img):
    """Grayscale + CLAHE -> robust to the glare/shading on stamped steel."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def disp_scale(img):
    return min(DISPLAY_W / img.shape[1], 1.0)


def to_display(img):
    s = disp_scale(img)
    return cv2.resize(img, None, fx=s, fy=s) if s < 1 else img.copy()


def click_points(img, n, title):
    """Click n points on a display-scaled image; returns FULL-RES coordinates."""
    s = disp_scale(img)
    disp = to_display(img)
    pts = []

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((x / s, y / s))
            cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(title, disp)

    cv2.imshow(title, disp)
    cv2.setMouseCallback(title, cb)
    while len(pts) < n:
        if cv2.waitKey(20) == 27:
            sys.exit("Cancelled.")
    cv2.waitKey(300)
    cv2.destroyWindow(title)
    return pts


def select_roi(img, title):
    s = disp_scale(img)
    r = cv2.selectROI(title, to_display(img), showCrosshair=True)
    cv2.destroyWindow(title)
    x, y, w, h = [int(round(v / s)) for v in r]
    if w == 0 or h == 0:
        sys.exit("Empty ROI - cancelled.")
    return [x, y, w, h]


def px_distance(p, q, axis="xy"):
    if axis == "x":
        return abs(q[0] - p[0])
    return float(np.hypot(q[0] - p[0], q[1] - p[1]))


def subpixel(res, loc):
    """Parabolic peak refinement of a template-match result."""
    x, y = loc
    dx = dy = 0.0
    if 0 < x < res.shape[1] - 1:
        l, c, r = res[y, x - 1], res[y, x], res[y, x + 1]
        d = l - 2 * c + r
        dx = 0.5 * (l - r) / d if d != 0 else 0.0
    if 0 < y < res.shape[0] - 1:
        t, c, b = res[y - 1, x], res[y, x], res[y + 1, x]
        d = t - 2 * c + b
        dy = 0.5 * (t - b) / d if d != 0 else 0.0
    return x + dx, y + dy


# ------------------------------------------------------------------ calibrate
def calibrate(args):
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Image not found: {args.image}")
    os.makedirs(CFG_DIR, exist_ok=True)
    print(f"Calibrating model {MODEL}: nominal {NOMINAL_MM} mm +/- {TOLERANCE_MM} mm")

    print("1) Drag a box around the WHOLE PART, press ENTER")
    part_roi = select_roi(img, "1) Whole part - ENTER to confirm")

    print("2) Click point A (catch plug) then point B (trimline)")
    (ax, ay), (bx, by) = click_points(img, 2, "2) Click catch plug then trimline")

    print("3) Drag a box around the catch plug (include its distinctive edges)")
    roi_a = select_roi(img, "3) Catch plug template - ENTER")
    print("4) Drag a box around the trimline edge")
    roi_b = select_roi(img, "4) Trimline template - ENTER")

    for (px, py), (x, y, w, h), name in [((ax, ay), roi_a, "A"), ((bx, by), roi_b, "B")]:
        if not (x <= px <= x + w and y <= py <= y + h):
            sys.exit(f"Point {name} must lie inside its feature box. Redo calibration.")

    dist_px = px_distance((ax, ay), (bx, by), MEASURE_AXIS)

    if args.scale_mm:
        print(f"5) Click both ends of the {args.scale_mm} mm reference")
        (sx1, sy1), (sx2, sy2) = click_points(img, 2, "5) Click both ends of reference")
        mm_per_px = args.scale_mm / float(np.hypot(sx2 - sx1, sy2 - sy1))
        print(f"   Golden part measures {dist_px * mm_per_px:.3f} mm "
              f"(nominal {NOMINAL_MM}) - verify this before production!")
    else:
        mm_per_px = NOMINAL_MM / dist_px
        print("   Scale derived from golden part (assumes it is exactly nominal).")

    gray = preprocess(img)
    cv2.imwrite(os.path.join(CFG_DIR, "reference.png"), img)
    for roi, name in [(roi_a, "A"), (roi_b, "B")]:
        x, y, w, h = roi
        cv2.imwrite(os.path.join(CFG_DIR, f"template_{name}.png"), gray[y:y + h, x:x + w])

    cfg = {
        "model": MODEL,
        "part_roi": part_roi,
        "roi_A": roi_a, "roi_B": roi_b,
        "offset_A": [ax - roi_a[0], ay - roi_a[1]],
        "offset_B": [bx - roi_b[0], by - roi_b[1]],
        "golden_px": dist_px,
        "mm_per_px": mm_per_px,
        "nominal_mm": NOMINAL_MM,
        "tolerance_mm": TOLERANCE_MM,
        "axis": MEASURE_AXIS,
    }
    with open(CFG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved {CFG_FILE}  |  {dist_px:.1f} px  |  {mm_per_px:.5f} mm/px")


# ------------------------------------------------------------------ gauge
class Gauge:
    def __init__(self, align=True):
        if not os.path.exists(CFG_FILE):
            sys.exit(f"No config at {CFG_FILE} - run calibration for {MODEL} first.")
        with open(CFG_FILE) as f:
            self.cfg = json.load(f)
        self.align_enabled = align
        self.ref = cv2.imread(os.path.join(CFG_DIR, "reference.png"))
        self.ref_gray = preprocess(self.ref)
        self.h, self.w = self.ref_gray.shape
        self.tmpl = {k: cv2.imread(os.path.join(CFG_DIR, f"template_{k}.png"), 0) for k in "AB"}

        # ORB features on the PART only (so the static fixture/table doesn't dominate)
        self.orb = cv2.ORB_create(4000)
        mask = np.zeros_like(self.ref_gray)
        x, y, w, h = self.cfg["part_roi"]
        mask[y:y + h, x:x + w] = 255
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(self.ref_gray, mask)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def align(self, frame):
        """Warp live frame into reference coordinates (handles part shift/rotation)."""
        gray = preprocess(frame)
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(kp) < 20:
            return None
        good = [m for m, n in (p for p in self.matcher.knnMatch(self.ref_des, des, k=2) if len(p) == 2)
                if m.distance < 0.75 * n.distance]
        if len(good) < 20:
            return None
        src = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([self.ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or inl.sum() < 15:
            return None
        return cv2.warpPerspective(frame, H, (self.w, self.h))

    def locate(self, gray, key):
        x, y, w, h = self.cfg[f"roi_{key}"]
        m = SEARCH_MARGIN_PX
        sx, sy = max(x - m, 0), max(y - m, 0)
        ex, ey = min(x + w + m, gray.shape[1]), min(y + h + m, gray.shape[0])
        area = gray[sy:ey, sx:ex]
        t = self.tmpl[key]
        if area.shape[0] < t.shape[0] or area.shape[1] < t.shape[1]:
            return None, 0.0
        res = cv2.matchTemplate(area, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        px, py = subpixel(res, loc)
        ox, oy = self.cfg[f"offset_{key}"]
        return (sx + px + ox, sy + py + oy), float(score)

    def measure(self, frame):
        # Camera/app may deliver a resized frame (e.g. 1024 wide). Bring it back to the
        # calibration resolution so mm/px stays valid; alignment then removes any
        # remaining offset, rotation or letterbox padding.
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[:2] != (self.h, self.w):
            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_CUBIC)

        if self.align_enabled:
            work = self.align(frame)
            if work is None:
                return {"status": "PART NOT FOUND", "mm": None}, frame.copy()
        else:
            work = frame

        gray = preprocess(work)
        vis = work.copy()
        pa, sa = self.locate(gray, "A")
        pb, sb = self.locate(gray, "B")
        result = {"score_A": sa, "score_B": sb, "mm": None}

        if pa is None or pb is None or min(sa, sb) < MIN_MATCH_SCORE:
            names = {"A": "CATCH PLUG", "B": "TRIMLINE"}
            missing = [names[k] for k, s in (("A", sa), ("B", sb)) if s < MIN_MATCH_SCORE]
            result["status"] = f"{' & '.join(missing)} NOT FOUND - CANNOT MEASURE"
            return result, vis

        dist_mm = px_distance(pa, pb, self.cfg.get("axis", "xy")) * self.cfg["mm_per_px"]
        dev = dist_mm - NOMINAL_MM
        result.update(mm=dist_mm, dev=dev,
                      status="OK" if abs(dev) <= TOLERANCE_MM + 1e-9 else "NG")

        col = (0, 200, 0) if result["status"] == "OK" else (0, 0, 255)
        a = tuple(map(int, pa))
        b = (int(pb[0]), a[1]) if self.cfg.get("axis") == "x" else tuple(map(int, pb))
        cv2.line(vis, a, b, col, 4)
        for p in (a, b):
            cv2.line(vis, (p[0], p[1] - 30), (p[0], p[1] + 30), col, 4)   # end ticks
        cv2.putText(vis, f"{dist_mm:.2f} mm", ((a[0] + b[0]) // 2 - 110, a[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4)
        return result, vis


# ------------------------------------------------------------------ alerting
class Alarm:
    """Debounced alarm: fires only after N consecutive bad frames."""

    def __init__(self, n=DEBOUNCE_FRAMES):
        self.hist = deque(maxlen=n)
        self.active = False

    def update(self, status):
        self.hist.append(status != "OK")
        new_state = len(self.hist) == self.hist.maxlen and all(self.hist)
        if new_state and not self.active:
            self.on_alarm()
        if not any(self.hist):
            self.active = False
        elif new_state:
            self.active = True
        return self.active

    def on_alarm(self):
        print("\a!!! ALERT: distance out of tolerance / feature missing")
        if os.name == "nt":
            import winsound
            threading.Thread(target=winsound.Beep, args=(2000, 600), daemon=True).start()
        # Raspberry Pi relay / tower light: drive your GPIO pin here, e.g.
        # GPIO.output(RELAY_RED, GPIO.HIGH)


def log_ng(result, vis):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cv2.imwrite(os.path.join(LOG_DIR, f"{MODEL}_{ts}.jpg"), vis)
    path = os.path.join(LOG_DIR, "ng_log.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "model", "status", "mm", "deviation", "score_A", "score_B"])
        w.writerow([ts, MODEL, result["status"], result.get("mm"), result.get("dev"),
                    result.get("score_A"), result.get("score_B")])


def draw_banner(vis, result, alarm_on):
    ok = result["status"] == "OK"
    col = (0, 160, 0) if ok else (0, 0, 220)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 190), col, -1)
    w = (255, 255, 255)
    if result.get("mm") is not None:
        lo, hi = NOMINAL_MM - TOLERANCE_MM, NOMINAL_MM + TOLERANCE_MM
        l1 = f"{MODEL}  {result['status']}   Measured: {result['mm']:.2f} mm"
        l2 = (f"Spec: {NOMINAL_MM} +/- {TOLERANCE_MM} mm ({lo:.2f} - {hi:.2f})   "
              f"Diff: {result['dev']:+.2f} mm")
        if not ok:
            l2 += "  OVERSIZE" if result["dev"] > 0 else "  UNDERSIZE"
    else:
        l1 = f"{MODEL}  NG   {result['status']}"
        l2 = f"Spec: {NOMINAL_MM} +/- {TOLERANCE_MM} mm"
    cv2.putText(vis, l1, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, w, 5)
    cv2.putText(vis, l2, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, w, 4)
    if alarm_on:
        cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), (0, 0, 255), 25)


# ------------------------------------------------------------------ run
def run(args):
    gauge = Gauge(align=not args.no_align)
    alarm = Alarm()
    src = args.source
    print(f"Model {MODEL}: nominal {NOMINAL_MM} mm +/- {TOLERANCE_MM} mm")

    if src.lower().endswith(IMG_EXT):
        frame = cv2.imread(src)
        if frame is None:
            sys.exit(f"Image not found: {src}")
        result, vis = gauge.measure(frame)
        draw_banner(vis, result, result["status"] != "OK")
        print(result)
        if result["status"] != "OK":
            log_ng(result, vis)
        cv2.imshow("Gauge", to_display(vis))
        cv2.waitKey(0)
        return

    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        sys.exit(f"Cannot open source: {src}")
    # Request full sensor resolution - accuracy depends on it
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)

    was_alarm = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result, vis = gauge.measure(frame)
        alarm_on = alarm.update(result["status"])
        if alarm_on and not was_alarm:
            log_ng(result, vis)
        was_alarm = alarm_on
        draw_banner(vis, result, alarm_on)
        cv2.imshow("Gauge", to_display(vis))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            cv2.imwrite(f"snap_{MODEL}_{datetime.now():%H%M%S}.jpg", vis)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--image", required=True)
    c.add_argument("--scale-mm", type=float, default=None,
                   help="known length of a ruler/gauge in the image (mm)")
    r = sub.add_parser("run")
    r.add_argument("--source", default="0")
    r.add_argument("--no-align", action="store_true",
                   help="skip alignment if the fixture holds the part rigidly")
    a = ap.parse_args()
    calibrate(a) if a.mode == "calibrate" else run(a)
