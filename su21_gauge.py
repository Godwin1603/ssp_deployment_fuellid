"""
SU21 Distance Gauge Module for SSP Fuel Door Inspection.
Measures catch plug -> trimline (cutout edge) distance on the SU21 fuel door inner panel.
Spec (SU21 CAD): 140.769 mm +/- 1.0 mm. Outside spec -> NG.
"""

import os
import csv
import json
import time
import logging
from datetime import datetime
import cv2
import numpy as np

logger = logging.getLogger("SU21Gauge")

class SU21DistanceGauge:
    def __init__(self, cfg_dir=None):
        if cfg_dir is None:
            # Look in SU21_gauge/gauge_cfg/SU21 first, then fallback to gauge_cfg/SU21
            base_dir = os.path.dirname(os.path.abspath(__file__))
            cand1 = os.path.join(base_dir, "SU21_gauge", "gauge_cfg", "SU21")
            cand2 = os.path.join(base_dir, "gauge_cfg", "SU21")
            cfg_dir = cand1 if os.path.exists(cand1) else cand2

        self.cfg_dir = cfg_dir
        self.cfg_file = os.path.join(cfg_dir, "config.json")
        if not os.path.exists(self.cfg_file):
            raise FileNotFoundError(f"Config file not found: {self.cfg_file}")

        with open(self.cfg_file, "r") as f:
            self.cfg = json.load(f)

        self.model_name = self.cfg.get("model", "SU21")
        self.nominal_mm = float(self.cfg.get("nominal_mm", 140.769))
        self.tolerance_mm = float(self.cfg.get("tolerance_mm", 1.0))
        self.mm_per_px = float(self.cfg.get("mm_per_px", 0.12145729))
        self.axis = self.cfg.get("axis", "x")
        self.min_match_score = 0.80
        self.search_margin_px = 80

        # Load reference image and precompute ORB features
        ref_path = os.path.join(cfg_dir, "reference.png")
        self.ref = cv2.imread(ref_path)
        if self.ref is None:
            raise FileNotFoundError(f"Reference image not found: {ref_path}")

        self.ref_gray = self.preprocess(self.ref)
        self.h, self.w = self.ref_gray.shape

        # Load templates A (catch plug) and B (trimline)
        self.tmpl = {}
        for k in ("A", "B"):
            tmpl_path = os.path.join(cfg_dir, f"template_{k}.png")
            t = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)
            if t is None:
                raise FileNotFoundError(f"Template {k} not found: {tmpl_path}")
            self.tmpl[k] = t

        # Feature matching on the PART only (so static tray does not dominate)
        self.orb = cv2.ORB_create(4000)
        mask = np.zeros_like(self.ref_gray)
        rx, ry, rw, rh = self.cfg["part_roi"]
        mask[ry:ry + rh, rx:rx + rw] = 255
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(self.ref_gray, mask)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        logger.info(f"[SU21Gauge] Initialized {self.model_name}: nominal {self.nominal_mm} +/- {self.tolerance_mm} mm (scale={self.mm_per_px:.6f} mm/px)")

    def preprocess(self, img):
        """Grayscale + CLAHE -> robust to glare and stamping reflections."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    def subpixel(self, res, loc):
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

    def locate(self, gray, key):
        """Locates template within taught ROI + search margin."""
        x, y, w, h = self.cfg[f"roi_{key}"]
        m = self.search_margin_px
        sx, sy = max(x - m, 0), max(y - m, 0)
        ex, ey = min(x + w + m, gray.shape[1]), min(y + h + m, gray.shape[0])
        area = gray[sy:ey, sx:ex]
        t = self.tmpl[key]
        if area.shape[0] < t.shape[0] or area.shape[1] < t.shape[1]:
            return None, 0.0
        res = cv2.matchTemplate(area, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        px, py = self.subpixel(res, loc)
        ox, oy = self.cfg[f"offset_{key}"]
        return (sx + px + ox, sy + py + oy), float(score)

    def measure(self, frame):
        """
        Runs full alignment, feature location, and distance calculation on a camera frame.
        Returns:
            dict containing status ('OK', 'NG', 'PART NOT FOUND', etc.),
            mm, dev, pt_a, pt_b (in frame's original pixel space),
            scores, and validity flag.
        """
        if frame is None:
            return {"status": "NO FRAME", "mm": None, "dev": None, "is_valid": False}

        orig_h, orig_w = frame.shape[:2]
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        need_resize = (orig_h != self.h or orig_w != self.w)
        if need_resize:
            scaled_frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        else:
            scaled_frame = frame

        gray = self.preprocess(scaled_frame)
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(kp) < 20:
            return {
                "status": "PART NOT FOUND",
                "mm": None, "dev": None,
                "nominal_mm": self.nominal_mm,
                "tolerance_mm": self.tolerance_mm,
                "score_A": 0.0, "score_B": 0.0,
                "pt_a": None, "pt_b": None,
                "is_valid": False
            }

        good = [m for m, n in (p for p in self.matcher.knnMatch(self.ref_des, des, k=2) if len(p) == 2)
                if m.distance < 0.75 * n.distance]
        if len(good) < 20:
            return {
                "status": "PART NOT FOUND",
                "mm": None, "dev": None,
                "nominal_mm": self.nominal_mm,
                "tolerance_mm": self.tolerance_mm,
                "score_A": 0.0, "score_B": 0.0,
                "pt_a": None, "pt_b": None,
                "is_valid": False
            }

        src = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([self.ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or inl.sum() < 15:
            return {
                "status": "ALIGNMENT FAILED",
                "mm": None, "dev": None,
                "nominal_mm": self.nominal_mm,
                "tolerance_mm": self.tolerance_mm,
                "score_A": 0.0, "score_B": 0.0,
                "pt_a": None, "pt_b": None,
                "is_valid": False
            }

        work = cv2.warpPerspective(scaled_frame, H, (self.w, self.h))
        work_gray = self.preprocess(work)

        pa, sa = self.locate(work_gray, "A")
        pb, sb = self.locate(work_gray, "B")
        if pa is None or pb is None or min(sa, sb) < self.min_match_score:
            missing = []
            if sa < self.min_match_score: missing.append("CATCH PLUG")
            if sb < self.min_match_score: missing.append("TRIMLINE")
            return {
                "status": f"{' & '.join(missing)} NOT FOUND",
                "mm": None, "dev": None,
                "nominal_mm": self.nominal_mm,
                "tolerance_mm": self.tolerance_mm,
                "score_A": float(sa), "score_B": float(sb),
                "pt_a": None, "pt_b": None,
                "is_valid": False
            }

        dist_px = abs(pb[0] - pa[0]) if self.axis == "x" else float(np.hypot(pb[0] - pa[0], pb[1] - pa[1]))
        dist_mm = dist_px * self.mm_per_px
        dev = dist_mm - self.nominal_mm
        status = "OK" if abs(dev) <= self.tolerance_mm + 1e-9 else "NG"

        # Project points back to original unwarped frame coordinates
        try:
            H_inv = np.linalg.inv(H)
            by_coord = pa[1] if self.axis == "x" else pb[1]
            pts_ref = np.array([[[pa[0], pa[1]]], [[pb[0], by_coord]]], dtype=np.float32)
            pts_orig = cv2.perspectiveTransform(pts_ref, H_inv).reshape(-1, 2)

            if need_resize:
                scale_x = orig_w / float(self.w)
                scale_y = orig_h / float(self.h)
                pt_a = (float(pts_orig[0][0] * scale_x), float(pts_orig[0][1] * scale_y))
                pt_b = (float(pts_orig[1][0] * scale_x), float(pts_orig[1][1] * scale_y))
            else:
                pt_a = (float(pts_orig[0][0]), float(pts_orig[0][1]))
                pt_b = (float(pts_orig[1][0]), float(pts_orig[1][1]))
        except Exception:
            pt_a = None
            pt_b = None

        return {
            "status": status,
            "mm": float(dist_mm),
            "dev": float(dev),
            "nominal_mm": float(self.nominal_mm),
            "tolerance_mm": float(self.tolerance_mm),
            "score_A": float(sa),
            "score_B": float(sb),
            "pt_a": pt_a,
            "pt_b": pt_b,
            "is_valid": True
        }

    @staticmethod
    def draw_overlay(img, result, scale=1.0):
        """Draws measurement line, end ticks, and status badge on an annotated frame."""
        if not result or not result.get("is_valid") or not result.get("pt_a") or not result.get("pt_b"):
            return

        pa = result["pt_a"]
        pb = result["pt_b"]
        d_mm = result["mm"]
        d_status = result["status"]
        color = (0, 220, 0) if d_status == "OK" else (0, 0, 255) # Green / Red

        p1 = (int(round(pa[0] * scale)), int(round(pa[1] * scale)))
        p2 = (int(round(pb[0] * scale)), int(round(pb[1] * scale)))

        # Measurement line between A and B
        thickness = max(2, int(round(3 * scale)))
        cv2.line(img, p1, p2, color, thickness)

        # End ticks
        tick = max(6, int(round(15 * scale)))
        cv2.line(img, (p1[0], p1[1] - tick), (p1[0], p1[1] + tick), color, thickness)
        cv2.line(img, (p2[0], p2[1] - tick), (p2[0], p2[1] + tick), color, thickness)

        # Dimension text badge
        txt = f"{d_mm:.2f} mm"
        font_scale = max(0.4, 0.6 * scale)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = max(th + 5, min(p1[1], p2[1]) - 8)
        cv2.rectangle(img, (mid_x - tw // 2 - 4, mid_y - th - 4), (mid_x + tw // 2 + 4, mid_y + 4), (0, 0, 0), -1)
        cv2.putText(img, txt, (mid_x - tw // 2, mid_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

        # Live feed status pill in corner
        badge_txt = f"SU21 GAUGE: {d_mm:.2f} mm [{'PASS' if d_status == 'OK' else 'FAIL'}]"
        (bw, bh), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bx = max(10, int(10 * scale))
        by = max(10, int(10 * scale))
        cv2.rectangle(img, (bx, by), (bx + bw + 12, by + bh + 10), (0, 0, 0), -1)
        cv2.rectangle(img, (bx, by), (bx + bw + 12, by + bh + 10), color, 1)
        cv2.putText(img, badge_txt, (bx + 6, by + bh + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def log_ng(self, result, frame):
        """Logs NG snapshot and CSV entry to ng_logs/."""
        try:
            os.makedirs("ng_logs", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_path = os.path.join("ng_logs", f"SU21_{ts}.jpg")
            cv2.imwrite(img_path, frame)
            csv_path = os.path.join("ng_logs", "ng_log.csv")
            new = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "model", "status", "mm", "deviation", "score_A", "score_B"])
                w.writerow([ts, self.model_name, result.get("status"), result.get("mm"), result.get("dev"),
                            result.get("score_A"), result.get("score_B")])
            logger.info(f"[SU21Gauge] Logged NG distance measurement to {img_path}")
        except Exception as e:
            logger.error(f"[SU21Gauge] Error writing ng_log: {e}")
