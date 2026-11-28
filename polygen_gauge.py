"""
Polygen Pose Distance Gauge for SSP Fuel Door Inspection.
Uses YOLOv8-Pose (POLYGEN_DIST_GAUGE.pt) to measure direct straight-line distance
between the two taught reference points on the Polygon fuel door back panel.
Spec: 111.112 mm +/- 1.0 mm (straight-line / xy).
"""

import os
import csv
import logging
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("PolygenGauge")

class PolygenPoseGauge:
    def __init__(self, model_path="POLYGEN_DIST_GAUGE.pt", nominal_mm=111.112, tolerance_mm=1.0, mm_per_px=0.12145729, axis="xy", conf_thresh=0.40):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Polygen Pose model not found at: {self.model_path}")

        logger.info(f"[PolygenGauge] Loading pose model from {self.model_path}...")
        self.model = YOLO(self.model_path)
        self.model_name = "POLYGON"
        self.nominal_mm = float(nominal_mm)
        self.tolerance_mm = float(tolerance_mm)
        self.mm_per_px = float(mm_per_px)
        self.axis = axis  # "xy" = direct straight line, "x" = horizontal
        self.conf_thresh = float(conf_thresh)

        logger.info(f"[PolygenGauge] Initialized {self.model_name} Gauge: nominal {self.nominal_mm} +/- {self.tolerance_mm} mm (axis={self.axis}, scale={self.mm_per_px:.6f} mm/px)")

    def measure(self, frame):
        """
        Runs YOLOv8-Pose inference on frame and calculates distance.
        Returns standard dictionary compatible with app.py and generate_report.py.
        """
        if frame is None:
            return {"status": "NO FRAME", "mm": None, "dev": None, "is_valid": False}

        try:
            results = self.model(frame, imgsz=1024, conf=self.conf_thresh, verbose=False)
        except Exception as e:
            logger.error(f"[PolygenGauge] Inference error: {e}")
            return {"status": "INFERENCE ERROR", "mm": None, "dev": None, "is_valid": False}

        if not results or len(results) == 0:
            return {
                "status": "PART NOT FOUND",
                "mm": None, "dev": None,
                "nominal_mm": self.nominal_mm,
                "tolerance_mm": self.tolerance_mm,
                "pt_a": None, "pt_b": None,
                "is_valid": False
            }

        # Check the detected keypoints
        best_r = results[0]
        if best_r.keypoints is None or len(best_r.keypoints.xy) == 0 or len(best_r.keypoints.xy[0]) < 2:
            return {
                "status": "KEYPOINTS NOT FOUND",
                "mm": None, "dev": None,
                "nominal_mm": self.nominal_mm,
                "tolerance_mm": self.tolerance_mm,
                "pt_a": None, "pt_b": None,
                "is_valid": False
            }

        kpts = best_r.keypoints.xy[0].cpu().numpy()
        conf = best_r.keypoints.conf[0].cpu().numpy() if best_r.keypoints.conf is not None else [1.0, 1.0]

        conf_a = float(conf[0])
        conf_b = float(conf[1])

        if conf_a < self.conf_thresh or conf_b < self.conf_thresh:
            return {
                "status": "LOW CONFIDENCE KEYPOINTS",
                "mm": None, "dev": None,
                "nominal_mm": self.nominal_mm,
                "tolerance_mm": self.tolerance_mm,
                "score_A": conf_a,
                "score_B": conf_b,
                "pt_a": None, "pt_b": None,
                "is_valid": False
            }

        pa = (float(kpts[0][0]), float(kpts[0][1]))
        pb = (float(kpts[1][0]), float(kpts[1][1]))

        # Calculate distance
        if self.axis == "x":
            dist_px = abs(pb[0] - pa[0])
        else:
            # Direct straight-line distance
            dist_px = float(np.hypot(pb[0] - pa[0], pb[1] - pa[1]))

        dist_mm = dist_px * self.mm_per_px
        dev = dist_mm - self.nominal_mm
        status = "OK" if abs(dev) <= self.tolerance_mm + 1e-9 else "NG"

        return {
            "status": status,
            "mm": float(dist_mm),
            "dev": float(dev),
            "nominal_mm": float(self.nominal_mm),
            "tolerance_mm": float(self.tolerance_mm),
            "dist_px": float(dist_px),
            "score_A": conf_a,
            "score_B": conf_b,
            "pt_a": pa,
            "pt_b": pb,
            "is_valid": True,
            "model_name": "POLYGON"
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
        dev = result.get("dev", 0.0)
        color = (0, 230, 0) if d_status == "OK" else (0, 0, 255) # Crisp Green / Red (BGR)

        p1 = (int(round(pa[0] * scale)), int(round(pa[1] * scale)))
        p2 = (int(round(pb[0] * scale)), int(round(pb[1] * scale)))

        # Measurement line between A and B
        thickness = max(2, int(round(3 * scale)))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

        # End ticks
        tick = max(8, int(round(16 * scale)))
        # Vector perpendicular to p1->p2 for tilted end ticks
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.hypot(dx, dy)
        if length > 0:
            nx = -dy / length
            ny = dx / length
            t1_a = (int(p1[0] + nx * tick), int(p1[1] + ny * tick))
            t1_b = (int(p1[0] - nx * tick), int(p1[1] - ny * tick))
            t2_a = (int(p2[0] + nx * tick), int(p2[1] + ny * tick))
            t2_b = (int(p2[0] - nx * tick), int(p2[1] - ny * tick))
            cv2.line(img, t1_a, t1_b, color, thickness, cv2.LINE_AA)
            cv2.line(img, t2_a, t2_b, color, thickness, cv2.LINE_AA)

        # High-contrast dimension text badge over the line
        font = cv2.FONT_HERSHEY_DUPLEX
        txt = f"{d_mm:.2f} mm"
        font_scale = 0.65 if scale < 0.6 else 0.85
        (tw, th), _ = cv2.getTextSize(txt, font, font_scale, 1)
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = max(th + 12, min(p1[1], p2[1]) - 14)
        pad_x, pad_y = 10, 6

        cv2.rectangle(img, (mid_x - tw // 2 - pad_x, mid_y - th - pad_y),
                           (mid_x + tw // 2 + pad_x, mid_y + pad_y), (15, 23, 42), -1)
        cv2.rectangle(img, (mid_x - tw // 2 - pad_x, mid_y - th - pad_y),
                           (mid_x + tw // 2 + pad_x, mid_y + pad_y), color, 1, cv2.LINE_AA)
        cv2.putText(img, txt, (mid_x - tw // 2, mid_y - 1), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Prominent top header banner
        badge_font_s = 0.55 if scale < 0.6 else 0.72
        sign = "+" if dev >= 0 else ""
        badge_title = f"POLY GAUGE: {d_mm:.2f} mm (Diff: {sign}{dev:.2f} mm) "
        badge_status = f"[{'PASS' if d_status == 'OK' else 'FAIL'}]"

        (bw1, bh1), _ = cv2.getTextSize(badge_title, font, badge_font_s, 1)
        (bw2, bh2), _ = cv2.getTextSize(badge_status, font, badge_font_s, 1)
        bx, by = 12, 12
        total_w = bw1 + bw2 + 24
        total_h = max(bh1, bh2) + 14

        cv2.rectangle(img, (bx, by), (bx + total_w, by + total_h), (15, 23, 42), -1)
        cv2.rectangle(img, (bx, by), (bx + total_w, by + total_h), color, 1, cv2.LINE_AA)
        cv2.putText(img, badge_title, (bx + 8, by + total_h - 5), font, badge_font_s, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, badge_status, (bx + bw1 + 10, by + total_h - 5), font, badge_font_s, color, 1, cv2.LINE_AA)

    def log_ng(self, result, frame):
        """Logs NG snapshot and CSV entry to ng_logs/."""
        try:
            os.makedirs("ng_logs", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            snap_path = os.path.join("ng_logs", f"poly_ng_{ts}.jpg")
            
            # Save annotated snapshot
            vis = frame.copy()
            self.draw_overlay(vis, result, scale=1.0)
            cv2.imwrite(snap_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Append CSV log
            csv_path = os.path.join("ng_logs", "polygen_distance_log.csv")
            file_exists = os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "status", "mm", "dev_mm", "nominal_mm", "tolerance_mm", "dist_px", "score_A", "score_B", "snapshot"])
                writer.writerow([
                    datetime.now().isoformat(),
                    result["status"],
                    f"{result['mm']:.3f}",
                    f"{result['dev']:.3f}",
                    f"{result['nominal_mm']:.3f}",
                    f"{result['tolerance_mm']:.3f}",
                    f"{result.get('dist_px', 0.0):.1f}",
                    f"{result.get('score_A', 0.0):.3f}",
                    f"{result.get('score_B', 0.0):.3f}",
                    snap_path
                ])
        except Exception as e:
            logger.error(f"[PolygenGauge] Failed to log NG: {e}")
