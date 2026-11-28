"""
Rectangle YOLO Distance & Bend Gauge for SSP Fuel Door Inspection.
Uses YOLO11-Seg (Rect_Dist_se2i-bn7i.pt) to inspect the back side of Rectangle fuel doors:
- Detects Catch_plug and trimline (su2i / wide / lwd) to measure distance.
- Directly detects 'catchplug_bend' to flag bent catch plug defects immediately.
- Spec (SU21 CAD): 140.769 mm +/- 1.0 mm.
"""

import os
import csv
import logging
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("RectYOLOGauge")

class RectYOLODistanceGauge:
    def __init__(self, model_path="Rect_Dist_se2i-bn7i.pt", nominal_mm=140.769, tolerance_mm=1.0, mm_per_px=0.12145729, axis="x", conf_thresh=0.35):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Rectangle Distance model not found at: {self.model_path}")

        logger.info(f"[RectGauge] Loading model from {self.model_path}...")
        self.model = YOLO(self.model_path)
        self.model_name = "RECTANGLE"
        self.nominal_mm = float(nominal_mm)
        self.tolerance_mm = float(tolerance_mm)
        self.mm_per_px = float(mm_per_px)
        self.axis = axis  # "x" for horizontal
        self.conf_thresh = float(conf_thresh)

        # Logging directory
        self.ng_dir = "ng_logs"
        os.makedirs(self.ng_dir, exist_ok=True)
        self.log_csv = os.path.join(self.ng_dir, "rect_distance_log.csv")
        self._init_csv()

        logger.info(f"[RectGauge] Initialized {self.model_name} Gauge: nominal {self.nominal_mm} +/- {self.tolerance_mm} mm (scale={self.mm_per_px:.6f} mm/px)")

    def _init_csv(self):
        if not os.path.exists(self.log_csv):
            try:
                with open(self.log_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "status", "mm", "dev_mm", "nominal_mm", "tolerance_mm", "dist_px", "variant", "defect", "snapshot"])
            except Exception as e:
                logger.error(f"[RectGauge] Failed to init log CSV: {e}")

    def measure(self, frame):
        """
        Runs YOLO inference on frame to find catch plug, trimline, and bend defects.
        """
        if frame is None:
            return {"status": "NO FRAME", "mm": None, "dev": None, "is_valid": False}

        try:
            results = self.model(frame, imgsz=1024, conf=self.conf_thresh, verbose=False)
        except Exception as e:
            logger.error(f"[RectGauge] Inference error: {e}")
            return {"status": "INFERENCE ERROR", "mm": None, "dev": None, "is_valid": False}

        if not results or len(results) == 0:
            return {
                "status": "PART NOT FOUND",
                "mm": None, "dev": None,
                "nominal_mm": self.nominal_mm,
                "tolerance_mm": self.tolerance_mm,
                "is_valid": False
            }

        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else []
        classes = [self.model.names[int(c)] for c in res.boxes.cls.cpu().numpy()] if len(res.boxes) > 0 else []
        confs = res.boxes.conf.cpu().numpy() if len(res.boxes) > 0 else []

        plug_box = None
        plug_conf = 0.0
        bend_box = None
        bend_conf = 0.0
        trim_box = None
        trim_conf = 0.0
        variant = "RECTANGLE"

        for b, c, conf in zip(boxes, classes, confs):
            if c == "Catch_plug" and conf > plug_conf:
                plug_box = b
                plug_conf = float(conf)
            elif c == "catchplug_bend" and conf > bend_conf:
                bend_box = b
                bend_conf = float(conf)
            elif c in ["su2i", "wide", "lwd"] and conf > trim_conf:
                trim_box = b
                trim_conf = float(conf)
                variant = c.upper()

        # If a bent catch plug was detected -> Immediate Defect Flag
        is_bent = (bend_box is not None)
        active_plug = bend_box if is_bent else plug_box
        active_conf = bend_conf if is_bent else plug_conf

        if active_plug is None and trim_box is None:
            return {
                "status": "FEATURES NOT FOUND",
                "mm": None, "dev": None,
                "nominal_mm": self.nominal_mm,
                "tolerance_mm": self.tolerance_mm,
                "is_valid": False
            }

        if active_plug is not None and trim_box is not None:
            # Measure distance from plug to trimline
            # Horizontal (x) axis
            pa = (float((active_plug[0] + active_plug[2]) / 2.0), float((active_plug[1] + active_plug[3]) / 2.0))
            pb = (float((trim_box[0] + trim_box[2]) / 2.0), float((trim_box[1] + trim_box[3]) / 2.0))

            if self.axis == "x":
                dist_px = abs(pb[0] - pa[0])
            else:
                dist_px = float(np.hypot(pb[0] - pa[0], pb[1] - pa[1]))

            dist_mm = dist_px * self.mm_per_px
            dev = dist_mm - self.nominal_mm
            
            # Status: If bent defect detected, it is always NG!
            if is_bent:
                status = "NG"
                defect_label = "catchplug_bend"
            else:
                status = "OK" if abs(dev) <= self.tolerance_mm else "NG"
                defect_label = "distance_out_of_spec" if status == "NG" else None

            return {
                "status": status,
                "mm": round(dist_mm, 2),
                "dev": round(dev, 2),
                "nominal_mm": self.nominal_mm,
                "tolerance_mm": self.tolerance_mm,
                "dist_px": round(dist_px, 1),
                "pt_a": pa,
                "pt_b": pb,
                "box_a": active_plug.tolist(),
                "box_b": trim_box.tolist(),
                "score_A": active_conf,
                "score_B": trim_conf,
                "is_valid": True,
                "is_bent": is_bent,
                "defect": defect_label,
                "variant": variant,
                "model_name": f"RECT ({variant})"
            }

        elif is_bent:
            # Bent detected even if trimline is occluded
            return {
                "status": "NG",
                "mm": None, "dev": None,
                "nominal_mm": self.nominal_mm,
                "tolerance_mm": self.tolerance_mm,
                "is_valid": True,
                "is_bent": True,
                "defect": "catchplug_bend",
                "variant": variant,
                "model_name": f"RECT ({variant})"
            }

        return {
            "status": "PARTIAL DETECTION",
            "mm": None, "dev": None,
            "nominal_mm": self.nominal_mm,
            "tolerance_mm": self.tolerance_mm,
            "is_valid": False
        }

    def draw_overlay(self, frame, res, scale=1.0):
        if not res or not res.get("is_valid"):
            return

        is_bent = res.get("is_bent", False)
        status = res.get("status", "UNKNOWN")
        is_ok = (status == "OK" and not is_bent)
        color = (0, 255, 0) if is_ok else (0, 0, 255)

        pt_a = res.get("pt_a")
        pt_b = res.get("pt_b")

        if pt_a and pt_b:
            p1 = (int(pt_a[0]), int(pt_a[1]))
            p2 = (int(pt_b[0]), int(pt_b[1]))

            # Draw measurement line & ticks
            cv2.line(frame, p1, p2, color, max(2, int(3 * scale)))
            tick_len = int(12 * scale)
            cv2.line(frame, (p1[0], p1[1] - tick_len), (p1[0], p1[1] + tick_len), color, max(2, int(3 * scale)))
            cv2.line(frame, (p2[0], p2[1] - tick_len), (p2[0], p2[1] + tick_len), color, max(2, int(3 * scale)))

            # Draw boxes around detected features
            box_a = res.get("box_a")
            box_b = res.get("box_b")
            if box_a:
                cv2.rectangle(frame, (int(box_a[0]), int(box_a[1])), (int(box_a[2]), int(box_a[3])), (0, 0, 255) if is_bent else (0, 255, 255), max(1, int(2 * scale)))
            if box_b:
                cv2.rectangle(frame, (int(box_b[0]), int(box_b[1])), (int(box_b[2]), int(box_b[3])), (255, 200, 0), max(1, int(2 * scale)))

            # Measurement badge
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2 - int(25 * scale)

            if is_bent:
                text = f"BENT PLUG: {res.get('mm', 0):.2f} mm [NG]"
            elif res.get("mm") is not None:
                text = f"{res['mm']:.2f} mm"
            else:
                text = f"{status}"

            font_scale = 0.9 * scale
            thick = max(2, int(2 * scale))
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)

            # Badge background
            pad = int(8 * scale)
            bx1, by1 = mid_x - tw // 2 - pad, mid_y - th - pad
            bx2, by2 = mid_x + tw // 2 + pad, mid_y + pad
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, max(1, int(2 * scale)))
            cv2.putText(frame, text, (mid_x - tw // 2, mid_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick, cv2.LINE_AA)

    def log_ng(self, res, frame):
        if not res or res.get("status") != "NG":
            return None

        ts = datetime.now()
        ts_str = ts.strftime("%Y%m%d_%H%M%S_%f")
        iso_ts = ts.isoformat()

        snap_filename = f"rect_ng_{ts_str}.jpg"
        snap_path = os.path.join(self.ng_dir, snap_filename)

        try:
            cv2.imwrite(snap_path, frame)
        except Exception as e:
            logger.error(f"[RectGauge] Failed to save NG snapshot: {e}")
            snap_path = ""

        try:
            with open(self.log_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    iso_ts,
                    res.get("status", "NG"),
                    f"{res.get('mm', 0.0):.3f}" if res.get('mm') is not None else "",
                    f"{res.get('dev', 0.0):.3f}" if res.get('dev') is not None else "",
                    f"{res.get('nominal_mm', self.nominal_mm):.3f}",
                    f"{res.get('tolerance_mm', self.tolerance_mm):.3f}",
                    f"{res.get('dist_px', 0.0):.1f}" if res.get('dist_px') is not None else "",
                    res.get("variant", "RECTANGLE"),
                    res.get("defect", "NG"),
                    snap_path
                ])
        except Exception as e:
            logger.error(f"[RectGauge] Failed to write NG CSV row: {e}")

        return snap_path
