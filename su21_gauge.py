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
        self.min_match_score = 0.60
        self.search_margin_px = 120

        # Load reference image and precompute ORB features
        ref_path = os.path.join(cfg_dir, "reference.png")
        self.ref = cv2.imread(ref_path)
        if self.ref is None:
            raise FileNotFoundError(f"Reference image not found: {ref_path}")

        self.ref_gray = self.preprocess(self.ref)
        self.h, self.w = self.ref_gray.shape

        # Load templates A (catch plug: normal + optional bend variants) and B (trimline)
        self.tmpl = {}
        for k in ("A", "B"):
            tmpl_path = os.path.join(cfg_dir, f"template_{k}.png")
            t = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)
            if t is None:
                raise FileNotFoundError(f"Template {k} not found: {tmpl_path}")
            self.tmpl[k] = t

        # Calibrated Template A box on reference.png -> tight aligned ROI for normal catch plug
        rx_a, ry_a, rw_a, rh_a = self.cfg["roi_A"]
        res_a_ref = cv2.matchTemplate(
            self.ref_gray[ry_a:ry_a + rh_a, rx_a:rx_a + rw_a],
            self.tmpl["A"],
            cv2.TM_CCOEFF_NORMED,
        )
        _, _, _, loc_a_ref = cv2.minMaxLoc(res_a_ref)
        ta_h, ta_w = self.tmpl["A"].shape[:2]
        self.ref_box_A = [rx_a + loc_a_ref[0], ry_a + loc_a_ref[1], ta_w, ta_h]
        aligned_roi_A = [
            max(self.ref_box_A[0] - 90, 0),
            max(self.ref_box_A[1] - 15, 0),
            ta_w + 180,
            ta_h + 30,
        ]

        self.tmpl_A_variants = [
            {
                "name": "normal",
                "tmpl": self.tmpl["A"],
                "tmpl_grad": self._grad_mag(self.tmpl["A"]),
                "roi": self.cfg["roi_A"],
                "aligned_roi": aligned_roi_A,
                "offset": self.cfg["offset_A"],
                "min_score": max(self.min_match_score, 0.65),
            }
        ]
        for extra in self.cfg.get("extra_templates_A", []):
            extra_path = os.path.join(cfg_dir, extra["file"])
            t_extra = cv2.imread(extra_path, cv2.IMREAD_GRAYSCALE)
            if t_extra is not None:
                self.tmpl_A_variants.append({
                    "name": extra.get("name", extra["file"]),
                    "tmpl": t_extra,
                    "tmpl_grad": self._grad_mag(t_extra),
                    "roi": extra["roi"],
                    "aligned_roi": extra["roi"],
                    "offset": extra["offset"],
                    "min_score": float(extra.get("min_score", self.min_match_score)),
                })
            else:
                logger.warning(f"[SU21Gauge] Optional Template A variant not found: {extra_path}")

        # Fixed reference location for Template B on aligned frames
        rx_b, ry_b, rw_b, rh_b = self.cfg["roi_B"]
        res_b_ref = cv2.matchTemplate(
            self.ref_gray[ry_b:ry_b + rh_b, rx_b:rx_b + rw_b],
            self.tmpl["B"],
            cv2.TM_CCOEFF_NORMED,
        )
        _, _, _, loc_b_ref = cv2.minMaxLoc(res_b_ref)
        tb_h, tb_w = self.tmpl["B"].shape[:2]
        self.ref_box_B = [rx_b + loc_b_ref[0], ry_b + loc_b_ref[1], tb_w, tb_h]
        if "fixed_pt_B" in self.cfg:
            fb = self.cfg["fixed_pt_B"]
            self.fixed_pt_B = (float(fb[0]), float(fb[1]))
        else:
            ox_b, oy_b = self.cfg["offset_B"]
            self.fixed_pt_B = (float(self.ref_box_B[0] + ox_b), float(self.ref_box_B[1] + oy_b))

        # Feature matching on the PART only (so static tray does not dominate)
        self.orb = cv2.ORB_create(4000)
        mask = np.zeros_like(self.ref_gray)
        rx, ry, rw, rh = self.cfg["part_roi"]
        mask[ry:ry + rh, rx:rx + rw] = 255
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(self.ref_gray, mask)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        logger.info(f"[SU21Gauge] Initialized {self.model_name}: nominal {self.nominal_mm} +/- {self.tolerance_mm} mm (scale={self.mm_per_px:.6f} mm/px)")

    @staticmethod
    def _grad_mag(gray):
        """Smoothed Sobel gradient magnitude invariant to lighting/glare polarity shifts."""
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.GaussianBlur(cv2.magnitude(gx, gy), (5, 5), 1.2)

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

    def _match_template(self, gray, tmpl, roi, offset, margin=None, guided_box=None, min_score=None):
        """Matches a single template inside roi (+ margin), optionally constrained near guided_box."""
        x, y, w, h = roi
        m = self.search_margin_px if margin is None else margin
        sx, sy = max(int(x - m), 0), max(int(y - m), 0)
        ex, ey = min(int(x + w + m), gray.shape[1]), min(int(y + h + m), gray.shape[0])

        if guided_box is not None:
            gx1, gy1, gx2, gy2 = guided_box
            pad = 120
            gsx, gsy = max(int(gx1 - pad), sx), max(int(gy1 - pad), sy)
            gex, gey = min(int(gx2 + pad), ex), min(int(gy2 + pad), ey)
            if (gex - gsx) >= tmpl.shape[1] and (gey - gsy) >= tmpl.shape[0]:
                sx, sy, ex, ey = gsx, gsy, gex, gey

        area = gray[sy:ey, sx:ex]
        if area.shape[0] < tmpl.shape[0] or area.shape[1] < tmpl.shape[1]:
            return None, 0.0
        res = cv2.matchTemplate(area, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)

        thresh = self.min_match_score if min_score is None else min_score
        # Dynamic recovery: if score is borderline (< thresh), test slight scale variations
        if score < thresh:
            best_score = score
            best_loc = loc
            for sc in (0.97, 0.985, 1.015, 1.03):
                tw = int(round(tmpl.shape[1] * sc))
                th = int(round(tmpl.shape[0] * sc))
                if area.shape[1] >= tw and area.shape[0] >= th:
                    t_scaled = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_LINEAR)
                    r_sc = cv2.matchTemplate(area, t_scaled, cv2.TM_CCOEFF_NORMED)
                    _, s_sc, _, l_sc = cv2.minMaxLoc(r_sc)
                    if s_sc > best_score:
                        best_score = s_sc
                        best_loc = l_sc
                        res = r_sc
            score = best_score
            loc = best_loc

        px, py = self.subpixel(res, loc)
        ox, oy = offset
        return (float(sx + px + ox), float(sy + py + oy)), float(score)

    def locate(self, gray, key, margin=None):
        """Locates template within taught ROI + search margin with subpixel refinement and dynamic recovery."""
        return self._match_template(
            gray,
            self.tmpl[key],
            self.cfg[f"roi_{key}"],
            self.cfg[f"offset_{key}"],
            margin=margin,
        )

    def locate_A(self, gray, margin=None, guided_box=None, aligned=False):
        """Locates catch plug across normal, bend-down, and bend-up Template A references."""
        best_pass = None  # (score, pt, name, min_score)
        best_any = None   # (score, pt, name, min_score)
        for var in self.tmpl_A_variants:
            roi = var["aligned_roi"] if aligned else var["roi"]
            var_margin = 0 if (aligned and margin is None) else margin
            min_sc = var["min_score"]
            pt, score = self._match_template(
                gray,
                var["tmpl"],
                roi,
                var["offset"],
                margin=var_margin,
                guided_box=guided_box,
                min_score=min_sc,
            )
            if pt is None:
                continue
            cand = (score, pt, var["name"], min_sc)
            if best_any is None or score > best_any[0]:
                best_any = cand
            if score >= min_sc and (best_pass is None or score > best_pass[0]):
                best_pass = cand
        chosen = best_pass or best_any
        if chosen is None:
            return None, 0.0, "normal", self.min_match_score
        return chosen[1], float(chosen[0]), chosen[2], float(chosen[3])

    def locate_A_gradient(self, work_gray, guided_box=None, margin=25, min_grad_score=0.45):
        """Edge-gradient fallback for reflective/glare catch plugs on ORB-aligned frames."""
        work_grad = self._grad_mag(work_gray)
        best_pass = None
        best_any = None
        for var in self.tmpl_A_variants:
            pt, score = self._match_template(
                work_grad,
                var["tmpl_grad"],
                var["aligned_roi"],
                var["offset"],
                margin=margin,
                guided_box=guided_box,
                min_score=min_grad_score,
            )
            if pt is None:
                continue
            cand = (score, pt, f"{var['name']}_grad", min_grad_score)
            if best_any is None or score > best_any[0]:
                best_any = cand
            if score >= min_grad_score and (best_pass is None or score > best_pass[0]):
                best_pass = cand
        chosen = best_pass or best_any
        if chosen is None:
            return None, 0.0, "normal_grad", min_grad_score
        return chosen[1], float(chosen[0]), chosen[2], float(chosen[3])

    def locate_B_aligned(self, work_gray):
        """
        Locates Template B on an ORB-aligned frame.
        Since Template B is always at a fixed position on the panel, searches tightly (+/-20 px)
        around its calibrated reference position and falls back to fixed_pt_B directly.
        """
        pb, sb = self._match_template(
            work_gray,
            self.tmpl["B"],
            self.ref_box_B,
            self.cfg["offset_B"],
            margin=20,
        )
        if pb is not None and sb >= self.min_match_score:
            return pb, sb
        return self.fixed_pt_B, 1.0

    @staticmethod
    def _plug_box_to_point(plug_box, plug_class=None):
        """Computes catch plug reference point directly from a YOLO Catch_plug/catchplug_bend/catchplug_wide box."""
        x1, y1, x2, y2 = plug_box
        cy = 0.5 * (y1 + y2)
        if plug_class == "catchplug_bend":
            cx = x1 + 0.845 * (x2 - x1)
        elif plug_class == "catchplug_wide":
            cx = x1 + 0.14 * (x2 - x1)
        else:
            cx = x1 + 0.62 * (x2 - x1)
        return (float(cx), float(cy))

    @staticmethod
    def _warp_box(box, H):
        """Warps an [x1, y1, x2, y2] bounding box through homography H into reference space."""
        if box is None or H is None:
            return None
        x1, y1, x2, y2 = box
        corners = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        return (
            float(np.min(warped[:, 0])),
            float(np.min(warped[:, 1])),
            float(np.max(warped[:, 0])),
            float(np.max(warped[:, 1])),
        )

    def _match_homography(self, kp, des):
        """Finds affine/perspective homography H and inlier count from precomputed ORB keypoints/descriptors."""
        min_inliers = 15
        if des is None or len(kp) < min_inliers:
            return None, 0
        good = [m for m, n in (p for p in self.matcher.knnMatch(self.ref_des, des, k=2) if len(p) == 2)
                if m.distance < 0.85 * n.distance]
        if len(good) < min_inliers:
            return None, 0
        src = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([self.ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

        M, inl_aff = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6.0)
        if M is not None and inl_aff is not None and int(inl_aff.sum()) >= min_inliers:
            sc = float(np.hypot(M[0, 0], M[1, 0]))
            if 0.85 <= sc <= 1.15:
                return np.vstack([M, [0, 0, 1]]), int(inl_aff.sum())

        H_cand, inl_h = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
        if H_cand is not None and inl_h is not None and int(inl_h.sum()) >= min_inliers:
            det = float(H_cand[0, 0] * H_cand[1, 1] - H_cand[0, 1] * H_cand[1, 0])
            if 0.70 <= det <= 1.40:
                return H_cand, int(inl_h.sum())
        return None, 0

    def _prepare_orb(self, frame):
        """Resizes frame (if needed), preprocesses grayscale, and extracts ORB features once."""
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
        return orig_w, orig_h, need_resize, scaled_frame, gray, kp, des

    def measure(self, frame, plug_box=None, plug_class=None, precomputed=None):
        """
        Runs full alignment, feature location, and distance calculation on a camera frame.
        Supports straight, bend-down, and bend-up catch plugs via multi-template A matching
        and optional YOLO Catch_plug / catchplug_bend guidance, while keeping Template B fixed
        on the aligned part.
        """
        if frame is None:
            return {"status": "NO FRAME", "mm": None, "dev": None, "is_valid": False}

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if precomputed is not None:
            orig_w, orig_h, need_resize, scaled_frame, gray, kp, des, H, inl = precomputed
        else:
            orig_w, orig_h, need_resize, scaled_frame, gray, kp, des = self._prepare_orb(frame)
            H, inl = self._match_homography(kp, des)

        scaled_plug_box = None
        if plug_box is not None:
            if need_resize:
                sx = self.w / float(orig_w)
                sy = self.h / float(orig_h)
                scaled_plug_box = (plug_box[0] * sx, plug_box[1] * sy, plug_box[2] * sx, plug_box[3] * sy)
            else:
                scaled_plug_box = plug_box

        # If aligned frame is available, measure on warped frame
        if H is not None:
            work = cv2.warpPerspective(scaled_frame, H, (self.w, self.h))
            work_gray = self.preprocess(work)
            warped_plug_box = self._warp_box(scaled_plug_box, H)

            pa, sa, var_a, thresh_a = self.locate_A(work_gray, guided_box=warped_plug_box, aligned=True)
            if pa is None or sa < thresh_a:
                pa_rec, sa_rec, var_rec, thresh_rec = self.locate_A(work_gray, margin=60, guided_box=warped_plug_box, aligned=True)
                if pa_rec is not None and sa_rec >= thresh_rec and (sa < thresh_a or sa_rec > sa):
                    pa, sa, var_a, thresh_a = pa_rec, sa_rec, var_rec, thresh_rec

            # Edge-gradient recovery when specular glare changes catch-plug brightness
            if (pa is None or sa < thresh_a) and inl >= 10:
                pa_g, sa_g, var_g, thresh_g = self.locate_A_gradient(work_gray, guided_box=warped_plug_box)
                if pa_g is not None and sa_g >= thresh_g:
                    pa, sa, var_a, thresh_a = pa_g, sa_g, var_g, thresh_g

            # YOLO catch plug anchor fallback if all Template A variants scored below threshold
            if (pa is None or sa < thresh_a) and warped_plug_box is not None:
                pa = self._plug_box_to_point(warped_plug_box, plug_class)
                sa = 0.85
                thresh_a = self.min_match_score
                var_a = f"yolo_{plug_class or 'catch_plug'}"

            # Template B is always in a fixed place on the aligned part
            pb, sb = self.locate_B_aligned(work_gray)

            if pa is not None and pb is not None and sa >= thresh_a:
                dist_px = abs(pb[0] - pa[0]) if self.axis == "x" else float(np.hypot(pb[0] - pa[0], pb[1] - pa[1]))
                dist_mm = dist_px * self.mm_per_px
                dev = dist_mm - self.nominal_mm

                # Physical plausibility gate: allow straight as well as bend-down / bend-up catch plugs
                if abs(dev) <= 70.0:
                    status = "OK" if abs(dev) <= self.tolerance_mm + 1e-9 else "NG"

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
                        "model_name": self.model_name,
                        "status": status,
                        "mm": float(dist_mm),
                        "dev": float(dev),
                        "nominal_mm": float(self.nominal_mm),
                        "tolerance_mm": float(self.tolerance_mm),
                        "score_A": float(sa),
                        "score_B": float(sb),
                        "template_A_variant": var_a,
                        "pt_a": pt_a,
                        "pt_b": pt_b,
                        "is_valid": True
                    }

        # Level 3: Direct Fixture Dynamic Recovery
        # When ORB feature alignment fails due to lighting or texture, search directly in the guided fixture tray
        pa_dir, sa_dir, var_dir, thresh_dir = self.locate_A(gray, margin=180, guided_box=scaled_plug_box)
        if (pa_dir is None or sa_dir < thresh_dir) and scaled_plug_box is not None:
            pa_dir = self._plug_box_to_point(scaled_plug_box, plug_class)
            sa_dir = 0.85
            thresh_dir = 0.60
            var_dir = f"yolo_{plug_class or 'catch_plug'}"
        pb_dir, sb_dir = self.locate(gray, "B", margin=180)
        # When ORB alignment failed, only trust Level 3 if YOLO confirmed the catch plug
        # or both template matches are high-confidence (prevents false NG on rims/hands/front hinges).
        level3_confirmed = (scaled_plug_box is not None) or (sa_dir >= 0.72 and sb_dir >= 0.70)
        if level3_confirmed and pa_dir is not None and pb_dir is not None and sa_dir >= thresh_dir and sb_dir >= 0.60:
            # Physical geometric plausibility gate: Catch plug and trimline align along expected axis
            max_dy = 220 if self.axis == "xy" else 100
            if abs(pa_dir[1] - pb_dir[1]) < max_dy:
                dist_px = abs(pb_dir[0] - pa_dir[0]) if self.axis == "x" else float(np.hypot(pb_dir[0] - pa_dir[0], pb_dir[1] - pa_dir[1]))
                dist_mm = dist_px * self.mm_per_px
                dev = dist_mm - self.nominal_mm
                if abs(dev) <= 70.0:  # Allow straight and bent catch plug measurements
                    status = "OK" if abs(dev) <= self.tolerance_mm + 1e-9 else "NG"

                    if need_resize:
                        scale_x = orig_w / float(self.w)
                        scale_y = orig_h / float(self.h)
                        pt_a = (float(pa_dir[0] * scale_x), float(pa_dir[1] * scale_y))
                        pt_b = (float(pb_dir[0] * scale_x), float(pb_dir[1] * scale_y))
                    else:
                        pt_a = (float(pa_dir[0]), float(pa_dir[1]))
                        pt_b = (float(pb_dir[0]), float(pb_dir[1]))

                    return {
                        "model_name": self.model_name,
                        "status": status,
                        "mm": float(dist_mm),
                        "dev": float(dev),
                        "nominal_mm": float(self.nominal_mm),
                        "tolerance_mm": float(self.tolerance_mm),
                        "score_A": float(sa_dir),
                        "score_B": float(sb_dir),
                        "template_A_variant": var_dir,
                        "pt_a": pt_a,
                        "pt_b": pt_b,
                        "is_valid": True,
                        "dynamic_recovery": True
                    }

        return {
            "model_name": self.model_name,
            "status": "ALIGNMENT FAILED",
            "mm": None, "dev": None,
            "nominal_mm": self.nominal_mm,
            "tolerance_mm": self.tolerance_mm,
            "score_A": 0.0, "score_B": 0.0,
            "pt_a": None, "pt_b": None,
            "is_valid": False
        }

    def count_inliers(self, frame):
        """Returns ORB RANSAC inlier count of frame against this gauge's reference image."""
        if frame is None:
            return 0
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[0] != self.h or frame.shape[1] != self.w:
            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        gray = self.preprocess(frame)
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(kp) < 15:
            return 0
        good = [m for m, n in (p for p in self.matcher.knnMatch(self.ref_des, des, k=2) if len(p) == 2)
                if m.distance < 0.85 * n.distance]
        if len(good) < 4:
            return 0
        src = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([self.ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        M, inl_aff = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6.0)
        if M is not None and inl_aff is not None:
            return int(inl_aff.sum())
        return 0

    @staticmethod
    def draw_overlay(img, result, scale=1.0):
        """Draws measurement line, end ticks, and status badge on an annotated frame with crisp anti-aliasing."""
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

        # Measurement line between A and B (crisp anti-aliased)
        thickness = max(2, int(round(3 * scale)))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

        # End ticks
        tick = max(8, int(round(16 * scale)))
        cv2.line(img, (p1[0], p1[1] - tick), (p1[0], p1[1] + tick), color, thickness, cv2.LINE_AA)
        cv2.line(img, (p2[0], p2[1] - tick), (p2[0], p2[1] + tick), color, thickness, cv2.LINE_AA)

        # High-contrast, sharp dimension text badge over the line
        font = cv2.FONT_HERSHEY_DUPLEX
        txt = f"{d_mm:.2f} mm"
        font_scale = 0.65 if scale < 0.6 else 0.85
        (tw, th), _ = cv2.getTextSize(txt, font, font_scale, 1)
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = max(th + 12, min(p1[1], p2[1]) - 14)
        pad_x, pad_y = 10, 6

        # Dark high-contrast background with colored border
        cv2.rectangle(img, (mid_x - tw // 2 - pad_x, mid_y - th - pad_y),
                           (mid_x + tw // 2 + pad_x, mid_y + pad_y), (15, 23, 42), -1)
        cv2.rectangle(img, (mid_x - tw // 2 - pad_x, mid_y - th - pad_y),
                           (mid_x + tw // 2 + pad_x, mid_y + pad_y), color, 1, cv2.LINE_AA)
        cv2.putText(img, txt, (mid_x - tw // 2, mid_y - 1), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Prominent top header banner: "SU21 GAUGE: 140.77 mm (Diff: +0.00 mm)  [PASS]"
        badge_font_s = 0.55 if scale < 0.6 else 0.72
        sign = "+" if dev >= 0 else ""
        gauge_label = result.get("model_name", "SU21")
        badge_title = f"{gauge_label} GAUGE: {d_mm:.2f} mm (Diff: {sign}{dev:.2f} mm) "
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
            img_path = os.path.join("ng_logs", f"{self.model_name}_{ts}.jpg")
            cv2.imwrite(img_path, frame)
            csv_path = os.path.join("ng_logs", "ng_log.csv")
            new = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "model", "status", "mm", "deviation", "score_A", "score_B"])
                w.writerow([ts, self.model_name, result.get("status"), result.get("mm"), result.get("dev"),
                            result.get("score_A"), result.get("score_B")])
            logger.info(f"[{self.model_name}Gauge] Logged NG distance measurement to {img_path}")
        except Exception as e:
            logger.error(f"[{self.model_name}Gauge] Error writing ng_log: {e}")


class RectangleDistanceGauge:
    """
    Multi-variant distance gauge dispatcher for Rectangle (SU21, LWB), Polygon (AI3), and Circle (CIRCLE_1, CIRCLE_2).
    Routes to the matching variant via YOLO/part hint or ORB reference inliers.
    """
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.gauges = {}

        su21_dir = os.path.join(base_dir, "SU21_gauge", "gauge_cfg", "SU21")
        if not os.path.exists(su21_dir):
            su21_dir = os.path.join(base_dir, "gauge_cfg", "SU21")
        if os.path.exists(os.path.join(su21_dir, "config.json")):
            self.gauges["SU21"] = SU21DistanceGauge(cfg_dir=su21_dir)

        lwb_dir = os.path.join(base_dir, "LWB_gauge", "gauge_cfg", "LWB")
        if os.path.exists(os.path.join(lwb_dir, "config.json")):
            self.gauges["LWB"] = SU21DistanceGauge(cfg_dir=lwb_dir)

        ai3_dir = os.path.join(base_dir, "Polygon_AI3_gauge", "gauge_cfg", "AI3")
        if os.path.exists(os.path.join(ai3_dir, "config.json")):
            self.gauges["AI3"] = SU21DistanceGauge(cfg_dir=ai3_dir)

        for cname in ("CIRCLE_1", "CIRCLE_2"):
            cdir = os.path.join(base_dir, "Circle_Guage", "gauge_cfg", cname)
            if os.path.exists(os.path.join(cdir, "config.json")):
                self.gauges[cname] = SU21DistanceGauge(cfg_dir=cdir)

        if not self.gauges:
            raise FileNotFoundError("No distance gauge configurations found (SU21, LWB, AI3, or CIRCLE).")

    @staticmethod
    def _normalize_hint(variant_hint):
        if not variant_hint:
            return None
        v = str(variant_hint).strip().upper()
        if v in ("LWD", "LWB"):
            return "LWB"
        if v in ("SU2I", "SU21"):
            return "SU21"
        if v in ("AI3", "POLYGEN", "POLYGON", "POLY"):
            return "AI3"
        if v in ("CIRCLE", "CIRCLE_1", "CIRCLE_2", "CRICLE"):
            return "CIRCLE"
        return None

    def _select_best_by_inliers(self, frame, candidates):
        """Picks best gauge from candidates, sharing ORB feature extraction across matching resolutions."""
        if len(candidates) == 1:
            return candidates[0], None
        if frame is None:
            return candidates[0], None

        orb_cache = {}
        best_gauge = candidates[0]
        best_inl = -1
        best_precomputed = None
        for g in candidates:
            dims = (g.w, g.h)
            if dims not in orb_cache:
                orb_cache[dims] = g._prepare_orb(frame)
            orig_w, orig_h, need_resize, scaled_frame, gray, kp, des = orb_cache[dims]
            H, inl = g._match_homography(kp, des)
            if inl > best_inl:
                best_inl = inl
                best_gauge = g
                best_precomputed = (orig_w, orig_h, need_resize, scaled_frame, gray, kp, des, H, inl)
        return best_gauge, best_precomputed

    def _select_gauge(self, frame, variant_hint=None):
        key = self._normalize_hint(variant_hint)
        if key == "CIRCLE":
            circle_candidates = [g for k, g in self.gauges.items() if k.startswith("CIRCLE")]
            if circle_candidates:
                return self._select_best_by_inliers(frame, circle_candidates)
        if key and key in self.gauges:
            return self.gauges[key], None

        # Auto-select between Rectangle variants (SU21 / LWB) by highest ORB inlier count
        rect_candidates = [g for k, g in self.gauges.items() if k in ("SU21", "LWB")]
        candidates = rect_candidates if rect_candidates else list(self.gauges.values())
        return self._select_best_by_inliers(frame, candidates)

    def measure(self, frame, plug_box=None, plug_class=None, variant_hint=None):
        gauge, precomputed = self._select_gauge(frame, variant_hint=variant_hint)
        return gauge.measure(frame, plug_box=plug_box, plug_class=plug_class, precomputed=precomputed)

    @staticmethod
    def draw_overlay(img, result, scale=1.0):
        SU21DistanceGauge.draw_overlay(img, result, scale=scale)

    def log_ng(self, result, frame):
        key = (result or {}).get("model_name", "SU21")
        if key == "CIRCLE":
            key = "CIRCLE_1"
        gauge = self.gauges.get(key) or next(iter(self.gauges.values()))
        gauge.log_ng(result, frame)
