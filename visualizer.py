"""
Visualizer and Dashboard HUD for Robodog 3D Vision & Obstacle Clearance.
Combines RGB overlay, Depth colormap, BEV 2D radar, and Elevation Profile.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

from config import RobodogGeometryConfig
from perception_engine import ObstacleDetectionResult


class VisionDashboard:
    """
    Renders an integrated 4-quadrant real-time HUD dashboard.
    """

    def __init__(self, config: Optional[RobodogGeometryConfig] = None):
        self.cfg = config or RobodogGeometryConfig()
        self.mouse_pos: Tuple[int, int] = (320, 240)
        self.hover_probe_info: str = ""

    def update_mouse_pos(self, x: int, y: int):
        self.mouse_pos = (x, y)

    def render_bev_map(
        self, result: ObstacleDetectionResult, width: int = 360, height: int = 300
    ) -> np.ndarray:
        """
        Renders a 2D Bird's Eye View (Top-Down) radar map.
        X_body: [-1.5m, +1.5m], Y_body: [0.0m, 3.0m]
        """
        bev = np.full((height, width, 3), (25, 25, 28), dtype=np.uint8)
        
        # Grid parameters
        origin_x = width // 2
        origin_y = height - 30
        max_y_m = 3.0
        scale_y = (origin_y - 20) / max_y_m  # pixels per meter
        scale_x = scale_y

        # Draw distance range arcs (0.5m, 1.0m, 1.5m, 2.0m, 2.5m, 3.0m)
        for dist_m in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            r_px = int(dist_m * scale_y)
            cv2.ellipse(
                bev, (origin_x, origin_y), (r_px, r_px), 0, 180, 360, (50, 55, 60), 1, cv2.LINE_AA
            )
            # Distance label
            cv2.putText(
                bev, f"{dist_m:.1f}m", (origin_x + 5, origin_y - r_px + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 130, 140), 1, cv2.LINE_AA
            )

        # Draw FOV lines (RealSense ~86 deg horizontal FOV -> +/- 43 deg)
        fov_angle_rad = np.radians(43.0)
        end_y = int(origin_y - max_y_m * scale_y)
        end_x_l = int(origin_x - np.tan(fov_angle_rad) * max_y_m * scale_x)
        end_x_r = int(origin_x + np.tan(fov_angle_rad) * max_y_m * scale_x)
        cv2.line(bev, (origin_x, origin_y), (end_x_l, end_y), (60, 65, 75), 1, cv2.LINE_AA)
        cv2.line(bev, (origin_x, origin_y), (end_x_r, end_y), (60, 65, 75), 1, cv2.LINE_AA)

        # Draw Robodog Step Corridor (width + safety margin)
        corridor_half_w = (self.cfg.robot_width_m / 2.0) + self.cfg.safety_lateral_margin_m
        corr_x_l = int(origin_x - corridor_half_w * scale_x)
        corr_x_r = int(origin_x + corridor_half_w * scale_x)
        lookahead_y = int(origin_y - self.cfg.corridor_forward_lookahead_m * scale_y)

        # Corridor fill overlay
        corr_overlay = bev.copy()
        cv2.rectangle(
            corr_overlay, (corr_x_l, lookahead_y), (corr_x_r, origin_y), (0, 70, 0), -1
        )
        cv2.addWeighted(corr_overlay, 0.25, bev, 0.75, 0, bev)
        cv2.rectangle(bev, (corr_x_l, lookahead_y), (corr_x_r, origin_y), (0, 180, 0), 1)

        # Plot obstacle points on BEV
        pts = result.bev_obstacle_points
        if len(pts) > 0:
            for pt in pts[::max(1, len(pts)//400)]:  # Subsample for smooth rendering
                px = int(origin_x + pt[0] * scale_x)
                py = int(origin_y - pt[1] * scale_y)
                if 0 <= px < width and 0 <= py < height:
                    # In corridor?
                    if abs(pt[0]) <= corridor_half_w and pt[1] <= self.cfg.corridor_forward_lookahead_m:
                        cv2.circle(bev, (px, py), 2, (0, 100, 255), -1)  # Orange in corridor
                    else:
                        cv2.circle(bev, (px, py), 1, (0, 220, 255), -1)  # Yellow outside

        # Draw Robodog icon at origin
        dog_w_px = int(self.cfg.robot_width_m * scale_x)
        dog_l_px = int(0.35 * scale_y)
        cv2.rectangle(
            bev, (origin_x - dog_w_px//2, origin_y - dog_l_px),
            (origin_x + dog_w_px//2, origin_y + 10), (0, 255, 120), -1
        )
        cv2.putText(
            bev, "ROBODOG", (origin_x - 30, origin_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 120), 1, cv2.LINE_AA
        )

        # Sector Distance Summary at top
        cv2.putText(
            bev, "HORIZONTAL SECTOR DISTANCES (BEV)", (10, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 210, 220), 1, cv2.LINE_AA
        )
        sec_str = " | ".join([f"{k[:4]}:{v:.1f}m" for k, v in result.sector_distances.items()])
        cv2.putText(
            bev, sec_str, (10, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 255), 1, cv2.LINE_AA
        )

        return bev

    def render_elevation_chart(
        self, result: ObstacleDetectionResult, width: int = 360, height: int = 180
    ) -> np.ndarray:
        """
        Renders the forward terrain height profile and required foot lift clearance curve.
        """
        chart = np.full((height, width, 3), (25, 25, 28), dtype=np.uint8)
        
        # Origin and axes
        margin_l = 45
        margin_r = 15
        margin_t = 25
        margin_b = 25
        
        plot_w = width - margin_l - margin_r
        plot_h = height - margin_t - margin_b
        ground_y = height - margin_b

        # Height scale: 0.0m to 0.30m (30 cm)
        max_h_m = 0.25
        scale_h = plot_h / max_h_m

        # Draw Grid & Height markers
        for h_cm in [5, 10, 15, 20]:
            y_px = int(ground_y - (h_cm / 100.0) * scale_h)
            cv2.line(chart, (margin_l, y_px), (width - margin_r, y_px), (45, 48, 52), 1)
            cv2.putText(
                chart, f"{h_cm}cm", (5, y_px + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (130, 140, 150), 1
            )

        # Ground line (Z = 0)
        cv2.line(chart, (margin_l, ground_y), (width - margin_r, ground_y), (100, 100, 100), 1)
        cv2.putText(chart, "0cm", (12, ground_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (130, 140, 150), 1)

        # Max Step Height threshold line (e.g. 12cm)
        max_step_y = int(ground_y - self.cfg.max_step_height_m * scale_h)
        cv2.line(
            chart, (margin_l, max_step_y), (width - margin_r, max_step_y), (0, 0, 220), 1, cv2.LINE_AA
        )
        cv2.putText(
            chart, f"MAX STEP ({self.cfg.max_step_height_m*100:.0f}cm)",
            (width - 125, max_step_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 100, 255), 1
        )

        # Camera Mounting Height level line
        cam_h_y = int(ground_y - result.calibrated_height_m * scale_h)
        if 0 <= cam_h_y < height:
            cv2.line(chart, (margin_l, cam_h_y), (width - margin_r, cam_h_y), (180, 180, 0), 1, cv2.LINE_AA)
            cv2.putText(
                chart, f"CAM ({result.calibrated_height_m*100:.1f}cm)",
                (margin_l + 5, cam_h_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (220, 220, 0), 1
            )

        # Title
        cv2.putText(
            chart, "FORWARD TERRAIN ELEVATION & FOOT CLEARANCE", (margin_l, 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA
        )

        # Plot Elevation Bins
        dists = result.elevation_profile_dist
        elevs = result.elevation_profile_height

        if len(dists) > 1 and len(elevs) == len(dists):
            max_d_m = self.cfg.corridor_forward_lookahead_m
            min_d_m = 0.2
            
            pts_poly = [(margin_l, ground_y)]
            line_pts = []
            lift_pts = []

            for d, h in zip(dists, elevs):
                px = int(margin_l + ((d - min_d_m) / (max_d_m - min_d_m)) * plot_w)
                clamped_h = min(max_h_m, max(0.0, h))
                py = int(ground_y - clamped_h * scale_h)
                
                # Foot lift height with safety margin
                lift_h = min(max_h_m, clamped_h + (self.cfg.foot_safety_margin_m if clamped_h > 0.02 else 0))
                py_lift = int(ground_y - lift_h * scale_h)

                px = max(margin_l, min(width - margin_r, px))
                pts_poly.append((px, py))
                line_pts.append((px, py))
                lift_pts.append((px, py_lift))

            pts_poly.append((width - margin_r, ground_y))

            # Fill terrain area
            poly_np = np.array(pts_poly, dtype=np.int32)
            cv2.fillPoly(chart, [poly_np], (60, 40, 30))
            
            # Draw terrain contour line
            for i in range(len(line_pts) - 1):
                cv2.line(chart, line_pts[i], line_pts[i+1], (0, 180, 255), 2, cv2.LINE_AA)

            # Draw Foot Lift clearance target line
            for i in range(len(lift_pts) - 1):
                cv2.line(chart, lift_pts[i], lift_pts[i+1], (0, 255, 100), 1, cv2.LINE_AA)

        # Distance labels at bottom
        for d_m in [0.5, 1.0, 1.5, 2.0]:
            px = int(margin_l + ((d_m - 0.2) / (2.0 - 0.2)) * plot_w)
            cv2.putText(
                chart, f"{d_m}m", (px - 10, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (120, 130, 140), 1
            )

        return chart

    def render_rgb_view(
        self, color_bgr: np.ndarray, result: ObstacleDetectionResult,
        depth_meters: np.ndarray, intrinsics: Dict[str, float]
    ) -> np.ndarray:
        """
        Renders RGB camera view augmented with bounding boxes, clearance tags, and corridor lines.
        """
        rgb_disp = color_bgr.copy()
        h, w = rgb_disp.shape[:2]

        # Draw Stepping Corridor boundary on ground in RGB perspective
        fx = intrinsics.get("fx", 385.0)
        fy = intrinsics.get("fy", 385.0)
        cx = intrinsics.get("cx", w / 2.0)
        cy = intrinsics.get("cy", h / 2.0)

        # Project corridor ground rails
        corridor_half_w = (self.cfg.robot_width_m / 2.0) + self.cfg.safety_lateral_margin_m
        cos_p = np.cos(np.radians(result.calibrated_pitch_deg))
        sin_p = np.sin(np.radians(result.calibrated_pitch_deg))
        cam_h = result.calibrated_height_m

        # Camera Horizon / Eye Level Line (Height = 13.5cm)
        v_horizon = int(cy - np.tan(np.radians(result.calibrated_pitch_deg)) * fy)
        if 0 <= v_horizon < h:
            cv2.line(rgb_disp, (0, v_horizon), (w, v_horizon), (180, 180, 0), 1, cv2.LINE_AA)
            cv2.putText(
                rgb_disp, f"CAMERA LEVEL ({cam_h*100:.1f}cm)", (10, max(15, v_horizon - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 0), 1, cv2.LINE_AA
            )

        for side in [-1, 1]:
            pts_rail = []
            for y_fwd in np.linspace(0.4, 2.2, 10):
                x_lat = side * corridor_half_w
                # Inverse transform from Ground Frame (X, Y, Z=0) to Camera Frame (Xc, Yc, Zc)
                # Zc = Y_fwd * cos_p + (cam_h) * sin_p
                # Yc = (cam_h) * cos_p - Y_fwd * sin_p  (downwards)
                zc = y_fwd * cos_p + cam_h * sin_p
                yc = cam_h * cos_p - y_fwd * sin_p
                xc = x_lat

                if zc > 0.1:
                    u = int(cx + (xc * fx) / zc)
                    v = int(cy + (yc * fy) / zc)
                    if 0 <= u < w and 0 <= v < h:
                        pts_rail.append((u, v))

            for i in range(len(pts_rail) - 1):
                cv2.line(rgb_disp, pts_rail[i], pts_rail[i+1], (0, 255, 0), 2, cv2.LINE_AA)

        # Draw Obstacle Bounding Boxes & Foot Lift Badges
        for box in result.obstacle_bboxes_2d:
            x, y, bw, bh = box["bbox"]
            dist_m = box["dist_m"]
            h_obs_cm = box["height_m"] * 100
            lift_cm = box["lift_m"] * 100
            can_step = box["can_step"]

            box_color = (0, 220, 0) if can_step else (0, 0, 255)  # Green if step over, Red if blocked

            # Bounding box
            cv2.rectangle(rgb_disp, (x, y), (x + bw, y + bh), box_color, 2)
            
            # Badge background
            badge_text = f"H:{h_obs_cm:.1f}cm | Lift:{lift_cm:.1f}cm | D:{dist_m:.2f}m"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(rgb_disp, (x, max(0, y - 22)), (x + tw + 8, y), (20, 20, 20), -1)
            cv2.putText(
                rgb_disp, badge_text, (x + 4, max(12, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_color, 1, cv2.LINE_AA
            )

        # Mouse probe marker on RGB
        mx, my = self.mouse_pos
        if 0 <= mx < w and 0 <= my < h:
            cv2.drawMarker(rgb_disp, (mx, my), (0, 255, 255), cv2.MARKER_CROSS, 16, 1)

        return rgb_disp

    def render_depth_view(
        self, depth_meters: np.ndarray, result: ObstacleDetectionResult
    ) -> np.ndarray:
        """
        Renders colorized depth map with live point inspection readout.
        """
        h, w = depth_meters.shape
        # Normalize depth 0.2m to 3.0m to 0..255
        clipped = np.clip(depth_meters, self.cfg.min_valid_depth_m, self.cfg.max_valid_depth_m)
        norm_depth = ((clipped - self.cfg.min_valid_depth_m) / (self.cfg.max_valid_depth_m - self.cfg.min_valid_depth_m) * 255).astype(np.uint8)
        
        # Colorize
        depth_color = cv2.applyColorMap(255 - norm_depth, cv2.COLORMAP_TURBO)
        # Invalidate missing depth (black)
        depth_color[depth_meters < self.cfg.min_valid_depth_m] = (15, 15, 15)

        # Mouse Probe computation at cursor
        mx, my = self.mouse_pos
        if 0 <= mx < w and 0 <= my < h:
            d_val = depth_meters[my, mx]
            if d_val > 0.1:
                # Deproject probe point
                fx = 385.0
                fy = 385.0
                cx = w / 2.0
                cy = h / 2.0
                xc = (mx - cx) * d_val / fx
                yc = (my - cy) * d_val / fy
                zc = d_val

                # Body frame
                pitch_rad = np.radians(result.calibrated_pitch_deg)
                cos_p = np.cos(pitch_rad)
                sin_p = np.sin(pitch_rad)
                x_body = xc
                y_body = zc * cos_p + yc * sin_p
                z_body = result.calibrated_height_m - (yc * cos_p - zc * sin_p)

                self.hover_probe_info = f"PROBE [{mx},{my}]: D={d_val:.2f}m | Lat={x_body*100:+.0f}cm, Fwd={y_body:.2f}m, Hgt={z_body*100:.1f}cm"
            else:
                self.hover_probe_info = f"PROBE [{mx},{my}]: No Depth"
            
            cv2.drawMarker(depth_color, (mx, my), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)

        # Overlay probe text
        cv2.rectangle(depth_color, (0, h - 26), (w, h), (15, 15, 18), -1)
        cv2.putText(
            depth_color, self.hover_probe_info, (10, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA
        )

        return depth_color

    def render_hud_telemetry_banner(
        self, result: ObstacleDetectionResult, fps: float, width: int = 1000, height: int = 80
    ) -> np.ndarray:
        """
        Renders the top digital telemetry banner with Robodog Action, Required Foot Lift, and Guidance.
        """
        banner = np.full((height, width, 3), (18, 20, 24), dtype=np.uint8)

        # Status badge background & color
        if result.decision == "CLEAR":
            badge_bg = (0, 140, 40)
            badge_text = "PATH CLEAR"
        elif result.decision == "STEP_OVER":
            badge_bg = (0, 130, 200)  # Orange
            badge_text = f"STEP OVER: LIFT FOOT +{result.required_foot_lift_m * 100:.1f} CM"
        else:
            badge_bg = (0, 0, 180)  # Red
            badge_text = f"OBSTACLE BLOCKED (STOP / DETOUR)"

        cv2.rectangle(banner, (10, 10), (450, 48), badge_bg, -1)
        cv2.putText(
            banner, badge_text, (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Telemetry stats
        info_l1 = f"Nearest Obs: {result.nearest_obs_distance_m:.2f}m ahead | Height: {result.nearest_obs_height_m*100:.1f}cm | Lift: {result.required_foot_lift_m*100:.1f}cm"
        cv2.putText(
            banner, info_l1, (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 210, 220), 1, cv2.LINE_AA
        )

        # Right side: Calibration & FPS
        pitch_str = "0.0° (Straight)" if abs(result.calibrated_pitch_deg) < 0.5 else f"{result.calibrated_pitch_deg:.1f}°"
        calib_str = f"H: {result.calibrated_height_m*100:.1f}cm | Pitch: {pitch_str} | FPS: {fps:.1f}"
        cv2.putText(
            banner, calib_str, (width - 450, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA
        )
        
        hints = "Keys: [C] Auto-Calib Ground | [P] Pitch+ | [O] Pitch- | [H] Hgt+ | [J] Hgt- | [S] Save | [Q] Quit"
        cv2.putText(
            banner, hints, (width - 480, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 150, 160), 1, cv2.LINE_AA
        )

        return banner

    def build_dashboard(
        self, color_bgr: np.ndarray, depth_meters: np.ndarray,
        result: ObstacleDetectionResult, intrinsics: Dict[str, float], fps: float
    ) -> np.ndarray:
        """
        Combines all panels into an integrated 1000x560 HUD dashboard.
        """
        rgb_panel = self.render_rgb_view(color_bgr, result, depth_meters, intrinsics)
        depth_panel = self.render_depth_view(depth_meters, result)

        bev_panel = self.render_bev_map(result, width=360, height=270)
        elev_panel = self.render_elevation_chart(result, width=360, height=210)
        right_sidebar = np.vstack([bev_panel, elev_panel])

        # Resize video views if needed
        rgb_panel = cv2.resize(rgb_panel, (320, 240))
        depth_panel = cv2.resize(depth_panel, (320, 240))
        video_col = np.vstack([rgb_panel, depth_panel])

        main_content = np.hstack([video_col, right_sidebar])
        
        # Ensure uniform width
        content_w = main_content.shape[1]
        banner = self.render_hud_telemetry_banner(result, fps, width=content_w, height=80)

        dashboard = np.vstack([banner, main_content])
        return dashboard
