"""
3D Spatial Perception Engine, Ground Plane Estimation, and Robodog Foot-Lift Calculation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

from config import RobodogGeometryConfig


@dataclass
class ObstacleDetectionResult:
    # Nearest obstacle in dog's path corridor
    has_obstacle_in_path: bool = False
    nearest_obs_distance_m: float = 0.0
    nearest_obs_lateral_m: float = 0.0
    nearest_obs_height_m: float = 0.0
    required_foot_lift_m: float = 0.0
    
    # State decision: "CLEAR", "STEP_OVER", "OBSTACLE_STOP", "STEER_LEFT", "STEER_RIGHT"
    decision: str = "CLEAR"
    status_text: str = "PATH CLEAR"
    recommendation: str = "Normal Gait"

    # Horizontal Sector Distances (meters to closest obstacle)
    sector_distances: Dict[str, float] = field(default_factory=dict)
    
    # Forward Elevation Profile (distance_m -> max_height_m)
    elevation_profile_dist: np.ndarray = field(default_factory=lambda: np.array([]))
    elevation_profile_height: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # 2D Bird's Eye View (BEV) Obstacle Points [X_body, Y_body]
    bev_obstacle_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    
    # Bounding boxes in image frame (u1, v1, u2, v2, dist_m, height_m, lift_m, step_feasible)
    obstacle_bboxes_2d: List[Dict[str, Any]] = field(default_factory=list)
    
    # Calibrated Ground Plane parameters (Pitch angle deg, Mount height m)
    calibrated_pitch_deg: float = 0.0
    calibrated_height_m: float = 0.0


class PerceptionEngine:
    """
    Transforms RealSense depth frames into 3D body-centric point clouds,
    fits ground planes, analyzes spatial distances horizontally and vertically,
    and calculates obstacle clearance and required foot lift heights.
    """

    def __init__(self, config: Optional[RobodogGeometryConfig] = None):
        self.cfg = config or RobodogGeometryConfig()
        
        # Current active pitch and height (can be updated dynamically or via RANSAC)
        self.camera_pitch_deg: float = self.cfg.camera_pitch_deg
        self.camera_height_m: float = self.cfg.camera_height_m
        
        # Precomputed pixel grid coordinates cache for fast deprojection
        self._cached_shape: Optional[Tuple[int, int]] = None
        self._u_grid: Optional[np.ndarray] = None
        self._v_grid: Optional[np.ndarray] = None
        
        # Sectors definitions (in degrees relative to forward heading)
        self.sectors = {
            "Far-Left": (-35.0, -20.0),
            "Left": (-20.0, -8.0),
            "Center": (-8.0, 8.0),
            "Right": (8.0, 20.0),
            "Far-Right": (20.0, 35.0)
        }

    def _init_pixel_grid(self, height: int, width: int):
        """Precomputes coordinate grids for ultra-fast vectorized deprojection."""
        if self._cached_shape != (height, width):
            u = np.arange(width, dtype=np.float32)
            v = np.arange(height, dtype=np.float32)
            self._u_grid, self._v_grid = np.meshgrid(u, v)
            self._cached_shape = (height, width)

    def deproject_to_camera_frame(
        self, depth_meters: np.ndarray, intrinsics: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized deprojection of 2D depth map to 3D Camera Coordinates (Xc, Yc, Zc).
        Returns:
            (Xc, Yc, Zc, valid_mask)
        """
        h, w = depth_meters.shape
        self._init_pixel_grid(h, w)
        
        fx = intrinsics.get("fx", 385.0)
        fy = intrinsics.get("fy", 385.0)
        cx = intrinsics.get("cx", w / 2.0)
        cy = intrinsics.get("cy", h / 2.0)

        # Depth range validity mask
        valid_mask = (depth_meters >= self.cfg.min_valid_depth_m) & (depth_meters <= self.cfg.max_valid_depth_m)

        Zc = np.where(valid_mask, depth_meters, np.nan)
        Xc = (self._u_grid - cx) * Zc / fx
        Yc = (self._v_grid - cy) * Zc / fy

        return Xc, Yc, Zc, valid_mask

    def transform_to_body_frame(
        self, Xc: np.ndarray, Yc: np.ndarray, Zc: np.ndarray, pitch_deg: float, height_m: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transforms 3D camera coordinates to Robodog Ground Body coordinates.
        X_body: Lateral offset from dog centerline (meters, Left < 0, Right > 0)
        Y_body: Forward distance along the ground plane (meters, Ahead > 0)
        Z_body: Vertical height above the ground plane (meters, Ground = 0.0)
        """
        pitch_rad = np.radians(pitch_deg)
        cos_p = np.cos(pitch_rad)
        sin_p = np.sin(pitch_rad)

        # Camera frame: +X right, +Y down, +Z forward
        # Body frame: +X right, +Y forward along ground, +Z up above ground
        X_body = Xc
        Y_body = Zc * cos_p + Yc * sin_p
        Z_body = height_m - (Yc * cos_p - Zc * sin_p)

        return X_body, Y_body, Z_body

    def auto_calibrate_ground_plane(
        self, Xc: np.ndarray, Yc: np.ndarray, Zc: np.ndarray, valid_mask: np.ndarray
    ) -> Tuple[float, float, bool]:
        """
        RANSAC Ground Plane Detection on lower region of the field of view.
        Returns:
            (estimated_pitch_deg, estimated_height_m, success)
        """
        h, w = valid_mask.shape
        # Focus on lower 40% of the image where floor is predominantly present
        lower_mask = np.zeros_like(valid_mask)
        lower_mask[int(h * 0.55):, int(w * 0.15):int(w * 0.85)] = True
        sample_mask = valid_mask & lower_mask

        xc_pts = Xc[sample_mask]
        yc_pts = Yc[sample_mask]
        zc_pts = Zc[sample_mask]

        if len(zc_pts) < 500:
            return self.camera_pitch_deg, self.camera_height_m, False

        # Subsample for fast RANSAC
        step = max(1, len(zc_pts) // 1000)
        pts = np.column_stack((xc_pts[::step], yc_pts[::step], zc_pts[::step]))

        best_inliers = 0
        best_plane = None
        num_samples = 40
        threshold = 0.02  # 2 cm tolerance

        for _ in range(num_samples):
            idx = np.random.choice(len(pts), 3, replace=False)
            p1, p2, p3 = pts[idx]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm

            # Ensure normal points generally upwards/towards camera floor (+Y component in camera frame)
            if normal[1] < 0:
                normal = -normal

            d = -np.dot(normal, p1)
            distances = np.abs(np.dot(pts, normal) + d)
            inliers = np.sum(distances < threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = (normal, d)

        if best_plane is not None and best_inliers > 0.4 * len(pts):
            normal, d = best_plane
            # Pitch is angle between optical axis (0,0,1) and ground plane normal projection
            # In camera frame, ground normal has approx (0, cos(pitch), sin(pitch)) or similar
            pitch_rad = np.arctan2(normal[2], normal[1])
            pitch_deg = np.degrees(pitch_rad)
            # Distance from camera origin (0,0,0) to floor plane is |d| / ||normal||
            height_m = abs(d)

            # Sanity check bounds
            if 0.0 <= pitch_deg <= 45.0 and 0.10 <= height_m <= 0.80:
                self.camera_pitch_deg = 0.85 * self.camera_pitch_deg + 0.15 * pitch_deg
                self.camera_height_m = 0.85 * self.camera_height_m + 0.15 * height_m
                return self.camera_pitch_deg, self.camera_height_m, True

        return self.camera_pitch_deg, self.camera_height_m, False

    def process_frame(
        self, depth_meters: np.ndarray, intrinsics: Dict[str, float], auto_ground_plane: bool = False
    ) -> ObstacleDetectionResult:
        """
        Full 3D perception pipeline:
        1. Deprojection into 3D
        2. Ground transformation
        3. Horizontal sector distance scanning
        4. Forward path corridor foot-lift clearance calculation
        5. 2D Image obstacle bounding box projection
        """
        result = ObstacleDetectionResult()
        
        # 1. Deprojection
        Xc, Yc, Zc, valid_mask = self.deproject_to_camera_frame(depth_meters, intrinsics)
        
        if not np.any(valid_mask):
            return result

        # Optional Auto ground calibration
        if auto_ground_plane:
            self.auto_calibrate_ground_plane(Xc, Yc, Zc, valid_mask)
            
        result.calibrated_pitch_deg = self.camera_pitch_deg
        result.calibrated_height_m = self.camera_height_m

        # 2. Body-frame transform (X: lateral, Y: forward, Z: height above ground)
        X_body, Y_body, Z_body = self.transform_to_body_frame(
            Xc, Yc, Zc, self.camera_pitch_deg, self.camera_height_m
        )

        # 3. Obstacle Points Filtering
        # Points with height > min_obstacle_height_m and < 1.2m above ground
        obstacle_mask = (
            valid_mask &
            (Z_body > self.cfg.min_obstacle_height_m) &
            (Z_body < 1.2) &
            (Y_body > 0.15) &
            (Y_body <= self.cfg.max_valid_depth_m)
        )

        # Extract obstacle coordinates
        obs_x = X_body[obstacle_mask]
        obs_y = Y_body[obstacle_mask]
        obs_z = Z_body[obstacle_mask]

        if len(obs_y) > 0:
            result.bev_obstacle_points = np.column_stack((obs_x, obs_y))
        else:
            result.bev_obstacle_points = np.empty((0, 2))

        # 4. Horizontal Sector Distance Calculation
        angles_deg = np.degrees(np.arctan2(X_body[valid_mask], Y_body[valid_mask]))
        distances_m = np.sqrt(X_body[valid_mask]**2 + Y_body[valid_mask]**2)
        is_obs = obstacle_mask[valid_mask]

        sector_dists = {}
        for sec_name, (a_min, a_max) in self.sectors.items():
            in_sector = (angles_deg >= a_min) & (angles_deg < a_max) & is_obs
            if np.any(in_sector):
                min_d = float(np.percentile(distances_m[in_sector], 5))  # 5th percentile to reject noise
                sector_dists[sec_name] = round(min_d, 2)
            else:
                sector_dists[sec_name] = round(self.cfg.max_valid_depth_m, 2)
        result.sector_distances = sector_dists

        # 5. Robodog Forward Stepping Corridor Analysis
        corridor_half_width = (self.cfg.robot_width_m / 2.0) + self.cfg.safety_lateral_margin_m
        in_corridor_mask = (
            obstacle_mask &
            (np.abs(X_body) <= corridor_half_width) &
            (Y_body <= self.cfg.corridor_forward_lookahead_m)
        )

        # 6. Elevation Profile along Forward Corridor
        bin_edges = np.linspace(0.2, self.cfg.corridor_forward_lookahead_m, self.cfg.num_elevation_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        elevation_heights = np.zeros(self.cfg.num_elevation_bins, dtype=np.float32)

        corr_y = Y_body[in_corridor_mask]
        corr_z = Z_body[in_corridor_mask]
        corr_x = X_body[in_corridor_mask]

        if len(corr_y) > 0:
            for i in range(len(bin_edges) - 1):
                in_bin = (corr_y >= bin_edges[i]) & (corr_y < bin_edges[i + 1])
                if np.any(in_bin):
                    # 95th percentile height in bin
                    elevation_heights[i] = float(np.percentile(corr_z[in_bin], 95))

        result.elevation_profile_dist = bin_centers
        result.elevation_profile_height = elevation_heights

        # 7. Foot Lift & Clearance Decision
        if len(corr_y) > 30:  # Valid obstacle cluster in corridor
            result.has_obstacle_in_path = True
            
            # Find closest cluster along Y
            closest_idx = np.argmin(corr_y)
            closest_y = float(corr_y[closest_idx])
            
            # Sample points within 25cm of closest obstacle contact
            nearby_pts = (corr_y >= closest_y) & (corr_y <= closest_y + 0.25)
            if np.any(nearby_pts):
                obs_height = float(np.percentile(corr_z[nearby_pts], 95))
                obs_lat = float(np.mean(corr_x[nearby_pts]))
            else:
                obs_height = float(corr_z[closest_idx])
                obs_lat = float(corr_x[closest_idx])

            result.nearest_obs_distance_m = round(closest_y, 2)
            result.nearest_obs_lateral_m = round(obs_lat, 2)
            result.nearest_obs_height_m = round(obs_height, 3)
            
            # Calculate required foot lift with safety margin
            required_lift = obs_height + self.cfg.foot_safety_margin_m
            result.required_foot_lift_m = round(required_lift, 3)

            # Determine feasibility
            if obs_height <= self.cfg.max_step_height_m:
                result.decision = "STEP_OVER"
                result.status_text = f"STEP OVER: Lift Foot {result.required_foot_lift_m * 100:.1f} cm"
                result.recommendation = f"Lift paw +{result.required_foot_lift_m * 100:.1f}cm @ {result.nearest_obs_distance_m:.2f}m ahead"
            else:
                result.decision = "OBSTACLE_STOP"
                # Determine detour direction from sector distances
                left_space = sector_dists.get("Left", 0) + sector_dists.get("Far-Left", 0)
                right_space = sector_dists.get("Right", 0) + sector_dists.get("Far-Right", 0)
                
                steer_dir = "STEER LEFT" if left_space >= right_space else "STEER RIGHT"
                result.status_text = f"WALL/OBSTACLE TOO HIGH ({obs_height * 100:.1f}cm) -> {steer_dir}"
                result.recommendation = f"H_obs ({obs_height*100:.1f}cm) > Max Step ({self.cfg.max_step_height_m*100:.0f}cm). {steer_dir}"
        else:
            result.has_obstacle_in_path = False
            result.decision = "CLEAR"
            result.status_text = "PATH CLEAR"
            result.recommendation = "Normal Walking Gait"
            result.nearest_obs_distance_m = 0.0
            result.nearest_obs_height_m = 0.0
            result.required_foot_lift_m = 0.0

        # 8. Compute 2D Bounding Boxes for Visualizer
        self._detect_2d_bounding_boxes(depth_meters, obstacle_mask, X_body, Y_body, Z_body, result)

        return result

    def _detect_2d_bounding_boxes(
        self, depth_meters: np.ndarray, obstacle_mask: np.ndarray,
        X_body: np.ndarray, Y_body: np.ndarray, Z_body: np.ndarray,
        result: ObstacleDetectionResult
    ):
        """Finds connected obstacle components and calculates bounding boxes with clearance info."""
        # Downsample mask for fast OpenCV contour detection
        obs_u8 = (obstacle_mask.astype(np.uint8)) * 255
        
        # Morphological opening/closing to group obstacle pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        cleaned = cv2.morphologyEx(obs_u8, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:  # Filter small noise
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            roi_mask = cleaned[y:y+h, x:x+w] > 0
            
            roi_y_body = Y_body[y:y+h, x:x+w][roi_mask]
            roi_z_body = Z_body[y:y+h, x:x+w][roi_mask]
            
            if len(roi_y_body) == 0:
                continue
                
            dist_m = float(np.percentile(roi_y_body, 10))
            height_m = float(np.percentile(roi_z_body, 95))
            lift_m = height_m + self.cfg.foot_safety_margin_m
            can_step = height_m <= self.cfg.max_step_height_m

            bboxes.append({
                "bbox": (x, y, w, h),
                "dist_m": round(dist_m, 2),
                "height_m": round(height_m, 3),
                "lift_m": round(lift_m, 3),
                "can_step": can_step
            })

        # Sort bboxes by distance (closest first)
        bboxes.sort(key=lambda b: b["dist_m"])
        result.obstacle_bboxes_2d = bboxes[:6]  # Top 6 closest
