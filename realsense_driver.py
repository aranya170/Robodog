"""
Intel RealSense D435 Camera Driver with RGB-D alignment & hardware post-processing filters.
"""

import time
import numpy as np
from typing import Optional, Tuple, Dict, Any
import pyrealsense2 as rs

from config import RealSenseConfig


class RealSenseCamera:
    """
    Interfaces with Intel RealSense D400-series depth cameras.
    Provides synchronized, aligned Color and Depth frames along with camera intrinsics.
    """

    def __init__(self, config: Optional[RealSenseConfig] = None):
        self.cfg = config or RealSenseConfig()
        self.pipeline: Optional[rs.pipeline] = None
        self.pipeline_profile: Optional[rs.pipeline_profile] = None
        self.align: Optional[rs.align] = None
        
        # RealSense Filters
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()
        self._configure_filters()

        self.depth_scale: float = 0.001  # Default 1mm per unit
        self.intrinsics: Optional[rs.intrinsics] = None
        self.is_running: bool = False

    def _configure_filters(self):
        """Configure post-processing filter parameters for clean depth edges."""
        if self.cfg.enable_spatial_filter:
            self.spatial_filter.set_option(rs.option.filter_magnitude, self.cfg.spatial_magnitude)
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, self.cfg.spatial_smooth_alpha)
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, self.cfg.spatial_smooth_delta)
        
        if self.cfg.enable_temporal_filter:
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, self.cfg.temporal_smooth_alpha)
            self.temporal_filter.set_option(rs.option.filter_smooth_delta, self.cfg.temporal_smooth_delta)
            
        if self.cfg.enable_hole_filling:
            self.hole_filling_filter.set_option(rs.option.holes_fill, self.cfg.hole_filling_mode)

    def start(self) -> bool:
        """Starts the RealSense pipeline and aligns streams to color."""
        try:
            self.pipeline = rs.pipeline()
            rs_config = rs.config()

            # Enable color & depth streams
            rs_config.enable_stream(
                rs.stream.depth,
                self.cfg.width,
                self.cfg.height,
                rs.format.z16,
                self.cfg.fps
            )
            rs_config.enable_stream(
                rs.stream.color,
                self.cfg.width,
                self.cfg.height,
                rs.format.bgr8,
                self.cfg.fps
            )

            # Start streaming
            self.pipeline_profile = self.pipeline.start(rs_config)
            
            # Get depth sensor scale
            depth_sensor = self.pipeline_profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

            # Enable IR emitter for accurate indoor depth
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1)
            if depth_sensor.supports(rs.option.laser_power):
                laser_range = depth_sensor.get_option_range(rs.option.laser_power)
                depth_sensor.set_option(rs.option.laser_power, laser_range.max)

            # Setup alignment object (align depth to color frame coordinate system)
            self.align = rs.align(rs.stream.color)

            # Warmup frames
            for _ in range(15):
                self.pipeline.wait_for_frames(5000)

            # Retrieve active intrinsics from aligned color stream
            color_profile = self.pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.intrinsics = color_profile.get_intrinsics()
            
            self.is_running = True
            print(f"[RealSense] Camera started successfully.")
            print(f"[RealSense] Resolution: {self.cfg.width}x{self.cfg.height} @ {self.cfg.fps} FPS")
            print(f"[RealSense] Depth Scale: {self.depth_scale} m/unit")
            print(f"[RealSense] Intrinsics: fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}, "
                  f"ppx={self.intrinsics.ppx:.2f}, ppy={self.intrinsics.ppy:.2f}")
            return True

        except Exception as e:
            print(f"[RealSense Error] Failed to initialize camera: {e}")
            self.is_running = False
            return False

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Fetches next synchronized frame pair.
        Returns:
            (success, color_bgr, depth_meters, raw_depth_filtered)
        """
        if not self.is_running or self.pipeline is None:
            return False, None, None, None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return False, None, None, None

            # Apply post-processing filters
            if self.cfg.enable_spatial_filter:
                depth_frame = self.spatial_filter.process(depth_frame)
            if self.cfg.enable_temporal_filter:
                depth_frame = self.temporal_filter.process(depth_frame)
            if self.cfg.enable_hole_filling:
                depth_frame = self.hole_filling_filter.process(depth_frame)

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            raw_depth = np.asanyarray(depth_frame.get_data())
            
            # Depth in meters (float32)
            depth_meters = raw_depth.astype(np.float32) * self.depth_scale

            return True, color_image, depth_meters, raw_depth

        except Exception as e:
            print(f"[RealSense Error] Error capturing frame: {e}")
            return False, None, None, None

    def get_intrinsics_dict(self) -> Dict[str, Any]:
        """Returns camera intrinsics as a dictionary."""
        if self.intrinsics is None:
            return {}
        return {
            "fx": self.intrinsics.fx,
            "fy": self.intrinsics.fy,
            "cx": self.intrinsics.ppx,
            "cy": self.intrinsics.ppy,
            "width": self.intrinsics.width,
            "height": self.intrinsics.height,
            "fov_h": np.degrees(2 * np.arctan(self.intrinsics.width / (2 * self.intrinsics.fx))),
            "fov_v": np.degrees(2 * np.arctan(self.intrinsics.height / (2 * self.intrinsics.fy)))
        }

    def stop(self):
        """Stops the camera pipeline."""
        if self.pipeline and self.is_running:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.is_running = False
            print("[RealSense] Pipeline stopped.")
