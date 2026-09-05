"""
Configuration parameters for Robodog 3D Vision & Obstacle Clearance System.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RealSenseConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    enable_color: bool = True
    enable_depth: bool = True
    
    # RealSense Post-Processing Filters
    enable_spatial_filter: bool = True
    enable_temporal_filter: bool = True
    enable_hole_filling: bool = True
    spatial_magnitude: int = 2
    spatial_smooth_alpha: float = 0.5
    spatial_smooth_delta: int = 20
    temporal_smooth_alpha: float = 0.4
    temporal_smooth_delta: int = 20
    hole_filling_mode: int = 1  # 1: farest from around, 2: nearest from around


@dataclass
class RobodogGeometryConfig:
    # Camera mounting position on the dog body
    camera_height_m: float = 0.135  # Height of camera above ground plane (13.5 cm)
    camera_pitch_deg: float = 0.0   # Camera pitch angle (0.0 = straight stream parallel to ground)
    camera_roll_deg: float = 0.0    # Roll angle (degrees)
    
    # Dog body dimensions
    robot_width_m: float = 0.28    # Width of the robodog body (meters)
    safety_lateral_margin_m: float = 0.08  # Extra safety margin on each side (meters)
    
    # Obstacle clearance & stepping capabilities
    max_step_height_m: float = 0.12  # Maximum height the dog can lift its foot to step over (e.g. 12 cm)
    foot_safety_margin_m: float = 0.03  # Extra foot lift margin above obstacle top (e.g. 3 cm)
    min_obstacle_height_m: float = 0.025  # Heights below this (2.5 cm) are considered flat ground / noise
    
    # Scanning ranges (meters)
    min_valid_depth_m: float = 0.18   # RealSense D435 min range ~0.18m
    max_valid_depth_m: float = 3.50   # Max range for obstacle detection (meters)
    
    # Corridors & Sectors
    corridor_forward_lookahead_m: float = 2.0  # Lookahead distance for forward stepping corridor
    num_elevation_bins: int = 20      # Number of longitudinal bins for terrain elevation chart


@dataclass
class TelemetryConfig:
    udp_enabled: bool = False
    udp_host: str = "127.0.0.1"
    udp_port: int = 9876
    publish_rate_hz: int = 30


@dataclass
class VisionConfig:
    realsense: RealSenseConfig = field(default_factory=RealSenseConfig)
    robot: RobodogGeometryConfig = field(default_factory=RobodogGeometryConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
