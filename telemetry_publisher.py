"""
Telemetry publisher for Robodog motion controller communication via UDP JSON packets.
"""

import json
import socket
import time
from typing import Optional

from config import TelemetryConfig
from perception_engine import ObstacleDetectionResult


class TelemetryPublisher:
    """
    Broadcasts obstacle clearance, distance metrics, and foot lift instructions
    to the Robodog gait/locomotion controller over UDP.
    """

    def __init__(self, config: Optional[TelemetryConfig] = None):
        self.cfg = config or TelemetryConfig()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_publish_time = 0.0

    def publish(self, result: ObstacleDetectionResult, fps: float = 0.0):
        if not self.cfg.udp_enabled:
            return

        now = time.time()
        if (now - self.last_publish_time) < (1.0 / self.cfg.publish_rate_hz):
            return

        payload = {
            "timestamp": now,
            "fps": round(fps, 1),
            "decision": result.decision,
            "status": result.status_text,
            "recommendation": result.recommendation,
            "obstacle_in_path": result.has_obstacle_in_path,
            "nearest_obstacle": {
                "distance_m": result.nearest_obs_distance_m,
                "lateral_offset_m": result.nearest_obs_lateral_m,
                "height_m": result.nearest_obs_height_m,
                "required_foot_lift_m": result.required_foot_lift_m,
                "required_foot_lift_cm": round(result.required_foot_lift_m * 100, 1)
            },
            "sectors_distance_m": result.sector_distances,
            "calibration": {
                "pitch_deg": round(result.calibrated_pitch_deg, 2),
                "mount_height_m": round(result.calibrated_height_m, 3)
            }
        }

        try:
            msg = json.dumps(payload).encode("utf-8")
            self.sock.sendto(msg, (self.cfg.udp_host, self.cfg.udp_port))
            self.last_publish_time = now
        except Exception as e:
            pass  # Non-blocking transmission

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass
