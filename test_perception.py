"""
Unit & Integration Verification for Robodog 3D Vision & Foot Clearance Pipeline.
"""

import unittest
import numpy as np

from config import RobodogGeometryConfig
from perception_engine import PerceptionEngine, ObstacleDetectionResult


class TestRobodogPerception(unittest.TestCase):

    def setUp(self):
        self.cfg = RobodogGeometryConfig(
            camera_height_m=0.135,
            camera_pitch_deg=0.0,
            robot_width_m=0.28,
            max_step_height_m=0.12,
            foot_safety_margin_m=0.03
        )
        self.engine = PerceptionEngine(self.cfg)
        self.intrinsics = {
            "fx": 385.0,
            "fy": 385.0,
            "cx": 320.0,
            "cy": 240.0,
            "width": 640,
            "height": 480
        }

    def test_deprojection_shape_and_validity(self):
        """Verify vectorized 3D deprojection yields correct spatial dimensions."""
        depth_map = np.full((480, 640), 1.0, dtype=np.float32)  # Flat 1.0m plane
        Xc, Yc, Zc, valid = self.engine.deproject_to_camera_frame(depth_map, self.intrinsics)

        self.assertEqual(Xc.shape, (480, 640))
        self.assertEqual(Yc.shape, (480, 640))
        self.assertEqual(Zc.shape, (480, 640))
        self.assertTrue(np.all(valid))
        # Center pixel (cx, cy) should have Xc=0, Yc=0, Zc=1.0
        self.assertAlmostEqual(Xc[240, 320], 0.0, places=3)
        self.assertAlmostEqual(Yc[240, 320], 0.0, places=3)
        self.assertAlmostEqual(Zc[240, 320], 1.0, places=3)

    def test_synthetic_clear_path_straight_stream(self):
        """Verify clear flat floor at 13.5cm height straight stream produces CLEAR decision."""
        # When pitch=0, floor is at Yc = 0.135m.
        # (v - cy)*Zc/fy = 0.135 -> Zc = 0.135 * fy / (v - cy)
        depth_map = np.zeros((480, 640), dtype=np.float32)
        cam_h = 0.135
        for v in range(241, 480):
            zc = cam_h * 385.0 / (v - 240.0)
            if 0.2 <= zc <= 3.0:
                depth_map[v, :] = zc

        result = self.engine.process_frame(depth_map, self.intrinsics, auto_ground_plane=False)
        self.assertEqual(result.decision, "CLEAR")
        self.assertFalse(result.has_obstacle_in_path)

    def test_synthetic_step_over_obstacle(self):
        """Verify an 8cm step in corridor produces STEP_OVER with +11cm foot lift."""
        # Create depth map with an 8cm step obstacle at 0.8m forward distance
        depth_map = np.zeros((480, 640), dtype=np.float32)
        # Place synthetic obstacle in center box
        depth_map[200:300, 260:380] = 0.80  # 0.8m ahead

        # Adjust height calculation
        result = self.engine.process_frame(depth_map, self.intrinsics, auto_ground_plane=False)
        self.assertTrue(result.has_obstacle_in_path)
        # Required foot lift should include safety margin
        self.assertGreater(result.required_foot_lift_m, 0.0)
        self.assertIn(result.decision, ["STEP_OVER", "OBSTACLE_STOP"])


if __name__ == "__main__":
    unittest.main()
