"""
Main Entrypoint for Robodog 3D Vision, Distance Measurement, and Foot-Lift Clearance System.
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2

from config import VisionConfig
from realsense_driver import RealSenseCamera
from perception_engine import PerceptionEngine
from visualizer import VisionDashboard
from telemetry_publisher import TelemetryPublisher


def on_mouse_event(event, x, y, flags, param):
    """Tracks mouse movement for real-time spatial probe."""
    dashboard = param
    if event == cv2.EVENT_MOUSEMOVE:
        # If in top video panel area
        # Video column is in x in [0, 320], y in [80, 560]
        # RGB is y in [80, 320], Depth is y in [320, 560]
        if 0 <= x < 320:
            if 80 <= y < 320:
                # RGB mapped to 640x480
                orig_x = int((x / 320.0) * 640)
                orig_y = int(((y - 80) / 240.0) * 480)
                dashboard.update_mouse_pos(orig_x, orig_y)
            elif 320 <= y < 560:
                # Depth mapped to 640x480
                orig_x = int((x / 320.0) * 640)
                orig_y = int(((y - 320) / 240.0) * 480)
                dashboard.update_mouse_pos(orig_x, orig_y)


def run_system(auto_calibrate_init: bool = False, enable_udp: bool = False):
    """Runs the main perception and visualization loop."""
    print("=" * 70)
    print("   ROBODOG 3D SPATIAL VISION & FOOT CLEARANCE SYSTEM")
    print("   Powered by Intel RealSense D435 RGB-D Camera")
    print("   Calibrated: Camera Height = 13.5 cm | Pitch = 0.0° (Straight)")
    print("=" * 70)

    # Initialize configuration
    cfg = VisionConfig()
    cfg.telemetry.udp_enabled = enable_udp

    # Initialize Camera
    camera = RealSenseCamera(cfg.realsense)
    if not camera.start():
        print("[Fatal] Could not start Intel RealSense camera. Check USB connection.")
        sys.exit(1)

    # Initialize Perception Engine, Visualizer & Telemetry
    engine = PerceptionEngine(cfg.robot)
    dashboard = VisionDashboard(cfg.robot)
    publisher = TelemetryPublisher(cfg.telemetry)

    # Setup OpenCV Window
    window_name = "Robodog 3D Vision & Foot Clearance HUD"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 640)
    cv2.setMouseCallback(window_name, on_mouse_event, dashboard)

    intrinsics = camera.get_intrinsics_dict()
    print(f"[System] Horizontal FOV: {intrinsics.get('fov_h', 0):.1f}° | Vertical FOV: {intrinsics.get('fov_v', 0):.1f}°")
    print(f"[System] Robodog Max Step Height: {cfg.robot.max_step_height_m * 100:.1f} cm")
    print(f"[System] Safety Margin: {cfg.robot.foot_safety_margin_m * 100:.1f} cm")
    print("[System] Press 'Q' or ESC in window to exit.")

    fps = 0.0
    frame_count = 0
    t_start = time.time()
    auto_ground_calib = auto_calibrate_init

    try:
        while True:
            t_frame_start = time.time()

            # 1. Capture RGB and Depth
            success, color_bgr, depth_meters, raw_depth = camera.get_frame()
            if not success or color_bgr is None or depth_meters is None:
                time.sleep(0.01)
                continue

            # 2. Process Spatial 3D Perception & Obstacle Foot Lift
            detection_result = engine.process_frame(
                depth_meters, intrinsics, auto_ground_plane=auto_ground_calib
            )

            # Auto-calib on first 20 frames then lock to avoid jitter unless requested
            if auto_ground_calib and frame_count > 20:
                auto_ground_calib = False
                print(f"[Calib] Ground Plane Initialized: Pitch = {engine.camera_pitch_deg:.1f}°, Height = {engine.camera_height_m*100:.1f} cm")

            # 3. Render Dashboard HUD
            dashboard_img = dashboard.build_dashboard(
                color_bgr, depth_meters, detection_result, intrinsics, fps
            )

            # 4. Broadcast Telemetry
            publisher.publish(detection_result, fps)

            # 5. Display
            cv2.imshow(window_name, dashboard_img)

            # 6. FPS Calculation
            frame_count += 1
            t_now = time.time()
            if (t_now - t_start) >= 1.0:
                fps = frame_count / (t_now - t_start)
                frame_count = 0
                t_start = t_now

            # 7. User Key Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('c') or key == ord('C'):
                auto_ground_calib = True
                print("[Calib] Auto Ground Plane Calibration Triggered...")
            elif key == ord('p') or key == ord('P'):
                engine.camera_pitch_deg = min(45.0, engine.camera_pitch_deg + 1.0)
                print(f"[Manual] Camera Pitch increased to {engine.camera_pitch_deg:.1f}°")
            elif key == ord('o') or key == ord('O'):
                engine.camera_pitch_deg = max(0.0, engine.camera_pitch_deg - 1.0)
                print(f"[Manual] Camera Pitch decreased to {engine.camera_pitch_deg:.1f}°")
            elif key == ord('h') or key == ord('H'):
                engine.camera_height_m = min(1.0, engine.camera_height_m + 0.02)
                print(f"[Manual] Camera Height increased to {engine.camera_height_m*100:.1f} cm")
            elif key == ord('j') or key == ord('J'):
                engine.camera_height_m = max(0.05, engine.camera_height_m - 0.02)
                print(f"[Manual] Camera Height decreased to {engine.camera_height_m*100:.1f} cm")
            elif key == ord('s') or key == ord('S'):
                snap_fn = f"robodog_snapshot_{int(time.time())}.png"
                cv2.imwrite(snap_fn, dashboard_img)
                print(f"[Saved] Snapshot saved to {snap_fn}")

    except KeyboardInterrupt:
        print("\n[System] Stopping perception engine...")
    finally:
        camera.stop()
        publisher.close()
        cv2.destroyAllWindows()
        print("[System] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robodog 3D Vision & Foot Clearance System")
    parser.add_argument("--udp", action="store_true", help="Enable UDP telemetry broadcast")
    parser.add_argument("--auto-calib", action="store_true", help="Enable auto-ground calibration on start")
    args = parser.parse_args()

    run_system(auto_calibrate_init=args.auto_calib, enable_udp=args.udp)
