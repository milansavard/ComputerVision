#!/usr/bin/env python3
"""
Demo: 3D Objects in AR Space

This demo shows how to add various 3D objects to your AR scene.
Works with the default camera calibration - no chessboard needed!

Usage:
    python examples/demo_3d_objects.py

Controls:
    1-5  - Switch between different object demos
    +/-  - Move object closer/farther
    WASD - Move object in X/Y plane
    R    - Reset object position
    M    - Toggle feature markers
    Q    - Quit
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
from typing import Optional

from video import VideoProcessor
from tracking.feature import FeatureTracker
from pose import PoseEstimator, PoseResult, CalibrationData
from overlay import OverlayRenderer, Object3D, Mesh3D
from utils import get_config


class Demo3DObjects:
    """Interactive demo for 3D AR objects."""

    def __init__(self):
        self.config = get_config()
        self.video: Optional[VideoProcessor] = None
        self.tracker: Optional[FeatureTracker] = None
        self.pose_estimator: Optional[PoseEstimator] = None
        self.renderer: Optional[OverlayRenderer] = None
        
        # Object position (in front of camera)
        self.object_position = np.array([0.0, 0.0, 0.5])  # 50cm in front
        self.object_scale = 0.08  # 8cm
        self.object_rotation = np.array([0.0, 0.0, 0.0])
        
        # Demo mode
        self.demo_mode = 1
        self.show_markers = True
        self.show_help = True
        
        # Animation
        self.frame_count = 0

    def initialize(self) -> bool:
        """Initialize all components."""
        print("\n" + "=" * 60)
        print("  NETHERGAZE - 3D Objects Demo")
        print("=" * 60)
        
        # Initialize video
        self.video = VideoProcessor(self.config)
        if not self.video.initialize():
            print("ERROR: Failed to initialize camera")
            print("Make sure your camera is connected and not in use.")
            return False
        print("✓ Camera initialized")
        
        # Initialize tracker
        self.tracker = FeatureTracker(self.config.get("feature_tracking", {}))
        print("✓ Feature tracker initialized")
        
        # Initialize pose estimator
        self.pose_estimator = PoseEstimator(self.config)
        if not self.pose_estimator.initialize():
            print("ERROR: Failed to initialize pose estimator")
            return False
        print("✓ Pose estimator initialized")
        
        # Initialize renderer
        self.renderer = OverlayRenderer(self.config.get("overlay", {}))
        self.renderer.initialize(self.pose_estimator.calibration)
        print("✓ Overlay renderer initialized")
        
        print("=" * 60)
        self._print_controls()
        
        return True

    def _print_controls(self):
        """Print control instructions."""
        print("\nCONTROLS:")
        print("  1 = Wireframe Cube")
        print("  2 = Solid Colored Box")
        print("  3 = Coordinate Axes")
        print("  4 = Pyramid")
        print("  5 = Spinning Animation")
        print("")
        print("  W/S = Move up/down")
        print("  A/D = Move left/right")
        print("  +/- = Move closer/farther")
        print("  R   = Reset position")
        print("  M   = Toggle feature markers")
        print("  H   = Toggle help overlay")
        print("  Q   = Quit")
        print("=" * 60 + "\n")

    def run(self):
        """Main demo loop."""
        if not self.initialize():
            return

        window_name = "NETHERGAZE - 3D Objects Demo"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)

        print("Point camera at a textured surface (books, posters, etc.)")
        print("The 3D object will appear anchored in the scene!\n")

        while True:
            # Capture frame
            frame = self.video.capture_frame()
            if frame is None:
                continue

            self.frame_count += 1

            # Track features
            tracking_result = self.tracker.process_frame(frame)
            
            # Estimate pose
            pose = None
            if tracking_result is not None:
                pose = self.pose_estimator.estimate_from_feature_tracks(tracking_result)

            # Draw feature markers
            if self.show_markers and tracking_result is not None:
                frame = self._draw_markers(frame, tracking_result)

            # Clear previous objects and add current demo object
            self.renderer.clear_3d_objects()
            self.renderer.clear_meshes()
            self._add_demo_object()

            # Render 3D objects
            if pose is not None and pose.success:
                frame = self.renderer.render(frame, pose)
                self._draw_pose_status(frame, pose, True)
            else:
                self._draw_pose_status(frame, pose, False)

            # Draw help overlay
            if self.show_help:
                self._draw_help(frame)

            # Display
            cv2.imshow(window_name, frame)

            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_key(key):
                break

        self.cleanup()

    def _add_demo_object(self):
        """Add the current demo object based on mode."""
        pos = self.object_position.copy()
        
        if self.demo_mode == 1:
            # Wireframe cube
            self.renderer.add_cube(
                position=pos,
                scale=self.object_scale,
                color=(0, 255, 255),  # Yellow
            )
            
        elif self.demo_mode == 2:
            # Solid colored box
            self.renderer.add_solid_box(
                position=pos,
                scale=self.object_scale,
                color=(0, 128, 255),  # Orange
                rotation=self.object_rotation,
            )
            
        elif self.demo_mode == 3:
            # Coordinate axes
            self.renderer.add_3d_object(Object3D(
                object_type="axes",
                position=pos,
                scale=self.object_scale * 2,
            ))
            
        elif self.demo_mode == 4:
            # Pyramid
            self.renderer.add_pyramid(
                position=pos,
                scale=self.object_scale,
                color=(255, 0, 255),  # Magenta
            )
            
        elif self.demo_mode == 5:
            # Spinning cube animation
            angle = (self.frame_count * 2) % 360
            self.renderer.add_solid_box(
                position=pos,
                scale=self.object_scale,
                color=(0, 255, 128),  # Cyan-green
                rotation=np.array([angle, angle * 0.5, 0.0]),
            )

    def _draw_markers(self, frame: np.ndarray, tracking_result) -> np.ndarray:
        """Draw tracked feature markers."""
        if tracking_result.keypoints is None:
            return frame
            
        for pt in tracking_result.keypoints[:200]:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        return frame

    def _draw_pose_status(self, frame: np.ndarray, pose: Optional[PoseResult], success: bool):
        """Draw pose tracking status."""
        h, w = frame.shape[:2]
        
        # Status indicator
        if success:
            color = (0, 255, 0)
            text = "TRACKING"
        else:
            color = (0, 0, 255)
            text = "SEARCHING..."
        
        cv2.rectangle(frame, (w - 150, 10), (w - 10, 40), (0, 0, 0), -1)
        cv2.putText(frame, text, (w - 145, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_help(self, frame: np.ndarray):
        """Draw help overlay."""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Demo mode name
        mode_names = {
            1: "Wireframe Cube",
            2: "Solid Box",
            3: "Coordinate Axes",
            4: "Pyramid",
            5: "Spinning Cube",
        }
        
        y = 30
        cv2.putText(frame, f"Mode {self.demo_mode}: {mode_names[self.demo_mode]}", 
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        y += 25
        cv2.putText(frame, f"Position: [{self.object_position[0]:.2f}, {self.object_position[1]:.2f}, {self.object_position[2]:.2f}]",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        y += 25
        cv2.putText(frame, "1-5: Change object", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y += 20
        cv2.putText(frame, "WASD: Move X/Y", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y += 20
        cv2.putText(frame, "+/-: Move Z (depth)", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y += 20
        cv2.putText(frame, "R: Reset | M: Markers | H: Help", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y += 20
        cv2.putText(frame, "Q: Quit", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def _handle_key(self, key: int) -> bool:
        """Handle keyboard input. Returns False to quit."""
        move_step = 0.02  # 2cm
        
        if key == ord('q') or key == 27:  # Q or ESC
            return False
            
        # Demo mode selection
        elif key == ord('1'):
            self.demo_mode = 1
            print("Mode 1: Wireframe Cube")
        elif key == ord('2'):
            self.demo_mode = 2
            print("Mode 2: Solid Box")
        elif key == ord('3'):
            self.demo_mode = 3
            print("Mode 3: Coordinate Axes")
        elif key == ord('4'):
            self.demo_mode = 4
            print("Mode 4: Pyramid")
        elif key == ord('5'):
            self.demo_mode = 5
            print("Mode 5: Spinning Animation")
            
        # Movement
        elif key == ord('w'):
            self.object_position[1] -= move_step  # Up (Y is inverted)
        elif key == ord('s'):
            self.object_position[1] += move_step  # Down
        elif key == ord('a'):
            self.object_position[0] -= move_step  # Left
        elif key == ord('d'):
            self.object_position[0] += move_step  # Right
        elif key == ord('+') or key == ord('='):
            self.object_position[2] -= move_step  # Closer
            self.object_position[2] = max(0.1, self.object_position[2])
        elif key == ord('-'):
            self.object_position[2] += move_step  # Farther
            
        # Reset
        elif key == ord('r'):
            self.object_position = np.array([0.0, 0.0, 0.5])
            self.object_rotation = np.array([0.0, 0.0, 0.0])
            print("Position reset")
            
        # Toggles
        elif key == ord('m'):
            self.show_markers = not self.show_markers
            print(f"Feature markers: {'ON' if self.show_markers else 'OFF'}")
        elif key == ord('h'):
            self.show_help = not self.show_help
            
        return True

    def cleanup(self):
        """Clean up resources."""
        if self.video:
            self.video.cleanup()
        if self.renderer:
            self.renderer.cleanup()
        cv2.destroyAllWindows()
        print("\nDemo ended. Goodbye!")


def main():
    demo = Demo3DObjects()
    demo.run()


if __name__ == "__main__":
    main()
