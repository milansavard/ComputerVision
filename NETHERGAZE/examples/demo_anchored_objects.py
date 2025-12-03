#!/usr/bin/env python3
"""
Demo: World-Anchored AR Objects

This demo shows objects that stay FIXED in 3D space - when you move the camera,
the objects stay where you placed them, just like real AR apps!

How it works:
1. Point camera at a textured surface
2. Press SPACE to "anchor" - this sets the world origin
3. Objects now stay fixed relative to that anchor point
4. Move camera around and watch objects stay in place!

Usage:
    python examples/demo_anchored_objects.py

Controls:
    SPACE  - Set anchor point (world origin) at current view
    1-4    - Add different objects at current anchor
    C      - Clear all placed objects
    M      - Toggle feature markers
    R      - Reset anchor (start over)
    Q      - Quit
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from video import VideoProcessor
from tracking.feature import FeatureTracker
from pose import PoseEstimator, PoseResult, CalibrationData
from overlay import OverlayRenderer, Object3D, Mesh3D
from utils import get_config


@dataclass
class AnchoredObject:
    """A 3D object anchored in world space."""
    object_type: str  # "cube", "pyramid", "axes", "sphere"
    world_position: np.ndarray  # Position in world coordinates
    scale: float = 0.05
    color: Tuple[int, int, int] = (0, 255, 255)
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))


class WorldAnchorSystem:
    """
    Manages world-space anchoring for AR objects.
    
    The key insight: we track camera pose RELATIVE to the first keyframe.
    Objects placed in "world space" are actually relative to that first frame.
    """
    
    def __init__(self):
        self.is_anchored = False
        self.anchor_pose: Optional[PoseResult] = None  # Pose when anchor was set
        self.world_origin_set = False
        
        # Accumulated camera pose (relative to world origin)
        self.world_R = np.eye(3)  # World rotation
        self.world_t = np.zeros((3, 1))  # World translation
        
        # Reference data from anchor frame
        self.anchor_features = None
        self.anchor_descriptors = None
        
    def set_anchor(self, pose: PoseResult, tracking_result):
        """Set the current view as the world origin."""
        if pose is None or not pose.success:
            return False
            
        self.is_anchored = True
        self.anchor_pose = pose
        self.world_origin_set = True
        
        # Store reference features for re-localization
        if tracking_result is not None:
            self.anchor_features = tracking_result.keypoints.copy() if tracking_result.keypoints is not None else None
            self.anchor_descriptors = tracking_result.descriptors.copy() if tracking_result.descriptors is not None else None
        
        # Reset world pose to identity (this is now "home")
        self.world_R = np.eye(3)
        self.world_t = np.zeros((3, 1))
        
        print("✓ Anchor set! Objects will now stay fixed in space.")
        return True
    
    def reset_anchor(self):
        """Clear the anchor and start over."""
        self.is_anchored = False
        self.anchor_pose = None
        self.world_origin_set = False
        self.world_R = np.eye(3)
        self.world_t = np.zeros((3, 1))
        print("Anchor cleared. Press SPACE to set a new anchor.")
    
    def update_world_pose(self, current_pose: PoseResult) -> bool:
        """
        Update the camera's position in world coordinates.
        
        Returns True if we have a valid world pose.
        """
        if not self.is_anchored or current_pose is None or not current_pose.success:
            return False
        
        if current_pose.rotation_matrix is None or current_pose.translation_vector is None:
            return False
        
        # The pose estimator gives us relative pose between frames
        # We accumulate this to get world pose
        # For simplicity, we use the current pose directly relative to anchor
        self.world_R = current_pose.rotation_matrix
        self.world_t = current_pose.translation_vector
        
        return True
    
    def world_to_camera(self, world_point: np.ndarray) -> np.ndarray:
        """Transform a point from world coordinates to camera coordinates."""
        # Camera sees: p_camera = R * p_world + t
        point = world_point.reshape(3, 1)
        camera_point = self.world_R @ point + self.world_t
        return camera_point.flatten()
    
    def get_camera_position_in_world(self) -> np.ndarray:
        """Get the camera's position in world coordinates."""
        # Camera position = -R^T * t
        return (-self.world_R.T @ self.world_t).flatten()


class AnchoredObjectsDemo:
    """Demo showing world-anchored AR objects."""
    
    def __init__(self, camera_id: int = 1, high_performance: bool = False):
        self.config = get_config()
        
        # Camera selection (default to Continuity Camera on index 1)
        self.config["camera_id"] = camera_id
        print(f"Using camera index: {camera_id}" + 
              (" (Continuity Camera)" if camera_id == 1 else " (Webcam)" if camera_id == 0 else ""))
        
        # Performance tuning
        if high_performance:
            # Higher FPS and more features for better tracking
            self.config["video_fps"] = 60
            self.config["video_width"] = 640  # Lower res = faster
            self.config["video_height"] = 480
            print("High performance mode: 60fps, 640x480")
        
        self.video: Optional[VideoProcessor] = None
        self.tracker: Optional[FeatureTracker] = None
        self.pose_estimator: Optional[PoseEstimator] = None
        self.renderer: Optional[OverlayRenderer] = None
        
        # World anchor system
        self.anchor_system = WorldAnchorSystem()
        
        # Placed objects (in world coordinates)
        self.placed_objects: List[AnchoredObject] = []
        
        # Current object type to place
        self.current_object_type = "cube"
        self.object_colors = {
            "cube": (0, 255, 255),      # Yellow
            "pyramid": (255, 0, 255),   # Magenta
            "axes": (255, 255, 255),    # White
            "box": (0, 128, 255),       # Orange
        }
        
        # Display options
        self.show_markers = True
        self.show_help = True
        self.show_grid = True
        
        # Tracking state
        self.last_tracking_result = None
        self.frame_count = 0

    def initialize(self) -> bool:
        """Initialize all components."""
        print("\n" + "=" * 60)
        print("  NETHERGAZE - World-Anchored AR Demo")
        print("=" * 60)
        
        # Initialize video
        self.video = VideoProcessor(self.config)
        if not self.video.initialize():
            print("ERROR: Failed to initialize camera")
            return False
        print("✓ Camera initialized")
        
        # Initialize tracker with enhanced settings for anchoring
        tracking_config = self.config.get("feature_tracking", {})
        tracking_config["max_features"] = 2000  # More features = better anchoring
        tracking_config["reacquire_threshold"] = 300  # Re-detect sooner if features drop
        tracking_config["optical_flow_win_size"] = 25  # Larger window for better flow
        tracking_config["keyframe_interval"] = 10  # More frequent keyframes
        self.tracker = FeatureTracker(tracking_config)
        print(f"✓ Feature tracker initialized (max {tracking_config['max_features']} features)")
        
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
        self._print_instructions()
        
        return True

    def _print_instructions(self):
        """Print usage instructions."""
        print("\nHOW TO USE:")
        print("1. Point camera at a textured surface (book, poster, etc.)")
        print("2. Wait for green 'TRACKING' indicator")
        print("3. Press SPACE to set the anchor point")
        print("4. Press 1-4 to place objects")
        print("5. Move camera around - objects stay in place!")
        print("")
        print("CONTROLS:")
        print("  SPACE = Set anchor (world origin)")
        print("  1 = Place cube")
        print("  2 = Place pyramid")
        print("  3 = Place coordinate axes")
        print("  4 = Place solid box")
        print("  C = Clear all objects")
        print("  G = Toggle ground grid")
        print("  M = Toggle feature markers")
        print("  R = Reset anchor")
        print("  Q = Quit")
        print("=" * 60 + "\n")

    def run(self):
        """Main demo loop."""
        if not self.initialize():
            return
        
        window_name = "NETHERGAZE - World-Anchored AR"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)
        
        while True:
            frame = self.video.capture_frame()
            if frame is None:
                continue
            
            self.frame_count += 1
            
            # Track features
            tracking_result = self.tracker.process_frame(frame)
            self.last_tracking_result = tracking_result
            
            # Estimate pose
            pose = None
            if tracking_result is not None:
                pose = self.pose_estimator.estimate_from_feature_tracks(tracking_result)
            
            # Update world pose if anchored
            tracking_good = pose is not None and pose.success
            world_pose_valid = False
            
            if self.anchor_system.is_anchored and tracking_good:
                world_pose_valid = self.anchor_system.update_world_pose(pose)
            
            # Draw features
            if self.show_markers and tracking_result is not None:
                frame = self._draw_markers(frame, tracking_result)
            
            # Render anchored objects
            if world_pose_valid:
                frame = self._render_anchored_objects(frame, pose)
            
            # Draw UI
            self._draw_status(frame, tracking_good, world_pose_valid)
            if self.show_help:
                self._draw_help(frame)
            
            cv2.imshow(window_name, frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_key(key, pose, tracking_result):
                break
        
        self.cleanup()

    def _render_anchored_objects(self, frame: np.ndarray, pose: PoseResult) -> np.ndarray:
        """Render all placed objects in world space."""
        if self.renderer.calibration is None:
            return frame
        
        # Draw ground grid at world origin
        if self.show_grid:
            frame = self._draw_world_grid(frame, pose)
        
        # Draw each placed object
        for obj in self.placed_objects:
            # Transform world position to camera coordinates
            camera_pos = self.anchor_system.world_to_camera(obj.world_position)
            
            # Only render if in front of camera
            if camera_pos[2] <= 0.05:
                continue
            
            # Create temporary 3D object at camera-relative position
            if obj.object_type == "cube":
                self._draw_wireframe_cube(frame, camera_pos, obj.scale, obj.color, pose)
            elif obj.object_type == "pyramid":
                self._draw_wireframe_pyramid(frame, camera_pos, obj.scale, obj.color, pose)
            elif obj.object_type == "axes":
                self._draw_axes(frame, camera_pos, obj.scale, pose)
            elif obj.object_type == "box":
                self._draw_solid_box(frame, camera_pos, obj.scale, obj.color, pose)
        
        return frame

    def _draw_wireframe_cube(self, frame: np.ndarray, position: np.ndarray, 
                              scale: float, color: Tuple[int, int, int], pose: PoseResult):
        """Draw a wireframe cube at the given camera-space position."""
        s = scale / 2
        p = position
        
        vertices = np.array([
            [p[0]-s, p[1]-s, p[2]-s], [p[0]+s, p[1]-s, p[2]-s],
            [p[0]+s, p[1]+s, p[2]-s], [p[0]-s, p[1]+s, p[2]-s],
            [p[0]-s, p[1]-s, p[2]+s], [p[0]+s, p[1]-s, p[2]+s],
            [p[0]+s, p[1]+s, p[2]+s], [p[0]-s, p[1]+s, p[2]+s],
        ], dtype=np.float64)
        
        edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4),
                 (0,4), (1,5), (2,6), (3,7)]
        
        self._draw_wireframe(frame, vertices, edges, color)

    def _draw_wireframe_pyramid(self, frame: np.ndarray, position: np.ndarray,
                                 scale: float, color: Tuple[int, int, int], pose: PoseResult):
        """Draw a wireframe pyramid."""
        s = scale / 2
        h = scale
        p = position
        
        vertices = np.array([
            [p[0]-s, p[1]+s, p[2]-s], [p[0]+s, p[1]+s, p[2]-s],
            [p[0]+s, p[1]+s, p[2]+s], [p[0]-s, p[1]+s, p[2]+s],
            [p[0], p[1]-h, p[2]],  # Apex
        ], dtype=np.float64)
        
        edges = [(0,1), (1,2), (2,3), (3,0), (0,4), (1,4), (2,4), (3,4)]
        
        self._draw_wireframe(frame, vertices, edges, color)

    def _draw_axes(self, frame: np.ndarray, position: np.ndarray, 
                   scale: float, pose: PoseResult):
        """Draw RGB coordinate axes."""
        p = position
        s = scale * 2
        
        vertices = np.array([
            [p[0], p[1], p[2]],      # Origin
            [p[0]+s, p[1], p[2]],    # X
            [p[0], p[1]-s, p[2]],    # Y (up)
            [p[0], p[1], p[2]+s],    # Z
        ], dtype=np.float64)
        
        pts_2d = self._project_points(vertices)
        if pts_2d is None:
            return
        
        origin = tuple(pts_2d[0].astype(int))
        cv2.line(frame, origin, tuple(pts_2d[1].astype(int)), (0, 0, 255), 2, cv2.LINE_AA)  # X red
        cv2.line(frame, origin, tuple(pts_2d[2].astype(int)), (0, 255, 0), 2, cv2.LINE_AA)  # Y green
        cv2.line(frame, origin, tuple(pts_2d[3].astype(int)), (255, 0, 0), 2, cv2.LINE_AA)  # Z blue

    def _draw_solid_box(self, frame: np.ndarray, position: np.ndarray,
                        scale: float, color: Tuple[int, int, int], pose: PoseResult):
        """Draw a solid colored box with simple shading."""
        s = scale / 2
        p = position
        
        # Define faces (as quads, front to back roughly)
        faces = [
            # Front face
            np.array([[p[0]-s, p[1]-s, p[2]+s], [p[0]+s, p[1]-s, p[2]+s],
                      [p[0]+s, p[1]+s, p[2]+s], [p[0]-s, p[1]+s, p[2]+s]]),
            # Top face
            np.array([[p[0]-s, p[1]-s, p[2]-s], [p[0]+s, p[1]-s, p[2]-s],
                      [p[0]+s, p[1]-s, p[2]+s], [p[0]-s, p[1]-s, p[2]+s]]),
            # Right face
            np.array([[p[0]+s, p[1]-s, p[2]-s], [p[0]+s, p[1]+s, p[2]-s],
                      [p[0]+s, p[1]+s, p[2]+s], [p[0]+s, p[1]-s, p[2]+s]]),
            # Left face
            np.array([[p[0]-s, p[1]-s, p[2]+s], [p[0]-s, p[1]+s, p[2]+s],
                      [p[0]-s, p[1]+s, p[2]-s], [p[0]-s, p[1]-s, p[2]-s]]),
        ]
        
        shades = [1.0, 0.9, 0.7, 0.5]  # Different brightness per face
        
        for face_verts, shade in zip(faces, shades):
            pts_2d = self._project_points(face_verts)
            if pts_2d is None:
                continue
            
            face_color = tuple(int(c * shade) for c in color)
            pts = pts_2d.astype(np.int32).reshape(-1, 1, 2)
            cv2.fillConvexPoly(frame, pts, face_color)

    def _draw_world_grid(self, frame: np.ndarray, pose: PoseResult) -> np.ndarray:
        """Draw a grid at the world origin (ground plane)."""
        grid_size = 0.3  # 30cm
        divisions = 6
        
        lines = []
        for i in range(-divisions, divisions + 1):
            offset = (i / divisions) * grid_size
            # Lines along X
            lines.append(([[-grid_size, 0, offset], [grid_size, 0, offset]], (80, 80, 80)))
            # Lines along Z
            lines.append((([[offset, 0, -grid_size], [offset, 0, grid_size]]), (80, 80, 80)))
        
        # Draw X and Z axis lines brighter
        lines.append(([[-grid_size, 0, 0], [grid_size, 0, 0]], (0, 0, 150)))  # X axis (red)
        lines.append((([[0, 0, -grid_size], [0, 0, grid_size]]), (150, 0, 0)))  # Z axis (blue)
        
        for line_pts, color in lines:
            pts_world = np.array(line_pts, dtype=np.float64)
            # Transform to camera space
            pts_camera = np.array([self.anchor_system.world_to_camera(p) for p in pts_world])
            
            # Only draw if in front of camera
            if np.all(pts_camera[:, 2] > 0.05):
                pts_2d = self._project_points(pts_camera)
                if pts_2d is not None:
                    pt1 = tuple(pts_2d[0].astype(int))
                    pt2 = tuple(pts_2d[1].astype(int))
                    cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)
        
        return frame

    def _draw_wireframe(self, frame: np.ndarray, vertices: np.ndarray,
                        edges: List[Tuple[int, int]], color: Tuple[int, int, int]):
        """Draw a wireframe from vertices and edges."""
        pts_2d = self._project_points(vertices)
        if pts_2d is None:
            return
        
        for i, j in edges:
            pt1 = tuple(pts_2d[i].astype(int))
            pt2 = tuple(pts_2d[j].astype(int))
            cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    def _project_points(self, points_3d: np.ndarray) -> Optional[np.ndarray]:
        """Project 3D points to 2D image coordinates."""
        if self.renderer.calibration is None:
            return None
        
        try:
            pts_2d, _ = cv2.projectPoints(
                points_3d.astype(np.float64),
                np.zeros(3),  # Points already in camera coords
                np.zeros(3),
                self.renderer.calibration.camera_matrix,
                self.renderer.calibration.dist_coeffs,
            )
            return pts_2d.reshape(-1, 2)
        except cv2.error:
            return None

    def _draw_markers(self, frame: np.ndarray, tracking_result) -> np.ndarray:
        """Draw tracked features."""
        if tracking_result.keypoints is None:
            return frame
        
        for pt in tracking_result.keypoints[:300]:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        return frame

    def _draw_status(self, frame: np.ndarray, tracking: bool, anchored: bool):
        """Draw status bar."""
        h, w = frame.shape[:2]
        
        # Background
        cv2.rectangle(frame, (w - 200, 5), (w - 5, 75), (0, 0, 0), -1)
        
        # Tracking status
        if tracking:
            cv2.putText(frame, "TRACKING", (w - 190, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "SEARCHING...", (w - 190, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Anchor status
        if self.anchor_system.is_anchored:
            cv2.putText(frame, "ANCHORED", (w - 190, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, f"Objects: {len(self.placed_objects)}", (w - 190, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "Press SPACE to anchor", (w - 190, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def _draw_help(self, frame: np.ndarray):
        """Draw help panel."""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (220, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y = 30
        cv2.putText(frame, "World-Anchored AR Demo", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        y += 25
        cv2.putText(frame, f"Object: {self.current_object_type.upper()}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        y += 20
        cv2.putText(frame, "SPACE: Set anchor", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y += 18
        cv2.putText(frame, "1-4: Place objects", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y += 18
        cv2.putText(frame, "C: Clear | R: Reset", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y += 18
        cv2.putText(frame, "G: Grid | M: Markers", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y += 18
        cv2.putText(frame, "H: Help | Q: Quit", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def _place_object(self, object_type: str):
        """Place an object at the world origin."""
        if not self.anchor_system.is_anchored:
            print("Set anchor first! (Press SPACE)")
            return
        
        # Place object at world origin (0, 0, 0) or offset based on count
        offset = len(self.placed_objects) * 0.08
        world_pos = np.array([offset, 0.0, 0.0])  # Spread along X axis
        
        obj = AnchoredObject(
            object_type=object_type,
            world_position=world_pos,
            scale=0.06,
            color=self.object_colors.get(object_type, (255, 255, 255)),
        )
        
        self.placed_objects.append(obj)
        print(f"Placed {object_type} at world position {world_pos}")

    def _handle_key(self, key: int, pose: PoseResult, tracking_result) -> bool:
        """Handle keyboard input."""
        if key == ord('q') or key == 27:
            return False
        
        elif key == ord(' '):  # SPACE - set anchor
            if pose is not None and pose.success:
                self.anchor_system.set_anchor(pose, tracking_result)
            else:
                print("Cannot set anchor - no tracking! Point at a textured surface.")
        
        elif key == ord('1'):
            self.current_object_type = "cube"
            self._place_object("cube")
        
        elif key == ord('2'):
            self.current_object_type = "pyramid"
            self._place_object("pyramid")
        
        elif key == ord('3'):
            self.current_object_type = "axes"
            self._place_object("axes")
        
        elif key == ord('4'):
            self.current_object_type = "box"
            self._place_object("box")
        
        elif key == ord('c'):
            self.placed_objects.clear()
            print("Cleared all objects")
        
        elif key == ord('r'):
            self.anchor_system.reset_anchor()
            self.placed_objects.clear()
        
        elif key == ord('g'):
            self.show_grid = not self.show_grid
        
        elif key == ord('m'):
            self.show_markers = not self.show_markers
        
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
        print("\nDemo ended!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="World-Anchored AR Demo")
    parser.add_argument("--camera", "-c", type=int, default=1,
                        help="Camera index (default: 1 for Continuity Camera, 0 for webcam)")
    parser.add_argument("--fast", action="store_true", 
                        help="High performance mode (60fps, more features)")
    args = parser.parse_args()
    
    demo = AnchoredObjectsDemo(camera_id=args.camera, high_performance=args.fast)
    demo.run()


if __name__ == "__main__":
    main()
