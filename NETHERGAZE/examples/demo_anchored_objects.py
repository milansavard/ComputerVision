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
    
    Simple approach: objects are stored in camera coordinates at placement time.
    The current pose is used directly to render objects.
    """
    
    def __init__(self):
        self.is_anchored = False
        
        # Current camera pose
        self.current_R = np.eye(3)
        self.current_t = np.zeros((3, 1))
        
    def set_anchor(self, pose: PoseResult, tracking_result):
        """Set the current view as the world origin."""
        self.is_anchored = True
        self.current_R = np.eye(3)
        self.current_t = np.zeros((3, 1))
        print("✓ Anchor set! Place objects with keys 1-5.")
        return True
    
    def reset_anchor(self):
        """Clear the anchor and start over."""
        self.is_anchored = False
        self.current_R = np.eye(3)
        self.current_t = np.zeros((3, 1))
        print("Anchor cleared. Press SPACE to set a new anchor.")
    
    def update_world_pose(self, current_pose: PoseResult) -> bool:
        """
        Update current camera pose.
        
        Returns True if we have a valid pose.
        """
        if not self.is_anchored:
            return False
        
        # Always return True when anchored - we'll render with whatever pose we have
        # Only update pose if we got a valid one
        if current_pose is not None and current_pose.success:
            if current_pose.rotation_matrix is not None:
                # Blend with previous to reduce jitter
                alpha = 0.3
                self.current_R = alpha * current_pose.rotation_matrix + (1 - alpha) * self.current_R
                # Re-orthonormalize
                U, _, Vt = np.linalg.svd(self.current_R)
                self.current_R = U @ Vt
            if current_pose.translation_vector is not None:
                alpha = 0.3
                new_t = current_pose.translation_vector.reshape(3, 1)
                self.current_t = alpha * new_t + (1 - alpha) * self.current_t
        
        return True
    
    def world_to_camera(self, world_point: np.ndarray) -> np.ndarray:
        """Transform a point from world coordinates to camera coordinates."""
        point = world_point.reshape(3, 1)
        
        # Clamp translation to reasonable bounds (objects shouldn't fly far away)
        t_clamped = np.clip(self.current_t, -0.5, 0.5)
        
        # Transform: camera_point = R @ world_point + t
        camera_point = self.current_R @ point + t_clamped
        return camera_point.flatten()
    
    def get_camera_position_in_world(self) -> np.ndarray:
        """Get the camera's position in world coordinates."""
        return (-self.current_R.T @ self.current_t).flatten()


class AnchoredObjectsDemo:
    """Demo showing world-anchored AR objects."""
    
    def __init__(self, high_performance: bool = False):
        self.config = get_config()
        
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
            "chair": (139, 90, 43),     # Brown/wood color
        }
        
        # Display options
        self.show_markers = True
        self.show_help = True
        self.show_grid = True
        
        # Tracking state
        self.last_tracking_result = None
        self.frame_count = 0
        
        # Statistics for pose info display
        self.tracking_successes = 0
        self.pose_successes = 0
        self.fps = 0.0
        self.last_frame_time = 0.0
        self.feature_count = 0

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
        tracking_config["max_features"] = 3000  # More features = better anchoring
        tracking_config["fast_threshold"] = 10  # Lower = more sensitive (default 20)
        tracking_config["quality_level"] = 0.005  # Lower = accept weaker features (default 0.01)
        tracking_config["min_distance"] = 5.0  # Allow features closer together (default 7)
        tracking_config["reacquire_threshold"] = 500  # Re-detect sooner if features drop
        tracking_config["optical_flow_win_size"] = 25  # Larger window for better flow
        tracking_config["keyframe_interval"] = 8  # More frequent keyframes
        tracking_config["orb_nlevels"] = 12  # More pyramid levels (default 8)
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
        print("4. Press 1-5 to place objects")
        print("5. Move camera around - objects stay in place!")
        print("")
        print("CONTROLS:")
        print("  SPACE = Set anchor (world origin)")
        print("  1 = Place cube")
        print("  2 = Place pyramid")
        print("  3 = Place coordinate axes")
        print("  4 = Place solid box")
        print("  5 = Place chair")
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
        
        import time
        
        while True:
            frame = self.video.capture_frame()
            if frame is None:
                continue
            
            self.frame_count += 1
            
            # Calculate FPS
            current_time = time.time()
            if self.last_frame_time > 0:
                dt = current_time - self.last_frame_time
                instant_fps = 1.0 / max(dt, 0.001)
                self.fps = 0.9 * self.fps + 0.1 * instant_fps if self.fps > 0 else instant_fps
            self.last_frame_time = current_time
            
            # Track features
            tracking_result = self.tracker.process_frame(frame)
            self.last_tracking_result = tracking_result
            
            # Update feature count
            if tracking_result is not None and tracking_result.keypoints is not None:
                self.feature_count = len(tracking_result.keypoints)
            
            # Estimate pose
            pose = None
            if tracking_result is not None:
                pose = self.pose_estimator.estimate_from_feature_tracks(tracking_result)
            
            # Update statistics
            tracking_good = tracking_result is not None and tracking_result.tracked_count > 0
            pose_good = pose is not None and pose.success
            
            if tracking_good:
                self.tracking_successes += 1
            if pose_good:
                self.pose_successes += 1
            
            # Update world pose if anchored
            world_pose_valid = False
            if self.anchor_system.is_anchored and pose_good:
                world_pose_valid = self.anchor_system.update_world_pose(pose)
            
            # Draw features
            if self.show_markers and tracking_result is not None:
                frame = self._draw_markers(frame, tracking_result)
            
            # Render anchored objects
            if world_pose_valid:
                frame = self._render_anchored_objects(frame, pose)
            
            # Draw UI
            self._draw_status(frame, pose_good, world_pose_valid)
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
            elif obj.object_type == "chair":
                self._draw_chair(frame, camera_pos, obj.scale, obj.color, pose)
        
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

    def _draw_chair(self, frame: np.ndarray, position: np.ndarray,
                    scale: float, color: Tuple[int, int, int], pose: PoseResult):
        """Draw a 3D wireframe chair."""
        s = scale  # Overall scale
        p = position
        
        # Chair dimensions
        seat_h = s * 0.4      # Seat height from ground
        seat_w = s * 0.8      # Seat width
        seat_d = s * 0.7      # Seat depth
        seat_t = s * 0.08     # Seat thickness
        leg_w = s * 0.06      # Leg width
        back_h = s * 0.9      # Backrest height from seat
        back_t = s * 0.06     # Backrest thickness
        
        # 4 Legs (vertical lines from ground to seat)
        legs = [
            # Front-left leg
            [[p[0]-seat_w/2+leg_w, p[1]+seat_h, p[2]+seat_d/2-leg_w],
             [p[0]-seat_w/2+leg_w, p[1], p[2]+seat_d/2-leg_w]],
            # Front-right leg
            [[p[0]+seat_w/2-leg_w, p[1]+seat_h, p[2]+seat_d/2-leg_w],
             [p[0]+seat_w/2-leg_w, p[1], p[2]+seat_d/2-leg_w]],
            # Back-left leg
            [[p[0]-seat_w/2+leg_w, p[1]+seat_h, p[2]-seat_d/2+leg_w],
             [p[0]-seat_w/2+leg_w, p[1], p[2]-seat_d/2+leg_w]],
            # Back-right leg
            [[p[0]+seat_w/2-leg_w, p[1]+seat_h, p[2]-seat_d/2+leg_w],
             [p[0]+seat_w/2-leg_w, p[1], p[2]-seat_d/2+leg_w]],
        ]
        
        # Seat (rectangle at seat height)
        seat_y = p[1] + seat_h
        seat_verts = np.array([
            [p[0]-seat_w/2, seat_y, p[2]-seat_d/2],
            [p[0]+seat_w/2, seat_y, p[2]-seat_d/2],
            [p[0]+seat_w/2, seat_y, p[2]+seat_d/2],
            [p[0]-seat_w/2, seat_y, p[2]+seat_d/2],
        ], dtype=np.float64)
        seat_edges = [(0,1), (1,2), (2,3), (3,0)]
        
        # Backrest (vertical rectangle at back of seat)
        back_bottom = seat_y
        back_top = seat_y - back_h
        back_verts = np.array([
            [p[0]-seat_w/2, back_bottom, p[2]-seat_d/2],
            [p[0]+seat_w/2, back_bottom, p[2]-seat_d/2],
            [p[0]+seat_w/2, back_top, p[2]-seat_d/2],
            [p[0]-seat_w/2, back_top, p[2]-seat_d/2],
        ], dtype=np.float64)
        back_edges = [(0,1), (1,2), (2,3), (3,0)]
        
        # Draw legs
        for leg in legs:
            leg_pts = np.array(leg, dtype=np.float64)
            pts_2d = self._project_points(leg_pts)
            if pts_2d is not None:
                pt1 = tuple(pts_2d[0].astype(int))
                pt2 = tuple(pts_2d[1].astype(int))
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
        
        # Draw seat
        self._draw_wireframe(frame, seat_verts, seat_edges, color)
        
        # Draw backrest
        self._draw_wireframe(frame, back_verts, back_edges, color)
        
        # Draw backrest vertical supports (two posts)
        supports = [
            [[p[0]-seat_w/2+leg_w*2, back_bottom, p[2]-seat_d/2],
             [p[0]-seat_w/2+leg_w*2, back_top, p[2]-seat_d/2]],
            [[p[0]+seat_w/2-leg_w*2, back_bottom, p[2]-seat_d/2],
             [p[0]+seat_w/2-leg_w*2, back_top, p[2]-seat_d/2]],
        ]
        for support in supports:
            support_pts = np.array(support, dtype=np.float64)
            pts_2d = self._project_points(support_pts)
            if pts_2d is not None:
                pt1 = tuple(pts_2d[0].astype(int))
                pt2 = tuple(pts_2d[1].astype(int))
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    def _draw_world_grid(self, frame: np.ndarray, pose: PoseResult) -> np.ndarray:
        """Draw a grid in front of the camera (ground plane)."""
        grid_size = 0.4  # 40cm
        divisions = 8
        z_offset = 0.5   # Grid starts 50cm in front of camera
        
        lines = []
        for i in range(-divisions, divisions + 1):
            frac = i / divisions
            x_pos = frac * grid_size
            z_start = z_offset
            z_end = z_offset + grid_size
            
            # Lines along Z (going away from camera)
            lines.append(([[x_pos, 0, z_start], [x_pos, 0, z_end]], (80, 80, 80)))
            
            # Lines along X (side to side)
            z_pos = z_offset + (i + divisions) / (2 * divisions) * grid_size
            lines.append(([[-grid_size, 0, z_pos], [grid_size, 0, z_pos]], (80, 80, 80)))
        
        # Draw center axis lines brighter
        lines.append(([[0, 0, z_offset], [0, 0, z_offset + grid_size]], (150, 0, 0)))  # Z axis (blue)
        lines.append(([[-grid_size, 0, z_offset + grid_size/2], [grid_size, 0, z_offset + grid_size/2]], (0, 0, 150)))  # X axis (red)
        
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
        """Draw status bar with pose statistics."""
        h, w = frame.shape[:2]
        
        # Right side status panel
        cv2.rectangle(frame, (w - 200, 5), (w - 5, 95), (0, 0, 0), -1)
        
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
        
        # Bottom-left pose statistics panel
        cv2.rectangle(frame, (5, h - 70), (200, h - 5), (0, 0, 0), -1)
        
        # Calculate rates
        tracking_rate = (self.tracking_successes / max(self.frame_count, 1)) * 100
        pose_rate = (self.pose_successes / max(self.frame_count, 1)) * 100
        
        # Display stats
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (15, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Features: {self.feature_count}", (15, h - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Track: {tracking_rate:.0f}% | Pose: {pose_rate:.0f}%", (15, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Object count (top-right panel)
        if self.anchor_system.is_anchored:
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
        cv2.putText(frame, "1-5: Place objects (5=chair)", (20, y),
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
        """Place an object at a fixed world position."""
        if not self.anchor_system.is_anchored:
            print("Set anchor first! (Press SPACE)")
            return
        
        # Place objects in a grid pattern at the world origin
        count = len(self.placed_objects)
        col = count % 3
        row = count // 3
        
        # Spread objects in X-Z plane (horizontal), at Y=0 (ground level)
        x = (col - 1) * 0.12  # -0.12, 0, 0.12
        z = 0.5 + row * 0.12   # Start 50cm away, spread back
        y = 0.0               # Ground level
        
        world_pos = np.array([x, y, z])
        
        obj = AnchoredObject(
            object_type=object_type,
            world_position=world_pos,
            scale=0.08,
            color=self.object_colors.get(object_type, (255, 255, 255)),
        )
        
        self.placed_objects.append(obj)
        print(f"Placed {object_type} at position ({x:.2f}, {y:.2f}, {z:.2f})")

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
        
        elif key == ord('5'):
            self.current_object_type = "chair"
            self._place_object("chair")
        
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
    parser.add_argument("--fast", action="store_true", 
                        help="High performance mode (60fps, more features)")
    args = parser.parse_args()
    
    demo = AnchoredObjectsDemo(high_performance=args.fast)
    demo.run()


if __name__ == "__main__":
    main()
