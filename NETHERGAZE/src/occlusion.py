"""
Occlusion handling module for NETHERGAZE.

Provides depth-aware rendering for proper AR occlusion:
- Sparse depth from triangulated features
- Depth interpolation/completion
- Occlusion mask generation
- Depth-aware overlay compositing

Note: For best results, use with depth sensors (RealSense, Kinect) or
stereo cameras. This module provides monocular depth approximation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pose import CalibrationData, PoseResult

LOGGER = logging.getLogger(__name__)


class DepthSource(Enum):
    """Source of depth information."""
    SPARSE_FEATURES = "sparse_features"  # From triangulated features
    DENSE_ESTIMATION = "dense_estimation"  # MiDaS or similar
    DEPTH_SENSOR = "depth_sensor"  # External depth camera
    GROUND_PLANE = "ground_plane"  # Assume flat ground


@dataclass
class OcclusionConfig:
    """Configuration for occlusion handling."""
    
    # Depth estimation
    depth_source: str = "sparse_features"
    default_depth: float = 5.0  # Default depth when unknown (meters)
    min_depth: float = 0.1  # Minimum valid depth
    max_depth: float = 20.0  # Maximum valid depth
    
    # Depth completion
    enable_depth_completion: bool = True
    completion_kernel_size: int = 15  # Kernel for depth interpolation
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    
    # Ground plane
    ground_plane_height: float = 1.5  # Camera height above ground
    ground_plane_enabled: bool = True
    
    # Occlusion rendering
    occlusion_threshold: float = 0.05  # Depth difference for occlusion (meters)
    edge_softness: float = 3.0  # Softness of occlusion edges (pixels)
    
    # Performance
    depth_scale: float = 0.5  # Scale factor for depth map computation


class DepthEstimator:
    """
    Estimates depth from various sources.
    
    Supports:
    - Sparse depth from triangulated features
    - Dense depth completion/interpolation
    - Ground plane assumption
    """
    
    def __init__(self, config: Optional[OcclusionConfig] = None):
        self.config = config or OcclusionConfig()
        self.calibration: Optional[CalibrationData] = None
        
        # Cached data
        self._last_depth_map: Optional[np.ndarray] = None
        self._last_frame_shape: Optional[Tuple[int, int]] = None
        
        # Ground plane
        self._ground_plane: Optional[np.ndarray] = None  # Normal and d
        
    def initialize(self, calibration: CalibrationData) -> bool:
        """Initialize with camera calibration."""
        self.calibration = calibration
        LOGGER.info("DepthEstimator initialized")
        return True

    def estimate_depth(
        self,
        frame: np.ndarray,
        pose: PoseResult,
        sparse_points_3d: Optional[np.ndarray] = None,
        sparse_points_2d: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Estimate depth map for a frame.
        
        Args:
            frame: BGR frame
            pose: Camera pose
            sparse_points_3d: Nx3 array of 3D points (optional)
            sparse_points_2d: Nx2 array of corresponding 2D points (optional)
            
        Returns:
            Depth map (float32, meters)
        """
        h, w = frame.shape[:2]
        
        # Compute at reduced resolution for performance
        scale = self.config.depth_scale
        small_h, small_w = int(h * scale), int(w * scale)
        
        # Start with default depth
        depth_map = np.full((small_h, small_w), self.config.default_depth, dtype=np.float32)
        
        # Add sparse depth from features
        if sparse_points_3d is not None and sparse_points_2d is not None:
            depth_map = self._add_sparse_depth(
                depth_map, sparse_points_3d, sparse_points_2d, scale
            )
        
        # Add ground plane depth
        if self.config.ground_plane_enabled and pose.success:
            depth_map = self._add_ground_plane_depth(depth_map, pose, scale)
        
        # Complete/interpolate depth
        if self.config.enable_depth_completion:
            depth_map = self._complete_depth(depth_map)
        
        # Upscale to original resolution
        if scale != 1.0:
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Clamp to valid range
        depth_map = np.clip(depth_map, self.config.min_depth, self.config.max_depth)
        
        self._last_depth_map = depth_map
        self._last_frame_shape = (h, w)
        
        return depth_map

    def _add_sparse_depth(
        self,
        depth_map: np.ndarray,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """Add sparse depth points to the depth map."""
        h, w = depth_map.shape
        
        for pt3d, pt2d in zip(points_3d, points_2d):
            x = int(pt2d[0] * scale)
            y = int(pt2d[1] * scale)
            
            if 0 <= x < w and 0 <= y < h:
                depth = pt3d[2]  # Z coordinate is depth
                if self.config.min_depth <= depth <= self.config.max_depth:
                    # Set depth in a small region around the point
                    radius = 3
                    y1, y2 = max(0, y - radius), min(h, y + radius + 1)
                    x1, x2 = max(0, x - radius), min(w, x + radius + 1)
                    depth_map[y1:y2, x1:x2] = depth
        
        return depth_map

    def _add_ground_plane_depth(
        self,
        depth_map: np.ndarray,
        pose: PoseResult,
        scale: float,
    ) -> np.ndarray:
        """Add depth from ground plane assumption."""
        if self.calibration is None:
            return depth_map
        
        h, w = depth_map.shape
        camera_height = self.config.ground_plane_height
        
        # Get camera intrinsics
        fx = self.calibration.camera_matrix[0, 0] * scale
        fy = self.calibration.camera_matrix[1, 1] * scale
        cx = self.calibration.camera_matrix[0, 2] * scale
        cy = self.calibration.camera_matrix[1, 2] * scale
        
        # For each pixel in the lower half (likely ground)
        for y in range(h // 2, h):
            for x in range(w):
                # Ray direction in camera coordinates
                ray_dir = np.array([
                    (x - cx) / fx,
                    (y - cy) / fy,
                    1.0
                ])
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                # If ray points downward, intersect with ground plane
                if ray_dir[1] > 0.1:  # Y is down in camera coords
                    # Distance to ground plane
                    t = camera_height / ray_dir[1]
                    depth = t * ray_dir[2]  # Z component
                    
                    if self.config.min_depth <= depth <= self.config.max_depth:
                        # Only update if current depth is default
                        if abs(depth_map[y, x] - self.config.default_depth) < 0.1:
                            depth_map[y, x] = depth
        
        return depth_map

    def _complete_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Complete sparse depth map using interpolation."""
        # Create mask of known depths (not default value)
        known_mask = np.abs(depth_map - self.config.default_depth) > 0.1
        
        if not np.any(known_mask):
            return depth_map
        
        # Convert to uint16 for OpenCV inpainting
        depth_normalized = (depth_map - self.config.min_depth) / (self.config.max_depth - self.config.min_depth)
        depth_uint16 = (depth_normalized * 65535).astype(np.uint16)
        
        # Inpaint unknown regions
        inpaint_mask = (~known_mask).astype(np.uint8) * 255
        
        try:
            # Use Telea inpainting
            inpainted = cv2.inpaint(
                depth_uint16.astype(np.float32),
                inpaint_mask,
                self.config.completion_kernel_size,
                cv2.INPAINT_TELEA,
            )
            
            # Convert back
            completed = (inpainted / 65535) * (self.config.max_depth - self.config.min_depth) + self.config.min_depth
            
            # Apply bilateral filter for smoothing
            completed = cv2.bilateralFilter(
                completed.astype(np.float32),
                self.config.completion_kernel_size,
                self.config.bilateral_sigma_color,
                self.config.bilateral_sigma_space,
            )
            
            return completed
            
        except cv2.error:
            return depth_map

    def get_last_depth_map(self) -> Optional[np.ndarray]:
        """Get the most recently computed depth map."""
        return self._last_depth_map


class OcclusionHandler:
    """
    Handles occlusion for AR overlays.
    
    Creates occlusion masks based on depth comparison between
    virtual objects and real-world depth.
    """
    
    def __init__(self, config: Optional[OcclusionConfig] = None):
        self.config = config or OcclusionConfig()
        self.depth_estimator = DepthEstimator(config)
        self.calibration: Optional[CalibrationData] = None

    def initialize(self, calibration: CalibrationData) -> bool:
        """Initialize occlusion handler."""
        self.calibration = calibration
        self.depth_estimator.initialize(calibration)
        LOGGER.info("OcclusionHandler initialized")
        return True

    def compute_occlusion_mask(
        self,
        frame: np.ndarray,
        object_depth: float,
        pose: PoseResult,
        sparse_points_3d: Optional[np.ndarray] = None,
        sparse_points_2d: Optional[np.ndarray] = None,
        object_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute occlusion mask for a virtual object at given depth.
        
        Args:
            frame: BGR frame
            object_depth: Depth of virtual object (meters)
            pose: Camera pose
            sparse_points_3d: 3D points for depth estimation
            sparse_points_2d: Corresponding 2D points
            object_mask: Optional mask of where virtual object appears
            
        Returns:
            Occlusion mask (0 = occluded, 255 = visible)
        """
        # Estimate scene depth
        scene_depth = self.depth_estimator.estimate_depth(
            frame, pose, sparse_points_3d, sparse_points_2d
        )
        
        # Compare depths
        # Object is visible where scene depth > object depth (object is closer)
        depth_diff = scene_depth - object_depth
        visible_mask = (depth_diff > -self.config.occlusion_threshold).astype(np.uint8) * 255
        
        # Apply edge softening
        if self.config.edge_softness > 0:
            visible_mask = cv2.GaussianBlur(
                visible_mask,
                (0, 0),
                self.config.edge_softness,
            )
        
        # Apply object mask if provided
        if object_mask is not None:
            visible_mask = cv2.bitwise_and(visible_mask, object_mask)
        
        return visible_mask

    def compute_depth_occlusion_mask(
        self,
        scene_depth: np.ndarray,
        object_depth_map: np.ndarray,
    ) -> np.ndarray:
        """
        Compute occlusion mask from two depth maps.
        
        Args:
            scene_depth: Depth map of the real scene
            object_depth_map: Depth map of virtual objects
            
        Returns:
            Occlusion mask (0 = occluded, 255 = visible)
        """
        # Object visible where it's closer than scene
        depth_diff = scene_depth - object_depth_map
        visible_mask = (depth_diff > -self.config.occlusion_threshold).astype(np.uint8) * 255
        
        # Soften edges
        if self.config.edge_softness > 0:
            visible_mask = cv2.GaussianBlur(
                visible_mask,
                (0, 0),
                self.config.edge_softness,
            )
        
        return visible_mask

    def apply_occlusion(
        self,
        background: np.ndarray,
        overlay: np.ndarray,
        occlusion_mask: np.ndarray,
        alpha: float = 1.0,
    ) -> np.ndarray:
        """
        Apply occlusion-aware blending.
        
        Args:
            background: Real camera frame
            overlay: Virtual overlay image
            occlusion_mask: Mask (0 = occluded, 255 = visible)
            alpha: Overall overlay opacity
            
        Returns:
            Composited result
        """
        # Normalize mask to 0-1
        mask_normalized = (occlusion_mask.astype(np.float32) / 255.0) * alpha
        
        # Handle grayscale mask
        if mask_normalized.ndim == 2:
            mask_normalized = mask_normalized[:, :, np.newaxis]
        
        # Blend
        result = (
            overlay.astype(np.float32) * mask_normalized +
            background.astype(np.float32) * (1 - mask_normalized)
        ).astype(np.uint8)
        
        return result

    def render_depth_map_visualization(
        self,
        depth_map: np.ndarray,
        colormap: int = cv2.COLORMAP_TURBO,
    ) -> np.ndarray:
        """
        Render depth map as colored visualization.
        
        Args:
            depth_map: Depth values in meters
            colormap: OpenCV colormap to use
            
        Returns:
            BGR visualization image
        """
        # Normalize to 0-255
        depth_normalized = (depth_map - self.config.min_depth) / (self.config.max_depth - self.config.min_depth)
        depth_normalized = np.clip(depth_normalized, 0, 1)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(depth_uint8, colormap)
        
        return colored


class DepthAwareOverlayRenderer:
    """
    Overlay renderer with depth-based occlusion.
    
    Extends basic overlay rendering with proper occlusion handling.
    """
    
    def __init__(self, config: Optional[OcclusionConfig] = None):
        self.config = config or OcclusionConfig()
        self.occlusion_handler = OcclusionHandler(config)
        self.calibration: Optional[CalibrationData] = None

    def initialize(self, calibration: CalibrationData) -> bool:
        """Initialize the renderer."""
        self.calibration = calibration
        self.occlusion_handler.initialize(calibration)
        return True

    def render_with_occlusion(
        self,
        frame: np.ndarray,
        overlay: np.ndarray,
        object_depth: float,
        pose: PoseResult,
        sparse_points_3d: Optional[np.ndarray] = None,
        sparse_points_2d: Optional[np.ndarray] = None,
        overlay_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render overlay with occlusion handling.
        
        Args:
            frame: Background camera frame
            overlay: Virtual overlay to composite
            object_depth: Depth of virtual object
            pose: Camera pose
            sparse_points_3d: 3D points for depth
            sparse_points_2d: Corresponding 2D points
            overlay_mask: Where overlay should appear
            
        Returns:
            Composited frame with occlusion
        """
        # Create overlay mask if not provided
        if overlay_mask is None:
            # Assume non-black pixels are the overlay
            overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            overlay_mask = (overlay_gray > 10).astype(np.uint8) * 255
        
        # Compute occlusion
        occlusion_mask = self.occlusion_handler.compute_occlusion_mask(
            frame,
            object_depth,
            pose,
            sparse_points_3d,
            sparse_points_2d,
            overlay_mask,
        )
        
        # Apply
        result = self.occlusion_handler.apply_occlusion(
            frame, overlay, occlusion_mask
        )
        
        return result

    def render_object_with_depth(
        self,
        frame: np.ndarray,
        object_vertices: np.ndarray,
        object_faces: np.ndarray,
        object_color: Tuple[int, int, int],
        pose: PoseResult,
        scene_depth: np.ndarray,
    ) -> np.ndarray:
        """
        Render a 3D object with per-face occlusion.
        
        Args:
            frame: Background frame
            object_vertices: Nx3 vertices
            object_faces: Mx3 face indices
            object_color: Base color
            pose: Camera pose
            scene_depth: Depth map of scene
            
        Returns:
            Frame with rendered object
        """
        if self.calibration is None or not pose.success:
            return frame
        
        if pose.rotation_vector is None or pose.translation_vector is None:
            return frame
        
        # Project vertices
        try:
            projected, _ = cv2.projectPoints(
                object_vertices,
                pose.rotation_vector,
                pose.translation_vector,
                self.calibration.camera_matrix,
                self.calibration.dist_coeffs,
            )
            pts_2d = projected.reshape(-1, 2)
        except cv2.error:
            return frame
        
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Sort faces by depth (back to front)
        face_depths = []
        for i, face in enumerate(object_faces):
            center_z = np.mean(object_vertices[face, 2])
            face_depths.append((i, center_z))
        face_depths.sort(key=lambda x: x[1], reverse=True)
        
        # Render each face with occlusion
        for face_idx, face_z in face_depths:
            face = object_faces[face_idx]
            triangle_pts = pts_2d[face].astype(np.int32)
            
            # Check bounds
            if np.any(triangle_pts < 0) or np.any(triangle_pts[:, 0] >= w) or np.any(triangle_pts[:, 1] >= h):
                continue
            
            # Create face mask
            face_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(face_mask, triangle_pts.reshape(-1, 1, 2), 255)
            
            # Get scene depth in face region
            face_pixels = np.where(face_mask > 0)
            if len(face_pixels[0]) == 0:
                continue
            
            avg_scene_depth = np.mean(scene_depth[face_pixels])
            
            # Check occlusion
            if face_z > avg_scene_depth + self.config.occlusion_threshold:
                # Object is behind scene - don't render
                continue
            
            # Render face
            cv2.fillConvexPoly(result, triangle_pts.reshape(-1, 1, 2), object_color)
        
        return result

