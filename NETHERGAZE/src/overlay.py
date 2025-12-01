"""
Virtual overlay rendering module.

This module handles rendering of virtual objects and overlays on camera frames
using OpenCV. Supports 2D annotations, wireframe 3D objects, textured meshes,
and compositing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from pose import CalibrationData, PoseResult

LOGGER = logging.getLogger(__name__)


@dataclass
class Overlay2D:
    """Definition of a 2D overlay element."""

    overlay_type: str  # "text", "rectangle", "circle", "line", "polygon"
    position: Tuple[int, int]  # Screen position (x, y)
    color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2
    # Type-specific parameters
    text: str = ""
    font_scale: float = 0.6
    size: Tuple[int, int] = (50, 50)  # For rectangles
    radius: int = 20  # For circles
    end_position: Optional[Tuple[int, int]] = None  # For lines
    points: Optional[np.ndarray] = None  # For polygons


@dataclass
class Object3D:
    """Definition of a 3D object to render."""

    object_type: str  # "cube", "axes", "pyramid", "grid", "custom", "mesh"
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 3D position
    scale: float = 0.05  # Size in meters
    color: Tuple[int, int, int] = (0, 255, 255)
    thickness: int = 2
    vertices: Optional[np.ndarray] = None  # Custom vertices (Nx3)
    edges: Optional[List[Tuple[int, int]]] = None  # Custom edge list
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Euler angles (degrees)


@dataclass
class Mesh3D:
    """
    3D mesh with optional texture for rendering.
    
    Supports:
    - Vertices and faces (triangles)
    - Per-vertex colors or texture mapping
    - OBJ file loading
    """
    
    vertices: np.ndarray  # Nx3 vertex positions
    faces: np.ndarray  # Mx3 face indices (triangles)
    
    # Optional texture data
    texture_coords: Optional[np.ndarray] = None  # Nx2 UV coordinates
    texture_image: Optional[np.ndarray] = None  # Texture image (BGR)
    
    # Per-vertex or per-face colors (if no texture)
    vertex_colors: Optional[np.ndarray] = None  # Nx3 BGR colors
    face_colors: Optional[np.ndarray] = None  # Mx3 BGR colors
    
    # Transform
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: float = 1.0
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Euler angles (degrees)
    
    # Rendering options
    render_mode: str = "solid"  # "solid", "wireframe", "textured"
    wireframe_color: Tuple[int, int, int] = (255, 255, 255)
    default_color: Tuple[int, int, int] = (128, 128, 128)

    @staticmethod
    def load_obj(filepath: str, texture_path: Optional[str] = None) -> "Mesh3D":
        """
        Load a mesh from an OBJ file.
        
        Args:
            filepath: Path to OBJ file
            texture_path: Optional path to texture image
            
        Returns:
            Loaded Mesh3D object
        """
        vertices = []
        faces = []
        texture_coords = []
        face_tex_indices = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                if parts[0] == 'v':
                    # Vertex position
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                
                elif parts[0] == 'vt':
                    # Texture coordinate
                    texture_coords.append([float(parts[1]), float(parts[2])])
                
                elif parts[0] == 'f':
                    # Face (can be "v", "v/vt", "v/vt/vn", "v//vn")
                    face_verts = []
                    face_texs = []
                    for i in range(1, len(parts)):
                        indices = parts[i].split('/')
                        face_verts.append(int(indices[0]) - 1)  # OBJ is 1-indexed
                        if len(indices) > 1 and indices[1]:
                            face_texs.append(int(indices[1]) - 1)
                    
                    # Triangulate if more than 3 vertices
                    for i in range(1, len(face_verts) - 1):
                        faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])
                        if face_texs:
                            face_tex_indices.append([face_texs[0], face_texs[i], face_texs[i + 1]])
        
        vertices_arr = np.array(vertices, dtype=np.float32)
        faces_arr = np.array(faces, dtype=np.int32)
        
        tex_coords_arr = None
        if texture_coords:
            tex_coords_arr = np.array(texture_coords, dtype=np.float32)
        
        texture_img = None
        if texture_path and os.path.exists(texture_path):
            texture_img = cv2.imread(texture_path)
        
        render_mode = "textured" if texture_img is not None else "solid"
        
        return Mesh3D(
            vertices=vertices_arr,
            faces=faces_arr,
            texture_coords=tex_coords_arr,
            texture_image=texture_img,
            render_mode=render_mode,
        )

    @staticmethod
    def create_box(
        width: float = 1.0,
        height: float = 1.0,
        depth: float = 1.0,
        color: Tuple[int, int, int] = (128, 128, 128),
    ) -> "Mesh3D":
        """Create a simple box mesh."""
        w, h, d = width / 2, height / 2, depth / 2
        
        vertices = np.array([
            [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],  # Back face
            [-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d],      # Front face
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Back
            [4, 6, 5], [4, 7, 6],  # Front
            [0, 4, 5], [0, 5, 1],  # Bottom
            [2, 6, 7], [2, 7, 3],  # Top
            [0, 3, 7], [0, 7, 4],  # Left
            [1, 5, 6], [1, 6, 2],  # Right
        ], dtype=np.int32)
        
        # Per-face colors (different shade per face for depth perception)
        face_colors = np.array([
            [color[0] * 0.8, color[1] * 0.8, color[2] * 0.8],  # Back (darker)
            [color[0] * 0.8, color[1] * 0.8, color[2] * 0.8],
            color, color,  # Front
            [color[0] * 0.6, color[1] * 0.6, color[2] * 0.6],  # Bottom
            [color[0] * 0.6, color[1] * 0.6, color[2] * 0.6],
            [color[0] * 1.0, color[1] * 1.0, color[2] * 1.0],  # Top (brightest)
            [color[0] * 1.0, color[1] * 1.0, color[2] * 1.0],
            [color[0] * 0.7, color[1] * 0.7, color[2] * 0.7],  # Left
            [color[0] * 0.7, color[1] * 0.7, color[2] * 0.7],
            [color[0] * 0.9, color[1] * 0.9, color[2] * 0.9],  # Right
            [color[0] * 0.9, color[1] * 0.9, color[2] * 0.9],
        ], dtype=np.uint8)
        
        return Mesh3D(
            vertices=vertices,
            faces=faces,
            face_colors=face_colors,
            default_color=color,
        )

    @staticmethod
    def create_plane(
        width: float = 1.0,
        height: float = 1.0,
        color: Tuple[int, int, int] = (128, 128, 128),
    ) -> "Mesh3D":
        """Create a simple plane mesh (for image projection)."""
        w, h = width / 2, height / 2
        
        vertices = np.array([
            [-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0],
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
        ], dtype=np.int32)
        
        texture_coords = np.array([
            [0, 1], [1, 1], [1, 0], [0, 0],
        ], dtype=np.float32)
        
        return Mesh3D(
            vertices=vertices,
            faces=faces,
            texture_coords=texture_coords,
            default_color=color,
        )

    def set_texture(self, image: np.ndarray):
        """Set texture image for the mesh."""
        self.texture_image = image
        self.render_mode = "textured"

    def get_transformed_vertices(self) -> np.ndarray:
        """Get vertices with position, rotation, and scale applied."""
        # Apply scale
        verts = self.vertices * self.scale
        
        # Apply rotation (Euler angles in degrees)
        if np.any(self.rotation != 0):
            rx, ry, rz = np.radians(self.rotation)
            
            # Rotation matrices
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)],
            ])
            Ry = np.array([
                [np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)],
            ])
            Rz = np.array([
                [np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz), np.cos(rz), 0],
                [0, 0, 1],
            ])
            
            R = Rz @ Ry @ Rx
            verts = (R @ verts.T).T
        
        # Apply translation
        verts = verts + self.position
        
        return verts.astype(np.float64)


@dataclass
class OverlayConfiguration:
    """Configuration for the overlay renderer."""

    enable_2d_overlays: bool = True
    enable_3d_overlays: bool = True
    enable_meshes: bool = True
    default_3d_color: Tuple[int, int, int] = (0, 255, 255)
    default_2d_color: Tuple[int, int, int] = (0, 255, 0)
    blend_alpha: float = 0.7
    antialiasing: bool = True
    
    # Mesh rendering options
    mesh_lighting: bool = True  # Simple lighting for solid meshes
    mesh_backface_culling: bool = True  # Hide back-facing triangles
    mesh_depth_sorting: bool = True  # Sort faces by depth


class OverlayRenderer:
    """
    Handles rendering of virtual overlays on camera frames.
    
    Supports:
    - 2D annotations (text, shapes) at screen coordinates
    - 3D wireframe objects projected using camera pose
    - Alpha blending of overlay layers
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize overlay renderer.
        
        Args:
            config: Configuration dictionary with overlay settings
        """
        cfg = config or {}
        self.config = OverlayConfiguration(
            enable_2d_overlays=cfg.get("enable_2d_overlays", True),
            enable_3d_overlays=cfg.get("enable_3d_overlays", True),
            default_3d_color=tuple(cfg.get("default_3d_color", (0, 255, 255))),
            default_2d_color=tuple(cfg.get("default_2d_color", (0, 255, 0))),
            blend_alpha=cfg.get("blend_alpha", 0.7),
            antialiasing=cfg.get("antialiasing", True),
        )

        self.calibration: Optional[CalibrationData] = None
        self.overlays_2d: List[Overlay2D] = []
        self.objects_3d: List[Object3D] = []
        self.meshes: List[Mesh3D] = []
        self.initialized = False

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def initialize(self, calibration: Optional[CalibrationData] = None) -> bool:
        """Initialize overlay rendering.
        
        Args:
            calibration: Camera calibration data for 3D projections
            
        Returns:
            True if initialization successful
        """
        self.calibration = calibration
        self.initialized = True
        LOGGER.info("Overlay renderer initialized")
        return True

    def set_calibration(self, calibration: CalibrationData):
        """Update camera calibration data."""
        self.calibration = calibration

    # ------------------------------------------------------------------ #
    # 2D Overlay Management
    # ------------------------------------------------------------------ #
    def add_2d_overlay(self, overlay: Overlay2D):
        """Add a 2D overlay element."""
        self.overlays_2d.append(overlay)

    def clear_2d_overlays(self):
        """Remove all 2D overlays."""
        self.overlays_2d.clear()

    def add_text(
        self,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = None,
        font_scale: float = 0.6,
        thickness: int = 2,
    ):
        """Convenience method to add a text overlay."""
        self.overlays_2d.append(
            Overlay2D(
                overlay_type="text",
                position=position,
                color=color or self.config.default_2d_color,
                text=text,
                font_scale=font_scale,
                thickness=thickness,
            )
        )

    # ------------------------------------------------------------------ #
    # 3D Object Management
    # ------------------------------------------------------------------ #
    def add_3d_object(self, obj: Object3D):
        """Add a 3D object to render."""
        self.objects_3d.append(obj)

    def clear_3d_objects(self):
        """Remove all 3D objects."""
        self.objects_3d.clear()

    def add_cube(
        self,
        position: np.ndarray = None,
        scale: float = 0.05,
        color: Tuple[int, int, int] = None,
    ):
        """Add a wireframe cube at the specified position."""
        self.objects_3d.append(
            Object3D(
                object_type="cube",
                position=position if position is not None else np.zeros(3),
                scale=scale,
                color=color or self.config.default_3d_color,
            )
        )

    def add_pyramid(
        self,
        position: np.ndarray = None,
        scale: float = 0.05,
        color: Tuple[int, int, int] = None,
    ):
        """Add a wireframe pyramid at the specified position."""
        self.objects_3d.append(
            Object3D(
                object_type="pyramid",
                position=position if position is not None else np.zeros(3),
                scale=scale,
                color=color or self.config.default_3d_color,
            )
        )

    def add_grid(
        self,
        position: np.ndarray = None,
        scale: float = 0.1,
        color: Tuple[int, int, int] = None,
    ):
        """Add a ground plane grid at the specified position."""
        self.objects_3d.append(
            Object3D(
                object_type="grid",
                position=position if position is not None else np.zeros(3),
                scale=scale,
                color=color or (100, 100, 100),
                thickness=1,
            )
        )

    # ------------------------------------------------------------------ #
    # 3D Mesh Management
    # ------------------------------------------------------------------ #
    def add_mesh(self, mesh: Mesh3D):
        """Add a 3D mesh to render."""
        self.meshes.append(mesh)

    def clear_meshes(self):
        """Remove all meshes."""
        self.meshes.clear()

    def load_mesh(
        self,
        filepath: str,
        texture_path: Optional[str] = None,
        position: np.ndarray = None,
        scale: float = 1.0,
        rotation: np.ndarray = None,
    ) -> Mesh3D:
        """
        Load and add a mesh from an OBJ file.
        
        Args:
            filepath: Path to OBJ file
            texture_path: Optional path to texture image
            position: 3D position
            scale: Scale factor
            rotation: Euler angles in degrees (x, y, z)
            
        Returns:
            The loaded mesh
        """
        mesh = Mesh3D.load_obj(filepath, texture_path)
        mesh.position = position if position is not None else np.zeros(3)
        mesh.scale = scale
        mesh.rotation = rotation if rotation is not None else np.zeros(3)
        self.meshes.append(mesh)
        return mesh

    def add_textured_plane(
        self,
        image: np.ndarray,
        position: np.ndarray = None,
        scale: float = 0.1,
        rotation: np.ndarray = None,
    ) -> Mesh3D:
        """
        Add a textured plane (billboard) to the scene.
        
        Args:
            image: Texture image (BGR)
            position: 3D position
            scale: Size in meters
            rotation: Euler angles in degrees
            
        Returns:
            The created mesh
        """
        # Create plane with aspect ratio matching image
        h, w = image.shape[:2]
        aspect = w / h
        mesh = Mesh3D.create_plane(width=scale * aspect, height=scale)
        mesh.set_texture(image)
        mesh.position = position if position is not None else np.zeros(3)
        mesh.rotation = rotation if rotation is not None else np.zeros(3)
        self.meshes.append(mesh)
        return mesh

    def add_solid_box(
        self,
        position: np.ndarray = None,
        scale: float = 0.05,
        color: Tuple[int, int, int] = (128, 128, 128),
        rotation: np.ndarray = None,
    ) -> Mesh3D:
        """
        Add a solid (shaded) box to the scene.
        
        Args:
            position: 3D position
            scale: Size in meters
            color: Base color (BGR)
            rotation: Euler angles in degrees
            
        Returns:
            The created mesh
        """
        mesh = Mesh3D.create_box(width=scale, height=scale, depth=scale, color=color)
        mesh.position = position if position is not None else np.zeros(3)
        mesh.scale = 1.0  # Already scaled in create_box
        mesh.rotation = rotation if rotation is not None else np.zeros(3)
        self.meshes.append(mesh)
        return mesh

    # ------------------------------------------------------------------ #
    # Main Rendering API
    # ------------------------------------------------------------------ #
    def render(self, frame: np.ndarray, pose: Optional[PoseResult] = None) -> np.ndarray:
        """Render all overlays onto the frame.
        
        Args:
            frame: Input BGR frame
            pose: Camera pose for 3D object projection
            
        Returns:
            Frame with overlays rendered
        """
        if frame is None:
            return frame

        # Create overlay layer for blending
        overlay_layer = frame.copy()

        # Render meshes first (back to front)
        if self.config.enable_meshes and pose is not None and pose.success:
            overlay_layer = self._render_meshes(overlay_layer, pose)

        # Render 3D wireframe objects
        if self.config.enable_3d_overlays and pose is not None and pose.success:
            overlay_layer = self._render_3d_objects(overlay_layer, pose)

        # Render 2D overlays on top
        if self.config.enable_2d_overlays:
            overlay_layer = self._render_2d_overlays(overlay_layer)

        return overlay_layer

    def render_with_blend(
        self,
        frame: np.ndarray,
        pose: Optional[PoseResult] = None,
        alpha: float = None,
    ) -> np.ndarray:
        """Render overlays with alpha blending to background.
        
        Args:
            frame: Input BGR frame
            pose: Camera pose for 3D projection
            alpha: Blend factor (0.0-1.0), uses config default if None
            
        Returns:
            Blended output frame
        """
        if frame is None:
            return frame

        alpha = alpha if alpha is not None else self.config.blend_alpha
        overlay_layer = np.zeros_like(frame)

        # Render onto transparent layer
        if self.config.enable_3d_overlays and pose is not None and pose.success:
            overlay_layer = self._render_3d_objects(overlay_layer, pose)

        if self.config.enable_2d_overlays:
            overlay_layer = self._render_2d_overlays(overlay_layer)

        # Blend layers
        return self.blend_layers(frame, overlay_layer, alpha)

    # ------------------------------------------------------------------ #
    # 2D Rendering Implementation
    # ------------------------------------------------------------------ #
    def _render_2d_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Render all 2D overlay elements onto the frame."""
        line_type = cv2.LINE_AA if self.config.antialiasing else cv2.LINE_8

        for overlay in self.overlays_2d:
            try:
                if overlay.overlay_type == "text":
                    cv2.putText(
                        frame,
                        overlay.text,
                        overlay.position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        overlay.font_scale,
                        overlay.color,
                        overlay.thickness,
                        line_type,
                    )

                elif overlay.overlay_type == "rectangle":
                    x, y = overlay.position
                    w, h = overlay.size
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        overlay.color,
                        overlay.thickness,
                        line_type,
                    )

                elif overlay.overlay_type == "circle":
                    cv2.circle(
                        frame,
                        overlay.position,
                        overlay.radius,
                        overlay.color,
                        overlay.thickness,
                        line_type,
                    )

                elif overlay.overlay_type == "line" and overlay.end_position:
                    cv2.line(
                        frame,
                        overlay.position,
                        overlay.end_position,
                        overlay.color,
                        overlay.thickness,
                        line_type,
                    )

                elif overlay.overlay_type == "polygon" and overlay.points is not None:
                    pts = overlay.points.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(
                        frame,
                        [pts],
                        isClosed=True,
                        color=overlay.color,
                        thickness=overlay.thickness,
                        lineType=line_type,
                    )

            except Exception as e:
                LOGGER.warning("Failed to render 2D overlay: %s", e)

        return frame

    # ------------------------------------------------------------------ #
    # 3D Rendering Implementation
    # ------------------------------------------------------------------ #
    def _render_3d_objects(self, frame: np.ndarray, pose: PoseResult) -> np.ndarray:
        """Render all 3D objects projected onto the frame."""
        if self.calibration is None:
            LOGGER.warning("Cannot render 3D objects without calibration data")
            return frame

        for obj in self.objects_3d:
            try:
                if obj.object_type == "cube":
                    frame = self._render_cube(frame, obj, pose)
                elif obj.object_type == "pyramid":
                    frame = self._render_pyramid(frame, obj, pose)
                elif obj.object_type == "grid":
                    frame = self._render_grid(frame, obj, pose)
                elif obj.object_type == "axes":
                    frame = self._render_axes(frame, obj, pose)
                elif obj.object_type == "custom" and obj.vertices is not None:
                    frame = self._render_custom(frame, obj, pose)
            except Exception as e:
                LOGGER.warning("Failed to render 3D object '%s': %s", obj.object_type, e)

        return frame

    def _project_points(
        self, points_3d: np.ndarray, pose: PoseResult
    ) -> Optional[np.ndarray]:
        """Project 3D points to 2D image coordinates."""
        if pose.rotation_vector is None or pose.translation_vector is None:
            return None

        try:
            image_points, _ = cv2.projectPoints(
                points_3d.astype(np.float64),
                pose.rotation_vector,
                pose.translation_vector,
                self.calibration.camera_matrix,
                self.calibration.dist_coeffs,
            )
            return image_points.reshape(-1, 2).astype(np.int32)
        except cv2.error as e:
            LOGGER.debug("Projection failed: %s", e)
            return None

    def _render_cube(self, frame: np.ndarray, obj: Object3D, pose: PoseResult) -> np.ndarray:
        """Render a wireframe cube."""
        s = obj.scale / 2
        p = obj.position

        # 8 vertices of a cube centered at position
        vertices = np.array([
            [p[0] - s, p[1] - s, p[2] - s],
            [p[0] + s, p[1] - s, p[2] - s],
            [p[0] + s, p[1] + s, p[2] - s],
            [p[0] - s, p[1] + s, p[2] - s],
            [p[0] - s, p[1] - s, p[2] + s],
            [p[0] + s, p[1] - s, p[2] + s],
            [p[0] + s, p[1] + s, p[2] + s],
            [p[0] - s, p[1] + s, p[2] + s],
        ], dtype=np.float64)

        # 12 edges of a cube
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
            (0, 4), (1, 5), (2, 6), (3, 7),  # Connecting edges
        ]

        return self._render_wireframe(frame, vertices, edges, obj.color, obj.thickness)

    def _render_pyramid(self, frame: np.ndarray, obj: Object3D, pose: PoseResult) -> np.ndarray:
        """Render a wireframe pyramid."""
        s = obj.scale / 2
        h = obj.scale
        p = obj.position

        # 5 vertices: 4 base corners + apex
        vertices = np.array([
            [p[0] - s, p[1] - s, p[2]],      # Base corner 0
            [p[0] + s, p[1] - s, p[2]],      # Base corner 1
            [p[0] + s, p[1] + s, p[2]],      # Base corner 2
            [p[0] - s, p[1] + s, p[2]],      # Base corner 3
            [p[0], p[1], p[2] + h],          # Apex
        ], dtype=np.float64)

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Base
            (0, 4), (1, 4), (2, 4), (3, 4),  # Sides to apex
        ]

        return self._render_wireframe(frame, vertices, edges, obj.color, obj.thickness)

    def _render_grid(self, frame: np.ndarray, obj: Object3D, pose: PoseResult) -> np.ndarray:
        """Render a ground plane grid."""
        s = obj.scale
        p = obj.position
        divisions = 5

        vertices = []
        edges = []
        idx = 0

        # Create grid lines
        for i in range(-divisions, divisions + 1):
            offset = (i / divisions) * s
            # Lines parallel to X
            vertices.append([p[0] - s, p[1] + offset, p[2]])
            vertices.append([p[0] + s, p[1] + offset, p[2]])
            edges.append((idx, idx + 1))
            idx += 2
            # Lines parallel to Y
            vertices.append([p[0] + offset, p[1] - s, p[2]])
            vertices.append([p[0] + offset, p[1] + s, p[2]])
            edges.append((idx, idx + 1))
            idx += 2

        vertices = np.array(vertices, dtype=np.float64)
        return self._render_wireframe(frame, vertices, edges, obj.color, obj.thickness)

    def _render_axes(self, frame: np.ndarray, obj: Object3D, pose: PoseResult) -> np.ndarray:
        """Render coordinate axes (RGB = XYZ)."""
        s = obj.scale
        p = obj.position

        vertices = np.array([
            [p[0], p[1], p[2]],          # Origin
            [p[0] + s, p[1], p[2]],      # X
            [p[0], p[1] + s, p[2]],      # Y
            [p[0], p[1], p[2] + s],      # Z
        ], dtype=np.float64)

        pts_2d = self._project_points(vertices, pose)
        if pts_2d is None or len(pts_2d) < 4:
            return frame

        origin = tuple(pts_2d[0])
        line_type = cv2.LINE_AA if self.config.antialiasing else cv2.LINE_8

        cv2.line(frame, origin, tuple(pts_2d[1]), (0, 0, 255), obj.thickness, line_type)  # X red
        cv2.line(frame, origin, tuple(pts_2d[2]), (0, 255, 0), obj.thickness, line_type)  # Y green
        cv2.line(frame, origin, tuple(pts_2d[3]), (255, 0, 0), obj.thickness, line_type)  # Z blue

        return frame

    def _render_custom(self, frame: np.ndarray, obj: Object3D, pose: PoseResult) -> np.ndarray:
        """Render a custom wireframe object."""
        if obj.vertices is None:
            return frame

        # Apply position offset and scale
        vertices = obj.vertices.astype(np.float64) * obj.scale + obj.position
        edges = obj.edges or []

        return self._render_wireframe(frame, vertices, edges, obj.color, obj.thickness)

    def _render_wireframe(
        self,
        frame: np.ndarray,
        vertices: np.ndarray,
        edges: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        thickness: int,
    ) -> np.ndarray:
        """Render a wireframe from vertices and edge list."""
        # Create a dummy pose at origin for projection
        # The vertices should already be in world coordinates
        from pose import PoseResult

        # We need to project using the current pose
        # But vertices are world coords, so we pass them directly to projectPoints
        if self.calibration is None:
            return frame

        try:
            # Project all vertices
            image_points, _ = cv2.projectPoints(
                vertices.astype(np.float64),
                np.zeros(3),  # No additional rotation
                np.zeros(3),  # No additional translation
                self.calibration.camera_matrix,
                self.calibration.dist_coeffs,
            )
            pts_2d = image_points.reshape(-1, 2).astype(np.int32)
        except cv2.error:
            return frame

        line_type = cv2.LINE_AA if self.config.antialiasing else cv2.LINE_8

        # Draw all edges
        for i, j in edges:
            if 0 <= i < len(pts_2d) and 0 <= j < len(pts_2d):
                pt1 = tuple(pts_2d[i])
                pt2 = tuple(pts_2d[j])
                cv2.line(frame, pt1, pt2, color, thickness, line_type)

        return frame

    # ------------------------------------------------------------------ #
    # Blending and Compositing
    # ------------------------------------------------------------------ #
    def blend_layers(
        self,
        background: np.ndarray,
        overlay: np.ndarray,
        alpha: float = 0.7,
    ) -> np.ndarray:
        """Blend overlay layer onto background.
        
        Args:
            background: Background frame
            overlay: Overlay frame (non-zero pixels are blended)
            alpha: Blend factor for overlay (0.0-1.0)
            
        Returns:
            Blended result
        """
        if background is None:
            return overlay
        if overlay is None:
            return background

        # Create mask from non-zero overlay pixels
        gray_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        mask = gray_overlay > 0

        # Blend where mask is true
        result = background.copy()
        result[mask] = cv2.addWeighted(
            background[mask].astype(np.float32),
            1 - alpha,
            overlay[mask].astype(np.float32),
            alpha,
            0,
        ).astype(np.uint8)

        return result

    def project_overlay(
        self,
        overlay_image: np.ndarray,
        frame: np.ndarray,
        corners_3d: np.ndarray,
        pose: PoseResult,
    ) -> np.ndarray:
        """Project a 2D overlay image onto a 3D surface.
        
        Args:
            overlay_image: Image to project
            frame: Target frame
            corners_3d: 4 corner points in 3D (4x3 array)
            pose: Camera pose
            
        Returns:
            Frame with projected overlay
        """
        if self.calibration is None or not pose.success:
            return frame

        # Project 3D corners to 2D
        pts_2d = self._project_points(corners_3d, pose)
        if pts_2d is None or len(pts_2d) < 4:
            return frame

        # Source corners (overlay image corners)
        h, w = overlay_image.shape[:2]
        src_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        # Compute homography and warp
        try:
            H, _ = cv2.findHomography(src_corners, pts_2d.astype(np.float32))
            if H is None:
                return frame

            warped = cv2.warpPerspective(overlay_image, H, (frame.shape[1], frame.shape[0]))

            # Create mask for blending
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts_2d, 255)

            # Blend warped overlay onto frame
            mask_3ch = cv2.merge([mask, mask, mask])
            result = np.where(mask_3ch > 0, warped, frame)

            return result

        except cv2.error as e:
            LOGGER.warning("Overlay projection failed: %s", e)
            return frame

    # ------------------------------------------------------------------ #
    # Mesh Rendering
    # ------------------------------------------------------------------ #
    def _render_meshes(self, frame: np.ndarray, pose: PoseResult) -> np.ndarray:
        """Render all meshes onto the frame."""
        if self.calibration is None:
            return frame

        for mesh in self.meshes:
            try:
                if mesh.render_mode == "wireframe":
                    frame = self._render_mesh_wireframe(frame, mesh, pose)
                elif mesh.render_mode == "textured" and mesh.texture_image is not None:
                    frame = self._render_mesh_textured(frame, mesh, pose)
                else:
                    frame = self._render_mesh_solid(frame, mesh, pose)
            except Exception as e:
                LOGGER.warning("Failed to render mesh: %s", e)

        return frame

    def _render_mesh_wireframe(
        self,
        frame: np.ndarray,
        mesh: Mesh3D,
        pose: PoseResult,
    ) -> np.ndarray:
        """Render mesh as wireframe."""
        vertices = mesh.get_transformed_vertices()
        pts_2d = self._project_points(vertices, pose)
        
        if pts_2d is None:
            return frame

        line_type = cv2.LINE_AA if self.config.antialiasing else cv2.LINE_8
        color = mesh.wireframe_color

        for face in mesh.faces:
            for i in range(3):
                j = (i + 1) % 3
                pt1 = tuple(pts_2d[face[i]].astype(int))
                pt2 = tuple(pts_2d[face[j]].astype(int))
                cv2.line(frame, pt1, pt2, color, 1, line_type)

        return frame

    def _render_mesh_solid(
        self,
        frame: np.ndarray,
        mesh: Mesh3D,
        pose: PoseResult,
    ) -> np.ndarray:
        """Render mesh with solid colored faces."""
        vertices = mesh.get_transformed_vertices()
        pts_2d = self._project_points(vertices, pose)
        
        if pts_2d is None:
            return frame

        # Get face depths for sorting (painter's algorithm)
        face_depths = []
        for i, face in enumerate(mesh.faces):
            center_z = np.mean(vertices[face, 2])
            face_depths.append((i, center_z))

        # Sort faces back to front (larger Z first in camera coords)
        if self.config.mesh_depth_sorting:
            face_depths.sort(key=lambda x: x[1], reverse=True)

        for face_idx, _ in face_depths:
            face = mesh.faces[face_idx]
            triangle_pts = pts_2d[face].astype(np.int32)
            
            # Backface culling
            if self.config.mesh_backface_culling:
                v0, v1, v2 = vertices[face]
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                # Check if face is pointing towards camera (negative Z)
                if normal[2] > 0:
                    continue

            # Get color
            if mesh.face_colors is not None and face_idx < len(mesh.face_colors):
                color = tuple(int(c) for c in mesh.face_colors[face_idx])
            else:
                color = mesh.default_color

            # Apply simple lighting if enabled
            if self.config.mesh_lighting:
                v0, v1, v2 = vertices[face]
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                normal = normal / (np.linalg.norm(normal) + 1e-8)
                
                # Light direction (from camera)
                light_dir = np.array([0, 0, -1])
                intensity = max(0.3, abs(np.dot(normal, light_dir)))
                
                color = tuple(int(c * intensity) for c in color)

            # Draw filled triangle
            cv2.fillConvexPoly(frame, triangle_pts.reshape(-1, 1, 2), color)

        return frame

    def _render_mesh_textured(
        self,
        frame: np.ndarray,
        mesh: Mesh3D,
        pose: PoseResult,
    ) -> np.ndarray:
        """Render mesh with texture mapping."""
        if mesh.texture_image is None or mesh.texture_coords is None:
            return self._render_mesh_solid(frame, mesh, pose)

        vertices = mesh.get_transformed_vertices()
        pts_2d = self._project_points(vertices, pose)
        
        if pts_2d is None:
            return frame

        tex_h, tex_w = mesh.texture_image.shape[:2]
        
        # Get face depths for sorting
        face_depths = []
        for i, face in enumerate(mesh.faces):
            center_z = np.mean(vertices[face, 2])
            face_depths.append((i, center_z))

        if self.config.mesh_depth_sorting:
            face_depths.sort(key=lambda x: x[1], reverse=True)

        for face_idx, _ in face_depths:
            face = mesh.faces[face_idx]
            
            # Backface culling
            if self.config.mesh_backface_culling:
                v0, v1, v2 = vertices[face]
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                if normal[2] > 0:
                    continue

            # Get projected vertices
            dst_pts = pts_2d[face].astype(np.float32)
            
            # Get texture coordinates
            tex_indices = face  # Assume same indices for simplicity
            if len(mesh.texture_coords) > max(tex_indices):
                src_pts = mesh.texture_coords[tex_indices].copy()
                src_pts[:, 0] *= tex_w
                src_pts[:, 1] = (1 - src_pts[:, 1]) * tex_h  # Flip Y
                src_pts = src_pts.astype(np.float32)
            else:
                continue

            # Compute affine transform and warp texture
            try:
                M = cv2.getAffineTransform(src_pts, dst_pts)
                warped = cv2.warpAffine(
                    mesh.texture_image,
                    M,
                    (frame.shape[1], frame.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                
                # Create mask for the triangle
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(mask, dst_pts.astype(np.int32).reshape(-1, 1, 2), 255)
                
                # Blend
                frame = np.where(mask[:, :, np.newaxis] > 0, warped, frame)
                
            except cv2.error:
                continue

        return frame

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def cleanup(self):
        """Clean up renderer resources."""
        self.clear_2d_overlays()
        self.clear_3d_objects()
        self.clear_meshes()
        self.initialized = False
        LOGGER.info("Overlay renderer cleaned up")
