"""
SLAM-lite mapping module for NETHERGAZE.

Provides lightweight 3D mapping capabilities:
- Sparse 3D point cloud from triangulated features
- Keyframe-based map management
- Loop closure detection (basic)
- Map persistence (save/load)

This is a simplified SLAM implementation suitable for small-scale AR applications.
For production SLAM, consider integrating ORB-SLAM3 or OpenVSLAM.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from pose import CalibrationData, PoseResult
from tracking.feature import TrackingFrameResult, Keyframe

LOGGER = logging.getLogger(__name__)


@dataclass
class MapPoint:
    """A 3D point in the map."""
    
    id: int
    position: np.ndarray  # 3D position (x, y, z)
    descriptor: Optional[np.ndarray] = None  # Feature descriptor
    observations: int = 1  # Number of times observed
    color: Optional[np.ndarray] = None  # RGB color
    normal: Optional[np.ndarray] = None  # Surface normal estimate
    
    # Quality metrics
    reprojection_error: float = 0.0
    last_seen_frame: int = 0
    created_frame: int = 0
    
    def merge(self, other: "MapPoint", alpha: float = 0.5):
        """Merge with another observation of the same point."""
        self.position = alpha * other.position + (1 - alpha) * self.position
        self.observations += 1
        if other.color is not None:
            if self.color is None:
                self.color = other.color
            else:
                self.color = (alpha * other.color + (1 - alpha) * self.color).astype(np.uint8)


@dataclass
class MapKeyframe:
    """A keyframe in the map with associated pose and observations."""
    
    id: int
    frame_index: int
    pose: PoseResult
    
    # Feature data
    keypoints: np.ndarray  # Nx2 keypoint positions
    descriptors: np.ndarray  # NxD descriptors
    
    # Map point associations (index -> map_point_id)
    point_associations: Dict[int, int] = field(default_factory=dict)
    
    # Connections to other keyframes
    covisible_keyframes: Set[int] = field(default_factory=set)
    
    # Metadata
    timestamp: float = 0.0
    quality_score: float = 0.0


@dataclass
class LoopCandidate:
    """A potential loop closure candidate."""
    
    query_keyframe_id: int
    match_keyframe_id: int
    num_matches: int
    geometric_score: float = 0.0
    verified: bool = False


@dataclass
class MapConfig:
    """Configuration for the mapping system."""
    
    # Point cloud management
    min_observations: int = 2  # Minimum observations to keep a point
    max_reprojection_error: float = 5.0  # pixels
    point_culling_interval: int = 20  # frames between culling
    
    # Keyframe selection
    min_keyframe_interval: int = 10  # Minimum frames between keyframes
    min_keyframe_parallax: float = 0.1  # Minimum baseline for triangulation
    max_keyframes: int = 50  # Maximum keyframes to keep
    
    # Loop closure
    enable_loop_closure: bool = True
    min_loop_matches: int = 30  # Minimum matches for loop candidate
    loop_closure_interval: int = 30  # Frames between loop closure checks
    
    # Triangulation
    triangulation_threshold: float = 0.9999  # Cosine threshold for triangulation
    min_triangulation_angle: float = 1.0  # Minimum angle in degrees
    
    # Bundle adjustment (simplified)
    enable_local_ba: bool = True
    local_ba_window: int = 5  # Number of keyframes for local BA


class SparseMap:
    """
    Sparse 3D map built from tracked features.
    
    Maintains:
    - 3D point cloud from triangulation
    - Keyframe poses and observations
    - Covisibility graph between keyframes
    """
    
    def __init__(self, config: Optional[MapConfig] = None):
        self.config = config or MapConfig()
        
        # Map data
        self.points: Dict[int, MapPoint] = {}
        self.keyframes: Dict[int, MapKeyframe] = {}
        
        # ID counters
        self._next_point_id: int = 0
        self._next_keyframe_id: int = 0
        
        # Current state
        self.current_keyframe_id: Optional[int] = None
        self.frame_count: int = 0
        self.last_keyframe_frame: int = -100
        
        # Calibration
        self.calibration: Optional[CalibrationData] = None
        
        # Loop closure
        self.loop_candidates: List[LoopCandidate] = []
        self.loop_closures: List[Tuple[int, int]] = []  # (kf1_id, kf2_id)
        
        # Statistics
        self.stats = {
            "total_points_created": 0,
            "total_points_culled": 0,
            "total_keyframes": 0,
            "loop_closures_found": 0,
        }

    def initialize(self, calibration: CalibrationData) -> bool:
        """Initialize the map with camera calibration."""
        self.calibration = calibration
        LOGGER.info("SparseMap initialized")
        return True

    def reset(self):
        """Reset the map to initial state."""
        self.points.clear()
        self.keyframes.clear()
        self._next_point_id = 0
        self._next_keyframe_id = 0
        self.current_keyframe_id = None
        self.frame_count = 0
        self.last_keyframe_frame = -100
        self.loop_candidates.clear()
        self.loop_closures.clear()
        LOGGER.info("SparseMap reset")

    # ------------------------------------------------------------------ #
    # Map Building
    # ------------------------------------------------------------------ #
    def process_frame(
        self,
        tracking_result: TrackingFrameResult,
        pose: PoseResult,
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Process a new frame and update the map.
        
        Args:
            tracking_result: Feature tracking result
            pose: Estimated camera pose
            frame: Optional BGR frame for color extraction
            
        Returns:
            True if a new keyframe was added
        """
        self.frame_count += 1
        
        if not pose.success or tracking_result.keypoints is None:
            return False
        
        # Check if we should add a keyframe
        should_add_kf = self._should_add_keyframe(tracking_result, pose)
        
        if should_add_kf:
            kf_id = self._add_keyframe(tracking_result, pose, frame)
            
            # Triangulate new points
            if self.current_keyframe_id is not None and kf_id != self.current_keyframe_id:
                self._triangulate_new_points(kf_id)
            
            self.current_keyframe_id = kf_id
            self.last_keyframe_frame = self.frame_count
            
            # Periodic maintenance
            if self.frame_count % self.config.point_culling_interval == 0:
                self._cull_points()
            
            # Loop closure detection
            if (self.config.enable_loop_closure and 
                self.frame_count % self.config.loop_closure_interval == 0):
                self._detect_loop_closure(kf_id)
            
            return True
        
        return False

    def _should_add_keyframe(
        self,
        tracking_result: TrackingFrameResult,
        pose: PoseResult,
    ) -> bool:
        """Determine if current frame should be a keyframe."""
        # Always add first keyframe
        if not self.keyframes:
            return True
        
        # Check minimum interval
        if self.frame_count - self.last_keyframe_frame < self.config.min_keyframe_interval:
            return False
        
        # Check if we have enough features
        if tracking_result.tracked_count < 50:
            return False
        
        # Check parallax to last keyframe
        if self.current_keyframe_id is not None and pose.translation_vector is not None:
            last_kf = self.keyframes.get(self.current_keyframe_id)
            if last_kf and last_kf.pose.translation_vector is not None:
                baseline = np.linalg.norm(
                    pose.translation_vector - last_kf.pose.translation_vector
                )
                if baseline < self.config.min_keyframe_parallax:
                    return False
        
        return True

    def _add_keyframe(
        self,
        tracking_result: TrackingFrameResult,
        pose: PoseResult,
        frame: Optional[np.ndarray] = None,
    ) -> int:
        """Add a new keyframe to the map."""
        kf_id = self._next_keyframe_id
        self._next_keyframe_id += 1
        
        keyframe = MapKeyframe(
            id=kf_id,
            frame_index=self.frame_count,
            pose=pose.copy(),
            keypoints=tracking_result.keypoints.copy(),
            descriptors=tracking_result.descriptors.copy() if tracking_result.descriptors is not None else np.array([]),
            timestamp=time.time(),
            quality_score=tracking_result.tracked_count / 1000.0,
        )
        
        self.keyframes[kf_id] = keyframe
        self.stats["total_keyframes"] += 1
        
        # Enforce max keyframes
        if len(self.keyframes) > self.config.max_keyframes:
            self._remove_oldest_keyframe()
        
        LOGGER.debug("Added keyframe %d with %d features", kf_id, len(keyframe.keypoints))
        return kf_id

    def _triangulate_new_points(self, new_kf_id: int):
        """Triangulate new 3D points between keyframes."""
        if self.calibration is None:
            return
        
        new_kf = self.keyframes.get(new_kf_id)
        if new_kf is None or new_kf.descriptors is None or len(new_kf.descriptors) == 0:
            return
        
        # Find best reference keyframe
        ref_kf = self._find_best_reference_keyframe(new_kf_id)
        if ref_kf is None:
            return
        
        # Match features between keyframes
        matches = self._match_keyframes(ref_kf, new_kf)
        if len(matches) < 10:
            return
        
        # Get matched points
        pts1 = ref_kf.keypoints[[m[0] for m in matches]]
        pts2 = new_kf.keypoints[[m[1] for m in matches]]
        
        # Build projection matrices
        P1 = self._build_projection_matrix(ref_kf.pose)
        P2 = self._build_projection_matrix(new_kf.pose)
        
        if P1 is None or P2 is None:
            return
        
        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = (points_4d[:3] / points_4d[3]).T
        
        # Add valid points to map
        num_added = 0
        for i, (pt3d, match) in enumerate(zip(points_3d, matches)):
            # Check point is in front of both cameras
            if pt3d[2] <= 0:
                continue
            
            # Check reprojection error
            err1 = self._reprojection_error(pt3d, pts1[i], ref_kf.pose)
            err2 = self._reprojection_error(pt3d, pts2[i], new_kf.pose)
            
            if err1 > self.config.max_reprojection_error or err2 > self.config.max_reprojection_error:
                continue
            
            # Create map point
            point = MapPoint(
                id=self._next_point_id,
                position=pt3d.astype(np.float64),
                descriptor=new_kf.descriptors[match[1]].copy() if new_kf.descriptors is not None else None,
                observations=2,
                reprojection_error=(err1 + err2) / 2,
                created_frame=self.frame_count,
                last_seen_frame=self.frame_count,
            )
            
            self.points[point.id] = point
            self._next_point_id += 1
            num_added += 1
            
            # Update keyframe associations
            ref_kf.point_associations[match[0]] = point.id
            new_kf.point_associations[match[1]] = point.id
        
        # Update covisibility
        if num_added > 10:
            ref_kf.covisible_keyframes.add(new_kf_id)
            new_kf.covisible_keyframes.add(ref_kf.id)
        
        self.stats["total_points_created"] += num_added
        LOGGER.debug("Triangulated %d new points between KF %d and KF %d", 
                    num_added, ref_kf.id, new_kf_id)

    def _find_best_reference_keyframe(self, query_kf_id: int) -> Optional[MapKeyframe]:
        """Find the best keyframe for triangulation with query."""
        query_kf = self.keyframes.get(query_kf_id)
        if query_kf is None:
            return None
        
        best_kf = None
        best_score = 0
        
        for kf_id, kf in self.keyframes.items():
            if kf_id == query_kf_id:
                continue
            
            # Check baseline
            if query_kf.pose.translation_vector is None or kf.pose.translation_vector is None:
                continue
            
            baseline = np.linalg.norm(
                query_kf.pose.translation_vector - kf.pose.translation_vector
            )
            
            # Score based on baseline and recency
            recency = 1.0 / (1 + abs(query_kf.frame_index - kf.frame_index))
            score = baseline * recency
            
            if score > best_score:
                best_score = score
                best_kf = kf
        
        return best_kf

    def _match_keyframes(
        self,
        kf1: MapKeyframe,
        kf2: MapKeyframe,
    ) -> List[Tuple[int, int]]:
        """Match features between two keyframes."""
        if kf1.descriptors is None or kf2.descriptors is None:
            return []
        
        if len(kf1.descriptors) == 0 or len(kf2.descriptors) == 0:
            return []
        
        # Use brute force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        try:
            matches = bf.match(kf1.descriptors, kf2.descriptors)
            return [(m.queryIdx, m.trainIdx) for m in matches]
        except cv2.error:
            return []

    def _build_projection_matrix(self, pose: PoseResult) -> Optional[np.ndarray]:
        """Build 3x4 projection matrix from pose."""
        if pose.rotation_matrix is None or pose.translation_vector is None:
            return None
        if self.calibration is None:
            return None
        
        RT = np.hstack((pose.rotation_matrix, pose.translation_vector))
        return self.calibration.camera_matrix @ RT

    def _reprojection_error(
        self,
        point_3d: np.ndarray,
        point_2d: np.ndarray,
        pose: PoseResult,
    ) -> float:
        """Compute reprojection error for a 3D point."""
        if pose.rotation_vector is None or pose.translation_vector is None:
            return float('inf')
        if self.calibration is None:
            return float('inf')
        
        try:
            projected, _ = cv2.projectPoints(
                point_3d.reshape(1, 3),
                pose.rotation_vector,
                pose.translation_vector,
                self.calibration.camera_matrix,
                self.calibration.dist_coeffs,
            )
            return float(np.linalg.norm(projected.reshape(2) - point_2d))
        except cv2.error:
            return float('inf')

    # ------------------------------------------------------------------ #
    # Map Maintenance
    # ------------------------------------------------------------------ #
    def _cull_points(self):
        """Remove low-quality points from the map."""
        to_remove = []
        
        for point_id, point in self.points.items():
            # Remove points not seen recently
            if self.frame_count - point.last_seen_frame > 50:
                to_remove.append(point_id)
                continue
            
            # Remove points with few observations
            if point.observations < self.config.min_observations:
                to_remove.append(point_id)
                continue
            
            # Remove points with high reprojection error
            if point.reprojection_error > self.config.max_reprojection_error * 2:
                to_remove.append(point_id)
        
        for point_id in to_remove:
            del self.points[point_id]
        
        self.stats["total_points_culled"] += len(to_remove)
        
        if to_remove:
            LOGGER.debug("Culled %d map points", len(to_remove))

    def _remove_oldest_keyframe(self):
        """Remove the oldest keyframe to maintain max size."""
        if not self.keyframes:
            return
        
        oldest_id = min(self.keyframes.keys())
        
        # Don't remove current keyframe
        if oldest_id == self.current_keyframe_id:
            return
        
        kf = self.keyframes[oldest_id]
        
        # Update covisibility
        for other_id in kf.covisible_keyframes:
            if other_id in self.keyframes:
                self.keyframes[other_id].covisible_keyframes.discard(oldest_id)
        
        del self.keyframes[oldest_id]
        LOGGER.debug("Removed oldest keyframe %d", oldest_id)

    # ------------------------------------------------------------------ #
    # Loop Closure
    # ------------------------------------------------------------------ #
    def _detect_loop_closure(self, query_kf_id: int):
        """Detect potential loop closures for a keyframe."""
        query_kf = self.keyframes.get(query_kf_id)
        if query_kf is None or query_kf.descriptors is None:
            return
        
        if len(query_kf.descriptors) == 0:
            return
        
        best_candidate = None
        best_matches = 0
        
        # Check against older keyframes (skip recent ones)
        for kf_id, kf in self.keyframes.items():
            # Skip recent keyframes
            if abs(kf.frame_index - query_kf.frame_index) < 30:
                continue
            
            if kf.descriptors is None or len(kf.descriptors) == 0:
                continue
            
            matches = self._match_keyframes(query_kf, kf)
            
            if len(matches) > best_matches:
                best_matches = len(matches)
                best_candidate = kf_id
        
        if best_matches >= self.config.min_loop_matches:
            candidate = LoopCandidate(
                query_keyframe_id=query_kf_id,
                match_keyframe_id=best_candidate,
                num_matches=best_matches,
            )
            
            # Geometric verification
            if self._verify_loop_closure(candidate):
                candidate.verified = True
                self.loop_closures.append((query_kf_id, best_candidate))
                self.stats["loop_closures_found"] += 1
                LOGGER.info("Loop closure detected between KF %d and KF %d (%d matches)",
                           query_kf_id, best_candidate, best_matches)

    def _verify_loop_closure(self, candidate: LoopCandidate) -> bool:
        """Verify loop closure with geometric consistency."""
        query_kf = self.keyframes.get(candidate.query_keyframe_id)
        match_kf = self.keyframes.get(candidate.match_keyframe_id)
        
        if query_kf is None or match_kf is None:
            return False
        
        # Get matched features
        matches = self._match_keyframes(query_kf, match_kf)
        if len(matches) < 8:
            return False
        
        pts1 = query_kf.keypoints[[m[0] for m in matches]]
        pts2 = match_kf.keypoints[[m[1] for m in matches]]
        
        # Compute fundamental matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
        
        if F is None or mask is None:
            return False
        
        inlier_ratio = np.sum(mask) / len(mask)
        candidate.geometric_score = inlier_ratio
        
        return inlier_ratio > 0.5

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def get_visible_points(
        self,
        pose: PoseResult,
        max_distance: float = 10.0,
    ) -> List[MapPoint]:
        """Get map points visible from a camera pose."""
        if pose.translation_vector is None:
            return []
        
        camera_pos = pose.translation_vector.flatten()
        visible = []
        
        for point in self.points.values():
            dist = np.linalg.norm(point.position - camera_pos)
            if dist < max_distance:
                visible.append(point)
        
        return visible

    def get_point_cloud(self) -> np.ndarray:
        """Get all map points as Nx3 array."""
        if not self.points:
            return np.empty((0, 3), dtype=np.float64)
        
        return np.array([p.position for p in self.points.values()], dtype=np.float64)

    def get_point_colors(self) -> Optional[np.ndarray]:
        """Get colors for all map points (Nx3 BGR)."""
        colors = []
        for p in self.points.values():
            if p.color is not None:
                colors.append(p.color)
            else:
                colors.append([128, 128, 128])
        
        if not colors:
            return None
        
        return np.array(colors, dtype=np.uint8)

    def get_keyframe_poses(self) -> List[Tuple[int, np.ndarray]]:
        """Get all keyframe poses as (id, 4x4 matrix) pairs."""
        poses = []
        for kf in self.keyframes.values():
            matrix = kf.pose.as_matrix()
            if matrix is not None:
                poses.append((kf.id, matrix))
        return poses

    def get_statistics(self) -> Dict:
        """Get map statistics."""
        return {
            **self.stats,
            "current_points": len(self.points),
            "current_keyframes": len(self.keyframes),
            "frame_count": self.frame_count,
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, filepath: str):
        """Save map to file."""
        data = {
            "version": "1.0",
            "config": {
                "min_observations": self.config.min_observations,
                "max_reprojection_error": self.config.max_reprojection_error,
                "max_keyframes": self.config.max_keyframes,
            },
            "points": [
                {
                    "id": p.id,
                    "position": p.position.tolist(),
                    "observations": p.observations,
                    "color": p.color.tolist() if p.color is not None else None,
                }
                for p in self.points.values()
            ],
            "keyframes": [
                {
                    "id": kf.id,
                    "frame_index": kf.frame_index,
                    "pose": {
                        "rvec": kf.pose.rotation_vector.tolist() if kf.pose.rotation_vector is not None else None,
                        "tvec": kf.pose.translation_vector.tolist() if kf.pose.translation_vector is not None else None,
                    },
                    "covisible": list(kf.covisible_keyframes),
                }
                for kf in self.keyframes.values()
            ],
            "statistics": self.stats,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        LOGGER.info("Map saved to %s (%d points, %d keyframes)",
                   filepath, len(self.points), len(self.keyframes))

    def load(self, filepath: str) -> bool:
        """Load map from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.reset()
            
            # Load points
            for p_data in data.get("points", []):
                point = MapPoint(
                    id=p_data["id"],
                    position=np.array(p_data["position"], dtype=np.float64),
                    observations=p_data.get("observations", 1),
                    color=np.array(p_data["color"], dtype=np.uint8) if p_data.get("color") else None,
                )
                self.points[point.id] = point
                self._next_point_id = max(self._next_point_id, point.id + 1)
            
            # Load keyframes (simplified - without full reconstruction)
            for kf_data in data.get("keyframes", []):
                pose_data = kf_data.get("pose", {})
                rvec = np.array(pose_data["rvec"]) if pose_data.get("rvec") else None
                tvec = np.array(pose_data["tvec"]) if pose_data.get("tvec") else None
                
                R = None
                if rvec is not None:
                    R, _ = cv2.Rodrigues(rvec)
                
                pose = PoseResult(
                    success=True,
                    rotation_vector=rvec,
                    translation_vector=tvec,
                    rotation_matrix=R,
                )
                
                kf = MapKeyframe(
                    id=kf_data["id"],
                    frame_index=kf_data.get("frame_index", 0),
                    pose=pose,
                    keypoints=np.array([]),
                    descriptors=np.array([]),
                    covisible_keyframes=set(kf_data.get("covisible", [])),
                )
                self.keyframes[kf.id] = kf
                self._next_keyframe_id = max(self._next_keyframe_id, kf.id + 1)
            
            self.stats = data.get("statistics", self.stats)
            
            LOGGER.info("Map loaded from %s (%d points, %d keyframes)",
                       filepath, len(self.points), len(self.keyframes))
            return True
            
        except Exception as e:
            LOGGER.error("Failed to load map: %s", e)
            return False


class MapVisualizer:
    """Visualize the sparse map."""
    
    @staticmethod
    def render_top_down(
        sparse_map: SparseMap,
        size: Tuple[int, int] = (500, 500),
        scale: float = 50.0,
    ) -> np.ndarray:
        """
        Render top-down view of the map.
        
        Args:
            sparse_map: The map to visualize
            size: Output image size
            scale: Pixels per meter
            
        Returns:
            BGR image
        """
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8) + 30
        
        center = np.array([size[0] // 2, size[1] // 2])
        
        # Draw points
        point_cloud = sparse_map.get_point_cloud()
        colors = sparse_map.get_point_colors()
        
        for i, pt in enumerate(point_cloud):
            x = int(center[0] + pt[0] * scale)
            y = int(center[1] - pt[2] * scale)  # Z is forward
            
            if 0 <= x < size[0] and 0 <= y < size[1]:
                color = tuple(int(c) for c in colors[i]) if colors is not None else (200, 200, 200)
                cv2.circle(img, (x, y), 2, color, -1)
        
        # Draw keyframe positions
        for kf_id, matrix in sparse_map.get_keyframe_poses():
            pos = matrix[:3, 3]
            x = int(center[0] + pos[0] * scale)
            y = int(center[1] - pos[2] * scale)
            
            if 0 <= x < size[0] and 0 <= y < size[1]:
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(img, str(kf_id), (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # Draw scale bar
        cv2.line(img, (20, size[1] - 20), (20 + int(scale), size[1] - 20), (255, 255, 255), 2)
        cv2.putText(img, "1m", (20, size[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw stats
        stats = sparse_map.get_statistics()
        cv2.putText(img, f"Points: {stats['current_points']}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f"Keyframes: {stats['current_keyframes']}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return img

    @staticmethod
    def project_map_points(
        sparse_map: SparseMap,
        pose: PoseResult,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (255, 0, 255),
    ) -> np.ndarray:
        """Project map points onto frame."""
        if sparse_map.calibration is None:
            return frame
        
        if pose.rotation_vector is None or pose.translation_vector is None:
            return frame
        
        point_cloud = sparse_map.get_point_cloud()
        if len(point_cloud) == 0:
            return frame
        
        try:
            projected, _ = cv2.projectPoints(
                point_cloud,
                pose.rotation_vector,
                pose.translation_vector,
                sparse_map.calibration.camera_matrix,
                sparse_map.calibration.dist_coeffs,
            )
            
            h, w = frame.shape[:2]
            for pt in projected.reshape(-1, 2):
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame, (x, y), 3, color, -1)
            
        except cv2.error:
            pass
        
        return frame

