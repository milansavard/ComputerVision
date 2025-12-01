"""
Camera/object pose estimation module.

Provides utilities for loading calibration data and estimating camera pose from
feature-track correspondences. Includes temporal smoothing and filtering for
stable AR overlays.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from tracking.feature import TrackingFrameResult

LOGGER = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """Container for camera calibration parameters."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


@dataclass
class PoseResult:
    """Structured container for pose estimation output."""

    success: bool
    rotation_vector: Optional[np.ndarray] = None
    translation_vector: Optional[np.ndarray] = None
    rotation_matrix: Optional[np.ndarray] = None
    method: str = ""
    inliers: int = 0
    reprojection_error: Optional[float] = None
    timestamp: Optional[float] = None
    is_smoothed: bool = False

    def as_matrix(self) -> Optional[np.ndarray]:
        """Return the 4x4 transformation matrix if pose is valid."""
        if not self.success or self.rotation_matrix is None or self.translation_vector is None:
            return None
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.rotation_matrix
        transform[:3, 3] = self.translation_vector.flatten()
        return transform

    def copy(self) -> PoseResult:
        """Create a copy of this pose result."""
        return PoseResult(
            success=self.success,
            rotation_vector=self.rotation_vector.copy() if self.rotation_vector is not None else None,
            translation_vector=self.translation_vector.copy() if self.translation_vector is not None else None,
            rotation_matrix=self.rotation_matrix.copy() if self.rotation_matrix is not None else None,
            method=self.method,
            inliers=self.inliers,
            reprojection_error=self.reprojection_error,
            timestamp=self.timestamp,
            is_smoothed=self.is_smoothed,
        )


@dataclass
class PoseFilterConfig:
    """Configuration for pose filtering and smoothing."""

    enable_smoothing: bool = True
    smoothing_alpha: float = 0.3  # EMA factor (0 = max smooth, 1 = no smooth)
    enable_outlier_rejection: bool = True
    max_translation_jump: float = 0.5  # Max allowed jump in meters
    max_rotation_jump: float = 0.5  # Max allowed rotation change in radians
    history_size: int = 10  # Number of poses to keep for median filtering
    use_median_filter: bool = False  # Use median instead of EMA
    min_inliers_threshold: int = 10  # Reject poses with fewer inliers


class PoseFilter:
    """
    Temporal filter for pose stabilization.
    
    Implements exponential moving average (EMA) smoothing and outlier rejection
    to produce stable pose estimates suitable for AR overlays.
    """

    def __init__(self, config: Optional[PoseFilterConfig] = None):
        self.config = config or PoseFilterConfig()
        self.pose_history: Deque[PoseResult] = deque(maxlen=self.config.history_size)
        self.smoothed_pose: Optional[PoseResult] = None
        self._ema_rotation: Optional[np.ndarray] = None
        self._ema_translation: Optional[np.ndarray] = None

    def reset(self):
        """Reset filter state."""
        self.pose_history.clear()
        self.smoothed_pose = None
        self._ema_rotation = None
        self._ema_translation = None

    def filter(self, pose: PoseResult) -> PoseResult:
        """Apply filtering to a pose estimate.
        
        Args:
            pose: Raw pose estimate
            
        Returns:
            Filtered/smoothed pose
        """
        if not pose.success:
            return pose

        # Check minimum inliers
        if self.config.enable_outlier_rejection:
            if pose.inliers < self.config.min_inliers_threshold:
                LOGGER.debug("Pose rejected: too few inliers (%d < %d)", pose.inliers, self.config.min_inliers_threshold)
                if self.smoothed_pose:
                    return self.smoothed_pose.copy()
                return PoseResult(success=False, method=pose.method)

        # Check for outlier jumps
        if self.config.enable_outlier_rejection and self.smoothed_pose:
            if self._is_outlier(pose):
                LOGGER.debug("Pose rejected as outlier")
                return self.smoothed_pose.copy()

        # Add to history
        self.pose_history.append(pose)

        # Apply smoothing
        if not self.config.enable_smoothing:
            self.smoothed_pose = pose.copy()
            return self.smoothed_pose

        if self.config.use_median_filter:
            smoothed = self._apply_median_filter(pose)
        else:
            smoothed = self._apply_ema_filter(pose)

        self.smoothed_pose = smoothed
        return smoothed

    def _is_outlier(self, pose: PoseResult) -> bool:
        """Check if pose is an outlier compared to recent history."""
        if self.smoothed_pose is None:
            return False

        if pose.translation_vector is None or self.smoothed_pose.translation_vector is None:
            return False

        # Check translation jump
        t_diff = np.linalg.norm(pose.translation_vector - self.smoothed_pose.translation_vector)
        if t_diff > self.config.max_translation_jump:
            LOGGER.debug("Translation jump: %.3f > %.3f", t_diff, self.config.max_translation_jump)
            return True

        # Check rotation jump (using rotation vectors)
        if pose.rotation_vector is not None and self.smoothed_pose.rotation_vector is not None:
            r_diff = np.linalg.norm(pose.rotation_vector - self.smoothed_pose.rotation_vector)
            if r_diff > self.config.max_rotation_jump:
                LOGGER.debug("Rotation jump: %.3f > %.3f", r_diff, self.config.max_rotation_jump)
                return True

        return False

    def _apply_ema_filter(self, pose: PoseResult) -> PoseResult:
        """Apply exponential moving average smoothing."""
        alpha = self.config.smoothing_alpha

        if pose.rotation_vector is None or pose.translation_vector is None:
            return pose.copy()

        # Initialize EMA state if needed
        if self._ema_rotation is None:
            self._ema_rotation = pose.rotation_vector.copy()
            self._ema_translation = pose.translation_vector.copy()
        else:
            # Update EMA
            self._ema_rotation = alpha * pose.rotation_vector + (1 - alpha) * self._ema_rotation
            self._ema_translation = alpha * pose.translation_vector + (1 - alpha) * self._ema_translation

        # Compute smoothed rotation matrix
        R_smoothed, _ = cv2.Rodrigues(self._ema_rotation)

        return PoseResult(
            success=True,
            rotation_vector=self._ema_rotation.copy(),
            translation_vector=self._ema_translation.copy(),
            rotation_matrix=R_smoothed,
            method=pose.method,
            inliers=pose.inliers,
            reprojection_error=pose.reprojection_error,
            timestamp=pose.timestamp,
            is_smoothed=True,
        )

    def _apply_median_filter(self, pose: PoseResult) -> PoseResult:
        """Apply median filtering over recent history."""
        if len(self.pose_history) < 3:
            return pose.copy()

        # Collect valid poses
        rotations = []
        translations = []
        for p in self.pose_history:
            if p.success and p.rotation_vector is not None and p.translation_vector is not None:
                rotations.append(p.rotation_vector.flatten())
                translations.append(p.translation_vector.flatten())

        if len(rotations) < 3:
            return pose.copy()

        # Compute element-wise median
        r_median = np.median(np.array(rotations), axis=0).reshape(3, 1)
        t_median = np.median(np.array(translations), axis=0).reshape(3, 1)

        R_median, _ = cv2.Rodrigues(r_median)

        return PoseResult(
            success=True,
            rotation_vector=r_median,
            translation_vector=t_median,
            rotation_matrix=R_median,
            method=pose.method,
            inliers=pose.inliers,
            reprojection_error=pose.reprojection_error,
            timestamp=pose.timestamp,
            is_smoothed=True,
        )


@dataclass
class ScaleEstimatorConfig:
    """Configuration for scale estimation."""

    method: str = "auto"  # "auto", "known_distance", "ground_plane", "object_size", "manual"
    manual_scale: float = 1.0  # Scale factor when method="manual"
    known_distance: Optional[float] = None  # Known distance in meters
    ground_plane_height: float = 1.5  # Expected camera height above ground (meters)
    reference_object_size: Optional[float] = None  # Known object size in meters
    scale_smoothing_alpha: float = 0.2  # EMA factor for scale updates
    min_scale: float = 0.001  # Minimum allowed scale
    max_scale: float = 100.0  # Maximum allowed scale
    consistency_threshold: float = 0.5  # Max allowed scale change ratio per frame


class ScaleEstimator:
    """
    Multi-method scale estimator for markerless tracking.
    
    Monocular visual odometry can only recover pose up to an unknown scale.
    This class provides multiple methods to estimate and maintain consistent scale:
    
    1. **Known Distance**: Use known distance between two tracked points
    2. **Ground Plane**: Assume camera at known height above ground
    3. **Object Size**: Use known physical size of a detected object
    4. **Manual**: User-specified fixed scale factor
    5. **Auto**: Automatically select best available method
    """

    def __init__(self, config: Optional[ScaleEstimatorConfig] = None):
        self.config = config or ScaleEstimatorConfig()
        
        # Scale state
        self._current_scale: float = self.config.manual_scale
        self._scale_history: Deque[float] = deque(maxlen=30)
        self._scale_confidence: float = 0.0
        
        # Reference points for known distance method
        self._reference_points_3d: Optional[np.ndarray] = None
        self._reference_distance: Optional[float] = None
        
        # Ground plane for ground plane method
        self._ground_normal: np.ndarray = np.array([0.0, 1.0, 0.0])  # Y-up convention
        
    def reset(self):
        """Reset scale estimator state."""
        self._current_scale = self.config.manual_scale
        self._scale_history.clear()
        self._scale_confidence = 0.0
        self._reference_points_3d = None
        self._reference_distance = None

    @property
    def current_scale(self) -> float:
        """Get current estimated scale."""
        return self._current_scale

    @property
    def scale_confidence(self) -> float:
        """Get confidence in current scale estimate (0-1)."""
        return self._scale_confidence

    def set_reference_distance(
        self,
        points_3d: np.ndarray,
        real_distance: float,
    ):
        """
        Set reference points with known real-world distance.
        
        Args:
            points_3d: Two 3D points (2x3 array)
            real_distance: Real-world distance between them in meters
        """
        if points_3d.shape[0] < 2:
            LOGGER.warning("Need at least 2 points for reference distance")
            return
            
        self._reference_points_3d = points_3d[:2].copy()
        self._reference_distance = real_distance
        LOGGER.info("Reference distance set: %.3f meters", real_distance)

    def estimate_scale(
        self,
        triangulated_points: Optional[np.ndarray] = None,
        pose: Optional["PoseResult"] = None,
        point_indices: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, float]:
        """
        Estimate scale using the configured method.
        
        Args:
            triangulated_points: 3D points from triangulation (Nx3)
            pose: Current pose estimate
            point_indices: Indices of points with known distance
            
        Returns:
            (scale, confidence) tuple
        """
        method = self.config.method.lower()
        
        if method == "manual":
            return self._estimate_manual()
        elif method == "known_distance":
            return self._estimate_from_known_distance(triangulated_points, point_indices)
        elif method == "ground_plane":
            return self._estimate_from_ground_plane(triangulated_points)
        elif method == "object_size":
            return self._estimate_from_object_size(triangulated_points)
        elif method == "auto":
            return self._estimate_auto(triangulated_points, pose, point_indices)
        else:
            LOGGER.warning("Unknown scale method: %s, using manual", method)
            return self._estimate_manual()

    def _estimate_manual(self) -> Tuple[float, float]:
        """Return manual scale with full confidence."""
        return self.config.manual_scale, 1.0

    def _estimate_from_known_distance(
        self,
        points_3d: Optional[np.ndarray],
        point_indices: Optional[Tuple[int, int]],
    ) -> Tuple[float, float]:
        """Estimate scale from known distance between two points."""
        if self.config.known_distance is None:
            return self._current_scale, 0.0
            
        if points_3d is None or point_indices is None:
            return self._current_scale, 0.0
            
        if len(points_3d) < max(point_indices) + 1:
            return self._current_scale, 0.0

        try:
            p1 = points_3d[point_indices[0]]
            p2 = points_3d[point_indices[1]]
            estimated_distance = np.linalg.norm(p2 - p1)
            
            if estimated_distance < 1e-6:
                return self._current_scale, 0.0
                
            scale = self.config.known_distance / estimated_distance
            scale = self._clamp_scale(scale)
            
            # Update with smoothing
            self._update_scale(scale, confidence=0.8)
            return self._current_scale, 0.8
            
        except Exception as e:
            LOGGER.debug("Known distance scale estimation failed: %s", e)
            return self._current_scale, 0.0

    def _estimate_from_ground_plane(
        self,
        points_3d: Optional[np.ndarray],
    ) -> Tuple[float, float]:
        """
        Estimate scale assuming camera is at known height above ground.
        
        This method finds the lowest points (assumed to be on ground) and
        uses the expected camera height to compute scale.
        """
        if points_3d is None or len(points_3d) < 10:
            return self._current_scale, 0.0

        try:
            # Find points likely on the ground plane (lowest Y values)
            # Assuming Y-down convention in camera frame
            y_coords = points_3d[:, 1]
            
            # Use bottom 10% of points as ground candidates
            threshold = np.percentile(y_coords, 90)  # Y-down: larger Y = lower
            ground_mask = y_coords >= threshold
            ground_points = points_3d[ground_mask]
            
            if len(ground_points) < 5:
                return self._current_scale, 0.0
            
            # Fit plane to ground points using RANSAC-like approach
            # For simplicity, use median Y as ground level
            ground_y = np.median(ground_points[:, 1])
            
            # Expected ground Y in scaled coordinates
            expected_ground_y = self.config.ground_plane_height
            
            if abs(ground_y) < 1e-6:
                return self._current_scale, 0.0
                
            scale = expected_ground_y / ground_y
            scale = self._clamp_scale(abs(scale))
            
            # Lower confidence due to assumptions
            self._update_scale(scale, confidence=0.5)
            return self._current_scale, 0.5
            
        except Exception as e:
            LOGGER.debug("Ground plane scale estimation failed: %s", e)
            return self._current_scale, 0.0

    def _estimate_from_object_size(
        self,
        points_3d: Optional[np.ndarray],
    ) -> Tuple[float, float]:
        """Estimate scale from known object size in scene."""
        if self.config.reference_object_size is None:
            return self._current_scale, 0.0
            
        if points_3d is None or len(points_3d) < 4:
            return self._current_scale, 0.0

        try:
            # Compute bounding box of all points as proxy for scene size
            # This is a rough heuristic - in practice, you'd detect specific objects
            min_coords = points_3d.min(axis=0)
            max_coords = points_3d.max(axis=0)
            bbox_size = np.linalg.norm(max_coords - min_coords)
            
            if bbox_size < 1e-6:
                return self._current_scale, 0.0
            
            # Very rough estimate - assumes object fills significant portion of view
            scale = self.config.reference_object_size / (bbox_size * 0.5)
            scale = self._clamp_scale(scale)
            
            # Low confidence due to rough heuristic
            self._update_scale(scale, confidence=0.3)
            return self._current_scale, 0.3
            
        except Exception as e:
            LOGGER.debug("Object size scale estimation failed: %s", e)
            return self._current_scale, 0.0

    def _estimate_auto(
        self,
        points_3d: Optional[np.ndarray],
        pose: Optional["PoseResult"],
        point_indices: Optional[Tuple[int, int]],
    ) -> Tuple[float, float]:
        """Automatically select best scale estimation method."""
        best_scale = self._current_scale
        best_confidence = 0.0
        
        # Try known distance first (highest confidence if available)
        if self.config.known_distance is not None and point_indices is not None:
            scale, conf = self._estimate_from_known_distance(points_3d, point_indices)
            if conf > best_confidence:
                best_scale, best_confidence = scale, conf
        
        # Try ground plane if we have points
        if best_confidence < 0.5 and points_3d is not None:
            scale, conf = self._estimate_from_ground_plane(points_3d)
            if conf > best_confidence:
                best_scale, best_confidence = scale, conf
        
        # Fall back to previous scale with decaying confidence
        if best_confidence < 0.3:
            # Decay confidence over time if no good estimate
            self._scale_confidence *= 0.95
            return self._current_scale, self._scale_confidence
        
        return best_scale, best_confidence

    def _update_scale(self, new_scale: float, confidence: float):
        """Update scale estimate with smoothing and consistency check."""
        # Check for consistency
        if len(self._scale_history) > 0:
            ratio = new_scale / self._current_scale if self._current_scale > 0 else float('inf')
            if ratio > (1 + self.config.consistency_threshold) or ratio < (1 - self.config.consistency_threshold):
                # Large jump - reduce influence
                confidence *= 0.3
                LOGGER.debug("Scale jump detected: %.3f -> %.3f", self._current_scale, new_scale)
        
        # EMA update weighted by confidence
        alpha = self.config.scale_smoothing_alpha * confidence
        self._current_scale = alpha * new_scale + (1 - alpha) * self._current_scale
        self._scale_confidence = max(self._scale_confidence, confidence)
        
        self._scale_history.append(new_scale)

    def _clamp_scale(self, scale: float) -> float:
        """Clamp scale to valid range."""
        return max(self.config.min_scale, min(self.config.max_scale, scale))

    def apply_scale(self, pose: "PoseResult") -> "PoseResult":
        """Apply current scale estimate to a pose."""
        if not pose.success or pose.translation_vector is None:
            return pose
            
        scaled_pose = pose.copy()
        scaled_pose.translation_vector = pose.translation_vector * self._current_scale
        return scaled_pose


class PoseEstimator:
    """
    Handles camera pose estimation from tracked features.
    
    Supports:
    - Feature-based pose from Essential matrix decomposition
    - Temporal smoothing and outlier filtering
    - Multi-method scale recovery for markerless tracking
    - Calibration from file or inline config
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.calibration_config = self.config.get("calibration", {})
        self.filter_config = self.config.get("pose_filter", {})
        self.scale_config = self.config.get("scale_estimation", {})

        self.calibration: Optional[CalibrationData] = None
        self.pose_filter: Optional[PoseFilter] = None
        self.scale_estimator: Optional[ScaleEstimator] = None
        self.initialized = False

        # Legacy scale state (kept for backwards compatibility)
        self._accumulated_scale: float = 1.0
        self._scale_samples: List[float] = []
        
        # Triangulated points cache for scale estimation
        self._last_triangulated_points: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Initialization / calibration
    # ------------------------------------------------------------------ #
    def initialize(self) -> bool:
        """Load calibration data and prepare estimator."""
        self.calibration = self._load_calibration(self.calibration_config)

        # Initialize pose filter
        filter_cfg = PoseFilterConfig(
            enable_smoothing=self.filter_config.get("enable_smoothing", True),
            smoothing_alpha=self.filter_config.get("smoothing_alpha", 0.3),
            enable_outlier_rejection=self.filter_config.get("enable_outlier_rejection", True),
            max_translation_jump=self.filter_config.get("max_translation_jump", 0.5),
            max_rotation_jump=self.filter_config.get("max_rotation_jump", 0.5),
            history_size=self.filter_config.get("history_size", 10),
            use_median_filter=self.filter_config.get("use_median_filter", False),
            min_inliers_threshold=self.filter_config.get("min_inliers_threshold", 10),
        )
        self.pose_filter = PoseFilter(filter_cfg)

        # Initialize scale estimator
        scale_cfg = ScaleEstimatorConfig(
            method=self.scale_config.get("method", "auto"),
            manual_scale=self.scale_config.get("manual_scale", 1.0),
            known_distance=self.scale_config.get("known_distance"),
            ground_plane_height=self.scale_config.get("ground_plane_height", 1.5),
            reference_object_size=self.scale_config.get("reference_object_size"),
            scale_smoothing_alpha=self.scale_config.get("scale_smoothing_alpha", 0.2),
            min_scale=self.scale_config.get("min_scale", 0.001),
            max_scale=self.scale_config.get("max_scale", 100.0),
            consistency_threshold=self.scale_config.get("consistency_threshold", 0.5),
        )
        self.scale_estimator = ScaleEstimator(scale_cfg)

        self.initialized = True
        LOGGER.info("Pose estimator initialized with calibration matrix:\n%s", self.calibration.camera_matrix)
        return True

    @staticmethod
    def _load_calibration(config: Dict) -> CalibrationData:
        """Load calibration data from config or external file."""
        calibration_file = config.get("calibration_file")

        if calibration_file:
            data = PoseEstimator._read_calibration_file(calibration_file)
        else:
            data = {
                "camera_matrix": config.get("camera_matrix"),
                "dist_coeffs": config.get("dist_coeffs"),
            }

        if data["camera_matrix"] is None:
            raise ValueError("Camera matrix must be provided for pose estimation.")

        camera_matrix = np.array(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
        dist_coeffs = PoseEstimator._normalize_dist_coeffs(data.get("dist_coeffs"))

        return CalibrationData(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    @staticmethod
    def _read_calibration_file(path: str) -> Dict:
        calib_path = Path(path)
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        with calib_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload

    @staticmethod
    def _normalize_dist_coeffs(coeffs: Optional[Sequence[float]]) -> np.ndarray:
        if coeffs is None:
            coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
        arr = np.array(coeffs, dtype=np.float64).reshape(-1, 1)
        return arr

    def _ensure_initialized(self):
        if not self.initialized or self.calibration is None:
            self.initialize()

    def reset_filter(self):
        """Reset the pose filter state."""
        if self.pose_filter:
            self.pose_filter.reset()

    def reset_scale(self):
        """Reset the scale estimator state."""
        if self.scale_estimator:
            self.scale_estimator.reset()
        self._accumulated_scale = 1.0
        self._scale_samples.clear()
        self._last_triangulated_points = None

    def set_known_distance(
        self,
        distance: float,
        point_indices: Optional[Tuple[int, int]] = None,
    ):
        """
        Set a known real-world distance for scale recovery.
        
        Args:
            distance: Known distance in meters
            point_indices: Optional indices of points with this distance
        """
        if self.scale_estimator:
            self.scale_estimator.config.known_distance = distance
            self.scale_estimator.config.method = "known_distance"
            LOGGER.info("Known distance set to %.3f meters", distance)

    def set_ground_plane_height(self, height: float):
        """
        Set expected camera height above ground plane.
        
        Args:
            height: Camera height in meters
        """
        if self.scale_estimator:
            self.scale_estimator.config.ground_plane_height = height
            LOGGER.info("Ground plane height set to %.3f meters", height)

    def set_manual_scale(self, scale: float):
        """
        Set a manual scale factor.
        
        Args:
            scale: Scale factor to apply to translations
        """
        if self.scale_estimator:
            self.scale_estimator.config.manual_scale = scale
            self.scale_estimator.config.method = "manual"
            self.scale_estimator._current_scale = scale
            LOGGER.info("Manual scale set to %.4f", scale)

    def get_scale_info(self) -> Dict:
        """
        Get current scale estimation information.
        
        Returns:
            Dictionary with scale, confidence, and method information
        """
        if self.scale_estimator:
            return {
                "scale": self.scale_estimator.current_scale,
                "confidence": self.scale_estimator.scale_confidence,
                "method": self.scale_estimator.config.method,
                "history_length": len(self.scale_estimator._scale_history),
            }
        return {
            "scale": self._accumulated_scale,
            "confidence": 0.5,
            "method": "legacy",
            "history_length": len(self._scale_samples),
        }

    def get_triangulated_points(self) -> Optional[np.ndarray]:
        """Get the last triangulated 3D points (for visualization/debugging)."""
        return self._last_triangulated_points

    # ------------------------------------------------------------------ #
    # Feature-based pose estimation
    # ------------------------------------------------------------------ #
    def estimate_from_feature_tracks(
        self,
        tracks: TrackingFrameResult,
        apply_filter: bool = True,
    ) -> PoseResult:
        """Estimate camera motion from feature correspondences between frames.
        
        Args:
            tracks: Feature tracking result with matches
            apply_filter: Whether to apply temporal smoothing
            
        Returns:
            PoseResult with estimated camera pose
        """
        self._ensure_initialized()

        if tracks is None or tracks.matches is None or len(tracks.matches) < 8:
            return PoseResult(success=False, method="markerless")

        pts_curr = tracks.matches[:, :2].astype(np.float64)
        pts_prev = tracks.matches[:, 2:].astype(np.float64)

        essential_matrix, inliers_mask = cv2.findEssentialMat(
            pts_curr,
            pts_prev,
            self.calibration.camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        if essential_matrix is None:
            return PoseResult(success=False, method="markerless")

        inliers_mask = inliers_mask if inliers_mask is not None else np.ones((pts_curr.shape[0], 1))
        inlier_count = int(np.count_nonzero(inliers_mask))

        if inlier_count < 5:
            return PoseResult(success=False, method="markerless")

        try:
            _, R, t, _ = cv2.recoverPose(
                essential_matrix,
                pts_curr,
                pts_prev,
                self.calibration.camera_matrix,
                mask=inliers_mask,
            )
        except cv2.error as exc:
            LOGGER.debug("recoverPose failed: %s", exc)
            return PoseResult(success=False, method="markerless")

        rvec, _ = cv2.Rodrigues(R)

        raw_pose = PoseResult(
            success=True,
            rotation_vector=rvec,
            translation_vector=t,
            rotation_matrix=R,
            method="markerless",
            inliers=inlier_count,
            timestamp=tracks.timestamp,
        )

        # Apply temporal filtering
        if apply_filter and self.pose_filter:
            return self.pose_filter.filter(raw_pose)

        return raw_pose

    def estimate_with_scale(
        self,
        tracks: TrackingFrameResult,
        known_distance: Optional[float] = None,
        point_indices: Optional[Tuple[int, int]] = None,
        use_new_estimator: bool = True,
    ) -> PoseResult:
        """Estimate pose with scale recovery.
        
        For markerless tracking, the translation is only known up to scale.
        This method attempts to recover absolute scale using multiple methods:
        - Known distance between points (if provided)
        - Ground plane assumption
        - Object size reference
        - Accumulated scale from previous estimates
        
        Args:
            tracks: Feature tracking result
            known_distance: Known real-world distance between two points (meters)
            point_indices: Indices of the two points with known distance
            use_new_estimator: Use the new ScaleEstimator (True) or legacy method (False)
            
        Returns:
            PoseResult with scaled translation
        """
        pose = self.estimate_from_feature_tracks(tracks, apply_filter=False)

        if not pose.success or pose.translation_vector is None:
            return pose

        # Triangulate points for scale estimation
        triangulated = self._triangulate_matches(tracks, pose)
        self._last_triangulated_points = triangulated

        if use_new_estimator and self.scale_estimator is not None:
            # Use new multi-method scale estimator
            if known_distance is not None:
                # Temporarily set known distance in config
                self.scale_estimator.config.known_distance = known_distance
            
            scale, confidence = self.scale_estimator.estimate_scale(
                triangulated_points=triangulated,
                pose=pose,
                point_indices=point_indices,
            )
            
            # Apply scale
            pose = self.scale_estimator.apply_scale(pose)
            LOGGER.debug("Scale estimate: %.4f (confidence: %.2f)", scale, confidence)
        else:
            # Legacy scale recovery method
            scale = self._accumulated_scale

            if known_distance is not None and point_indices is not None:
                estimated_scale = self._recover_scale_from_points(
                    tracks, pose, known_distance, point_indices
                )
                if estimated_scale is not None:
                    self._scale_samples.append(estimated_scale)
                    if len(self._scale_samples) > 10:
                        self._scale_samples = self._scale_samples[-10:]
                    scale = np.median(self._scale_samples)
                    self._accumulated_scale = scale

            pose.translation_vector = pose.translation_vector * scale

        # Apply filter after scaling
        if self.pose_filter:
            pose = self.pose_filter.filter(pose)

        return pose

    def _triangulate_matches(
        self,
        tracks: TrackingFrameResult,
        pose: PoseResult,
    ) -> Optional[np.ndarray]:
        """
        Triangulate matched points between frames.
        
        Args:
            tracks: Feature tracking result with matches
            pose: Estimated pose (R, t)
            
        Returns:
            Nx3 array of 3D points, or None if triangulation fails
        """
        if tracks.matches is None or len(tracks.matches) < 8:
            return None
            
        if pose.rotation_matrix is None or pose.translation_vector is None:
            return None

        try:
            pts_curr = tracks.matches[:, :2].astype(np.float64)
            pts_prev = tracks.matches[:, 2:].astype(np.float64)

            # Create projection matrices
            # P1 is identity (reference frame)
            P1 = self.calibration.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
            # P2 uses the estimated pose
            P2 = self.calibration.camera_matrix @ np.hstack((pose.rotation_matrix, pose.translation_vector))

            # Triangulate all points
            points_4d = cv2.triangulatePoints(P1, P2, pts_prev.T, pts_curr.T)
            
            # Convert from homogeneous coordinates
            points_3d = (points_4d[:3] / points_4d[3]).T

            # Filter out points at infinity or behind camera
            valid_mask = (
                (np.abs(points_4d[3]) > 1e-6) &  # Not at infinity
                (points_3d[:, 2] > 0)  # In front of camera
            )
            
            valid_points = points_3d[valid_mask]
            
            if len(valid_points) < 5:
                return None
                
            return valid_points

        except cv2.error as e:
            LOGGER.debug("Triangulation failed: %s", e)
            return None

    def _recover_scale_from_points(
        self,
        tracks: TrackingFrameResult,
        pose: PoseResult,
        known_distance: float,
        point_indices: Tuple[int, int],
    ) -> Optional[float]:
        """Recover scale from known distance between two tracked points."""
        if tracks.matches is None or len(tracks.matches) < max(point_indices) + 1:
            return None

        try:
            # Triangulate the two points
            pts1 = tracks.matches[list(point_indices), :2].astype(np.float64)
            pts2 = tracks.matches[list(point_indices), 2:].astype(np.float64)

            # Create projection matrices
            P1 = self.calibration.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = self.calibration.camera_matrix @ np.hstack((pose.rotation_matrix, pose.translation_vector))

            # Triangulate
            points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            points_3d = (points_4d[:3] / points_4d[3]).T

            # Compute distance between triangulated points
            estimated_distance = np.linalg.norm(points_3d[0] - points_3d[1])

            if estimated_distance > 0:
                return known_distance / estimated_distance

        except Exception as e:
            LOGGER.debug("Scale recovery failed: %s", e)

        return None

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #
    def _reprojection_error(
        self,
        object_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        image_points: np.ndarray,
    ) -> float:
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.calibration.camera_matrix,
            self.calibration.dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        error = np.linalg.norm(projected - image_points, axis=1).mean()
        return float(error)

    def project_points(
        self, points_3d: np.ndarray, pose: PoseResult
    ) -> Optional[np.ndarray]:
        """Project 3D points into the image using the provided pose."""
        if not pose or not pose.success or pose.rotation_vector is None or pose.translation_vector is None:
            return None
        image_points, _ = cv2.projectPoints(
            points_3d,
            pose.rotation_vector,
            pose.translation_vector,
            self.calibration.camera_matrix,
            self.calibration.dist_coeffs,
        )
        return image_points.reshape(-1, 2)

    def project_axes(self, pose: PoseResult, axis_length: float = 0.05) -> Optional[np.ndarray]:
        """Project canonical XYZ axes for visualisation."""
        axes = np.array(
            [
                [0.0, 0.0, 0.0],
                [axis_length, 0.0, 0.0],
                [0.0, axis_length, 0.0],
                [0.0, 0.0, axis_length],
            ],
            dtype=np.float64,
        )
        return self.project_points(axes, pose)

    def get_pose_quality(self, pose: PoseResult) -> float:
        """Compute a quality score for the pose estimate (0-1)."""
        if not pose.success:
            return 0.0

        # Base quality on inlier count
        inlier_score = min(pose.inliers / 100.0, 1.0)

        # Penalize if reprojection error is high
        error_score = 1.0
        if pose.reprojection_error is not None:
            error_score = max(0.0, 1.0 - pose.reprojection_error / 5.0)

        return inlier_score * error_score

    def decompose_pose(self, pose: PoseResult) -> Optional[Dict]:
        """Decompose pose into interpretable components.
        
        Returns:
            Dictionary with:
            - euler_angles: (roll, pitch, yaw) in degrees
            - position: (x, y, z) translation
            - distance: distance from camera
        """
        if not pose.success or pose.rotation_matrix is None or pose.translation_vector is None:
            return None

        # Extract Euler angles from rotation matrix
        R = pose.rotation_matrix
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0

        t = pose.translation_vector.flatten()

        return {
            "euler_angles": (np.degrees(roll), np.degrees(pitch), np.degrees(yaw)),
            "position": (t[0], t[1], t[2]),
            "distance": float(np.linalg.norm(t)),
        }
