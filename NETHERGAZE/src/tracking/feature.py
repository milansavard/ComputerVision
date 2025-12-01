"""
Feature-based tracking utilities for markerless workflows.

Supports multiple detector/descriptor combinations:
- ORB (default, fast and robust)
- FAST + BRIEF (very fast, less distinctive)
- AKAZE (good for scale/rotation invariance)
- BRISK (balanced speed/accuracy)
- SIFT (most robust, requires opencv-contrib)
- Good Features to Track + ORB descriptors
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class DetectorType(Enum):
    """Supported feature detector types."""
    ORB = "orb"
    FAST_BRIEF = "fast_brief"
    AKAZE = "akaze"
    BRISK = "brisk"
    SIFT = "sift"
    GFTT_ORB = "gftt_orb"  # Good Features To Track + ORB descriptors


class MatcherType(Enum):
    """Supported descriptor matcher types."""
    BF_HAMMING = "bf_hamming"  # For binary descriptors (ORB, BRIEF, BRISK, AKAZE)
    BF_L2 = "bf_l2"  # For float descriptors (SIFT)
    FLANN = "flann"  # Fast approximate matching


@dataclass
class TrackingConfiguration:
    """Configuration for the feature tracker."""

    # Detector selection
    method: str = "orb"
    
    # Common parameters
    max_features: int = 1000
    quality_level: float = 0.01
    min_distance: float = 7.0
    
    # ORB-specific
    fast_threshold: int = 20
    orb_scale_factor: float = 1.2
    orb_nlevels: int = 8
    orb_edge_threshold: int = 31
    orb_patch_size: int = 31
    
    # FAST-specific
    fast_nonmax_suppression: bool = True
    fast_type: int = 2  # cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
    
    # AKAZE-specific
    akaze_threshold: float = 0.001
    akaze_noctaves: int = 4
    akaze_noctave_layers: int = 4
    
    # BRISK-specific
    brisk_threshold: int = 30
    brisk_octaves: int = 3
    brisk_pattern_scale: float = 1.0
    
    # SIFT-specific (if available)
    sift_noctave_layers: int = 3
    sift_contrast_threshold: float = 0.04
    sift_edge_threshold: float = 10.0
    sift_sigma: float = 1.6
    
    # Optical flow
    use_optical_flow: bool = True
    optical_flow_win_size: int = 21
    optical_flow_max_level: int = 3
    optical_flow_criteria_eps: float = 0.03
    optical_flow_criteria_count: int = 30
    optical_flow_min_eig_threshold: float = 0.001
    
    # Adaptive optical flow
    adaptive_optical_flow: bool = True
    optical_flow_quality_threshold: float = 0.3  # Min forward-backward consistency
    
    # Keyframe management
    reacquire_threshold: int = 300
    keyframe_interval: int = 15
    max_keyframes: int = 5
    min_keyframe_features: int = 200
    keyframe_quality_threshold: float = 0.5  # Min feature quality for keyframe
    
    # Matching
    matcher_type: str = "bf_hamming"
    match_ratio_threshold: float = 0.75  # Lowe's ratio test threshold
    min_match_distance: float = 30.0  # Max descriptor distance for match
    
    # Grid-based feature distribution
    use_grid_detection: bool = False
    grid_rows: int = 4
    grid_cols: int = 4

    def __post_init__(self):
        if self.reacquire_threshold <= 0:
            self.reacquire_threshold = max(int(self.max_features * 0.3), 50)


@dataclass
class Keyframe:
    """Persisted keyframe used for relocalisation."""

    keypoints: np.ndarray  # shape (N, 2)
    descriptors: np.ndarray
    frame_index: int
    quality_score: float = 0.0
    feature_responses: Optional[np.ndarray] = None  # Keypoint response values


@dataclass
class TrackingFrameResult:
    """Result container for feature tracking on a frame."""

    keypoints: Optional[np.ndarray] = None
    descriptors: Optional[np.ndarray] = None
    matches: Optional[np.ndarray] = None
    source: str = ""
    tracked_count: int = 0
    reacquired: bool = False
    timestamp: Optional[float] = None
    
    # Extended metrics
    detector_type: str = ""
    optical_flow_quality: float = 0.0
    keyframe_match_count: int = 0
    feature_responses: Optional[np.ndarray] = None


@dataclass
class TrackingMetrics:
    """Accumulated tracking metrics for analysis."""
    
    total_frames: int = 0
    optical_flow_frames: int = 0
    detection_frames: int = 0
    reacquisition_count: int = 0
    avg_features_tracked: float = 0.0
    avg_optical_flow_quality: float = 0.0
    keyframes_created: int = 0


class FeatureDetectorFactory:
    """Factory for creating feature detectors and descriptors."""

    @staticmethod
    def create_detector(
        detector_type: str,
        config: TrackingConfiguration,
    ) -> Tuple[Optional[cv2.Feature2D], Optional[cv2.Feature2D], str]:
        """
        Create detector and descriptor extractor.
        
        Returns:
            (detector, descriptor_extractor, matcher_norm)
        """
        dtype = detector_type.lower()
        
        if dtype == "orb":
            detector = cv2.ORB_create(
                nfeatures=config.max_features,
                scaleFactor=config.orb_scale_factor,
                nlevels=config.orb_nlevels,
                edgeThreshold=config.orb_edge_threshold,
                patchSize=config.orb_patch_size,
                fastThreshold=config.fast_threshold,
            )
            return detector, detector, "hamming"
        
        elif dtype == "fast_brief":
            detector = cv2.FastFeatureDetector_create(
                threshold=config.fast_threshold,
                nonmaxSuppression=config.fast_nonmax_suppression,
                type=config.fast_type,
            )
            try:
                descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            except AttributeError:
                LOGGER.warning("BRIEF not available, falling back to ORB descriptors")
                descriptor = cv2.ORB_create(nfeatures=config.max_features)
            return detector, descriptor, "hamming"
        
        elif dtype == "akaze":
            detector = cv2.AKAZE_create(
                threshold=config.akaze_threshold,
                nOctaves=config.akaze_noctaves,
                nOctaveLayers=config.akaze_noctave_layers,
            )
            return detector, detector, "hamming"
        
        elif dtype == "brisk":
            detector = cv2.BRISK_create(
                thresh=config.brisk_threshold,
                octaves=config.brisk_octaves,
                patternScale=config.brisk_pattern_scale,
            )
            return detector, detector, "hamming"
        
        elif dtype == "sift":
            try:
                detector = cv2.SIFT_create(
                    nfeatures=config.max_features,
                    nOctaveLayers=config.sift_noctave_layers,
                    contrastThreshold=config.sift_contrast_threshold,
                    edgeThreshold=config.sift_edge_threshold,
                    sigma=config.sift_sigma,
                )
                return detector, detector, "l2"
            except AttributeError:
                LOGGER.warning("SIFT not available, falling back to ORB")
                return FeatureDetectorFactory.create_detector("orb", config)
        
        elif dtype == "gftt_orb":
            # Good Features To Track for detection, ORB for description
            descriptor = cv2.ORB_create(nfeatures=config.max_features)
            return None, descriptor, "hamming"  # None detector = use GFTT
        
        else:
            LOGGER.warning("Unknown detector type '%s', using ORB", dtype)
            return FeatureDetectorFactory.create_detector("orb", config)

    @staticmethod
    def create_matcher(matcher_type: str, norm: str) -> cv2.DescriptorMatcher:
        """Create a descriptor matcher."""
        if matcher_type == "flann":
            if norm == "hamming":
                # FLANN for binary descriptors
                index_params = dict(
                    algorithm=6,  # FLANN_INDEX_LSH
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1,
                )
            else:
                # FLANN for float descriptors
                index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # Brute force matcher
            norm_type = cv2.NORM_HAMMING if norm == "hamming" else cv2.NORM_L2
            return cv2.BFMatcher(norm_type, crossCheck=False)


class FeatureTracker:
    """
    Advanced feature tracker with multiple detector/descriptor options,
    adaptive optical-flow, keyframe management, and tracking metrics.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg_dict = dict(config or {})
        self.config = TrackingConfiguration(**{
            k: v for k, v in cfg_dict.items()
            if k in TrackingConfiguration.__dataclass_fields__
        })
        
        # Build detector and descriptor extractor
        self.detector, self.descriptor_extractor, self._norm = (
            FeatureDetectorFactory.create_detector(self.config.method, self.config)
        )
        
        # Build matcher
        self.matcher = FeatureDetectorFactory.create_matcher(
            self.config.matcher_type, self._norm
        )
        
        # For backwards compatibility
        self.bf_matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING if self._norm == "hamming" else cv2.NORM_L2,
            crossCheck=True
        )

        # Tracking state
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self.prev_descriptors: Optional[np.ndarray] = None
        self.prev_keypoints_cv: Optional[List[cv2.KeyPoint]] = None

        self.frame_index: int = 0
        self.keyframes: List[Keyframe] = []
        
        # Metrics
        self.metrics = TrackingMetrics()
        
        LOGGER.info(
            "FeatureTracker initialized: detector=%s, matcher=%s",
            self.config.method,
            self.config.matcher_type,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def reset(self, clear_keyframes: bool = True):
        """Reset tracker state."""
        self.prev_gray = None
        self.prev_points = None
        self.prev_descriptors = None
        self.prev_keypoints_cv = None
        self.frame_index = 0
        if clear_keyframes:
            self.keyframes.clear()
        self.metrics = TrackingMetrics()

    def get_metrics(self) -> TrackingMetrics:
        """Get accumulated tracking metrics."""
        return self.metrics

    def get_available_detectors(self) -> List[str]:
        """Get list of available detector types."""
        available = ["orb", "akaze", "brisk", "gftt_orb"]
        
        # Check for optional detectors
        try:
            cv2.SIFT_create()
            available.append("sift")
        except AttributeError:
            pass
        
        try:
            cv2.xfeatures2d.BriefDescriptorExtractor_create()
            available.append("fast_brief")
        except AttributeError:
            pass
        
        return available

    def set_detector(self, detector_type: str):
        """Change the feature detector at runtime."""
        self.config.method = detector_type
        self.detector, self.descriptor_extractor, self._norm = (
            FeatureDetectorFactory.create_detector(detector_type, self.config)
        )
        self.matcher = FeatureDetectorFactory.create_matcher(
            self.config.matcher_type, self._norm
        )
        self.bf_matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING if self._norm == "hamming" else cv2.NORM_L2,
            crossCheck=True
        )
        LOGGER.info("Detector changed to: %s", detector_type)

    def process_frame(self, frame: np.ndarray) -> TrackingFrameResult:
        """Process a frame and return tracked features."""
        if frame is None or frame.size == 0:
            raise ValueError("Frame cannot be empty.")

        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_index += 1
        self.metrics.total_frames += 1

        keypoints_array: Optional[np.ndarray] = None
        descriptors: Optional[np.ndarray] = None
        matches: Optional[np.ndarray] = None
        keypoints_cv: Optional[List[cv2.KeyPoint]] = None
        used_optical_flow = False
        optical_flow_quality = 0.0

        # Try optical flow tracking first
        if (
            self.config.use_optical_flow
            and self.prev_gray is not None
            and self.prev_points is not None
            and len(self.prev_points) > 0
        ):
            tracked, tracked_prev, quality = self._track_optical_flow_adaptive(
                self.prev_gray, gray, self.prev_points
            )
            if tracked is not None and len(tracked) > 0:
                keypoints_array = tracked
                matches = np.hstack((tracked, tracked_prev)).astype(np.float32)
                used_optical_flow = True
                optical_flow_quality = quality
                self.metrics.optical_flow_frames += 1

        # Check if reacquisition needed
        need_reacquire = (
            keypoints_array is None
            or len(keypoints_array) < self.config.reacquire_threshold
        )

        if need_reacquire:
            keypoints_cv, descriptors = self._detect_features(gray)
            keypoints_array = self._keypoints_to_array(keypoints_cv)
            matches = self._match_keyframes_advanced(descriptors, keypoints_cv)
            used_optical_flow = False
            self.metrics.detection_frames += 1
            
            if used_optical_flow is False and self.prev_points is not None:
                self.metrics.reacquisition_count += 1

            self._maybe_add_keyframe(keypoints_array, descriptors, keypoints_cv)
            self.prev_descriptors = descriptors
            self.prev_keypoints_cv = keypoints_cv
        else:
            descriptors = None

        # Update state
        if keypoints_array is not None and len(keypoints_array):
            self.prev_points = keypoints_array.reshape(-1, 1, 2).astype(np.float32)
        else:
            self.prev_points = None

        self.prev_gray = gray

        # Update running average
        if keypoints_array is not None:
            n = self.metrics.total_frames
            self.metrics.avg_features_tracked = (
                (self.metrics.avg_features_tracked * (n - 1) + len(keypoints_array)) / n
            )
        if optical_flow_quality > 0:
            n = self.metrics.optical_flow_frames
            self.metrics.avg_optical_flow_quality = (
                (self.metrics.avg_optical_flow_quality * (n - 1) + optical_flow_quality) / n
            )

        # Get feature responses if available
        responses = None
        if keypoints_cv is not None:
            responses = np.array([kp.response for kp in keypoints_cv], dtype=np.float32)

        result = TrackingFrameResult(
            keypoints=keypoints_array if keypoints_array is not None else np.empty((0, 2), dtype=np.float32),
            descriptors=descriptors,
            matches=matches,
            source="optical_flow" if used_optical_flow and not need_reacquire else "detection",
            tracked_count=int(len(keypoints_array) if keypoints_array is not None else 0),
            reacquired=need_reacquire,
            timestamp=self.frame_index,
            detector_type=self.config.method,
            optical_flow_quality=optical_flow_quality,
            keyframe_match_count=len(matches) if matches is not None else 0,
            feature_responses=responses,
        )
        return result

    # ------------------------------------------------------------------ #
    # Feature Detection
    # ------------------------------------------------------------------ #
    def _detect_features(self, gray: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Detect features using the configured detector."""
        if self.config.use_grid_detection:
            return self._detect_features_grid(gray)
        
        if self.detector is not None:
            keypoints = self.detector.detect(gray, None)
            # Limit to max_features by response
            if len(keypoints) > self.config.max_features:
                keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
                keypoints = keypoints[:self.config.max_features]
            keypoints, descriptors = self.descriptor_extractor.compute(gray, keypoints)
        else:
            # Use Good Features to Track
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.config.max_features,
                qualityLevel=self.config.quality_level,
                minDistance=self.config.min_distance,
            )
            if corners is None:
                return [], None
            keypoints = [
                cv2.KeyPoint(x=float(pt[0][0]), y=float(pt[0][1]), size=31)
                for pt in corners
            ]
            keypoints, descriptors = self.descriptor_extractor.compute(gray, keypoints)
        
        return keypoints or [], descriptors

    def _detect_features_grid(self, gray: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Detect features using grid-based distribution."""
        h, w = gray.shape
        cell_h = h // self.config.grid_rows
        cell_w = w // self.config.grid_cols
        features_per_cell = self.config.max_features // (self.config.grid_rows * self.config.grid_cols)
        
        all_keypoints = []
        
        for row in range(self.config.grid_rows):
            for col in range(self.config.grid_cols):
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                
                if row == self.config.grid_rows - 1:
                    y2 = h
                if col == self.config.grid_cols - 1:
                    x2 = w
                
                cell = gray[y1:y2, x1:x2]
                
                if self.detector is not None:
                    keypoints = self.detector.detect(cell, None)
                else:
                    corners = cv2.goodFeaturesToTrack(
                        cell,
                        maxCorners=features_per_cell,
                        qualityLevel=self.config.quality_level,
                        minDistance=self.config.min_distance,
                    )
                    if corners is None:
                        continue
                    keypoints = [
                        cv2.KeyPoint(x=float(pt[0][0]), y=float(pt[0][1]), size=31)
                        for pt in corners
                    ]
                
                # Sort by response and limit
                keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
                keypoints = keypoints[:features_per_cell]
                
                # Offset to global coordinates
                for kp in keypoints:
                    kp.pt = (kp.pt[0] + x1, kp.pt[1] + y1)
                    all_keypoints.append(kp)
        
        if not all_keypoints:
            return [], None
        
        # Compute descriptors
        all_keypoints, descriptors = self.descriptor_extractor.compute(gray, all_keypoints)
        return all_keypoints or [], descriptors

    # ------------------------------------------------------------------ #
    # Optical Flow
    # ------------------------------------------------------------------ #
    def _track_optical_flow_adaptive(
        self,
        prev_gray: np.ndarray,
        gray: np.ndarray,
        prev_points: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Track features with adaptive optical flow and quality filtering.
        
        Returns:
            (tracked_points, prev_points, quality_score)
        """
        lk_params = dict(
            winSize=(self.config.optical_flow_win_size, self.config.optical_flow_win_size),
            maxLevel=self.config.optical_flow_max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.config.optical_flow_criteria_count,
                self.config.optical_flow_criteria_eps,
            ),
            minEigThreshold=self.config.optical_flow_min_eig_threshold,
        )
        
        # Forward tracking
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_points, None, **lk_params
        )
        
        if next_pts is None or status is None:
            return None, None, 0.0

        status_flat = status.flatten()
        
        if self.config.adaptive_optical_flow:
            # Backward tracking for consistency check
            back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray, prev_gray, next_pts, None, **lk_params
            )
            
            if back_pts is not None and back_status is not None:
                # Check forward-backward consistency
                fb_error = np.linalg.norm(
                    prev_points.reshape(-1, 2) - back_pts.reshape(-1, 2),
                    axis=1
                )
                consistency_mask = fb_error < (self.config.optical_flow_win_size * 0.5)
                status_flat = status_flat & consistency_mask.astype(np.uint8) & back_status.flatten()

        good_new = next_pts[status_flat == 1].reshape(-1, 2)
        good_prev = prev_points[status_flat == 1].reshape(-1, 2)

        if len(good_new) == 0:
            return None, None, 0.0

        # Compute quality score
        quality = len(good_new) / len(prev_points) if len(prev_points) > 0 else 0.0

        return good_new, good_prev, quality

    # Legacy method for backwards compatibility
    def _track_optical_flow(
        self,
        prev_gray: np.ndarray,
        gray: np.ndarray,
        prev_points: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        tracked, prev, _ = self._track_optical_flow_adaptive(prev_gray, gray, prev_points)
        return tracked, prev

    # ------------------------------------------------------------------ #
    # Keyframe Matching
    # ------------------------------------------------------------------ #
    def _match_keyframes_advanced(
        self,
        descriptors: Optional[np.ndarray],
        keypoints_cv: List[cv2.KeyPoint],
    ) -> Optional[np.ndarray]:
        """Match against keyframes with ratio test."""
        if descriptors is None or descriptors.size == 0 or not self.keyframes or not keypoints_cv:
            return None

        best_matches = None
        best_keyframe = None
        best_match_count = 0

        for keyframe in reversed(self.keyframes):
            try:
                # Use knnMatch for ratio test
                knn_matches = self.matcher.knnMatch(descriptors, keyframe.descriptors, k=2)
                
                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in knn_matches:
                    if len(match_pair) >= 2:
                        m, n = match_pair
                        if m.distance < self.config.match_ratio_threshold * n.distance:
                            good_matches.append(m)
                    elif len(match_pair) == 1:
                        # Only one match found, use if distance is good
                        if match_pair[0].distance < self.config.min_match_distance:
                            good_matches.append(match_pair[0])
                
                if len(good_matches) > best_match_count:
                    best_matches = good_matches
                    best_keyframe = keyframe
                    best_match_count = len(good_matches)
                    
            except cv2.error as e:
                LOGGER.debug("Matching failed for keyframe %d: %s", keyframe.frame_index, e)
                continue

        if not best_matches or best_keyframe is None:
            return None

        # Sort by distance
        best_matches = sorted(best_matches, key=lambda m: m.distance)
        
        curr_pts = np.array([keypoints_cv[m.queryIdx].pt for m in best_matches], dtype=np.float32)
        ref_pts = best_keyframe.keypoints[[m.trainIdx for m in best_matches]]
        
        return np.hstack((curr_pts, ref_pts)).astype(np.float32)

    # Legacy method for backwards compatibility
    def _match_keyframes(
        self,
        descriptors: Optional[np.ndarray],
        keypoints_cv: List[cv2.KeyPoint],
    ) -> Optional[np.ndarray]:
        return self._match_keyframes_advanced(descriptors, keypoints_cv)

    # ------------------------------------------------------------------ #
    # Keyframe Management
    # ------------------------------------------------------------------ #
    def _maybe_add_keyframe(
        self,
        keypoints: Optional[np.ndarray],
        descriptors: Optional[np.ndarray],
        keypoints_cv: Optional[List[cv2.KeyPoint]] = None,
    ):
        """Add a keyframe if conditions are met."""
        if (
            descriptors is None
            or keypoints is None
            or len(keypoints) < self.config.min_keyframe_features
        ):
            return

        should_add = not self.keyframes or (self.frame_index % self.config.keyframe_interval == 0)
        if not should_add:
            return

        # Compute quality score from keypoint responses
        quality_score = 0.0
        responses = None
        if keypoints_cv is not None:
            responses = np.array([kp.response for kp in keypoints_cv], dtype=np.float32)
            if len(responses) > 0:
                quality_score = float(np.mean(responses))

        # Check quality threshold
        if quality_score < self.config.keyframe_quality_threshold and self.keyframes:
            return

        keyframe = Keyframe(
            keypoints=keypoints.astype(np.float32).copy(),
            descriptors=descriptors.copy(),
            frame_index=self.frame_index,
            quality_score=quality_score,
            feature_responses=responses,
        )
        self.keyframes.append(keyframe)
        self.metrics.keyframes_created += 1
        
        if len(self.keyframes) > self.config.max_keyframes:
            self.keyframes.pop(0)
        
        LOGGER.debug(
            "Added keyframe %d with %d features (quality: %.3f)",
            self.frame_index,
            len(keypoints),
            quality_score,
        )

    def get_keyframe_info(self) -> List[Dict]:
        """Get information about stored keyframes."""
        return [
            {
                "frame_index": kf.frame_index,
                "num_features": len(kf.keypoints),
                "quality_score": kf.quality_score,
            }
            for kf in self.keyframes
        ]

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _keypoints_to_array(keypoints: Optional[List[cv2.KeyPoint]]) -> np.ndarray:
        if not keypoints:
            return np.empty((0, 2), dtype=np.float32)
        return np.array([kp.pt for kp in keypoints], dtype=np.float32)
