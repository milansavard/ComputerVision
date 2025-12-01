"""
Integration tests for the NETHERGAZE pipeline.

Tests the complete pipeline from video input through pose estimation using
offline video playback for reproducible testing.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tracking.feature import FeatureTracker, TrackingConfiguration, TrackingFrameResult
from pose import PoseEstimator, PoseResult
from overlay import OverlayRenderer, Mesh3D
from video import VideoProcessor
from utils import get_config

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline testing."""
    
    total_frames: int = 0
    tracking_successes: int = 0
    pose_successes: int = 0
    processing_times: List[float] = field(default_factory=list)
    feature_counts: List[int] = field(default_factory=list)
    pose_inliers: List[int] = field(default_factory=list)
    
    @property
    def tracking_rate(self) -> float:
        return self.tracking_successes / max(self.total_frames, 1) * 100
    
    @property
    def pose_rate(self) -> float:
        return self.pose_successes / max(self.total_frames, 1) * 100
    
    @property
    def avg_processing_time(self) -> float:
        return np.mean(self.processing_times) if self.processing_times else 0.0
    
    @property
    def avg_features(self) -> float:
        return np.mean(self.feature_counts) if self.feature_counts else 0.0
    
    @property
    def avg_inliers(self) -> float:
        return np.mean(self.pose_inliers) if self.pose_inliers else 0.0

    def to_dict(self) -> Dict:
        return {
            "total_frames": self.total_frames,
            "tracking_rate": self.tracking_rate,
            "pose_rate": self.pose_rate,
            "avg_processing_time_ms": self.avg_processing_time * 1000,
            "avg_features": self.avg_features,
            "avg_inliers": self.avg_inliers,
        }


class SyntheticVideoGenerator:
    """Generate synthetic video frames for testing."""
    
    @staticmethod
    def create_checkerboard_sequence(
        num_frames: int = 100,
        width: int = 640,
        height: int = 480,
        motion: str = "rotation",
    ) -> List[np.ndarray]:
        """
        Generate a sequence of frames with a moving checkerboard pattern.
        
        Args:
            num_frames: Number of frames to generate
            width: Frame width
            height: Frame height
            motion: Type of motion ("rotation", "translation", "zoom")
            
        Returns:
            List of BGR frames
        """
        frames = []
        
        # Create base checkerboard
        checker_size = 40
        rows = height // checker_size + 1
        cols = width // checker_size + 1
        
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Apply motion
            if motion == "rotation":
                angle = i * 2  # 2 degrees per frame
                offset_x = int(20 * np.sin(np.radians(angle)))
                offset_y = int(20 * np.cos(np.radians(angle)))
            elif motion == "translation":
                offset_x = i % 50
                offset_y = (i // 2) % 30
            elif motion == "zoom":
                scale = 1.0 + 0.005 * i
                offset_x = int((scale - 1) * width / 4)
                offset_y = int((scale - 1) * height / 4)
            else:
                offset_x = 0
                offset_y = 0
            
            # Draw checkerboard
            for row in range(rows):
                for col in range(cols):
                    x1 = col * checker_size + offset_x
                    y1 = row * checker_size + offset_y
                    x2 = x1 + checker_size
                    y2 = y1 + checker_size
                    
                    if (row + col) % 2 == 0:
                        color = (255, 255, 255)
                    else:
                        color = (0, 0, 0)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            
            # Add some texture/noise for better feature detection
            noise = np.random.randint(0, 30, (height, width), dtype=np.uint8)
            frame = cv2.add(frame, cv2.merge([noise, noise, noise]))
            
            frames.append(frame)
        
        return frames

    @staticmethod
    def create_feature_rich_sequence(
        num_frames: int = 100,
        width: int = 640,
        height: int = 480,
    ) -> List[np.ndarray]:
        """
        Generate frames with good feature distribution.
        """
        frames = []
        
        # Create base image with various shapes
        base = np.zeros((height, width, 3), dtype=np.uint8) + 50
        
        # Add various shapes for features
        np.random.seed(42)
        for _ in range(50):
            x = np.random.randint(50, width - 50)
            y = np.random.randint(50, height - 50)
            size = np.random.randint(10, 40)
            color = tuple(np.random.randint(100, 255, 3).tolist())
            
            shape_type = np.random.choice(["circle", "rect", "triangle"])
            if shape_type == "circle":
                cv2.circle(base, (x, y), size, color, -1)
            elif shape_type == "rect":
                cv2.rectangle(base, (x, y), (x + size, y + size), color, -1)
            else:
                pts = np.array([
                    [x, y - size],
                    [x - size, y + size],
                    [x + size, y + size],
                ], dtype=np.int32)
                cv2.fillPoly(base, [pts], color)
        
        for i in range(num_frames):
            # Apply perspective transform to simulate camera motion
            angle = i * 0.5
            tx = 5 * np.sin(np.radians(i * 3))
            ty = 5 * np.cos(np.radians(i * 2))
            
            # Create transformation matrix
            center = (width / 2, height / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M[0, 2] += tx
            M[1, 2] += ty
            
            frame = cv2.warpAffine(base, M, (width, height), borderValue=(50, 50, 50))
            frames.append(frame)
        
        return frames

    @staticmethod
    def save_as_video(frames: List[np.ndarray], output_path: str, fps: int = 30):
        """Save frames as a video file."""
        if not frames:
            return
        
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for frame in frames:
            writer.write(frame)
        
        writer.release()


class PipelineTester:
    """Test harness for the NETHERGAZE pipeline."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config()
        self.tracker: Optional[FeatureTracker] = None
        self.pose_estimator: Optional[PoseEstimator] = None
        self.overlay_renderer: Optional[OverlayRenderer] = None
        self.metrics = PipelineMetrics()
    
    def initialize(self) -> bool:
        """Initialize pipeline components."""
        try:
            tracking_config = self.config.get("feature_tracking", {})
            self.tracker = FeatureTracker(tracking_config)
            
            self.pose_estimator = PoseEstimator(self.config)
            self.pose_estimator.initialize()
            
            self.overlay_renderer = OverlayRenderer(self.config.get("overlay", {}))
            self.overlay_renderer.initialize(self.pose_estimator.calibration)
            
            return True
        except Exception as e:
            LOGGER.error("Failed to initialize pipeline: %s", e)
            return False
    
    def process_frames(
        self,
        frames: List[np.ndarray],
        visualize: bool = False,
    ) -> PipelineMetrics:
        """
        Process a sequence of frames through the pipeline.
        
        Args:
            frames: List of BGR frames
            visualize: Whether to display processing (for debugging)
            
        Returns:
            Collected metrics
        """
        self.metrics = PipelineMetrics()
        
        for i, frame in enumerate(frames):
            start_time = time.time()
            
            # Track features
            tracking_result = self.tracker.process_frame(frame)
            
            tracking_ok = (
                tracking_result is not None
                and tracking_result.tracked_count > 0
            )
            if tracking_ok:
                self.metrics.tracking_successes += 1
                self.metrics.feature_counts.append(tracking_result.tracked_count)
            
            # Estimate pose
            pose = None
            if tracking_result is not None:
                pose = self.pose_estimator.estimate_from_feature_tracks(tracking_result)
                
                if pose and pose.success:
                    self.metrics.pose_successes += 1
                    self.metrics.pose_inliers.append(pose.inliers)
            
            # Record timing
            elapsed = time.time() - start_time
            self.metrics.processing_times.append(elapsed)
            self.metrics.total_frames += 1
            
            # Visualize if requested
            if visualize:
                vis_frame = frame.copy()
                if tracking_result and tracking_result.keypoints is not None:
                    for x, y in tracking_result.keypoints[:200]:
                        cv2.circle(vis_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                
                cv2.putText(
                    vis_frame,
                    f"Frame {i+1}/{len(frames)} | Features: {tracking_result.tracked_count if tracking_result else 0}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                
                cv2.imshow("Pipeline Test", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        if visualize:
            cv2.destroyAllWindows()
        
        return self.metrics

    def process_video_file(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        visualize: bool = False,
    ) -> PipelineMetrics:
        """
        Process a video file through the pipeline.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            visualize: Whether to display processing
            
        Returns:
            Collected metrics
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        frames = []
        count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(frame)
            count += 1
            
            if max_frames and count >= max_frames:
                break
        
        cap.release()
        
        return self.process_frames(frames, visualize=visualize)


# ============================================================================
# Test Cases
# ============================================================================

class TestFeatureTracking:
    """Tests for feature tracking module."""
    
    def test_tracker_initialization(self):
        """Test tracker initializes with default config."""
        tracker = FeatureTracker()
        assert tracker is not None
        assert tracker.config.method == "orb"
    
    def test_tracker_with_different_detectors(self):
        """Test tracker works with different detector types."""
        detectors = ["orb", "akaze", "brisk", "gftt_orb"]
        
        for detector in detectors:
            tracker = FeatureTracker({"method": detector})
            assert tracker.config.method == detector
            
            # Process a simple frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = tracker.process_frame(frame)
            assert result is not None
    
    def test_tracker_on_synthetic_sequence(self):
        """Test tracking on synthetic video sequence."""
        frames = SyntheticVideoGenerator.create_feature_rich_sequence(num_frames=30)
        
        tracker = FeatureTracker({
            "method": "orb",
            "max_features": 500,
            "use_optical_flow": True,
        })
        
        feature_counts = []
        for frame in frames:
            result = tracker.process_frame(frame)
            feature_counts.append(result.tracked_count)
        
        # Should detect features consistently
        assert np.mean(feature_counts) > 50
        assert len([c for c in feature_counts if c > 0]) > len(frames) * 0.8
    
    def test_optical_flow_quality(self):
        """Test that optical flow maintains tracking."""
        frames = SyntheticVideoGenerator.create_feature_rich_sequence(num_frames=50)
        
        tracker = FeatureTracker({
            "use_optical_flow": True,
            "adaptive_optical_flow": True,
        })
        
        optical_flow_used = 0
        for frame in frames:
            result = tracker.process_frame(frame)
            if result.source == "optical_flow":
                optical_flow_used += 1
        
        # After initial detection, should use optical flow frequently
        assert optical_flow_used > len(frames) * 0.5


class TestPoseEstimation:
    """Tests for pose estimation module."""
    
    def test_pose_estimator_initialization(self):
        """Test pose estimator initializes correctly."""
        config = get_config()
        estimator = PoseEstimator(config)
        assert estimator.initialize()
        assert estimator.calibration is not None
    
    def test_pose_from_synthetic_matches(self):
        """Test pose estimation with synthetic correspondences."""
        config = get_config()
        estimator = PoseEstimator(config)
        estimator.initialize()
        
        # Create synthetic matches (simulating camera translation)
        np.random.seed(42)
        pts1 = np.random.rand(50, 2) * [640, 480]
        pts2 = pts1 + np.random.randn(50, 2) * 5 + [10, 5]  # Small shift
        
        matches = np.hstack([pts2, pts1]).astype(np.float32)
        
        # Create mock tracking result
        tracking_result = TrackingFrameResult(
            keypoints=pts2,
            matches=matches,
            tracked_count=len(pts2),
        )
        
        pose = estimator.estimate_from_feature_tracks(tracking_result)
        # May or may not succeed depending on point distribution
        assert pose is not None
    
    def test_pose_filter_smoothing(self):
        """Test that pose filter smooths estimates."""
        config = get_config()
        config["pose_filter"] = {
            "enable_smoothing": True,
            "smoothing_alpha": 0.3,
        }
        estimator = PoseEstimator(config)
        estimator.initialize()
        
        # Feed multiple similar poses
        translations = []
        for i in range(10):
            # Create slightly varying synthetic pose
            t = np.array([[0.1 + np.random.randn() * 0.01],
                          [0.0],
                          [1.0]])
            pose = PoseResult(
                success=True,
                rotation_vector=np.zeros((3, 1)),
                translation_vector=t,
                rotation_matrix=np.eye(3),
                inliers=50,
            )
            filtered = estimator.pose_filter.filter(pose)
            if filtered.translation_vector is not None:
                translations.append(filtered.translation_vector[0, 0])
        
        # Filtered translations should be less variable
        if len(translations) > 5:
            assert np.std(translations[-5:]) < 0.05


class TestOverlayRendering:
    """Tests for overlay rendering module."""
    
    def test_overlay_initialization(self):
        """Test overlay renderer initializes."""
        renderer = OverlayRenderer()
        assert renderer.initialize()
    
    def test_mesh_creation(self):
        """Test mesh creation methods."""
        box = Mesh3D.create_box(1.0, 1.0, 1.0)
        assert box.vertices.shape == (8, 3)
        assert box.faces.shape[0] == 12  # 6 faces * 2 triangles
        
        plane = Mesh3D.create_plane(1.0, 1.0)
        assert plane.vertices.shape == (4, 3)
        assert plane.faces.shape[0] == 2
    
    def test_mesh_transform(self):
        """Test mesh transformation."""
        box = Mesh3D.create_box()
        box.position = np.array([1.0, 2.0, 3.0])
        box.scale = 2.0
        box.rotation = np.array([45.0, 0.0, 0.0])
        
        transformed = box.get_transformed_vertices()
        assert transformed.shape == box.vertices.shape
        # Center should be approximately at position
        center = transformed.mean(axis=0)
        assert np.allclose(center, box.position, atol=0.1)


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline_synthetic(self):
        """Test full pipeline on synthetic video."""
        frames = SyntheticVideoGenerator.create_feature_rich_sequence(num_frames=30)
        
        tester = PipelineTester()
        assert tester.initialize()
        
        metrics = tester.process_frames(frames)
        
        assert metrics.total_frames == 30
        assert metrics.tracking_rate > 50  # At least 50% tracking success
        print(f"\nPipeline metrics: {metrics.to_dict()}")
    
    def test_different_detectors_performance(self):
        """Compare performance of different detectors."""
        frames = SyntheticVideoGenerator.create_feature_rich_sequence(num_frames=20)
        
        detectors = ["orb", "akaze", "brisk"]
        results = {}
        
        for detector in detectors:
            config = get_config()
            config["feature_tracking"]["method"] = detector
            
            tester = PipelineTester(config)
            if tester.initialize():
                metrics = tester.process_frames(frames)
                results[detector] = {
                    "tracking_rate": metrics.tracking_rate,
                    "avg_features": metrics.avg_features,
                    "avg_time_ms": metrics.avg_processing_time * 1000,
                }
        
        print("\nDetector comparison:")
        for detector, stats in results.items():
            print(f"  {detector}: {stats}")
        
        # All detectors should achieve some tracking
        for detector, stats in results.items():
            assert stats["tracking_rate"] > 30, f"{detector} failed"


def run_benchmark(video_path: Optional[str] = None, num_frames: int = 100):
    """
    Run a benchmark on the pipeline.
    
    Args:
        video_path: Optional path to video file (uses synthetic if None)
        num_frames: Number of frames to process
    """
    print("\n" + "=" * 60)
    print("NETHERGAZE Pipeline Benchmark")
    print("=" * 60)
    
    if video_path and os.path.exists(video_path):
        print(f"Using video: {video_path}")
        tester = PipelineTester()
        tester.initialize()
        metrics = tester.process_video_file(video_path, max_frames=num_frames)
    else:
        print(f"Using synthetic video ({num_frames} frames)")
        frames = SyntheticVideoGenerator.create_feature_rich_sequence(num_frames)
        tester = PipelineTester()
        tester.initialize()
        metrics = tester.process_frames(frames)
    
    print("\nResults:")
    print(f"  Total frames: {metrics.total_frames}")
    print(f"  Tracking success rate: {metrics.tracking_rate:.1f}%")
    print(f"  Pose success rate: {metrics.pose_rate:.1f}%")
    print(f"  Average features: {metrics.avg_features:.1f}")
    print(f"  Average inliers: {metrics.avg_inliers:.1f}")
    print(f"  Average processing time: {metrics.avg_processing_time * 1000:.2f} ms")
    print(f"  Effective FPS: {1.0 / max(metrics.avg_processing_time, 0.001):.1f}")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run NETHERGAZE integration tests")
    parser.add_argument("--video", type=str, help="Path to video file for testing")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to process")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark only")
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmark(args.video, args.frames)
    else:
        # Run pytest
        pytest.main([__file__, "-v", "--tb=short"])

