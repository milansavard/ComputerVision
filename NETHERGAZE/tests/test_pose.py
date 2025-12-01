"""
Tests for pose estimation functionality.
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pose import PoseEstimator, PoseResult  # type: ignore
from tracking.feature import TrackingFrameResult  # type: ignore


class TestPoseEstimator(unittest.TestCase):
    """Test cases for markerless pose estimation."""

    def setUp(self):
        calibration = {
            "camera_matrix": [
                [800.0, 0.0, 320.0],
                [0.0, 800.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        self.config = {
            "axis_length": 0.08,
            "calibration": calibration,
        }
        self.estimator = PoseEstimator(self.config)
        self.estimator.initialize()

    def test_pose_estimator_initialization(self):
        """Ensure calibration matrix is loaded."""
        self.assertTrue(self.estimator.initialized)
        self.assertIsNotNone(self.estimator.calibration)
        self.assertEqual(self.estimator.calibration.camera_matrix.shape, (3, 3))

    def test_project_axes(self):
        """Projected axes should lie near the expected image location."""
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.array([[0.0], [0.0], [0.4]], dtype=np.float64)
        pose = PoseResult(
            success=True,
            rotation_vector=rvec,
            translation_vector=tvec,
            rotation_matrix=np.eye(3, dtype=np.float64),
            method="markerless",
            inliers=100,
        )
        projected_axes = self.estimator.project_axes(pose, axis_length=self.config["axis_length"])

        self.assertIsNotNone(projected_axes)
        self.assertEqual(projected_axes.shape, (4, 2))
        # Check that axes are projected near the image centre (cx=320, cy=240)
        cx, cy = self.estimator.calibration.camera_matrix[0, 2], self.estimator.calibration.camera_matrix[1, 2]
        self.assertLess(np.abs(projected_axes[0][0] - cx), 20)
        self.assertLess(np.abs(projected_axes[0][1] - cy), 20)

    def test_pose_from_feature_tracks(self):
        """Recover relative pose from synthetic feature correspondences."""
        num_points = 200
        points_3d = np.random.uniform(-0.1, 0.1, (num_points, 3)).astype(np.float64)
        points_3d[:, 2] += 0.6

        rvec1 = np.zeros((3, 1))
        tvec1 = np.zeros((3, 1))

        rvec2 = np.array([[0.02], [-0.015], [0.01]], dtype=np.float64)
        tvec2 = np.array([[0.05], [0.0], [0.0]], dtype=np.float64)

        K = self.estimator.calibration.camera_matrix
        dist = self.estimator.calibration.dist_coeffs

        img1, _ = cv2.projectPoints(points_3d, rvec1, tvec1, K, dist)
        img2, _ = cv2.projectPoints(points_3d, rvec2, tvec2, K, dist)

        pts1 = img2.reshape(-1, 2)
        pts0 = img1.reshape(-1, 2)

        matches = np.hstack((pts1, pts0)).astype(np.float32)
        tracking_result = TrackingFrameResult(
            keypoints=None,
            descriptors=None,
            matches=matches,
        )

        pose = self.estimator.estimate_from_feature_tracks(tracking_result)
        self.assertTrue(pose.success)
        self.assertEqual(pose.method, "markerless")
        self.assertGreater(pose.inliers, 20)

        recovered_direction = pose.translation_vector.flatten() / np.linalg.norm(pose.translation_vector)
        expected_direction = tvec2.flatten() / np.linalg.norm(tvec2)
        cosine_similarity = np.dot(recovered_direction, expected_direction)
        self.assertGreater(abs(cosine_similarity), 0.8)


if __name__ == "__main__":
    unittest.main()

