"""
Tests for feature-based markerless tracking utilities.
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from tracking.feature import FeatureTracker  # type: ignore


class TestFeatureTracker(unittest.TestCase):
    """Validate feature tracker behaviour with optical flow and keyframes."""

    def make_tracker(self, **overrides) -> FeatureTracker:
        config = {
            "method": "orb",
            "max_features": 600,
            "reacquire_threshold": 20,
            "keyframe_interval": 1,
            "min_keyframe_features": 10,
        }
        config.update(overrides)
        return FeatureTracker(config)

    def test_process_frame_returns_keypoints(self):
        """Tracker should return keypoints for textured frames."""
        tracker = self.make_tracker()

        synthetic = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(synthetic, "NETHERGAZE", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.circle(synthetic, (240, 120), 50, (255, 255, 255), -1)

        result = tracker.process_frame(synthetic)
        self.assertGreater(result.tracked_count, 0)
        self.assertEqual(result.source, "detection")

    def test_optical_flow_tracks_motion(self):
        """Subsequent frames should use optical flow to track features."""
        tracker = self.make_tracker(reacquire_threshold=0)

        base_frame = np.zeros((320, 320, 3), dtype=np.uint8)
        for y in range(40, 280, 40):
            for x in range(40, 280, 40):
                cv2.circle(base_frame, (x, y), 5, (255, 255, 255), -1)
        cv2.putText(base_frame, "FLOW", (70, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        tracker.process_frame(base_frame)
        matrix = np.float32([[1, 0, 5], [0, 1, 0]])
        shifted_frame = cv2.warpAffine(base_frame, matrix, (320, 320), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        result = tracker.process_frame(shifted_frame)

        self.assertIsNotNone(result.matches)
        self.assertGreater(result.matches.shape[0], 0)
        self.assertEqual(result.source, "optical_flow")

        delta = result.matches[:, :2] - result.matches[:, 2:]
        mean_shift = delta.mean(axis=0)
        self.assertAlmostEqual(mean_shift[0], 5.0, delta=1.5)

    def test_keyframe_reacquire_matches_map(self):
        """After reset the tracker should use stored keyframes for matches."""
        tracker = self.make_tracker(reacquire_threshold=200)

        textured = np.zeros((320, 320, 3), dtype=np.uint8)
        cv2.putText(textured, "KF", (110, 180), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 5)

        first = tracker.process_frame(textured)
        self.assertFalse(first.keypoints.size == 0)
        self.assertGreater(len(tracker.keyframes), 0)

        tracker.reset(clear_keyframes=False)

        second = tracker.process_frame(textured)
        self.assertTrue(second.reacquired)
        self.assertIsNotNone(second.matches)
        self.assertGreater(second.matches.shape[0], 0)

    def test_reset_clears_state(self):
        """Reset should drop prior frame state while keeping config."""
        tracker = self.make_tracker()

        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        tracker.process_frame(frame)
        tracker.reset()

        self.assertIsNone(tracker.prev_gray)
        self.assertIsNone(tracker.prev_points)


if __name__ == "__main__":
    unittest.main()

