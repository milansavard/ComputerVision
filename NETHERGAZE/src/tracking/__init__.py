"""
Tracking subpackage.

Provides marker-based and markerless tracking utilities with multiple
detector/descriptor combinations.

Supported detectors:
- ORB (default, fast and robust)
- FAST + BRIEF (very fast)
- AKAZE (scale/rotation invariant)
- BRISK (balanced)
- SIFT (most robust, requires opencv-contrib)
- GFTT + ORB (Good Features To Track with ORB descriptors)
"""

from .feature import (
    DetectorType,
    FeatureDetectorFactory,
    FeatureTracker,
    Keyframe,
    MatcherType,
    TrackingConfiguration,
    TrackingFrameResult,
    TrackingMetrics,
)

__all__ = [
    "DetectorType",
    "FeatureDetectorFactory",
    "FeatureTracker",
    "Keyframe",
    "MatcherType",
    "TrackingConfiguration",
    "TrackingFrameResult",
    "TrackingMetrics",
]

