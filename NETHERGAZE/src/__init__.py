"""
NETHERGAZE - Computer Vision Toolkit.

This package provides functionality for:
- Real-time video capture
- Markerless tracking
- Camera pose estimation
- Virtual overlay rendering
- SLAM/mapping
- Occlusion handling
"""

from .tracking import FeatureTracker, TrackingConfiguration, TrackingFrameResult
from .pose import (
    CalibrationData,
    PoseEstimator,
    PoseFilter,
    PoseFilterConfig,
    PoseResult,
    ScaleEstimator,
    ScaleEstimatorConfig,
)
from .overlay import Mesh3D, Object3D, Overlay2D, OverlayConfiguration, OverlayRenderer
from .mapping import MapConfig, MapPoint, MapKeyframe, SparseMap, MapVisualizer
from .occlusion import (
    DepthEstimator,
    DepthSource,
    OcclusionConfig,
    OcclusionHandler,
    DepthAwareOverlayRenderer,
)
from .ui import UserInterface
from .video import VideoProcessor

__version__ = "0.2.0"
__author__ = "Milan Savard"

__all__ = [
    # Video & UI
    "VideoProcessor",
    "UserInterface",
    # Tracking
    "FeatureTracker",
    "TrackingConfiguration",
    "TrackingFrameResult",
    # Pose
    "CalibrationData",
    "PoseEstimator",
    "PoseFilter",
    "PoseFilterConfig",
    "PoseResult",
    "ScaleEstimator",
    "ScaleEstimatorConfig",
    # Overlay
    "Mesh3D",
    "Object3D",
    "Overlay2D",
    "OverlayConfiguration",
    "OverlayRenderer",
    # Mapping/SLAM
    "MapConfig",
    "MapPoint",
    "MapKeyframe",
    "SparseMap",
    "MapVisualizer",
    # Occlusion
    "DepthEstimator",
    "DepthSource",
    "OcclusionConfig",
    "OcclusionHandler",
    "DepthAwareOverlayRenderer",
]
