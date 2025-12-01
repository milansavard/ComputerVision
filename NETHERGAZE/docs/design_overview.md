# NETHERGAZE Design Overview

## Project Description

NETHERGAZE is a markerless tracking toolkit built to power immersive AR overlays using feature-based vision pipelines.

## System Architecture

### High-Level Components

1. **Video Processing (`video.py`)**
   - Camera capture and frame preprocessing
   - Backend selection logic (AVFoundation, V4L2, etc.)

2. **Tracking Subpackage (`tracking/`)**
   - `feature.py`: ORB / GFTT-based keypoint detection, LK optical-flow updates, keyframe management for markerless tracking
   - Future modules: learned feature extraction, dense optical flow, global map management

3. **Pose Estimation (`pose.py`)**
   - Camera calibration loading (inline config or external JSON)
   - Essential-matrix based pose recovery for feature tracks
   - Optional PnP support for future reference targets
   - Temporal filtering and refinement (planned)

4. **Overlay Rendering (`overlay.py`)**
   - 2D HUD overlays
   - Optional 3D rendering pipeline (OpenGL / PyOpenGL)
   - Compositing and blending strategies

5. **User Interface (`ui.py`)**
   - OpenCV window creation, event handling, control toggles

6. **Application Orchestration (`main.py`)**
   - Future: runtime configuration, pipeline management, telemetry/logging hooks

7. **Utilities (`utils.py`)**
   - Configuration management, logging bootstrap, common helpers

## Data Flow

```
Camera → VideoProcessor → FeatureTracker → Pose → Overlay → UI Display
```

## Technical Requirements

- Python 3.9+
- `opencv-python` + `opencv-contrib-python` (ORB, optical flow)
- NumPy / SciPy for math operations
- PyOpenGL / pygame (planned for 3D overlays)

## Implementation Checklist

- [x] Video capture loop + UI shell
- [x] Markerless tracking scaffold (`tracking.feature`)
- [x] Camera calibration ingestion + PnP / Essential matrix pose solver
- [ ] Pose refinement (temporal smoothing, bundle adjustment)
- [ ] Overlay rendering MVP
- [ ] SLAM-lite map for persistent markerless anchors
- [ ] Full application orchestrator (`main.py`)

## Future Enhancements

- Hybrid tracking mode that blends different feature sources (e.g., learned keypoints)
- Multi-camera fusion
- GPU-accelerated feature extraction and rendering
- Configurable plugin architecture for custom trackers/overlays
