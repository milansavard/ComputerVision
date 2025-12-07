# NETHERGAZE Design Overview

## Project Description

NETHERGAZE is a markerless augmented reality system that places virtual 3D objects in real-world scenes using natural feature tracking. Built entirely on classical computer vision techniques without neural networks.

## System Architecture

### High-Level Components

1. **Video Processing (`video.py`)**
   - Camera capture with multi-backend support (AVFoundation, Continuity Camera)
   - Configurable resolution and FPS
   - Frame preprocessing pipeline

2. **Feature Tracking (`tracking/feature.py`)**
   - ORB (Oriented FAST and Rotated BRIEF) feature detection
   - Lucas-Kanade optical flow for frame-to-frame tracking
   - Keyframe management with automatic re-detection
   - Configurable thresholds for robust tracking

3. **Pose Estimation (`pose.py`)**
   - Essential Matrix decomposition with RANSAC
   - Camera pose recovery (rotation + translation)
   - Temporal smoothing via Exponential Moving Average
   - Outlier rejection based on motion thresholds
   - Scale estimation (ground plane, known distance methods)

4. **Overlay Rendering (`overlay.py`)**
   - 3D wireframe objects (cube, pyramid, axes, box)
   - Solid shaded rendering with face sorting
   - Camera projection using calibration matrix
   - Alpha blending and compositing

5. **World Anchoring (`examples/demo_anchored_objects.py`)**
   - Anchor point system for establishing world origin
   - Objects stay fixed in 3D space as camera moves
   - Multiple object types including 3D chair
   - Real-time pose statistics display

6. **SLAM/Mapping (`mapping.py`)**
   - Sparse 3D point cloud from triangulated features
   - Keyframe management with covisibility graph
   - Loop closure detection and verification
   - Map persistence (save/load to JSON)

7. **Occlusion Handling (`occlusion.py`)**
   - Depth estimation from tracked features
   - Ground plane assumption
   - Occlusion mask generation
   - Depth-aware overlay rendering

8. **User Interface (`ui.py`)**
   - OpenCV window with keyboard controls
   - Live statistics display (FPS, tracking rate, pose rate)
   - Help overlay with control reference

9. **Application Entry (`main.py`)**
   - CLI interface with --fast and --verbose options
   - Launches world-anchored AR demo directly

## Data Flow

```
Camera → VideoProcessor → FeatureTracker → PoseEstimator → WorldAnchor → OverlayRenderer → Display
              ↓                  ↓                ↓
         Preprocessing    Optical Flow      Essential Matrix
                          Keyframes          RANSAC + EMA
```

## Technical Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| Vision Library | OpenCV 4.8+ (with contrib) |
| Feature Detection | ORB |
| Tracking | Lucas-Kanade Optical Flow |
| Pose | Essential Matrix + RANSAC |
| Rendering | OpenCV drawing (cv2.line, cv2.projectPoints) |
| Smoothing | Exponential Moving Average |

## Implementation Checklist

- [x] Video capture with multi-backend support
- [x] ORB feature detection and matching
- [x] Lucas-Kanade optical flow tracking
- [x] Keyframe management with re-detection
- [x] Essential Matrix pose estimation
- [x] Temporal smoothing (EMA filter)
- [x] Outlier rejection
- [x] 3D wireframe rendering
- [x] World-anchored objects
- [x] Multiple object types (cube, pyramid, axes, box, chair)
- [x] Real-time statistics display
- [x] SLAM/mapping module
- [x] Occlusion handling
- [x] Integration tests

## Key Parameters

Optimized tracking parameters for stable AR:

```python
max_features = 3000        # Detect many features
fast_threshold = 10        # Sensitive corner detection
quality_level = 0.005      # Accept weaker features
min_distance = 5.0         # Allow features closer together
reacquire_threshold = 500  # Re-detect features early
keyframe_interval = 8      # Frequent keyframe updates
orb_nlevels = 12          # More pyramid levels for scale
```

## Usage

```bash
python src/main.py              # Run AR demo
python src/main.py --fast       # High performance mode
```

Controls: SPACE (anchor), 1-5 (place objects), C (clear), Q (quit)
