# NETHERGAZE Implementation Progress

## Completed Components ✅

### 1. Utilities (`src/utils.py`)
- Centralised logging setup with sensible defaults
- Configuration loader tuned for markerless tracking (feature + pose + overlay settings)
- Helper utilities (timestamping, directory creation)
- Pose filter and overlay configuration defaults

### 2. Video + UI Loop
- `src/video.py` (VideoProcessor) handles capture, preprocessing, and cleanup
- `src/ui.py` (UserInterface) renders frames, keyboard controls, and overlays
- `examples/run_demo.py` integrates the loop with runtime mode switching

### 3. Tracking Foundations
- Markerless tracker in `src/tracking/feature.py` adds ORB detection, LK optical-flow updates, keyframe map, and robustness thresholds
- Package exports updated (`src/__init__.py`) to expose markerless stack
- Demo renders tracked features + pose overlays in real time

### 4. Pose Estimation (`src/pose.py`)
- Calibration loader (inline config or external JSON)
- Markerless pose from feature correspondences (Essential matrix + recoverPose)
- Axis projection helper for visual overlays
- **Temporal smoothing** via EMA filter with configurable alpha
- **Outlier rejection** based on translation/rotation jump thresholds
- **Median filtering** option for additional robustness
- **Multi-method scale recovery** via `ScaleEstimator` class:
  - Known distance between tracked points
  - Ground plane assumption (camera at known height)
  - Object size reference
  - Manual scale factor
  - Auto-selection of best method
- **Point triangulation** for 3D reconstruction and scale estimation
- Pose decomposition helper (Euler angles, position, distance)
- Unit tests in `tests/test_pose.py`

### 7. Camera Calibration Tool (`examples/calibrate_camera.py`)
- **Interactive capture mode** with live chessboard detection
- **Batch calibration** from existing image files
- **Preview mode** with undistortion visualization
- Corner sub-pixel refinement for accuracy
- JSON export compatible with pipeline calibration loader
- CLI with configurable board size and square dimensions

### 5. Overlay Rendering (`src/overlay.py`)
- 2D overlay primitives (text, rectangles, circles, lines, polygons)
- 3D wireframe objects (cube, pyramid, axes, grid, custom)
- Projection using camera calibration and pose
- Alpha blending and layer compositing
- Overlay image projection onto 3D surfaces (homography warp)
- Configuration via `OverlayConfiguration` dataclass

### 6. Pipeline Orchestration (`src/main.py`)
- Full pipeline: Capture → Track → Pose → Overlay → Display
- CLI argument parsing (--config, --camera, --video, --width, --height, --verbose)
- Runtime statistics (FPS, tracking/pose success rates)
- Graceful shutdown via signal handlers
- Video file playback support
- NETHERGAZEApp class encapsulates all pipeline logic

## Roadmap (Markerless Focus)

### ✅ Completed
1. **Camera Calibration Tooling** - Interactive chessboard calibration with JSON export
2. **Scale Recovery** - Multi-method scale estimation (known distance, ground plane, object size)
3. **Pose Refinement** - Temporal smoothing, outlier rejection, median filtering

### ✅ Recently Completed
4. **Markerless Tracking Iteration**
   - Multiple detector/descriptor combos: ORB, FAST+BRIEF, AKAZE, BRISK, SIFT, GFTT+ORB
   - Adaptive optical flow with forward-backward consistency check
   - Grid-based feature distribution option
   - Keyframe quality scoring and metrics tracking
   - Runtime detector switching

5. **Advanced Overlay Rendering**
   - `Mesh3D` class for textured 3D model rendering
   - OBJ file loading with texture support
   - Solid shaded rendering with lighting
   - Textured rendering with UV mapping
   - Factory methods: `create_box()`, `create_plane()`
   - Backface culling and depth sorting

6. **Integration Tests** (`tests/test_integration.py`)
   - Synthetic video generation for reproducible testing
   - Full pipeline benchmarking
   - Detector comparison tests
   - Tracking and pose metrics collection

### ✅ Recently Completed (Continued)

7. **SLAM/Mapping Module** (`src/mapping.py`)
   - `SparseMap` class for 3D point cloud management
   - Keyframe-based map building with covisibility graph
   - Automatic point triangulation between keyframes
   - Loop closure detection and verification
   - Map persistence (save/load to JSON)
   - `MapVisualizer` for top-down and projected views

8. **Occlusion Handling** (`src/occlusion.py`)
   - `DepthEstimator` for sparse-to-dense depth estimation
   - Ground plane depth assumption
   - Depth completion/interpolation
   - `OcclusionHandler` for mask generation
   - `DepthAwareOverlayRenderer` for proper AR occlusion
   - Per-face depth sorting for 3D objects

### 📋 Future Enhancements
1. **Dense Depth Integration**
   - MiDaS or similar neural depth estimation
   - Depth sensor support (RealSense, Kinect)
2. **Advanced SLAM**
   - Bundle adjustment optimization
   - Relocalization from saved maps
   - Multi-session mapping

## Testing Status

### Automated
- `tests/test_tracking_feature.py` validates feature extraction/matching skeleton
- `tests/test_pose.py` covers calibration + pose recovery from feature correspondences

### Pending
- Overlay rendering visual regression tests
- End-to-end integration test harness
- Pose filter unit tests

## How to Test Right Now

### Quick Demo
```bash
cd /Users/milansavard/Desktop/GitHub/ComputerVision/NETHERGAZE
python3 examples/run_demo.py
```

### Full Pipeline with CLI
```bash
# Default camera
python3 src/main.py

# Specific camera
python3 src/main.py --camera 1

# Video file
python3 src/main.py --video path/to/video.mp4

# Custom config + verbose logging
python3 src/main.py --config config.json --verbose
```

Inside the demo window:
- Press `m` to toggle feature overlay
- Press `a` to toggle pose axes
- Press `p` to pause/resume
- Press `h` for control cheat sheet, `q` to quit

### Camera Calibration
```bash
# Interactive capture mode (recommended for first-time calibration)
python3 examples/calibrate_camera.py --capture --output calibration.json

# Calibrate from existing chessboard images
python3 examples/calibrate_camera.py --images ./calib_images/*.jpg --output calibration.json

# Preview undistorted camera feed with saved calibration
python3 examples/calibrate_camera.py --preview --config calibration.json

# Customize board size and square dimensions
python3 examples/calibrate_camera.py --capture --board-size 7x5 --square-size 30
```

Calibration controls:
- `SPACE` - Capture frame (when chessboard detected)
- `c` - Run calibration
- `s` - Save calibration to file
- `u` - Toggle undistorted preview
- `r` - Reset captures
- `q` - Quit

## Configuration Cheatsheet

`src/utils.get_config()` now exposes:

```python
{
    "feature_tracking": {
        "method": "orb",
        "max_features": 1000,
        "quality_level": 0.01,
        "min_distance": 7.0,
        "fast_threshold": 20,
        "use_optical_flow": True,
        "optical_flow_win_size": 21,
        "optical_flow_max_level": 3,
        "optical_flow_criteria_eps": 0.03,
        "optical_flow_criteria_count": 30,
        "reacquire_threshold": 200,
        "keyframe_interval": 15,
        "max_keyframes": 6,
        "min_keyframe_features": 160,
    },
    "calibration": {
        "calibration_file": None,
        "camera_matrix": [[800.0, 0.0, 320.0],
                          [0.0, 800.0, 240.0],
                          [0.0, 0.0, 1.0]],
        "dist_coeffs": [0, 0, 0, 0, 0],
    },
    "pose_filter": {
        "enable_smoothing": True,
        "smoothing_alpha": 0.3,
        "enable_outlier_rejection": True,
        "max_translation_jump": 0.5,
        "max_rotation_jump": 0.5,
        "history_size": 10,
        "use_median_filter": False,
        "min_inliers_threshold": 10,
    },
    "scale_estimation": {
        "method": "auto",  # "auto", "known_distance", "ground_plane", "manual"
        "manual_scale": 1.0,
        "known_distance": None,
        "ground_plane_height": 1.5,
        "scale_smoothing_alpha": 0.2,
    },
    "overlay": {
        "enable_2d_overlays": True,
        "enable_3d_overlays": True,
        "default_3d_color": [0, 255, 255],
        "default_2d_color": [0, 255, 0],
        "blend_alpha": 0.7,
        "antialiasing": True,
    },
    "axis_length": 0.05,
    ...
}
```

Override these values via JSON config file or CLI options to experiment with markerless tuning.

## Notes

- Keep `opencv-contrib-python` installed for ORB/OpenCV extras
- Markerless path is intentionally modular so SLAM components can slot in
- Document new experiments in `docs/` as you iterate (see roadmap updates)
