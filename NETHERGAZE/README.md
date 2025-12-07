# NETHERGAZE

A markerless augmented reality system that places virtual 3D objects in real-world scenes using natural feature tracking.

## Project Overview

NETHERGAZE is a complete AR pipeline built entirely on classical computer vision techniques. It uses ORB feature detection and Lucas-Kanade optical flow to track camera movement, then renders 3D wireframe objects that stay anchored in world space.

## Features

- **Markerless AR** - No fiducial markers needed, works on any textured surface
- **World-Anchored Objects** - Virtual objects stay fixed in 3D space as the camera moves
- **Real-time Tracking** - ORB feature detection + Lucas-Kanade optical flow
- **Multiple 3D Objects** - Cubes, pyramids, coordinate axes, boxes, and chairs
- **Live Statistics** - FPS, feature count, tracking rate, and pose success rate
- **Pose Estimation** - Essential Matrix + RANSAC with temporal smoothing

## Getting Started

```bash
git clone https://github.com/.../ComputerVision.git
cd ComputerVision/NETHERGAZE
python3 -m venv .venv && source .venv/bin/activate  # recommended
pip install -r requirements.txt
```

Run the AR demo:

```bash
python src/main.py                # Default settings
python src/main.py --fast         # High performance mode (60fps)
```

## How to Use

1. **Point camera** at a textured surface (book, poster, keyboard)
2. **Wait for green "TRACKING"** indicator
3. **Press SPACE** to set the anchor point (world origin)
4. **Press 1-5** to place 3D objects in the scene
5. **Move camera around** - objects stay fixed in 3D space!

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Set anchor point (world origin) |
| `1` | Place wireframe cube |
| `2` | Place pyramid |
| `3` | Place RGB coordinate axes |
| `4` | Place solid box |
| `5` | Place 3D chair |
| `C` | Clear all placed objects |
| `G` | Toggle ground grid |
| `M` | Toggle feature markers |
| `R` | Reset anchor |
| `H` | Toggle help overlay |
| `Q` / `ESC` | Quit |

## Directory Structure

```
NETHERGAZE/
├── README.md
├── requirements.txt
├── examples/
│   └── demo_anchored_objects.py  # Main AR demo
├── notebooks/
│   ├── project_overview.ipynb    # Detailed documentation
│   └── images/                   # Demo screenshots
├── src/
│   ├── main.py                   # Entry point
│   ├── video.py                  # Video capture
│   ├── pose.py                   # Pose estimation
│   ├── overlay.py                # 3D rendering
│   ├── mapping.py                # SLAM/mapping
│   ├── occlusion.py              # Depth handling
│   ├── tracking/
│   │   └── feature.py            # ORB + optical flow
│   └── ...
└── tests/
```

## Technical Details

- **Feature Detection**: ORB (Oriented FAST and Rotated BRIEF)
- **Tracking**: Lucas-Kanade optical flow with keyframe re-detection
- **Pose**: Essential Matrix decomposition with RANSAC
- **Rendering**: OpenCV drawing functions (cv2.line, cv2.projectPoints)
- **Smoothing**: Exponential Moving Average (EMA) filtering

## Requirements

- Python 3.8+
- OpenCV with contrib modules
- NumPy
- macOS/Linux (Continuity Camera supported on macOS)

## Author

Milan Savard - CS366 F25 Final Project
