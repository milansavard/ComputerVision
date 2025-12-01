# NETHERGAZE

An experimental computer-vision toolkit focused entirely on markerless tracking and augmented-reality overlays.

## Project Overview

The project began as a marker-based demo. It has now shifted entirely to feature tracking and markerless pose estimation; the legacy marker path has been removed so development can concentrate on the markerless pipeline.

## Features

- Camera capture + preprocessing (`VideoProcessor`)
- OpenCV-based UI with keyboard controls (`UserInterface`)
- **Markerless tracking** (`tracking.feature.FeatureTracker`) with ORB detection, LK optical flow, and keyframe relocalisation
- **Pose estimation** (`pose.PoseEstimator`) that consumes feature tracks and projects axes/overlays
- Demo script that visualises tracked features and pose in real time

## Getting Started

```bash
git clone https://github.com/.../ComputerVision.git
cd ComputerVision/NETHERGAZE
python3 -m venv .venv && source .venv/bin/activate  # recommended
pip install -r requirements.txt
```

Run the live demo:

```bash
python examples/run_demo.py
```

## Roadmap Highlights

- Refine pose smoothing and calibration workflows (capture tool, filtering)
- Expand feature tracker (optical flow, keyframe maps, relocalisation)
- Build overlay renderer capable of 2D/3D compositing
- Wire everything together in `src/main.py` for a cohesive pipeline

See `docs/design_overview.md` and `PROGRESS.md` for the detailed plan.

## Directory Structure

```
NETHERGAZE/
├── README.md
├── requirements.txt
├── data/
├── docs/
├── examples/
├── src/
│   ├── tracking/
│   └── ...
└── tests/
```

## Contributing / Notes

- Ensure `opencv-contrib-python` is installed for ORB/OpenCV extras.
- Use virtual environments to avoid macOS “externally managed” Python issues.
- Document experiments and findings in `docs/` as the markerless pipeline matures.
