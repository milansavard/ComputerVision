# NETHERGAZE Testing Guide

## Quick Start Testing

### Test 0: Verify OpenCV Installation
Test that OpenCV is installed and working (no camera required).

```bash
cd /Users/milansavard/Desktop/GitHub/ComputerVision/NETHERGAZE/
python3 examples/test_opencv.py
```

**Expected output:**
- OpenCV version displayed
- A window appears with a blue circle and green rectangle
- Press any key to close
- Success message

**Success criteria:**
- ✅ OpenCV version shows (e.g., 4.x.x)
- ✅ Window displays test image
- ✅ No errors occur

---

### Test 1: Camera Access Test
Test camera access before running the main demo.

```bash
python3 examples/test_camera.py
```

**Expected output:**
- Detects your operating system
- Tests different camera backends
- Opens camera window showing live video
- Frame counter updates in real-time

**Success criteria:**
- ✅ Camera backend found and working
- ✅ Frames captured successfully
- ✅ Live video displays
- ✅ Frame counter updates

If this fails, follow the troubleshooting steps displayed.

---

### Test 2: World-Anchored AR Demo
Test the full AR demo with world-anchored objects.

```bash
python3 src/main.py
```

**Expected output:**
- A window titled "NETHERGAZE - World-Anchored AR" appears
- Live video feed with green feature points (ORB features)
- Help overlay in top-left corner
- Statistics in bottom-left (FPS, Features, Track %, Pose %)

**Test the controls:**
- Point at textured surface (book, poster, keyboard)
- Wait for green "TRACKING" indicator
- Press `SPACE` - Sets anchor point, shows "ANCHORED"
- Press `1` - Places a wireframe cube
- Press `2-5` - Places other objects (pyramid, axes, box, chair)
- Move camera - Objects should stay in place
- Press `C` - Clears all objects
- Press `Q` or `ESC` - Application exits cleanly

**Success criteria:**
- ✅ Window opens without errors
- ✅ Green feature points visible
- ✅ Tracking indicator shows "TRACKING"
- ✅ Objects can be placed after anchoring
- ✅ Application exits cleanly

---

### Test 2: Module Imports
Test that all implemented modules import correctly.

```bash
cd /Users/milansavard/Desktop/GitHub/ComputerVision/NETHERGAZE
python3 -c "from src import utils, video, ui; print('All modules imported successfully!')"
```

**Expected output:**
```
All modules imported successfully!
```

---

### Test 3: Configuration System
Test configuration loading and validation.

```bash
python3 -c "
from src.utils import setup_logging, get_config, validate_config
setup_logging()
config = get_config()
print('Config valid:', validate_config(config))
print('Camera ID:', config['camera_id'])
print('Resolution:', config['video_width'], 'x', config['video_height'])
"
```

**Expected output:**
```
2025-XX-XX XX:XX:XX - src.utils - INFO - Logging initialized
2025-XX-XX XX:XX:XX - root - INFO - Configuration validated successfully
Config valid: True
Camera ID: 0
Resolution: 640 x 480
```

---

## Troubleshooting

### Issue: "Failed to open camera 0"

**Possible causes:**
1. No camera connected
2. Camera already in use by another application
3. Camera permissions not granted (macOS)

**Solutions:**
- Check that a camera is connected
- Close other applications using the camera (Zoom, Teams, etc.)
- On macOS: Go to System Preferences → Security & Privacy → Camera and grant permission to Terminal/Python

### Issue: "ModuleNotFoundError: No module named 'cv2'"

**Solution:**
```bash
pip3 install opencv-python opencv-contrib-python numpy
```

### Issue: Window appears but is black/frozen

**Possible causes:**
1. Camera initialization failed silently
2. Frame capture is failing

**Debug:**
```bash
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
print('Camera opened:', cap.isOpened())
ret, frame = cap.read()
print('Frame captured:', ret)
if ret:
    print('Frame shape:', frame.shape)
cap.release()
"
```

### Issue: Application won't close

**Solution:**
- Try pressing `q` or `ESC`
- If stuck, use `Ctrl+C` in terminal
- As last resort, force quit the window

---

## Performance Testing

### Frame Rate Test
Check if you're getting reasonable frame rates:

The AR demo displays FPS in the bottom-left corner. Run:

```bash
python3 src/main.py
```

**Expected FPS:** 25-30 fps (normal mode), 50-60 fps (with --fast flag)

For high performance mode:
```bash
python3 src/main.py --fast
```

---

## Unit Tests

```bash
cd /Users/milansavard/Desktop/GitHub/ComputerVision/NETHERGAZE
python3 -m unittest tests/test_tracking_feature.py tests/test_pose.py
```

---

## Integration Testing Checklist

When more features are implemented, test these scenarios:

### Camera Functionality
- [ ] Camera initializes correctly
- [ ] Frames are captured consistently
- [ ] Different resolutions work (320x240, 640x480, 1280x720)
- [ ] Video file input works (alternative to camera)

### UI Functionality  
- [ ] Window opens and closes properly
- [ ] All keyboard shortcuts work
- [ ] Status text is readable
- [ ] Window resizing works
- [ ] Multiple runs don't leak resources

### Markerless Tracking
- [ ] Feature points remain tracked during motion
- [ ] Optical flow handoff is smooth
- [ ] Reacquisition works after rapid motion
- [ ] Keyframe map maintains useful matches

### Pose Estimation (Future)
- [ ] Pose is calculated correctly
- [ ] Pose updates smoothly without jitter
- [ ] Translation direction aligns with expected motion
- [ ] Camera calibration affects results

### Overlay Rendering (Future)
- [ ] Simple overlays render correctly
- [ ] Overlays track markers smoothly
- [ ] Multiple overlays work simultaneously
- [ ] Overlay blending looks correct

---

## Known Limitations

Current implementation:
- Pose relies on essential matrix (scale is relative, not metric)
- Best results on textured surfaces (books, posters, keyboards)
- Objects may drift during rapid camera movement
- No recording functionality
- Wireframe rendering only (no textured 3D models)
