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

### Test 2: Basic Video Capture Demo
Test the full demo with camera and UI.

```bash
python3 examples/run_demo.py
```

**Expected output:**
- A window titled "NETHERGAZE" appears
- Live video feed from your camera is displayed
- Green text overlay shows "Press 'h' for help, 'q' to quit"

**Test the controls:**
- Press `h` - Help message appears in console
- Press `p` - Video pauses (see "PAUSED" text in red)
- Press `p` again - Video resumes
- Press `q` or `ESC` - Application exits cleanly

**Success criteria:**
- ✅ Window opens without errors
- ✅ Video feed is visible and updating
- ✅ All keyboard controls work
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

Add this to the demo loop (after line 54 in `run_demo.py`):

```python
import time
start_time = time.time()
frame_count = 0

# In the main loop:
frame_count += 1
if frame_count % 30 == 0:
    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    print(f"FPS: {fps:.2f}")
```

**Expected FPS:** 15-30 fps (depends on camera and system)

---

## Unit Tests (Future)

Once unit tests are implemented:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_marker_detect.py

# Run with coverage
pytest --cov=src tests/
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

### Marker Detection (Future)
- [ ] ArUco markers are detected
- [ ] Multiple markers can be tracked
- [ ] Marker IDs are correctly identified
- [ ] Detection works at various distances
- [ ] Detection works at various angles

### Pose Estimation (Future)
- [ ] Pose is calculated correctly
- [ ] Pose updates smoothly
- [ ] Multiple marker poses work
- [ ] Camera calibration affects results

### Overlay Rendering (Future)
- [ ] Simple overlays render correctly
- [ ] Overlays track markers smoothly
- [ ] Multiple overlays work simultaneously
- [ ] Overlay blending looks correct

---

## Known Limitations

Current implementation:
- No marker detection yet
- No pose estimation yet
- No overlay rendering yet
- Basic video processing only
- No recording functionality
- No advanced preprocessing

These will be addressed in future implementations.
