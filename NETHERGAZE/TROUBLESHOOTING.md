# NETHERGAZE Troubleshooting Guide

## Camera Issues on macOS

### Problem: "OpenCV: not authorized to capture video (status 0), requesting..."

**This is a macOS camera permissions issue.**

### Solution Steps:

#### Step 1: Grant Terminal Camera Access
1. Open **System Settings** (or System Preferences)
2. Go to **Privacy & Security** → **Camera**
3. Look for **Terminal** (or your Python app) in the list
4. Enable the toggle to grant camera access
5. **Restart Terminal** (important!)

#### Step 2: Reset Camera Permissions (if needed)
If granting permission doesn't work, reset it:

```bash
# Reset camera permissions for all apps
tccutil reset Camera

# Then grant permission again in System Settings
```

#### Step 3: Test Camera in Another App
Verify your camera works:
- Open **Photo Booth** or **FaceTime**
- Check if camera LED turns on
- If camera doesn't work there either, there's a hardware/driver issue

#### Step 4: Check for Other Apps Using Camera
Close these apps if they're running:
- Zoom
- Microsoft Teams  
- Skype
- Google Meet (in browser)
- OBS Studio
- Any other video apps

#### Step 5: Try Different Camera
If using external camera:
- Try unplugging and replugging
- Try a different USB port
- Try the built-in camera (camera_id: 0)
- Try external camera (camera_id: 1)

---

## Testing Camera Access

### Quick Camera Test
```bash
cd /Users/milansavard/Desktop/GitHub/ComputerVision/NETHERGAZE/
python3 examples/test_camera.py
```

This script will:
- Detect your operating system
- Test different camera backends
- Show helpful error messages
- Display live video if working

---

## Common Errors and Solutions

### Error: "Failed to open camera 0"

**Causes:**
1. Camera not connected
2. Camera permissions not granted (macOS)
3. Camera in use by another app
4. Wrong camera ID

**Solutions:**
```bash
# Test with camera 0 (usually built-in)
python3 examples/test_camera.py

# If that fails, try camera 1 (usually external)
# Edit config in src/utils.py and change camera_id to 1
```

---

### Error: "Camera opened but cannot capture frames"

**Causes:**
1. Permissions granted but not applied (need restart)
2. Camera hardware issue
3. Driver issue

**Solutions:**
1. **Restart Terminal completely** (Cmd+Q to quit, then reopen)
2. Test camera in Photo Booth
3. Restart your computer
4. Check for macOS updates

---

### Error: "ModuleNotFoundError: No module named 'cv2'"

**Solution:**
```bash
pip3 install opencv-python opencv-contrib-python numpy
```

If that doesn't work:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install packages
pip install -r requirements.txt
```

---

### Error: Window appears but is black

**Causes:**
1. Frame capture failing silently
2. Camera initializing slowly

**Solutions:**
1. Run the camera test: `python3 examples/test_camera.py`
2. Check Terminal output for errors
3. Try increasing wait time before first frame

---

### Error: "AVCaptureDeviceTypeExternal is deprecated"

**This is just a warning - not an error!**

Apple is deprecating an old API. The camera should still work.
The warning can be ignored for now. We've updated the code to use the newer API.

---

## Checking Your Setup

### 1. Verify OpenCV Installation
```bash
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

Expected output: `OpenCV version: 4.x.x` (or higher)

### 2. Test OpenCV Without Camera
```bash
python3 examples/test_opencv.py
```

This creates a test image - should work without camera.

### 3. List Available Cameras
```python
import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i}: Available")
        cap.release()
    else:
        print(f"Camera {i}: Not available")
```

---

## Platform-Specific Notes

### macOS (Darwin)
- Uses AVFoundation backend
- Requires camera permissions
- May show deprecation warnings (can be ignored)
- Built-in camera is usually ID 0
- External cameras are usually ID 1+

### Windows
- Uses DirectShow backend
- Usually doesn't require special permissions
- May need drivers for external cameras

### Linux
- Uses V4L2 backend
- May need to add user to video group
- Check `/dev/video*` devices exist

---

## Getting Help

If none of these solutions work:

1. **Collect Information:**
   ```bash
   python3 -c "import cv2, platform; print(f'OS: {platform.system()} {platform.release()}'); print(f'OpenCV: {cv2.__version__}')"
   ```

2. **Run All Tests:**
   ```bash
   python3 examples/test_opencv.py
   python3 examples/test_camera.py
   ```

3. **Check Logs:**
   - Look for ERROR messages in Terminal output
   - Note exact error messages

4. **System Info:**
   - macOS version
   - Python version: `python3 --version`
   - Camera type (built-in vs external)

---

## Success Indicators

You know it's working when:

✅ `test_opencv.py` shows a test image  
✅ `test_camera.py` shows live video with frame counter  
✅ `run_demo.py` shows live video with controls  
✅ No error messages in Terminal  
✅ Camera LED turns on  
✅ Frame counter updates smoothly  

---

## Quick Reference

| Problem | Quick Fix |
|---------|-----------|
| Camera permission | System Settings → Privacy → Camera → Enable Terminal |
| Need to restart | Close Terminal completely (Cmd+Q), reopen |
| Wrong camera | Try camera ID 0, 1, 2 |
| Camera in use | Close Zoom, Teams, etc. |
| Black screen | Run `test_camera.py` first |
| Module error | `pip3 install opencv-python` |

---

## Still Not Working?

The camera test scripts will give you specific guidance based on what fails.
Start with:

```bash
cd /Users/milansavard/Desktop/GitHub/ComputerVision/NETHERGAZE/
python3 examples/test_opencv.py    # Test OpenCV
python3 examples/test_camera.py    # Test camera
python3 examples/run_demo.py       # Run main demo
```

Each test will tell you exactly what's wrong and how to fix it!
