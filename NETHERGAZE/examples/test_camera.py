"""
Test camera access and basic video capture.

This script tests if your camera is accessible and working.
"""

import cv2
import sys
import os
import platform

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import setup_logging

def test_camera_access():
    """Test camera access."""
    print("=" * 60)
    print("Camera Access Test")
    print("=" * 60)
    
    print(f"\nSystem: {platform.system()} {platform.release()}")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Test different camera backends
    print("\n1. Testing camera backends...")
    
    if platform.system() == 'Darwin':  # macOS
        print("   Detected macOS - trying AVFoundation backend")
        backends_to_try = [
            (cv2.CAP_AVFOUNDATION, "AVFoundation (recommended for macOS)"),
            (cv2.CAP_ANY, "Default backend")
        ]
    else:
        backends_to_try = [
            (cv2.CAP_ANY, "Default backend"),
            (cv2.CAP_DSHOW, "DirectShow (Windows)"),
            (cv2.CAP_V4L2, "V4L2 (Linux)")
        ]
    
    working_backend = None
    cap = None
    
    for backend, name in backends_to_try:
        try:
            print(f"\n   Testing {name}...")
            cap = cv2.VideoCapture(0, backend)
            
            if cap.isOpened():
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"   ✓ SUCCESS with {name}")
                    print(f"   Frame shape: {frame.shape}")
                    working_backend = (backend, name)
                    break
                else:
                    print(f"   ✗ Camera opened but cannot capture frames")
                    cap.release()
            else:
                print(f"   ✗ Failed to open camera with {name}")
                
        except Exception as e:
            print(f"   ✗ Error with {name}: {e}")
            if cap is not None:
                cap.release()
    
    if working_backend is None:
        print("\n" + "=" * 60)
        print("ERROR: Cannot access camera!")
        print("=" * 60)
        print("\nTroubleshooting steps:")
        if platform.system() == 'Darwin':
            print("\n macOS-specific:")
            print("  1. Check System Settings → Privacy & Security → Camera")
            print("  2. Enable camera access for Terminal")
            print("  3. Restart Terminal after granting permissions")
            print("  4. Try running: 'tccutil reset Camera' (resets permissions)")
        print("\n  - Make sure camera is connected")
        print("  - Close other apps using the camera")
        print("  - Try different USB ports")
        print("  - Check if camera works in Photo Booth / Camera app")
        print("=" * 60)
        return False
    
    # If we got here, camera is working!
    print("\n2. Testing video capture...")
    
    # Set properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"   Resolution: {actual_width}x{actual_height}")
    print(f"   FPS: {actual_fps}")
    
    # Capture a few test frames
    print("\n3. Capturing test frames...")
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"   Frame {i+1}: ✓ {frame.shape}")
        else:
            print(f"   Frame {i+1}: ✗ Failed")
    
    # Display live video
    print("\n4. Testing live display...")
    print("   Opening camera window - press 'q' to quit")
    
    cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Test", 640, 480)
    
    frame_count = 0
    while frame_count < 300:  # Max 10 seconds at 30fps
        ret, frame = cap.read()
        
        if not ret:
            print("   Failed to capture frame!")
            break
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        cv2.imshow("Camera Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("SUCCESS: Camera is working correctly!")
    print("=" * 60)
    print(f"\nWorking backend: {working_backend[1]}")
    print(f"Frames captured: {frame_count}")
    print("\nYour camera is ready for NETHERGAZE!")
    print("Run 'python3 src/main.py' to start the AR demo.")
    print()
    
    return True

def main():
    """Main entry point."""
    setup_logging()
    
    try:
        success = test_camera_access()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

