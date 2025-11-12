"""
Test OpenCV installation and basic functionality without requiring camera.

This script creates a test image and displays it to verify OpenCV is working.
"""

import cv2
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import setup_logging

def test_opencv_basic():
    """Test basic OpenCV functionality."""
    print("=" * 60)
    print("OpenCV Installation Test")
    print("=" * 60)
    
    # Check OpenCV version
    print(f"\n1. OpenCV version: {cv2.__version__}")
    
    # Create a test image
    print("\n2. Creating test image...")
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw some shapes
    cv2.rectangle(test_image, (50, 50), (590, 430), (0, 255, 0), 2)
    cv2.circle(test_image, (320, 240), 100, (255, 0, 0), -1)
    cv2.putText(test_image, "OpenCV Test Image", (150, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(test_image, "Press any key to continue", (180, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    print("   ✓ Test image created successfully")
    
    # Display the image
    print("\n3. Testing display window...")
    window_name = "OpenCV Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)
    cv2.imshow(window_name, test_image)
    
    print("   ✓ Display window created")
    print("\n   A window should appear with a blue circle and green rectangle.")
    print("   Press any key in the window to close it...")
    
    # Wait for key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n4. Testing image operations...")
    
    # Test some basic operations
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    print(f"   ✓ Grayscale conversion: {gray.shape}")
    
    blurred = cv2.GaussianBlur(test_image, (5, 5), 0)
    print(f"   ✓ Gaussian blur: {blurred.shape}")
    
    edges = cv2.Canny(gray, 100, 200)
    print(f"   ✓ Edge detection: {edges.shape}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: OpenCV is working correctly!")
    print("=" * 60)
    print("\nNext step: Test camera with 'python3 examples/test_camera.py'")
    print()
    
    return True

def main():
    """Main entry point."""
    setup_logging()
    
    try:
        success = test_opencv_basic()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

