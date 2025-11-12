"""
Demo script for running a minimal NETHERGAZE demonstration.

This script provides a simple way to test the NETHERGAZE system.
"""

import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import setup_logging, get_config
from video import VideoProcessor
from ui import UserInterface


def run_basic_demo():
    """Run a basic video capture and display demonstration."""
    print("NETHERGAZE - Basic Demo")
    print("=" * 40)
    print("This demo showcases basic video capture and display.")
    print("Press 'h' for help, 'q' to quit")
    print("=" * 40)
    
    # Set up logging
    setup_logging()
    
    # Load configuration
    config = get_config()
    
    # Initialize components
    video_processor = VideoProcessor(config)
    ui = UserInterface(config)
    
    # Initialize video capture
    if not video_processor.initialize():
        print("\n" + "=" * 60)
        print("ERROR: Failed to initialize video processor")
        print("=" * 60)
        print("\nPossible solutions:")
        print("1. Make sure you have a camera connected")
        print("2. On macOS: Grant camera permissions to Terminal/Python")
        print("   - Go to: System Settings → Privacy & Security → Camera")
        print("   - Enable access for Terminal or your Python app")
        print("3. Close other apps that might be using the camera")
        print("4. Try unplugging and replugging your camera")
        print("5. Restart your computer if permission issues persist")
        print("=" * 60 + "\n")
        return False
    
    # Initialize UI
    if not ui.initialize():
        print("Failed to initialize user interface")
        video_processor.cleanup()
        return False
    
    print("\nDemo running... Press 'q' to quit\n")
    
    # Main loop
    try:
        while True:
            # Capture frame
            frame = video_processor.capture_frame()
            
            if frame is None:
                print("Failed to capture frame")
                break
            
            # Display frame
            ui.display_frame(frame)
            
            # Handle events
            if not ui.handle_events():
                break
                
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        video_processor.cleanup()
        ui.cleanup()
        print("Demo complete!")
    
    return True


def main():
    """Main entry point for demo."""
    try:
        success = run_basic_demo()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
