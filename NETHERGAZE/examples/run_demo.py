"""
Demo script for running a markerless NETHERGAZE demonstration.

This script provides a simple way to test the feature-tracking pipeline and
visualize the recovered camera pose in the live camera feed.
"""

import logging
import os
import sys

import cv2

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pose import PoseEstimator, PoseResult  # type: ignore
from tracking import FeatureTracker, TrackingFrameResult  # type: ignore
from ui import UserInterface  # type: ignore
from utils import get_config, setup_logging  # type: ignore
from video import VideoProcessor  # type: ignore


LOGGER = logging.getLogger(__name__)


def _draw_feature_annotations(frame, result: TrackingFrameResult):
    """Overlay tracked feature keypoints and matches for markerless mode."""
    if result is None or frame is None:
        return

    if result.keypoints is not None and result.keypoints.size:
        for x, y in result.keypoints:
            cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 0), -1, cv2.LINE_AA)

    if result.matches is not None and result.matches.size:
        for x1, y1, x0, y0 in result.matches[:200]:
            cv2.line(
                frame,
                (int(x0), int(y0)),
                (int(x1), int(y1)),
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )

    height = frame.shape[0]
    feature_count = 0 if result.keypoints is None else len(result.keypoints)
    cv2.putText(
        frame,
        f"Features tracked: {feature_count}",
        (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_pose_overlay(frame, pose: PoseResult, estimator: PoseEstimator, axis_scale: float):
    """Draw pose axes and translation vector on the frame."""
    if not pose or not pose.success:
        return

    projected_axes = estimator.project_axes(pose, axis_scale)
    if projected_axes is None or len(projected_axes) < 4:
        return

    origin = tuple(int(coord) for coord in projected_axes[0])
    x_axis = tuple(int(coord) for coord in projected_axes[1])
    y_axis = tuple(int(coord) for coord in projected_axes[2])
    z_axis = tuple(int(coord) for coord in projected_axes[3])

    cv2.line(frame, origin, x_axis, (0, 0, 255), 2, cv2.LINE_AA)  # X axis - red
    cv2.line(frame, origin, y_axis, (0, 255, 0), 2, cv2.LINE_AA)  # Y axis - green
    cv2.line(frame, origin, z_axis, (255, 0, 0), 2, cv2.LINE_AA)  # Z axis - blue

    if pose.translation_vector is not None:
        t = pose.translation_vector.flatten()
        text = f"t: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m"
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


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

    pose_estimator = PoseEstimator(config)
    pose_estimator.initialize()
    feature_tracker = FeatureTracker(config.get("feature_tracking", {}))
    LOGGER.info("Running demo in markerless feature-tracking mode.")
    
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

            tracking_result: TrackingFrameResult = None
            try:
                tracking_result = feature_tracker.process_frame(frame)
            except Exception as exc:
                LOGGER.error("Feature tracking error: %s", exc)
                tracking_result = None

            if tracking_result and ui.show_markers:
                _draw_feature_annotations(frame, tracking_result)
            pose = (
                pose_estimator.estimate_from_feature_tracks(tracking_result)
                if tracking_result
                else None
            )
            if pose and pose.success:
                _draw_pose_overlay(
                    frame,
                    pose,
                    pose_estimator,
                    axis_scale=float(config.get("axis_length", 0.05)),
                )
            
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
