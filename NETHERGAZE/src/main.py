"""
Main entry point for the NETHERGAZE application.

This module orchestrates the complete pipeline:
1. Video input capture/preprocessing
2. Feature-based tracking (markerless)
3. Pose estimation
4. Virtual overlay rendering
5. Output display

Usage:
    python main.py                      # Run with defaults
    python main.py --config config.json # Use custom config
    python main.py --camera 1           # Use camera index 1
    python main.py --video path/to.mp4  # Process video file
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from overlay import OverlayRenderer
from pose import PoseEstimator, PoseResult
from tracking.feature import FeatureTracker, TrackingFrameResult
from ui import UserInterface
from utils import get_config, setup_logging
from video import VideoProcessor

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Runtime statistics for the pipeline."""

    frames_processed: int = 0
    tracking_successes: int = 0
    pose_successes: int = 0
    start_time: float = field(default_factory=time.time)
    last_frame_time: float = 0.0
    fps: float = 0.0

    def update(self, tracking_ok: bool, pose_ok: bool):
        """Update statistics after processing a frame."""
        self.frames_processed += 1
        if tracking_ok:
            self.tracking_successes += 1
        if pose_ok:
            self.pose_successes += 1

        now = time.time()
        if self.last_frame_time > 0:
            dt = now - self.last_frame_time
            # Exponential moving average for FPS
            instant_fps = 1.0 / max(dt, 0.001)
            self.fps = 0.9 * self.fps + 0.1 * instant_fps if self.fps > 0 else instant_fps
        self.last_frame_time = now

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def tracking_rate(self) -> float:
        return self.tracking_successes / max(self.frames_processed, 1) * 100

    @property
    def pose_rate(self) -> float:
        return self.pose_successes / max(self.frames_processed, 1) * 100


class NETHERGAZEApp:
    """
    Main application class orchestrating the NETHERGAZE pipeline.
    
    Pipeline stages:
        Capture → Preprocess → Track Features → Estimate Pose → Render Overlay → Display
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the application.
        
        Args:
            config: Configuration dictionary. If None, defaults are loaded.
        """
        self.config = config or get_config()
        self.running = False

        # Pipeline components (initialized lazily)
        self.video_processor: Optional[VideoProcessor] = None
        self.feature_tracker: Optional[FeatureTracker] = None
        self.pose_estimator: Optional[PoseEstimator] = None
        self.overlay_renderer: Optional[OverlayRenderer] = None
        self.ui: Optional[UserInterface] = None

        # Runtime state
        self.stats = PipelineStats()
        self.last_pose: Optional[PoseResult] = None
        self.video_file_mode = False

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def initialize(self, video_file: Optional[str] = None) -> bool:
        """Initialize all pipeline components.
        
        Args:
            video_file: Optional path to video file (uses camera if None)
            
        Returns:
            True if all components initialized successfully
        """
        LOGGER.info("Initializing NETHERGAZE pipeline...")

        # Video processor
        self.video_processor = VideoProcessor(self.config)
        if video_file:
            self.video_file_mode = True
            if not self.video_processor.load_video_file(video_file):
                LOGGER.error("Failed to load video file: %s", video_file)
                return False
        else:
            if not self.video_processor.initialize():
                LOGGER.error("Failed to initialize video capture")
                self._print_camera_help()
                return False

        # Feature tracker
        tracking_config = self.config.get("feature_tracking", {})
        self.feature_tracker = FeatureTracker(tracking_config)
        LOGGER.info("Feature tracker initialized with method: %s", tracking_config.get("method", "orb"))

        # Pose estimator
        self.pose_estimator = PoseEstimator(self.config)
        if not self.pose_estimator.initialize():
            LOGGER.error("Failed to initialize pose estimator")
            return False

        # Overlay renderer
        overlay_config = self.config.get("overlay", {})
        self.overlay_renderer = OverlayRenderer(overlay_config)
        self.overlay_renderer.initialize(self.pose_estimator.calibration)

        # User interface
        self.ui = UserInterface(self.config)
        if not self.ui.initialize():
            LOGGER.error("Failed to initialize UI")
            return False

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        LOGGER.info("Pipeline initialization complete")
        return True

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        LOGGER.info("Received signal %s, shutting down...", signum)
        self.running = False

    def _print_camera_help(self):
        """Print helpful camera troubleshooting information."""
        print("\n" + "=" * 60)
        print("CAMERA INITIALIZATION FAILED")
        print("=" * 60)
        print("\nTroubleshooting steps:")
        print("1. Ensure a camera is connected")
        print("2. Grant camera permissions:")
        print("   - macOS: System Settings → Privacy & Security → Camera")
        print("   - Linux: Check /dev/video* permissions")
        print("3. Close other applications using the camera")
        print("4. Try a different camera index: --camera 1")
        print("5. Try a video file instead: --video path/to/video.mp4")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------ #
    # Main Run Loop
    # ------------------------------------------------------------------ #
    def run(self) -> bool:
        """Run the main application loop.
        
        Returns:
            True if exited normally, False on error
        """
        if self.video_processor is None or self.ui is None:
            LOGGER.error("Pipeline not initialized. Call initialize() first.")
            return False

        self.running = True
        self.stats = PipelineStats()

        print("\n" + "=" * 50)
        print("NETHERGAZE - Markerless AR Pipeline")
        print("=" * 50)
        print("Controls:")
        print("  q/ESC - Quit")
        print("  m     - Toggle feature overlay")
        print("  a     - Toggle axes display")
        print("  p     - Pause/Resume")
        print("  h     - Help")
        print("=" * 50 + "\n")

        try:
            while self.running:
                success = self._process_frame()
                if not success:
                    break

                # Handle UI events
                if not self.ui.handle_events():
                    LOGGER.info("User requested exit")
                    break

                # In video file mode, check for end of file
                if self.video_file_mode:
                    # Small delay to not burn CPU
                    cv2.waitKey(1)

        except KeyboardInterrupt:
            LOGGER.info("Interrupted by user")
        except Exception as e:
            LOGGER.exception("Pipeline error: %s", e)
            return False
        finally:
            self._print_stats()
            self.cleanup()

        return True

    def _process_frame(self) -> bool:
        """Process a single frame through the pipeline.
        
        Returns:
            True to continue, False to stop
        """
        # 1. Capture frame
        frame = self.video_processor.capture_frame()
        if frame is None:
            if self.video_file_mode:
                LOGGER.info("End of video file")
                return False
            LOGGER.warning("Failed to capture frame")
            return True  # Continue trying

        # 2. Track features
        tracking_result: Optional[TrackingFrameResult] = None
        try:
            tracking_result = self.feature_tracker.process_frame(frame)
        except Exception as e:
            LOGGER.warning("Tracking error: %s", e)

        tracking_ok = tracking_result is not None and tracking_result.tracked_count > 0

        # 3. Estimate pose
        pose: Optional[PoseResult] = None
        if tracking_result is not None:
            try:
                pose = self.pose_estimator.estimate_from_feature_tracks(tracking_result)
                if pose and pose.success:
                    self.last_pose = pose
            except Exception as e:
                LOGGER.warning("Pose estimation error: %s", e)

        pose_ok = pose is not None and pose.success

        # 4. Draw annotations
        annotated_frame = self._draw_annotations(frame, tracking_result, pose)

        # 5. Render overlays
        if self.overlay_renderer and pose and pose.success:
            annotated_frame = self.overlay_renderer.render(annotated_frame, pose)

        # 6. Add stats overlay
        annotated_frame = self._draw_stats_overlay(annotated_frame)

        # 7. Display
        self.ui.display_frame(annotated_frame)

        # Update statistics
        self.stats.update(tracking_ok, pose_ok)

        return True

    def _draw_annotations(
        self,
        frame: np.ndarray,
        tracking_result: Optional[TrackingFrameResult],
        pose: Optional[PoseResult],
    ) -> np.ndarray:
        """Draw tracking and pose annotations on the frame."""
        if frame is None:
            return frame

        # Draw tracked features
        if tracking_result and self.ui and self.ui.show_markers:
            frame = self._draw_feature_overlay(frame, tracking_result)

        # Draw pose axes
        if pose and pose.success and self.ui and self.ui.show_axes:
            frame = self._draw_pose_axes(frame, pose)

        return frame

    def _draw_feature_overlay(
        self, frame: np.ndarray, result: TrackingFrameResult
    ) -> np.ndarray:
        """Draw tracked feature points and matches."""
        if result.keypoints is not None and result.keypoints.size:
            for x, y in result.keypoints:
                cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 0), -1, cv2.LINE_AA)

        if result.matches is not None and result.matches.size:
            for match in result.matches[:200]:
                x1, y1, x0, y0 = match
                cv2.line(
                    frame,
                    (int(x0), int(y0)),
                    (int(x1), int(y1)),
                    (0, 165, 255),
                    1,
                    cv2.LINE_AA,
                )

        return frame

    def _draw_pose_axes(self, frame: np.ndarray, pose: PoseResult) -> np.ndarray:
        """Draw coordinate axes using the estimated pose."""
        axis_length = float(self.config.get("axis_length", 0.05))
        projected = self.pose_estimator.project_axes(pose, axis_length)

        if projected is None or len(projected) < 4:
            return frame

        origin = tuple(int(c) for c in projected[0])
        x_end = tuple(int(c) for c in projected[1])
        y_end = tuple(int(c) for c in projected[2])
        z_end = tuple(int(c) for c in projected[3])

        cv2.line(frame, origin, x_end, (0, 0, 255), 2, cv2.LINE_AA)  # X - Red
        cv2.line(frame, origin, y_end, (0, 255, 0), 2, cv2.LINE_AA)  # Y - Green
        cv2.line(frame, origin, z_end, (255, 0, 0), 2, cv2.LINE_AA)  # Z - Blue

        # Draw translation vector text
        if pose.translation_vector is not None:
            t = pose.translation_vector.flatten()
            text = f"t: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]"
            cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def _draw_stats_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw runtime statistics on the frame."""
        if frame is None:
            return frame

        h = frame.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        color = (200, 200, 200)
        thickness = 1

        # Bottom-left statistics
        y = h - 40
        cv2.putText(frame, f"FPS: {self.stats.fps:.1f}", (10, y), font, scale, color, thickness, cv2.LINE_AA)
        y += 18
        cv2.putText(
            frame,
            f"Track: {self.stats.tracking_rate:.0f}% | Pose: {self.stats.pose_rate:.0f}%",
            (10, y),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        return frame

    def _print_stats(self):
        """Print final statistics to console."""
        print("\n" + "=" * 50)
        print("Session Statistics")
        print("=" * 50)
        print(f"  Frames processed: {self.stats.frames_processed}")
        print(f"  Elapsed time:     {self.stats.elapsed:.1f}s")
        print(f"  Average FPS:      {self.stats.frames_processed / max(self.stats.elapsed, 0.001):.1f}")
        print(f"  Tracking success: {self.stats.tracking_rate:.1f}%")
        print(f"  Pose success:     {self.stats.pose_rate:.1f}%")
        print("=" * 50 + "\n")

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def cleanup(self):
        """Clean up all pipeline resources."""
        LOGGER.info("Cleaning up pipeline...")

        if self.overlay_renderer:
            self.overlay_renderer.cleanup()

        if self.ui:
            self.ui.cleanup()

        if self.video_processor:
            self.video_processor.cleanup()

        LOGGER.info("Pipeline cleanup complete")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="NETHERGAZE - Markerless Augmented Reality Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with default camera
  python main.py --camera 1               # Use camera index 1
  python main.py --video demo.mp4         # Process video file
  python main.py --config my_config.json  # Use custom config
  python main.py --verbose                # Enable debug logging
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--camera", "-cam",
        type=int,
        default=None,
        help="Camera index to use (default: 0)",
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Path to video file (overrides camera)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Video capture width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Video capture height",
    )
    parser.add_argument(
        "--verbose", "-V",
        action="store_true",
        help="Enable verbose/debug logging",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    # Load configuration
    config = get_config(args.config)

    # Apply CLI overrides
    if args.camera is not None:
        config["camera_id"] = args.camera
    if args.width:
        config["video_width"] = args.width
    if args.height:
        config["video_height"] = args.height

    # Create and run application
    app = NETHERGAZEApp(config)

    video_file = args.video
    if video_file and not Path(video_file).exists():
        LOGGER.error("Video file not found: %s", video_file)
        sys.exit(1)

    if not app.initialize(video_file=video_file):
        LOGGER.error("Failed to initialize application")
        sys.exit(1)

    success = app.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
