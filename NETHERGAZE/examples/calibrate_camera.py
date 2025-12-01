"""
Camera Calibration Tool for NETHERGAZE.

This script provides utilities to calibrate a camera using a chessboard pattern.
Proper calibration improves pose estimation accuracy significantly.

Usage:
    # Interactive capture mode (recommended)
    python calibrate_camera.py --capture --output calibration.json

    # Calibrate from existing images
    python calibrate_camera.py --images ./calib_images/*.jpg --output calibration.json

    # Live preview with calibration applied
    python calibrate_camera.py --preview --config calibration.json

Controls (capture mode):
    SPACE - Capture current frame
    c     - Run calibration with captured frames
    s     - Save calibration to file
    u     - Undistort preview toggle
    r     - Reset (clear captured frames)
    q     - Quit
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import get_config, setup_logging  # type: ignore
from video import VideoProcessor  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Container for calibration results."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]
    rms_error: float
    num_images: int
    rvecs: List[np.ndarray]
    tvecs: List[np.ndarray]

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.flatten().tolist(),
            "image_size": list(self.image_size),
            "rms_error": self.rms_error,
            "num_images": self.num_images,
            "focal_length": {
                "fx": float(self.camera_matrix[0, 0]),
                "fy": float(self.camera_matrix[1, 1]),
            },
            "principal_point": {
                "cx": float(self.camera_matrix[0, 2]),
                "cy": float(self.camera_matrix[1, 2]),
            },
        }

    def save(self, filepath: str):
        """Save calibration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info("Calibration saved to %s", filepath)

    @staticmethod
    def load(filepath: str) -> "CalibrationResult":
        """Load calibration from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        return CalibrationResult(
            camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
            image_size=tuple(data["image_size"]),
            rms_error=data.get("rms_error", 0.0),
            num_images=data.get("num_images", 0),
            rvecs=[],
            tvecs=[],
        )


class ChessboardCalibrator:
    """
    Camera calibrator using chessboard pattern detection.
    
    The chessboard should have known dimensions. Common sizes:
    - 9x6 internal corners (10x7 squares)
    - 7x5 internal corners (8x6 squares)
    """

    def __init__(
        self,
        board_size: Tuple[int, int] = (9, 6),
        square_size: float = 0.025,
    ):
        """
        Initialize the calibrator.
        
        Args:
            board_size: (columns, rows) of internal chessboard corners
            square_size: Physical size of each square in meters
        """
        self.board_size = board_size
        self.square_size = square_size

        # Prepare object points template
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
        self.objp *= square_size

        # Storage for calibration data
        self.obj_points: List[np.ndarray] = []
        self.img_points: List[np.ndarray] = []
        self.image_size: Optional[Tuple[int, int]] = None

        # Calibration result
        self.result: Optional[CalibrationResult] = None

        # Corner refinement criteria
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

    def reset(self):
        """Clear all captured calibration data."""
        self.obj_points.clear()
        self.img_points.clear()
        self.result = None
        LOGGER.info("Calibration data cleared")

    @property
    def num_captures(self) -> int:
        """Number of captured calibration frames."""
        return len(self.obj_points)

    def detect_chessboard(
        self, image: np.ndarray, draw: bool = True
    ) -> Tuple[bool, np.ndarray, Optional[np.ndarray]]:
        """
        Detect chessboard corners in an image.
        
        Args:
            image: Input BGR image
            draw: Whether to draw detected corners on the image
            
        Returns:
            (success, annotated_image, corners)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        output = image.copy()

        # Find chessboard corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, flags)

        if ret:
            # Refine corner positions
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), self.criteria
            )

            if draw:
                cv2.drawChessboardCorners(output, self.board_size, corners_refined, ret)

            return True, output, corners_refined

        return False, output, None

    def add_calibration_image(
        self, image: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Process an image and add to calibration set if chessboard is found.
        
        Args:
            image: Input BGR image
            
        Returns:
            (success, corners) - success indicates if chessboard was found
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        if self.image_size is None:
            self.image_size = (gray.shape[1], gray.shape[0])

        ret, _, corners = self.detect_chessboard(image, draw=False)

        if ret and corners is not None:
            self.obj_points.append(self.objp.copy())
            self.img_points.append(corners)
            LOGGER.info("Added calibration frame %d", self.num_captures)
            return True, corners

        return False, None

    def calibrate(self, min_images: int = 10) -> Optional[CalibrationResult]:
        """
        Run camera calibration on captured images.
        
        Args:
            min_images: Minimum number of images required
            
        Returns:
            CalibrationResult if successful, None otherwise
        """
        if self.num_captures < min_images:
            LOGGER.warning(
                "Not enough images for calibration: %d < %d",
                self.num_captures,
                min_images,
            )
            return None

        if self.image_size is None:
            LOGGER.error("No image size recorded")
            return None

        LOGGER.info("Running calibration with %d images...", self.num_captures)

        try:
            rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.obj_points,
                self.img_points,
                self.image_size,
                None,
                None,
            )

            self.result = CalibrationResult(
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                image_size=self.image_size,
                rms_error=rms,
                num_images=self.num_captures,
                rvecs=list(rvecs),
                tvecs=list(tvecs),
            )

            LOGGER.info("Calibration complete!")
            LOGGER.info("  RMS reprojection error: %.4f pixels", rms)
            LOGGER.info("  Focal length: fx=%.1f, fy=%.1f", camera_matrix[0, 0], camera_matrix[1, 1])
            LOGGER.info("  Principal point: cx=%.1f, cy=%.1f", camera_matrix[0, 2], camera_matrix[1, 2])

            return self.result

        except cv2.error as e:
            LOGGER.error("Calibration failed: %s", e)
            return None

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Apply undistortion using calibration result."""
        if self.result is None:
            return image

        return cv2.undistort(
            image,
            self.result.camera_matrix,
            self.result.dist_coeffs,
        )


class InteractiveCalibrator:
    """Interactive calibration session with live camera feed."""

    def __init__(
        self,
        board_size: Tuple[int, int] = (9, 6),
        square_size: float = 0.025,
        camera_id: int = 0,
    ):
        self.calibrator = ChessboardCalibrator(board_size, square_size)
        self.camera_id = camera_id
        self.video_processor: Optional[VideoProcessor] = None
        self.window_name = "NETHERGAZE Camera Calibration"
        self.show_undistorted = False
        self.running = False

    def initialize(self) -> bool:
        """Initialize video capture."""
        config = get_config()
        config["camera_id"] = self.camera_id
        self.video_processor = VideoProcessor(config)

        if not self.video_processor.initialize():
            LOGGER.error("Failed to initialize camera")
            return False

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        return True

    def run(self, output_path: str = "calibration.json", min_images: int = 10):
        """
        Run interactive calibration session.
        
        Args:
            output_path: Path to save calibration JSON
            min_images: Minimum images before calibration is allowed
        """
        if not self.initialize():
            return

        self.running = True
        print("\n" + "=" * 60)
        print("NETHERGAZE Camera Calibration")
        print("=" * 60)
        print(f"Board size: {self.calibrator.board_size}")
        print(f"Square size: {self.calibrator.square_size * 1000:.1f} mm")
        print(f"Minimum images: {min_images}")
        print("=" * 60)
        print("\nControls:")
        print("  SPACE - Capture frame (when chessboard detected)")
        print("  c     - Run calibration")
        print("  s     - Save calibration to file")
        print("  u     - Toggle undistorted view")
        print("  r     - Reset (clear all captures)")
        print("  q     - Quit")
        print("=" * 60 + "\n")

        while self.running:
            frame = self.video_processor.capture_frame()
            if frame is None:
                continue

            # Detect chessboard
            detected, display_frame, corners = self.calibrator.detect_chessboard(frame)

            # Apply undistortion if enabled and calibrated
            if self.show_undistorted and self.calibrator.result is not None:
                display_frame = self.calibrator.undistort(display_frame)

            # Draw status overlay
            self._draw_status(display_frame, detected)

            cv2.imshow(self.window_name, display_frame)

            # Handle key input
            key = cv2.waitKey(1) & 0xFF
            self._handle_key(key, frame, detected, output_path, min_images)

        self.cleanup()

    def _draw_status(self, frame: np.ndarray, detected: bool):
        """Draw status information on frame."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Status bar background
        cv2.rectangle(frame, (0, 0), (w, 80), (40, 40, 40), -1)

        # Detection status
        status_color = (0, 255, 0) if detected else (0, 0, 255)
        status_text = "CHESSBOARD DETECTED" if detected else "Searching for chessboard..."
        cv2.putText(frame, status_text, (10, 25), font, 0.6, status_color, 2)

        # Capture count
        count_text = f"Captures: {self.calibrator.num_captures}"
        cv2.putText(frame, count_text, (10, 50), font, 0.5, (255, 255, 255), 1)

        # Calibration status
        if self.calibrator.result is not None:
            calib_text = f"Calibrated (RMS: {self.calibrator.result.rms_error:.3f}px)"
            cv2.putText(frame, calib_text, (10, 70), font, 0.5, (0, 255, 255), 1)

            if self.show_undistorted:
                cv2.putText(frame, "[UNDISTORTED]", (w - 150, 25), font, 0.5, (255, 255, 0), 1)

        # Instructions
        if detected:
            cv2.putText(
                frame,
                "Press SPACE to capture",
                (w // 2 - 100, h - 20),
                font,
                0.6,
                (0, 255, 0),
                2,
            )

    def _handle_key(
        self,
        key: int,
        frame: np.ndarray,
        detected: bool,
        output_path: str,
        min_images: int,
    ):
        """Handle keyboard input."""
        if key == ord("q") or key == 27:  # q or ESC
            self.running = False

        elif key == ord(" ") and detected:  # SPACE - capture
            success, _ = self.calibrator.add_calibration_image(frame)
            if success:
                print(f"Captured frame {self.calibrator.num_captures}")

        elif key == ord("c"):  # c - calibrate
            if self.calibrator.num_captures >= min_images:
                result = self.calibrator.calibrate(min_images)
                if result:
                    print("\nCalibration successful!")
                    print(f"  RMS error: {result.rms_error:.4f} pixels")
                    print(f"  fx={result.camera_matrix[0,0]:.1f}, fy={result.camera_matrix[1,1]:.1f}")
            else:
                print(f"Need at least {min_images} images (have {self.calibrator.num_captures})")

        elif key == ord("s"):  # s - save
            if self.calibrator.result is not None:
                self.calibrator.result.save(output_path)
                print(f"Saved calibration to {output_path}")
            else:
                print("No calibration to save. Press 'c' to calibrate first.")

        elif key == ord("u"):  # u - toggle undistort
            if self.calibrator.result is not None:
                self.show_undistorted = not self.show_undistorted
                print(f"Undistorted view: {self.show_undistorted}")

        elif key == ord("r"):  # r - reset
            self.calibrator.reset()
            self.show_undistorted = False
            print("Reset calibration data")

    def cleanup(self):
        """Clean up resources."""
        if self.video_processor:
            self.video_processor.cleanup()
        cv2.destroyAllWindows()


def calibrate_from_images(
    image_paths: List[str],
    board_size: Tuple[int, int] = (9, 6),
    square_size: float = 0.025,
    output_path: str = "calibration.json",
    visualize: bool = False,
) -> Optional[CalibrationResult]:
    """
    Calibrate camera from a list of images.
    
    Args:
        image_paths: List of paths to calibration images
        board_size: Chessboard internal corner count
        square_size: Physical square size in meters
        output_path: Where to save calibration JSON
        visualize: Whether to show detected corners
        
    Returns:
        CalibrationResult if successful
    """
    calibrator = ChessboardCalibrator(board_size, square_size)

    print(f"\nProcessing {len(image_paths)} images...")

    for i, path in enumerate(image_paths):
        image = cv2.imread(path)
        if image is None:
            LOGGER.warning("Could not read image: %s", path)
            continue

        success, corners = calibrator.add_calibration_image(image)

        if visualize:
            _, display, _ = calibrator.detect_chessboard(image)
            cv2.imshow("Calibration", display)
            cv2.waitKey(100)

        status = "✓" if success else "✗"
        print(f"  [{status}] {os.path.basename(path)}")

    if visualize:
        cv2.destroyAllWindows()

    print(f"\nFound chessboard in {calibrator.num_captures}/{len(image_paths)} images")

    result = calibrator.calibrate()
    if result:
        result.save(output_path)
        print(f"\nCalibration saved to {output_path}")

    return result


def preview_calibration(config_path: str, camera_id: int = 0):
    """
    Preview undistorted camera feed using saved calibration.
    
    Args:
        config_path: Path to calibration JSON file
        camera_id: Camera index
    """
    if not os.path.exists(config_path):
        print(f"Calibration file not found: {config_path}")
        return

    result = CalibrationResult.load(config_path)
    print(f"Loaded calibration from {config_path}")
    print(f"  RMS error: {result.rms_error:.4f}")
    print(f"  Image size: {result.image_size}")

    config = get_config()
    config["camera_id"] = camera_id
    video = VideoProcessor(config)

    if not video.initialize():
        print("Failed to initialize camera")
        return

    window_name = "Calibration Preview (Press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nShowing undistorted preview. Press 'q' to quit.")

    while True:
        frame = video.capture_frame()
        if frame is None:
            continue

        # Side-by-side comparison
        undistorted = cv2.undistort(frame, result.camera_matrix, result.dist_coeffs)

        # Add labels
        cv2.putText(frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(undistorted, "Undistorted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        combined = np.hstack([frame, undistorted])
        cv2.imshow(window_name, combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.cleanup()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NETHERGAZE Camera Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive capture and calibration
  python calibrate_camera.py --capture

  # Calibrate from existing images
  python calibrate_camera.py --images ./calib_images/*.jpg

  # Preview with existing calibration
  python calibrate_camera.py --preview --config calibration.json

Chessboard Tips:
  - Use a flat, rigid chessboard pattern
  - Capture from multiple angles and distances
  - Cover all areas of the image frame
  - Avoid motion blur - hold steady when capturing
  - A 9x6 internal corners board (10x7 squares) works well
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--capture",
        action="store_true",
        help="Interactive capture mode with live camera",
    )
    mode.add_argument(
        "--images",
        type=str,
        nargs="+",
        help="Calibrate from existing image files (glob patterns supported)",
    )
    mode.add_argument(
        "--preview",
        action="store_true",
        help="Preview undistorted feed with existing calibration",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="calibration.json",
        help="Output path for calibration JSON (default: calibration.json)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to existing calibration file (for preview mode)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--board-size",
        type=str,
        default="9x6",
        help="Chessboard internal corners as WxH (default: 9x6)",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=25.0,
        help="Chessboard square size in mm (default: 25.0)",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=10,
        help="Minimum images required for calibration (default: 10)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show detected corners when processing images",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    # Parse board size
    try:
        w, h = map(int, args.board_size.lower().split("x"))
        board_size = (w, h)
    except ValueError:
        print(f"Invalid board size format: {args.board_size}")
        print("Expected format: WxH (e.g., 9x6)")
        sys.exit(1)

    # Convert square size from mm to meters
    square_size = args.square_size / 1000.0

    if args.capture:
        # Interactive capture mode
        calibrator = InteractiveCalibrator(
            board_size=board_size,
            square_size=square_size,
            camera_id=args.camera,
        )
        calibrator.run(output_path=args.output, min_images=args.min_images)

    elif args.images:
        # Batch calibration from images
        # Expand glob patterns
        image_paths = []
        for pattern in args.images:
            image_paths.extend(glob.glob(pattern))

        if not image_paths:
            print("No images found matching the provided patterns")
            sys.exit(1)

        calibrate_from_images(
            image_paths=image_paths,
            board_size=board_size,
            square_size=square_size,
            output_path=args.output,
            visualize=args.visualize,
        )

    elif args.preview:
        # Preview mode
        config_path = args.config or args.output
        preview_calibration(config_path, camera_id=args.camera)


if __name__ == "__main__":
    main()

