"""
Video input and preprocessing utilities.

This module handles video capture, frame processing, and video-related operations.
"""

import cv2
import numpy as np
import logging


class VideoProcessor:
    """Handles video input capture and preprocessing."""
    
    def __init__(self, config=None):
        """Initialize video processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.cap = None
        self.logger = logging.getLogger(__name__)
        
        # Video settings from config
        self.camera_id = self.config.get('camera_id', 0)
        self.width = self.config.get('video_width', 640)
        self.height = self.config.get('video_height', 480)
        self.fps = self.config.get('video_fps', 30)
        
        # Preprocessing settings
        self.enable_preprocessing = self.config.get('enable_preprocessing', False)
        self.blur_kernel = self.config.get('blur_kernel', (5, 5))
        
    def initialize(self):
        """Initialize video capture.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize camera capture with AVFoundation on macOS
            import platform
            if platform.system() == 'Darwin':  # macOS
                # Use AVFoundation backend on macOS
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_AVFOUNDATION)
            else:
                self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Try to read a test frame to verify camera is working
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self.logger.error("Camera opened but cannot capture frames")
                self.logger.error("Please check camera permissions in System Settings")
                self.cap.release()
                self.cap = None
                return False
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            self.logger.info(f"Test frame captured successfully: {test_frame.shape}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Video processor initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def capture_frame(self):
        """Capture a frame from the video source.
        
        Returns:
            np.ndarray or None: Captured frame or None if failed
        """
        if self.cap is None or not self.cap.isOpened():
            return None
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning("Failed to capture frame")
                return None
                
            # Apply preprocessing if enabled
            if self.enable_preprocessing:
                frame = self.preprocess_frame(frame)
                
            return frame
            
        except Exception as e:
            self.logger.error(f"Frame capture error: {e}")
            return None
    
    def preprocess_frame(self, frame):
        """Apply preprocessing to the frame.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            np.ndarray: Preprocessed frame
        """
        try:
            # Apply Gaussian blur to reduce noise
            if self.blur_kernel[0] > 0:
                frame = cv2.GaussianBlur(frame, self.blur_kernel, 0)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Frame preprocessing error: {e}")
            return frame
    
    def load_video_file(self, filepath):
        """Load a video file instead of camera.
        
        Args:
            filepath: Path to video file
            
        Returns:
            bool: True if load successful, False otherwise
        """
        try:
            if self.cap is not None:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(filepath)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video file: {filepath}")
                return False
                
            self.logger.info(f"Video file loaded: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Video file load error: {e}")
            return False
    
    def get_frame_info(self):
        """Get information about the current video stream.
        
        Returns:
            dict: Frame information
        """
        if self.cap is None:
            return {}
            
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
        }
    
    def cleanup(self):
        """Clean up video resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.logger.info("Video processor cleaned up")
