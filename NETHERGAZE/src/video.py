"""
Video input and preprocessing utilities.

This module handles video capture, frame processing, and video-related operations.
"""

import logging
import platform
from typing import List, Optional

import cv2
import numpy as np


class VideoProcessor:
    """Handles video input capture and preprocessing."""
    
    def __init__(self, config=None):
        """Initialize video processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.cap: Optional[cv2.VideoCapture] = None
        self.logger = logging.getLogger(__name__)
        
        # Video settings from config
        self.camera_id = self.config.get('camera_id', 0)
        self.width = self.config.get('video_width', 640)
        self.height = self.config.get('video_height', 480)
        self.fps = self.config.get('video_fps', 30)
        
        # Backend priority list
        self.backend_priority = self._resolve_backend_priority(
            self.config.get('camera_backend_priority')
        )
        self.selected_backend: Optional[int] = None
        
        # Preprocessing settings
        self.enable_preprocessing = self.config.get('enable_preprocessing', False)
        self.blur_kernel = self.config.get('blur_kernel', (5, 5))
        
        # Frame warmup attempts
        self.max_init_attempts = self.config.get('camera_init_attempts', 10)
    
    @staticmethod
    def _resolve_backend_priority(user_priority: Optional[List[int]]) -> List[int]:
        """Determine backend priority order based on platform and config."""
        if user_priority:
            return user_priority
        
        system = platform.system()
        backends: List[int] = []
        
        def add_backend(name: str):
            value = getattr(cv2, name, None)
            if value is not None:
                backends.append(value)
        
        if system == 'Darwin':
            add_backend('CAP_AVFOUNDATION')
            add_backend('CAP_QT')
        elif system == 'Windows':
            add_backend('CAP_DSHOW')
            add_backend('CAP_MSMF')
        else:
            add_backend('CAP_V4L2')
            add_backend('CAP_GSTREAMER')
        
        add_backend('CAP_ANY')
        return backends or [cv2.CAP_ANY]
    
    @staticmethod
    def _backend_name(backend: Optional[int]) -> str:
        """Return human-readable name for backend constant."""
        if backend is None:
            return "Unknown"
        
        for attr in dir(cv2):
            if attr.startswith("CAP_") and getattr(cv2, attr) == backend:
                return attr
        return f"Backend({backend})"
    
    def set_camera_id(self, camera_id: int):
        """Update camera ID."""
        self.camera_id = camera_id
    
    def initialize(self):
        """Initialize video capture.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        self.cleanup()
        
        for backend in self.backend_priority:
            try:
                self.logger.info(
                    "Attempting to initialize camera %s using backend %s",
                    self.camera_id,
                    self._backend_name(backend),
                )
                cap = cv2.VideoCapture(self.camera_id, backend)
                
                if not cap.isOpened():
                    self.logger.warning(
                        "Failed to open camera %s with backend %s",
                        self.camera_id,
                        self._backend_name(backend),
                    )
                    cap.release()
                    continue
                
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                
                # Warm up and verify frame capture
                test_frame = self._warmup_camera(cap)
                if test_frame is None:
                    self.logger.warning(
                        "Camera opened but failed to provide frames (backend %s)",
                        self._backend_name(backend),
                    )
                    cap.release()
                    continue
                
                # Success
                self.cap = cap
                self.selected_backend = backend
                
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                
                self.logger.info(
                    "Camera initialized with backend %s: %sx%s @ %sfps",
                    self._backend_name(backend),
                    actual_width,
                    actual_height,
                    actual_fps,
                )
                self.logger.debug(
                    "Initial frame shape: %s", test_frame.shape if test_frame is not None else None
                )
                
                return True
            
            except Exception as e:
                self.logger.error(
                    "Error initializing camera %s with backend %s: %s",
                    self.camera_id,
                    self._backend_name(backend),
                    e,
                )
                import traceback
                traceback.print_exc()
        
        self.logger.error(
            "Unable to initialize camera %s with available backends: %s",
            self.camera_id,
            [self._backend_name(b) for b in self.backend_priority],
        )
        self.logger.error(
            "Please verify camera permissions and that no other application is using the camera."
        )
        return False
    
    def _warmup_camera(self, cap: cv2.VideoCapture) -> Optional[np.ndarray]:
        """Capture a few frames to allow camera to warm up."""
        frame: Optional[np.ndarray] = None
        for attempt in range(1, self.max_init_attempts + 1):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                if frame.mean() == 0:
                    # Completely black frame - continue warming up
                    self.logger.debug(
                        "Warmup frame %s captured but appears black; retrying...", attempt
                    )
                    continue
                return frame
        return None
    
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
            'backend': self._backend_name(self.selected_backend),
        }
    
    def cleanup(self):
        """Clean up video resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.logger.info("Video processor cleaned up")
