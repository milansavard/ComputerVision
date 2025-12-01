"""
Shared helper functions and utilities.

This module contains common utility functions used across the project.
"""

import logging
import os
import json
from datetime import datetime
from pathlib import Path


def setup_logging(level=logging.INFO):
    """Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")


def get_config(config_path=None):
    """Load configuration from file or return defaults.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        dict: Configuration dictionary
    """
    # Default configuration
    default_config = {
        # Video settings
        'camera_id': 0,
        'video_width': 640,
        'video_height': 480,
        'video_fps': 30,
        'camera_backend_priority': None,
        'camera_init_attempts': 10,
        
        # Preprocessing
        'enable_preprocessing': True,
        'blur_kernel': (5, 5),
        'contrast_alpha': 1.0,
        'brightness_beta': 0,
        
        # Tracking
        'axis_length': 0.05,  # meters, used for pose visualization
        'feature_tracking': {
            # Detector selection: 'orb', 'fast_brief', 'akaze', 'brisk', 'sift', 'gftt_orb'
            'method': 'orb',
            'max_features': 1000,
            'quality_level': 0.01,
            'min_distance': 7.0,
            
            # ORB-specific
            'fast_threshold': 20,
            'orb_scale_factor': 1.2,
            'orb_nlevels': 8,
            
            # AKAZE-specific
            'akaze_threshold': 0.001,
            
            # Optical flow
            'use_optical_flow': True,
            'optical_flow_win_size': 21,
            'optical_flow_max_level': 3,
            'optical_flow_criteria_eps': 0.03,
            'optical_flow_criteria_count': 30,
            'adaptive_optical_flow': True,  # Forward-backward consistency check
            
            # Keyframe management
            'reacquire_threshold': 200,
            'keyframe_interval': 15,
            'max_keyframes': 6,
            'min_keyframe_features': 160,
            'keyframe_quality_threshold': 0.5,
            
            # Matching
            'matcher_type': 'bf_hamming',  # 'bf_hamming', 'bf_l2', 'flann'
            'match_ratio_threshold': 0.75,  # Lowe's ratio test
            
            # Grid-based detection for better feature distribution
            'use_grid_detection': False,
            'grid_rows': 4,
            'grid_cols': 4,
        },
        
        # Calibration / pose estimation
        'calibration': {
            'calibration_file': None,  # Optional path to JSON file with camera_matrix/dist_coeffs
            'camera_matrix': [
                [800.0, 0.0, 320.0],
                [0.0, 800.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            'dist_coeffs': [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        
        # Pose filtering/smoothing
        'pose_filter': {
            'enable_smoothing': True,
            'smoothing_alpha': 0.3,  # EMA factor (0 = max smooth, 1 = no smooth)
            'enable_outlier_rejection': True,
            'max_translation_jump': 0.5,  # meters
            'max_rotation_jump': 0.5,  # radians
            'history_size': 10,
            'use_median_filter': False,
            'min_inliers_threshold': 10,
        },
        
        # Scale estimation for markerless tracking
        'scale_estimation': {
            'method': 'auto',  # 'auto', 'known_distance', 'ground_plane', 'object_size', 'manual'
            'manual_scale': 1.0,  # Scale factor when method='manual'
            'known_distance': None,  # Known distance in meters (if available)
            'ground_plane_height': 1.5,  # Expected camera height above ground (meters)
            'reference_object_size': None,  # Known object size in meters
            'scale_smoothing_alpha': 0.2,  # EMA factor for scale updates
            'min_scale': 0.001,  # Minimum allowed scale
            'max_scale': 100.0,  # Maximum allowed scale
            'consistency_threshold': 0.5,  # Max allowed scale change ratio per frame
        },
        
        # Overlay rendering
        'overlay': {
            'enable_2d_overlays': True,
            'enable_3d_overlays': True,
            'default_3d_color': [0, 255, 255],  # Cyan
            'default_2d_color': [0, 255, 0],  # Green
            'blend_alpha': 0.7,
            'antialiasing': True,
        },
        
        # Display
        'display_width': 640,
        'display_height': 480,
        'show_markers': True,
        'show_axes': True,
    }
    
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
                logging.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
    
    return default_config


def save_config(config, config_path):
    """Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
        
    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logging.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save config to {config_path}: {e}")
        return False


def validate_config(config):
    """Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_keys = ['camera_id', 'video_width', 'video_height']
    
    for key in required_keys:
        if key not in config:
            logging.error(f"Missing required config key: {key}")
            return False
    
    # Validate numeric values
    if config['video_width'] <= 0 or config['video_height'] <= 0:
        logging.error("Video dimensions must be positive")
        return False
    
    logging.info("Configuration validated successfully")
    return True


def get_timestamp():
    """Get current timestamp string.
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_directory(path):
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
        
    Returns:
        bool: True if created or exists, False on error
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Failed to create directory {path}: {e}")
        return False
