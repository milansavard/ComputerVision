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
        
        # Preprocessing
        'enable_preprocessing': True,
        'blur_kernel': (5, 5),
        'contrast_alpha': 1.0,
        'brightness_beta': 0,
        
        # Marker detection
        'aruco_dict_type': 'DICT_4X4_50',
        'marker_size': 0.05,  # meters
        
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
