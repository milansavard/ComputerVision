"""
User interface module.

This module handles user interaction and control interface.
"""

import cv2
import logging


class UserInterface:
    """Handles user interaction and control interface using OpenCV."""
    
    def __init__(self, config=None):
        """Initialize user interface.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Window settings
        self.window_name = "NETHERGAZE"
        self.display_width = self.config.get('display_width', 640)
        self.display_height = self.config.get('display_height', 480)
        
        # Control flags
        self.show_markers = self.config.get('show_markers', True)
        self.show_axes = self.config.get('show_axes', True)
        self.paused = False
        
    def initialize(self):
        """Initialize user interface.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Create main display window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
            
            self.logger.info(f"UI initialized: {self.display_width}x{self.display_height}")
            return True
            
        except Exception as e:
            self.logger.error(f"UI initialization failed: {e}")
            return False
    
    def display_frame(self, frame):
        """Display the processed frame.
        
        Args:
            frame: Frame to display (numpy array)
        """
        if frame is None:
            return
            
        try:
            # Add status text
            self._add_status_text(frame)
            
            # Display the frame
            cv2.imshow(self.window_name, frame)
            
        except Exception as e:
            self.logger.error(f"Frame display error: {e}")
    
    def handle_events(self):
        """Handle user input events.
        
        Returns:
            bool: True to continue running, False to exit
        """
        try:
            # Wait for key press (1ms delay for non-blocking)
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key presses
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                self.logger.info("User requested exit")
                return False
            
            elif key == ord('m'):  # Toggle marker display
                self.show_markers = not self.show_markers
                self.logger.info(f"Marker display: {self.show_markers}")
            
            elif key == ord('a'):  # Toggle axes display
                self.show_axes = not self.show_axes
                self.logger.info(f"Axes display: {self.show_axes}")
            
            elif key == ord('p'):  # Pause/resume
                self.paused = not self.paused
                self.logger.info(f"Paused: {self.paused}")
            
            elif key == ord('h'):  # Show help
                self._print_help()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Event handling error: {e}")
            return False
    
    def _add_status_text(self, frame):
        """Add status text overlay to frame.
        
        Args:
            frame: Frame to add text to
        """
        try:
            # Add status text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            color = (0, 255, 0)  # Green
            
            # Status messages
            y_offset = 20
            if self.paused:
                cv2.putText(frame, "PAUSED", (10, y_offset), 
                           font, font_scale, (0, 0, 255), thickness)
                y_offset += 20
            
            cv2.putText(frame, "Press 'h' for help, 'q' to quit", (10, y_offset),
                       font, font_scale, color, thickness)
            
        except Exception as e:
            self.logger.error(f"Status text error: {e}")
    
    def _print_help(self):
        """Print help information to console."""
        help_text = """
        NETHERGAZE Controls:
        ====================
        q / ESC - Quit application
        m       - Toggle marker display
        a       - Toggle axes display
        p       - Pause/Resume
        h       - Show this help
        """
        print(help_text)
    
    def create_control_panel(self):
        """Create control panel for user interaction.
        
        Note: This is a placeholder for future GUI enhancements.
        Currently using keyboard controls only.
        """
        # TODO: Add trackbars or GUI controls if needed
        pass
    
    def cleanup(self):
        """Clean up UI resources."""
        try:
            cv2.destroyAllWindows()
            self.logger.info("UI cleaned up")
        except Exception as e:
            self.logger.error(f"UI cleanup error: {e}")
