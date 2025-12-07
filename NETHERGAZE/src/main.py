"""
Main entry point for the NETHERGAZE application.

This module runs the world-anchored AR demo with pose statistics display.

Usage:
    python main.py                      # Run with defaults
    python main.py --fast               # High performance mode
    python main.py --verbose            # Enable debug logging
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Add examples directory to path for demo import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from utils import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="NETHERGAZE - Markerless Augmented Reality with World-Anchored Objects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                # Run the AR demo
  python main.py --fast         # High performance mode (60fps)
  python main.py --verbose      # Enable debug logging

Controls (in demo):
  SPACE  - Set anchor point (world origin)
  1-5    - Place objects (1=cube, 2=pyramid, 3=axes, 4=box, 5=chair)
  C      - Clear all objects
  G      - Toggle ground grid
  M      - Toggle feature markers
  R      - Reset anchor
  H      - Toggle help overlay
  Q      - Quit
        """,
    )

    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="High performance mode (60fps, optimized settings)",
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

    LOGGER.info("Starting NETHERGAZE...")

    # Import and run the anchored objects demo
    try:
        from demo_anchored_objects import AnchoredObjectsDemo
        
        demo = AnchoredObjectsDemo(high_performance=args.fast)
        demo.run()
        
    except ImportError as e:
        LOGGER.error("Failed to import demo: %s", e)
        print("\nError: Could not load the AR demo.")
        print("Make sure you're running from the NETHERGAZE directory.")
        sys.exit(1)
    except Exception as e:
        LOGGER.exception("Application error: %s", e)
        sys.exit(1)

    LOGGER.info("NETHERGAZE exited normally")
    sys.exit(0)


if __name__ == "__main__":
    main()
