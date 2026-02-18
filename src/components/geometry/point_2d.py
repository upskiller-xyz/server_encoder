from typing import Tuple
from dataclasses import dataclass
from src.core import GRAPHICS_CONSTANTS


@dataclass
class Point2D:
    """Represents a 2D point in meters"""
    x: float
    y: float

    def to_pixel(self, resolution: float = 0.1) -> Tuple[int, int]:
        """
        Convert point from meters to pixels

        Args:
            resolution: Meters per pixel (default 0.1m = 10cm)

        Returns:
            (x_pixel, y_pixel) tuple
        """
        return (GRAPHICS_CONSTANTS.get_pixel_value(self.x), GRAPHICS_CONSTANTS.get_pixel_value(self.y))
