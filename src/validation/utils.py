"""Validation utility functions (SRP: helper functions for validation checks)"""
from typing import Dict, Any
from src.core.enums import ParameterName, REQUIRED_WINDOW_COORDINATES, REQUIRED_WINDOW_2D_COORDINATES


class ValidationUtils:
    """Utility class for common validation checks"""

    @staticmethod
    def has_window_coordinates(parameters: Dict[str, Any], require_3d: bool = True) -> bool:
        """
        Check if parameters contain window coordinates.

        Args:
            parameters: Parameters dict to check
            require_3d: Whether to require 3D coordinates (x1, y1, z1, x2, y2, z2)
                       If False, only 2D coordinates are required (x1, y1, x2, y2)

        Returns:
            True if window coordinates are present (either as window_geometry or flat)
        """
        # Check for nested window_geometry object
        if ParameterName.WINDOW_GEOMETRY.value in parameters:
            return True

        # Check for flat coordinate structure
        required_coords = REQUIRED_WINDOW_COORDINATES if require_3d else REQUIRED_WINDOW_2D_COORDINATES
        return all(k in parameters for k in required_coords)

    @staticmethod
    def has_flat_window_coordinates(parameters: Dict[str, Any]) -> bool:
        """
        Check if parameters contain flat window coordinates (x1, y1, z1, x2, y2, z2).

        This is a convenience method for has_window_coordinates with require_3d=True.

        Args:
            parameters: Parameters dict to check

        Returns:
            True if flat 3D window coordinates are present
        """
        return all(k in parameters for k in REQUIRED_WINDOW_COORDINATES)

    @staticmethod
    def has_2d_window_coordinates(parameters: Dict[str, Any]) -> bool:
        """
        Check if parameters contain 2D window coordinates (x1, y1, x2, y2).

        Args:
            parameters: Parameters dict to check

        Returns:
            True if 2D window coordinates are present
        """
        return all(k in parameters for k in REQUIRED_WINDOW_2D_COORDINATES)
