"""
Custom exceptions for the window encoder system.

This module defines custom exception classes for specific error conditions
that can occur during window geometry validation and encoding.
"""

from typing import Optional


class WindowEncoderException(Exception):
    """Base exception class for all window encoder errors"""
    pass


class GeometryValidationError(WindowEncoderException):
    """Base exception for geometry validation errors"""
    pass


class WindowNotOnPolygonError(GeometryValidationError):
    """
    Exception raised when window edge does not lie on room polygon border.

    This error indicates that the window's bounding box does not align with
    any edge of the room polygon within the specified tolerance.
    """

    def __init__(
        self,
        window_bbox: tuple[float, float, float, float],
        direction_angle: float,
        tolerance: float,
        polygon_edges: Optional[list] = None,
        window_edges: Optional[list] = None
    ):
        """
        Initialize WindowNotOnPolygonError.

        Args:
            window_bbox: Window bounding box as (x1, y1, x2, y2)
            direction_angle: Window direction angle in radians
            tolerance: Placement tolerance in meters
            polygon_edges: List of polygon edges (optional)
            window_edges: List of window candidate edges (optional)
        """
        self.window_bbox = window_bbox
        self.direction_angle = direction_angle
        self.tolerance = tolerance
        self.polygon_edges = polygon_edges
        self.window_edges = window_edges

        x1, y1, x2, y2 = window_bbox

        # Handle None direction_angle
        if direction_angle is not None:
            direction_degrees = direction_angle * 180 / 3.14159265359
            angle_info = f"with direction_angle={direction_angle:.3f} rad ({direction_degrees:.1f}°). "
        else:
            angle_info = "(direction_angle not set). "

        message = (
            f"Window edge not on polygon border. "
            f"Window bounding box from ({x1:.3f}, {y1:.3f}) to ({x2:.3f}, {y2:.3f}) "
            f"{angle_info}"
            f"Expected edges: ({x1:.1f}, {y1:.1f})->({x2:.1f}, {y1:.1f}) "
            f"or ({x1:.1f}, {y2:.1f})->({x2:.1f}, {y2:.1f}) "
            f"to lie on polygon edge (tolerance: {tolerance}m)."
        )

        if polygon_edges:
            message += f"\nPolygon has {len(polygon_edges)} edges."
        if window_edges:
            message += f"\nWindow has {len(window_edges)} candidate edges."

        super().__init__(message)


class WindowHeightValidationError(GeometryValidationError):
    """
    Exception raised when window z-coordinates are outside floor-roof bounds.

    This error indicates that the window extends beyond the valid vertical
    range defined by the floor and roof heights.
    """

    def __init__(
        self,
        window_bottom: float,
        window_top: float,
        floor_height: float,
        roof_height: float,
        error_type: str = "unknown"
    ):
        """
        Initialize WindowHeightValidationError.

        Args:
            window_bottom: Window bottom z-coordinate
            window_top: Window top z-coordinate
            floor_height: Floor height
            roof_height: Roof height
            error_type: Type of error ("below_floor" or "above_roof")
        """
        self.window_bottom = window_bottom
        self.window_top = window_top
        self.floor_height = floor_height
        self.roof_height = roof_height
        self.error_type = error_type

        if error_type == "below_floor":
            message = (
                f"Window bottom ({window_bottom:.2f}m) is below floor ({floor_height:.2f}m). "
                f"Window z-coordinates must be between floor and roof heights."
            )
        elif error_type == "above_roof":
            message = (
                f"Window top ({window_top:.2f}m) is above roof ({roof_height:.2f}m). "
                f"Window z-coordinates must be between floor and roof heights."
            )
        else:
            message = (
                f"Window height validation failed. "
                f"Window range: ({window_bottom:.2f}m to {window_top:.2f}m), "
                f"Valid range: ({floor_height:.2f}m to {roof_height:.2f}m)."
            )

        super().__init__(message)


class ParameterValidationError(WindowEncoderException):
    """Exception raised when required parameters are missing or invalid"""

    def __init__(self, parameter_name: str, details: Optional[str] = None):
        """
        Initialize ParameterValidationError.

        Args:
            parameter_name: Name of the invalid/missing parameter
            details: Additional details about the validation error
        """
        self.parameter_name = parameter_name
        self.details = details

        message = f"Parameter validation failed for '{parameter_name}'"
        if details:
            message += f": {details}"

        super().__init__(message)


class EncodingError(WindowEncoderException):
    """Base exception for encoding-related errors"""
    pass


class RegionEncodingError(EncodingError):
    """Exception raised when region encoding fails"""

    def __init__(self, region_type: str, details: str):
        """
        Initialize RegionEncodingError.

        Args:
            region_type: Type of region that failed to encode
            details: Details about the encoding failure
        """
        self.region_type = region_type
        self.details = details

        message = f"Failed to encode {region_type} region: {details}"
        super().__init__(message)
