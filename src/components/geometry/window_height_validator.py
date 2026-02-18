from typing import Tuple, Dict, Any
from src.components.geometry.window_geometry import WindowGeometry
from src.core import GRAPHICS_CONSTANTS
from src.core import WindowHeightValidationError


class WindowHeightValidator:
    """Validates that window z-coordinates lie between room floor and roof."""

    @classmethod
    def validate_window_height_bounds(
        cls,
        window_geometry: WindowGeometry,
        floor_height: float,
        roof_height: float
    ) -> Tuple[bool, str]:
        """
        Validate that window z-coordinates are within floor-roof bounds.

        Args:
            window_geometry: Window geometry with z1, z2 coordinates
            floor_height: Floor height (floor_height_above_terrain)
            roof_height: Roof height (floor_height_above_terrain + height_roof_over_floor)

        Returns:
            Tuple of (is_valid, error_message)
            - (True, "") if window is within bounds
            - (False, error_message) if window extends beyond floor or roof
        """
        try:
            z1 = window_geometry.z1
            z2 = window_geometry.z2

            window_bottom = min(z1, z2)
            window_top = max(z1, z2)

            # Check if window bottom is below floor
            if window_bottom < floor_height - GRAPHICS_CONSTANTS.WINDOW_HEIGHT_TOLERANCE:
                raise WindowHeightValidationError(
                window_bottom=window_bottom,
                window_top=window_top,
                floor_height=floor_height,
                roof_height=roof_height,
                error_type="below_floor"
            )

        # Check if window top is above roof
            if window_top > roof_height + GRAPHICS_CONSTANTS.WINDOW_HEIGHT_TOLERANCE:
                raise WindowHeightValidationError(
                window_bottom=window_bottom,
                window_top=window_top,
                floor_height=floor_height,
                roof_height=roof_height,
                error_type="above_roof"
                )

            return True, ""

        except (KeyError, ValueError, AttributeError) as e:
            return False, f"Error validating window height: {type(e).__name__}: {str(e)}"

    @classmethod
    def validate_from_parameters(
        cls,
        window_geometry_data: Dict[str, Any],
        floor_height: float,
        roof_height: float
    ) -> Tuple[bool, str]:
        """
        Validate window height from parameter dictionaries.

        Args:
            window_geometry_data: Dict with x1, y1, z1, x2, y2, z2
            floor_height: Floor height above terrain
            roof_height: Roof height (floor + height_roof_over_floor)
            tolerance: Numerical tolerance for comparisons

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Parse window geometry
            window_geom = WindowGeometry.from_dict(window_geometry_data)

            return cls.validate_window_height_bounds(window_geom, floor_height, roof_height)

        except (KeyError, ValueError, AttributeError, TypeError) as e:
            return False, f"Error parsing height data: {type(e).__name__}: {str(e)}"
