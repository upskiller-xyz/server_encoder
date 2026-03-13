import logging
from typing import Tuple, Dict, Any
from src.components.geometry.window_geometry import WindowGeometry
from src.core import GRAPHICS_CONSTANTS
from src.core import WindowHeightValidationError

logger = logging.getLogger(__name__)


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
        Windows within WINDOW_HEIGHT_CORRECTION_TOLERANCE (15cm) are clamped
        to floor/roof. Windows beyond that tolerance raise an error.

        Args:
            window_geometry: Window geometry with z1, z2 coordinates
            floor_height: Floor height (floor_height_above_terrain)
            roof_height: Roof height (floor_height_above_terrain + height_roof_over_floor)

        Returns:
            Tuple of (is_valid, error_message)
            - (True, "") if window is within bounds (or was clamped)
            - (False, error_message) if window extends beyond tolerance
        """
        try:
            z1 = window_geometry.z1
            z2 = window_geometry.z2
            tolerance = GRAPHICS_CONSTANTS.WINDOW_HEIGHT_CORRECTION_TOLERANCE

            window_bottom = min(z1, z2)
            window_top = max(z1, z2)

            # Check if window bottom is below floor
            if window_bottom < floor_height:
                deviation = floor_height - window_bottom
                if deviation > tolerance:
                    raise WindowHeightValidationError(
                        window_bottom=window_bottom,
                        window_top=window_top,
                        floor_height=floor_height,
                        roof_height=roof_height,
                        error_type="below_floor"
                    )
                logger.info(
                    "Window bottom (%.2fm) clamped to floor (%.2fm) — deviation: %.3fm",
                    window_bottom, floor_height, deviation
                )
                cls._clamp_z(window_geometry, z1, z2, floor_height, is_bottom=True)

            # Check if window top is above roof
            if window_top > roof_height:
                deviation = window_top - roof_height
                if deviation > tolerance:
                    raise WindowHeightValidationError(
                        window_bottom=window_bottom,
                        window_top=window_top,
                        floor_height=floor_height,
                        roof_height=roof_height,
                        error_type="above_roof"
                    )
                logger.info(
                    "Window top (%.2fm) clamped to roof (%.2fm) — deviation: %.3fm",
                    window_top, roof_height, deviation
                )
                cls._clamp_z(window_geometry, z1, z2, roof_height, is_bottom=False)

            return True, ""

        except (KeyError, ValueError, AttributeError) as e:
            return False, f"Error validating window height: {type(e).__name__}: {str(e)}"

    @staticmethod
    def _clamp_z(
        window_geometry: WindowGeometry,
        z1: float,
        z2: float,
        target: float,
        is_bottom: bool
    ) -> None:
        """Clamp the bottom or top z-coordinate of the window to the target value."""
        if is_bottom:
            if z1 <= z2:
                window_geometry.z1 = target
            else:
                window_geometry.z2 = target
        else:
            if z1 >= z2:
                window_geometry.z1 = target
            else:
                window_geometry.z2 = target

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

        except WindowHeightValidationError as e:
            return False, str(e)
        except (KeyError, ValueError, AttributeError, TypeError) as e:
            return False, f"Error parsing height data: {type(e).__name__}: {str(e)}"
