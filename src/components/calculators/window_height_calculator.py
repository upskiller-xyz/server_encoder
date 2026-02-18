from typing import Dict, Any
from src.components.calculators.i_parameter_calculator import IParameterCalculator
from src.core import ParameterName
from src.components.geometry import WindowGeometry


class WindowHeightCalculator(IParameterCalculator):
    """
    Calculator for window_height parameter

    Formula:
    - If min(z1, z2) >= floor_height_above_terrain:
        window_height = abs(z2 - z1)  (normal case)
    - If min(z1, z2) < floor_height_above_terrain:
        window_height = max(z1, z2) - floor_height_above_terrain  (window starts at floor)

    Where:
    - z1, z2: Window bottom and top Z coordinates
    - floor_height_above_terrain: Height of floor above terrain (optional for this calc)
    """

    def can_calculate(self, parameters: Dict[str, Any]) -> bool:
        """Check if we have window Z coordinates"""
        has_window_geometry = ParameterName.WINDOW_GEOMETRY.value in parameters
        has_z_coords = ParameterName.Z1.value in parameters and ParameterName.Z2.value in parameters

        return has_window_geometry or has_z_coords

    def calculate(self, parameters: Dict[str, Any]) -> float:
        """
        Calculate window height

        Args:
            parameters: Must contain z1, z2 (or window_geometry)
                       Optional: floor_height_above_terrain for floor adjustment

        Returns:
            Calculated window height in meters

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        try:
            # Extract Z coordinates - handle both dict and WindowGeometry object
            if ParameterName.WINDOW_GEOMETRY.value in parameters:
                window_geom = parameters[ParameterName.WINDOW_GEOMETRY.value]
                if isinstance(window_geom, WindowGeometry):
                    z1 = window_geom.z1
                    z2 = window_geom.z2
                else:
                    # Dict format
                    z1 = float(window_geom[ParameterName.Z1.value])
                    z2 = float(window_geom[ParameterName.Z2.value])
            else:
                z1 = float(parameters[ParameterName.Z1.value])
                z2 = float(parameters[ParameterName.Z2.value])

            window_bottom = min(z1, z2)
            window_top = max(z1, z2)

            # Check if floor height is available
            if ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value in parameters:
                floor_height = float(parameters[ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value])

                window_height = window_top - window_bottom
                # If window bottom is below floor, calculate height from floor
                if window_bottom < floor_height:
                    window_height = window_top - floor_height

            else:
                # No floor height available, use full window height
                window_height = abs(z2 - z1)

            return window_height

        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"Cannot calculate window_height. "
                f"Required: z1, z2. "
                f"Error: {type(e).__name__}: {str(e)}"
            )

    def get_parameter_name(self) -> str:
        """Get parameter name"""
        return ParameterName.WINDOW_HEIGHT.value
