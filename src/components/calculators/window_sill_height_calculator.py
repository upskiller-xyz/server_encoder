from src.components.calculators.i_parameter_calculator import IParameterCalculator
from src.core import ParameterName
from src.components.geometry import WindowGeometry
from typing import Dict, Any


class WindowSillHeightCalculator(IParameterCalculator):
    """
    Calculator for window_sill_height parameter

    Formula:
    - window_sill_height = max(0, min(z1, z2) - floor_height_above_terrain)
    - Capped to 0 if window bottom is below floor level

    Where:
    - z1, z2: Window bottom and top Z coordinates
    - floor_height_above_terrain: Height of floor above terrain
    """

    def can_calculate(self, parameters: Dict[str, Any]) -> bool:
        """Check if we have window geometry and floor height"""
        has_window_geometry = ParameterName.WINDOW_GEOMETRY.value in parameters
        has_z_coords = ParameterName.Z1.value in parameters and ParameterName.Z2.value in parameters
        has_floor_height = ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value in parameters

        return (has_window_geometry or has_z_coords) and has_floor_height

    def calculate(self, parameters: Dict[str, Any]) -> float:
        """
        Calculate window sill height

        Args:
            parameters: Must contain z1, z2 (or window_geometry) and floor_height_above_terrain

        Returns:
            Calculated window sill height in meters (minimum 0)

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
                    z1 = float(window_geom.get(ParameterName.Z1.value, 0))
                    z2 = float(window_geom.get(ParameterName.Z2.value, 0))
            else:
                z1 = float(parameters.get(ParameterName.Z1.value, 0))
                z2 = float(parameters.get(ParameterName.Z2.value, 0))

            floor_height = float(parameters[ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value])

            # Calculate: min(z1, z2) - floor_height, capped at 0
            window_bottom = min(z1, z2)
            window_sill_height = max(0.0, window_bottom - floor_height)

            return window_sill_height

        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"Cannot calculate window_sill_height. "
                f"Required: z1, z2, floor_height_above_terrain. "
                f"Error: {type(e).__name__}: {str(e)}"
            )

    def get_parameter_name(self) -> str:
        """Get parameter name"""
        return ParameterName.WINDOW_SILL_HEIGHT.value
