from typing import Dict, Any
from abc import ABC, abstractmethod
from src.components.enums import ParameterName
from src.components.geometry import WindowGeometry


class IParameterCalculator(ABC):
    """
    Abstract base class for parameter calculators (Strategy Pattern)

    Calculates derived parameters from input data.
    """

    @abstractmethod
    def can_calculate(self, parameters: Dict[str, Any]) -> bool:
        """
        Check if this calculator can compute from given parameters

        Args:
            parameters: Input parameters

        Returns:
            True if calculator has required inputs
        """
        pass

    @abstractmethod
    def calculate(self, parameters: Dict[str, Any]) -> Any:
        """
        Calculate the parameter value

        Args:
            parameters: Input parameters

        Returns:
            Calculated value

        Raises:
            ValueError: If calculation cannot be performed
        """
        pass

    @abstractmethod
    def get_parameter_name(self) -> str:
        """Get the name of the parameter this calculator produces"""
        pass


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


class ParameterCalculatorRegistry:
    """
    Registry of parameter calculators (Factory + Registry Pattern)

    Manages automatic calculation of derived parameters.
    """

    # Available calculators (order matters: dependencies should be calculated first)
    _CALCULATORS = [
        WindowHeightCalculator(),         # No dependencies
        WindowSillHeightCalculator(),     # Depends on floor_height_above_terrain
    ]

    @classmethod
    def calculate_derived_parameters(
        cls,
        parameters: Dict[str, Any],
        logger=None
    ) -> Dict[str, Any]:
        """
        Calculate all derived parameters that can be computed

        Args:
            parameters: Input parameters
            logger: Optional logger for warnings

        Returns:
            Dictionary with calculated parameter values

        Raises:
            ValueError: If strict mode (logger=None) and calculation fails
        """
        result = {}

        for calculator in cls._CALCULATORS:
            param_name = calculator.get_parameter_name()

            # Skip if parameter already provided
            if param_name in parameters:
                continue

            # Try to calculate
            if calculator.can_calculate(parameters):
                try:
                    calculated_value = calculator.calculate(parameters)
                    result[param_name] = calculated_value

                    if logger:
                        logger.debug(
                            f"Auto-calculated {param_name} = {calculated_value}"
                        )
                except ValueError as e:
                    if logger is None:
                        # Strict mode: raise error
                        raise
                    else:
                        # Lenient mode: log warning
                        logger.warning(
                            f"Could not auto-calculate {param_name}: {str(e)}"
                        )

        return result
