"""
Reflectance Parameters Model

Encapsulates all reflectance parameters with validation.
Follows the Strategy Pattern for organizing related parameters.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from src.core import ParameterName


@dataclass
class ReflectanceParameters:
    """
    Encapsulates reflectance parameters for various surfaces.

    All reflectance values must be between 0 and 1 (representing 0-100% reflectance).
    """
    # Room surface reflectances
    ceiling_reflectance: Optional[float] = None
    floor_reflectance: Optional[float] = None
    wall_reflectance: Optional[float] = None

    # Background/environment reflectances
    facade_reflectance: Optional[float] = None
    terrain_reflectance: Optional[float] = None

    # Obstruction parameters
    balcony_reflectance: Optional[float] = None
    context_reflectance: Optional[float] = None

    # Window frame reflectance
    window_frame_reflectance: Optional[float] = None

    def validate(self) -> Tuple[bool, str]:
        """
        Validate all reflectance parameters.

        Returns:
            (is_valid, error_message) tuple
        """
        reflectances = {
            ParameterName.CEILING_REFLECTANCE.value: self.ceiling_reflectance,
            ParameterName.FLOOR_REFLECTANCE.value: self.floor_reflectance,
            ParameterName.WALL_REFLECTANCE.value: self.wall_reflectance,
            ParameterName.FACADE_REFLECTANCE.value: self.facade_reflectance,
            ParameterName.TERRAIN_REFLECTANCE.value: self.terrain_reflectance,
            ParameterName.BALCONY_REFLECTANCE.value: self.balcony_reflectance,
            ParameterName.CONTEXT_REFLECTANCE.value: self.context_reflectance,
            ParameterName.WINDOW_FRAME_REFLECTANCE.value: self.window_frame_reflectance,
        }

        for name, value in reflectances.items():
            if value is not None and not 0 <= value <= 1:
                return False, f"{name} must be between 0 and 1, got {value}"

        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format, excluding None values.

        Returns:
            Dictionary with all non-None reflectance parameters
        """
        result = {}

        if self.ceiling_reflectance is not None:
            result[ParameterName.CEILING_REFLECTANCE.value] = self.ceiling_reflectance
        if self.floor_reflectance is not None:
            result[ParameterName.FLOOR_REFLECTANCE.value] = self.floor_reflectance
        if self.wall_reflectance is not None:
            result[ParameterName.WALL_REFLECTANCE.value] = self.wall_reflectance
        if self.facade_reflectance is not None:
            result[ParameterName.FACADE_REFLECTANCE.value] = self.facade_reflectance
        if self.terrain_reflectance is not None:
            result[ParameterName.TERRAIN_REFLECTANCE.value] = self.terrain_reflectance
        if self.balcony_reflectance is not None:
            result[ParameterName.BALCONY_REFLECTANCE.value] = self.balcony_reflectance
        if self.context_reflectance is not None:
            result[ParameterName.CONTEXT_REFLECTANCE.value] = self.context_reflectance
        if self.window_frame_reflectance is not None:
            result[ParameterName.WINDOW_FRAME_REFLECTANCE.value] = self.window_frame_reflectance

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReflectanceParameters':
        """
        Parse dictionary into ReflectanceParameters.

        Args:
            data: Dictionary with reflectance parameters

        Returns:
            ReflectanceParameters instance
        """
        return cls(
            ceiling_reflectance=float(data[ParameterName.CEILING_REFLECTANCE.value]) if ParameterName.CEILING_REFLECTANCE.value in data else None,
            floor_reflectance=float(data[ParameterName.FLOOR_REFLECTANCE.value]) if ParameterName.FLOOR_REFLECTANCE.value in data else None,
            wall_reflectance=float(data[ParameterName.WALL_REFLECTANCE.value]) if ParameterName.WALL_REFLECTANCE.value in data else None,
            facade_reflectance=float(data[ParameterName.FACADE_REFLECTANCE.value]) if ParameterName.FACADE_REFLECTANCE.value in data else None,
            terrain_reflectance=float(data[ParameterName.TERRAIN_REFLECTANCE.value]) if ParameterName.TERRAIN_REFLECTANCE.value in data else None,
            balcony_reflectance=float(data[ParameterName.BALCONY_REFLECTANCE.value]) if ParameterName.BALCONY_REFLECTANCE.value in data else None,
            context_reflectance=float(data[ParameterName.CONTEXT_REFLECTANCE.value]) if ParameterName.CONTEXT_REFLECTANCE.value in data else None,
            window_frame_reflectance=float(data[ParameterName.WINDOW_FRAME_REFLECTANCE.value]) if ParameterName.WINDOW_FRAME_REFLECTANCE.value in data else None,
        )
