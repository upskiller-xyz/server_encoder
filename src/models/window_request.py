"""
Window Request Model

Represents a single window with all its parameters.
Provides validation and conversion to internal formats.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union, List
from src.core import ParameterName
from src.models.reflectance_parameters import ReflectanceParameters


@dataclass
class WindowRequest:
    """
    Window parameters from API request

    Represents a single window with all its parameters.
    Provides validation and conversion to internal formats.
    """
    # Required coordinates
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float

    # Required window parameters
    window_frame_ratio: float

    # Optional parameters (auto-calculated or defaulted)
    window_sill_height: Optional[float] = None
    window_height: Optional[float] = None
    direction_angle: Optional[float] = None

    # Reflectance parameters
    reflectance: ReflectanceParameters = field(default_factory=ReflectanceParameters)

    # Obstruction bar parameters (optional per window)
    # Can be either a scalar (float) or an array (List[float])
    horizon: Optional[Union[float, List[float]]] = None
    zenith: Optional[Union[float, List[float]]] = None

    def validate(self) -> Tuple[bool, str]:
        """
        Validate window parameters

        Returns:
            (is_valid, error_message) tuple
        """
        # Validate frame ratio
        if not 0 <= self.window_frame_ratio <= 1:
            return False, f"window_frame_ratio must be between 0 and 1, got {self.window_frame_ratio}"

        # Validate window dimensions are positive
        width = abs(self.x2 - self.x1)
        height = abs(self.z2 - self.z1)
        depth = abs(self.y2 - self.y1)

        if width <= 0 or height <= 0:
            return False, f"Window must have positive width and height"

        # Validate optional parameters if provided
        if self.window_sill_height is not None and self.window_sill_height < 0:
            return False, f"window_sill_height must be non-negative, got {self.window_sill_height}"

        if self.window_height is not None and self.window_height <= 0:
            return False, f"window_height must be positive, got {self.window_height}"

        if self.horizon is not None:
            if isinstance(self.horizon, list):
                if not all(0 <= h <= 90 for h in self.horizon):
                    return False, f"horizon array values must be between 0 and 90"
            else:
                if not 0 <= self.horizon <= 90:
                    return False, f"horizon must be between 0 and 90, got {self.horizon}"

        if self.zenith is not None:
            if isinstance(self.zenith, list):
                if not all(0 <= z <= 90 for z in self.zenith):
                    return False, f"zenith array values must be between 0 and 90"
            else:
                if not 0 <= self.zenith <= 90:
                    return False, f"zenith must be between 0 and 90, got {self.zenith}"

        # Validate reflectance parameters
        is_valid, error = self.reflectance.validate()
        if not is_valid:
            return False, f"Reflectance error: {error}"

        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = {
            ParameterName.X1.value: self.x1,
            ParameterName.Y1.value: self.y1,
            ParameterName.Z1.value: self.z1,
            ParameterName.X2.value: self.x2,
            ParameterName.Y2.value: self.y2,
            ParameterName.Z2.value: self.z2,
            ParameterName.WINDOW_FRAME_RATIO.value: self.window_frame_ratio,
        }

        # Add optional parameters if present
        if self.window_sill_height is not None:
            result[ParameterName.WINDOW_SILL_HEIGHT.value] = self.window_sill_height
        if self.window_height is not None:
            result[ParameterName.WINDOW_HEIGHT.value] = self.window_height
        if self.direction_angle is not None:
            result[ParameterName.DIRECTION_ANGLE.value] = self.direction_angle
        if self.horizon is not None:
            result[ParameterName.HORIZON.value] = self.horizon
        if self.zenith is not None:
            result[ParameterName.ZENITH.value] = self.zenith

        # Add reflectance parameters
        result.update(self.reflectance.to_dict())

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WindowRequest':
        """
        Parse dictionary into WindowRequest

        Args:
            data: Raw request dictionary

        Returns:
            WindowRequest instance

        Raises:
            ValueError: If required fields are missing
        """
        # Check required fields
        required = [
            ParameterName.X1.value,
            ParameterName.Y1.value,
            ParameterName.Z1.value,
            ParameterName.X2.value,
            ParameterName.Y2.value,
            ParameterName.Z2.value,
            ParameterName.WINDOW_FRAME_RATIO.value
        ]
        missing = [field for field in required if field not in data]
        if missing:
            raise ValueError(f"Missing required window fields: {', '.join(missing)}")

        # Parse reflectance parameters
        reflectance = ReflectanceParameters.from_dict(data)

        return cls(
            x1=float(data[ParameterName.X1.value]),
            y1=float(data[ParameterName.Y1.value]),
            z1=float(data[ParameterName.Z1.value]),
            x2=float(data[ParameterName.X2.value]),
            y2=float(data[ParameterName.Y2.value]),
            z2=float(data[ParameterName.Z2.value]),
            window_frame_ratio=float(data[ParameterName.WINDOW_FRAME_RATIO.value]),
            window_sill_height=float(data[ParameterName.WINDOW_SILL_HEIGHT.value]) if ParameterName.WINDOW_SILL_HEIGHT.value in data else None,
            window_height=float(data[ParameterName.WINDOW_HEIGHT.value]) if ParameterName.WINDOW_HEIGHT.value in data else None,
            reflectance=reflectance,
            direction_angle=float(data[ParameterName.DIRECTION_ANGLE.value]) if ParameterName.DIRECTION_ANGLE.value in data else None,
            horizon=WindowRequest._parse_obstruction_value(data, ParameterName.HORIZON.value),
            zenith=WindowRequest._parse_obstruction_value(data, ParameterName.ZENITH.value),
        )

    @staticmethod
    def _parse_obstruction_value(data: Dict[str, Any], key: str) -> Optional[Union[float, List[float]]]:
        """
        Parse obstruction value (horizon or zenith) from dictionary.
        
        Can be either a scalar float or an array of floats.
        
        Args:
            data: Dictionary to parse from
            key: Key to look up
            
        Returns:
            Float, List[float], or None
        """
        if key not in data:
            return None
        
        value = data[key]
        if isinstance(value, list):
            return [float(v) for v in value]
        else:
            return float(value)
