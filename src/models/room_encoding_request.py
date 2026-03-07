"""
Room Encoding Request Model

Represents a complete room encoding request with all parameters.
Provides validation and conversion to internal formats.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np
from src.core import ModelType, ParameterName
from src.core.enums import REQUIRED_WINDOW_COORDINATES
from src.models.window_request import WindowRequest
from src.models.reflectance_parameters import ReflectanceParameters


def _parse_float_or_list(value: Any) -> Optional[Union[float, List[float]]]:
    """
    Parse value as either float or list of floats
    
    Args:
        value: Value to parse (can be int, float, or list)
        
    Returns:
        Parsed value as float, list, or None
    """
    if value is None:
        return None
    if isinstance(value, (list, np.ndarray)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


@dataclass
class RoomEncodingRequest:
    """
    Complete room encoding request from API

    Represents all parameters needed to encode a room image.
    Provides validation and conversion to internal formats.
    """
    # Model configuration
    model_type: ModelType

    # Required room parameters
    height_roof_over_floor: float
    floor_height_above_terrain: float

    # Room geometry (optional but recommended)
    room_polygon: Optional[List[List[float]]] = None

    # Windows (can be single window or multiple)
    windows: Dict[str, WindowRequest] = field(default_factory=dict)

    # Optional reflectance parameters
    reflectance: ReflectanceParameters = field(default_factory=ReflectanceParameters)

    # Optional background parameters
    window_orientation: Optional[float] = None

    # Global obstruction parameters (can be overridden per window)
    # Can be float (single value) or list (array of values for obstruction pattern)
    horizon: Optional[Union[float, List[float]]] = None
    zenith: Optional[Union[float, List[float]]] = None

    def validate(self) -> Tuple[bool, str]:
        """
        Validate request parameters

        Returns:
            (is_valid, error_message) tuple
        """
        # Validate roof height
        if self.height_roof_over_floor <= 0:
            return False, f"height_roof_over_floor must be positive, got {self.height_roof_over_floor}"

        # Validate floor height
        if self.floor_height_above_terrain < 0:
            return False, f"floor_height_above_terrain must be non-negative, got {self.floor_height_above_terrain}"

        # Validate room polygon if provided
        if self.room_polygon is not None:
            if len(self.room_polygon) < 3:
                return False, "room_polygon must have at least 3 vertices"
            for i, vertex in enumerate(self.room_polygon):
                if len(vertex) < 2:
                    return False, f"room_polygon vertex {i} must have at least 2 coordinates"

        # Validate windows
        if not self.windows:
            return False, "At least one window is required"

        for window_id, window in self.windows.items():
            is_valid, error = window.validate()
            if not is_valid:
                return False, f"Window '{window_id}': {error}"

        # Validate reflectance parameters
        is_valid, error = self.reflectance.validate()
        if not is_valid:
            return False, f"Reflectance error: {error}"

        # Validate orientation
        if self.window_orientation is not None and not 0 <= self.window_orientation <= 360:
            return False, f"window_orientation must be between 0 and 360, got {self.window_orientation}"

        return True, ""

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert to flat dictionary format (for backward compatibility)

        Returns:
            Flat dictionary with all parameters
        """
        result = {
            ParameterName.HEIGHT_ROOF_OVER_FLOOR.value: self.height_roof_over_floor,
            ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value: self.floor_height_above_terrain,
        }

        # Add room polygon if present
        if self.room_polygon is not None:
            result[ParameterName.ROOM_POLYGON.value] = self.room_polygon

        # Add optional background parameters
        if self.window_orientation is not None:
            result[ParameterName.WINDOW_ORIENTATION.value] = self.window_orientation

        # Add global obstruction parameters
        if self.horizon is not None:
            result[ParameterName.HORIZON.value] = self.horizon
        if self.zenith is not None:
            result[ParameterName.ZENITH.value] = self.zenith

        # Add reflectance parameters
        result.update(self.reflectance.to_dict())

        # Add windows
        if len(self.windows) == 1:
            # Single window: merge into flat structure
            window = list(self.windows.values())[0]
            result.update(window.to_dict())
        else:
            # Multiple windows: nested structure
            result[ParameterName.WINDOWS.value] = {
                window_id: window.to_dict()
                for window_id, window in self.windows.items()
            }

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RoomEncodingRequest':
        """
        Parse raw request dictionary into RoomEncodingRequest

        Args:
            data: Raw API request dictionary

        Returns:
            RoomEncodingRequest instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Parse model type
        model_type_str = data.get(ParameterName.MODEL_TYPE.value)
        if not model_type_str:
            raise ValueError(f"Missing required field: {ParameterName.MODEL_TYPE.value}")

        try:
            model_type = ModelType(model_type_str)
        except ValueError:
            raise ValueError(f"Invalid model_type: {model_type_str}")

        # Get parameters section
        params = data.get(ParameterName.PARAMETERS.value, {})

        # Parse required fields
        if ParameterName.HEIGHT_ROOF_OVER_FLOOR.value not in params:
            raise ValueError(f"Missing required field: {ParameterName.HEIGHT_ROOF_OVER_FLOOR.value}")
        if ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value not in params:
            raise ValueError(f"Missing required field: {ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value}")

        # Parse windows
        windows = {}

        # Check for single window (flat structure)
        if all(k in params for k in REQUIRED_WINDOW_COORDINATES):
            windows['window_0'] = WindowRequest.from_dict(params)
        # Check for multiple windows (nested structure)
        elif ParameterName.WINDOWS.value in params:
            windows_data = params[ParameterName.WINDOWS.value]
            if not isinstance(windows_data, dict):
                raise ValueError(f"'{ParameterName.WINDOWS.value}' must be a dictionary")
            for window_id, window_data in windows_data.items():
                windows[window_id] = WindowRequest.from_dict(window_data)
        else:
            raise ValueError(f"No windows found in request. Need either flat window coordinates or '{ParameterName.WINDOWS.value}' dictionary")

        # Parse reflectance parameters
        reflectance = ReflectanceParameters.from_dict(params)

        return cls(
            model_type=model_type,
            height_roof_over_floor=float(params[ParameterName.HEIGHT_ROOF_OVER_FLOOR.value]),
            floor_height_above_terrain=float(params[ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value]),
            room_polygon=params.get(ParameterName.ROOM_POLYGON.value),
            windows=windows,
            reflectance=reflectance,
            window_orientation=float(params[ParameterName.WINDOW_ORIENTATION.value]) if ParameterName.WINDOW_ORIENTATION.value in params else None,
            horizon=_parse_float_or_list(params.get(ParameterName.HORIZON.value)),
            zenith=_parse_float_or_list(params.get(ParameterName.ZENITH.value)),
        )
