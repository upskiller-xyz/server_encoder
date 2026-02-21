from typing import Tuple, Any, Dict
import numpy as np
from src.core import RegionType, ParameterName, EncodingScheme
from src.components.region_encoders.base_region_encoder import BaseRegionEncoder
from src.components.region_encoders.validation_helpers import validate_required_parameters
from src.components.geometry import WindowGeometry
from src.core import GRAPHICS_CONSTANTS
from src.components.calculators import ParameterCalculatorRegistry


class WindowRegionEncoder(BaseRegionEncoder):
    """
    Encodes window region parameters

    Window is viewed from top (plan view):
    - Located 12 pixels from right edge (8 pixels from obstruction bar)
    - Appears as vertical line/rectangle
    - Width (horizontal) = wall thickness (~0.3m = 3px)
    - Height (vertical) = window width in 3D space (x2-x1)

    CORRECTED CHANNEL MAPPINGS:
    - Red: sill_height (0-5m input → 0-1 normalized)
    - Green: frame_ratio (1-0 input → 0-1 normalized, REVERSED)
    - Blue: window_height (0.2-5m input → 0.99-0.01 normalized, REVERSED)
    - Alpha: window_frame_reflectance (0-1 input → 0-1 normalized, optional, default=0.8)
    """

    def __init__(self, encoding_scheme: EncodingScheme = EncodingScheme.RGB):
        super().__init__(RegionType.WINDOW, encoding_scheme)

    def _update_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Calculate derived parameters for window region
        calculated_params = ParameterCalculatorRegistry.calculate_derived_parameters(params)
        # Update parameters with calculated values
        params.update(calculated_params)
        return params


    def _get_area_mask(self, image, parameters, model_type) -> np.ndarray[Any, Any]:
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        x_start, y_start, x_end, y_end = self._get_window_bounds(
            image, parameters
        )
        mask[y_start:y_end, x_start:x_end] = True
        return mask

    def _validate_required_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate required window parameters using list comprehension"""
        missing = validate_required_parameters(self._region_type, parameters)

        # Also need window geometry (either window_geometry or individual coords)
        has_geometry = (
            ParameterName.WINDOW_GEOMETRY.value in parameters or
            all(key in parameters for key in [
                ParameterName.X1.value, ParameterName.Y1.value, ParameterName.Z1.value,
                ParameterName.X2.value, ParameterName.Y2.value, ParameterName.Z2.value
            ])
        )

        if not has_geometry:
            missing.append("window geometry (x1,y1,z1,x2,y2,z2 or window_geometry)")

        if missing:
            raise ValueError(f"Missing required window parameters: {', '.join(missing)}")

    def _get_window_bounds(
        self,
        image: np.ndarray,
        parameters: Dict[str, Any]
    ) -> tuple:
        """
        Get window bounds in pixels from geometry

        Args:
            image: Image array
            parameters: Parameters including window geometry

        Returns:
            (x_start, y_start, x_end, y_end) tuple
        """
        height, width = image.shape[:2]

        # Get window geometry (already normalized to WindowGeometry class at entry point)
        if ParameterName.WINDOW_GEOMETRY.value in parameters:
            window_geom: WindowGeometry = parameters[ParameterName.WINDOW_GEOMETRY.value]
        else:
            # Create from individual coordinates
            window_geom = WindowGeometry(
                x1=parameters[ParameterName.X1.value],
                y1=parameters[ParameterName.Y1.value],
                z1=parameters[ParameterName.Z1.value],
                x2=parameters[ParameterName.X2.value],
                y2=parameters[ParameterName.Y2.value],
                z2=parameters[ParameterName.Z2.value],
                direction_angle=parameters.get(ParameterName.DIRECTION_ANGLE.value, None) #type: ignore
            )

        # Get pixel bounds from geometry
        x_start, y_start, x_end, y_end = window_geom.get_pixel_bounds(image_size=width)

        # Snap window to room's facade edge if available (preserves width, eliminates gap)
        
        x_start, x_end = self._snap_to_wall(x_start, x_end, parameters)
        
        # Enforce border (must remain background)
        border = GRAPHICS_CONSTANTS.BORDER_PX
        x_start = max(x_start, border)
        y_start = max(y_start, border)
        x_end = min(x_end, width - border)
        y_end = min(y_end, height - border)

        return (x_start, y_start, x_end, y_end)
    
    def _snap_to_wall(self, x1: float, x2: float, parameters: Dict[str, Any]) -> Tuple[float, float]:
        room_facade_right_edge = parameters.get(ParameterName.RIGHT_WALL.value)
        
        if room_facade_right_edge is not None:
            window_width = x2 - x1
            x1 = room_facade_right_edge + 1
            x2 = x1 + window_width
        return x1, x2

