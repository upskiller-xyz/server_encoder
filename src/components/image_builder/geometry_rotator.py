from typing import Optional
import copy
import math
from src.core import ParameterName, GRAPHICS_CONSTANTS
from src.models import EncodingParameters
from src.components.geometry import WindowGeometry, RoomPolygon, Point2D


class GeometryRotator:
    """
    Rotates window and room geometry (Single Responsibility)
    """

    @staticmethod
    def rotate_if_needed(
        all_parameters: EncodingParameters,
        window_geom: WindowGeometry,
        room_polygon: Optional[RoomPolygon]
    ) -> EncodingParameters:
        """
        Rotate geometry if window is not pointing right (0 degrees)

        Args:
            all_parameters: All parameters grouped by region
            window_geom: WindowGeometry object with direction_angle set
            room_polygon: RoomPolygon object or None

        Returns:
            Parameters with rotated geometry
        """
        # If direction_angle is None, treat as 0 (no rotation needed)
        if window_geom.direction_angle is None:
            all_parameters.set_global(ParameterName.DIRECTION_ANGLE.value, 0.0)
            return all_parameters

        direction_angle_degrees = window_geom.direction_angle * 180 / math.pi  # Convert to degrees
        all_parameters.set_global(ParameterName.DIRECTION_ANGLE.value, window_geom.direction_angle)

        # If already pointing right (within tolerance), no rotation needed
        if abs(direction_angle_degrees) < GRAPHICS_CONSTANTS.WINDOW_PLACEMENT_TOLERANCE:
            return all_parameters

        # Rotation angle is negative of direction angle (rotate opposite direction to align to 0°)
        rotation_angle = -direction_angle_degrees
        origin = Point2D(0, 0)

        # Calculate wall thickness BEFORE rotation (it's invariant under rotation)

        # Make a deep copy of parameters to avoid modifying original
        rotated_params = copy.deepcopy(all_parameters)

        # Rotate window geometry
        rotated_window = window_geom.rotate(rotation_angle, origin)
        window_params_copy = rotated_params.window.parameters

        GeometryRotator._update_window_coords(window_params_copy, rotated_window, window_geom.wall_thickness)

        # Rotate room polygon if present
        if room_polygon is not None:
            rotated_polygon = room_polygon.rotate(rotation_angle, origin)

            room_params = rotated_params.room.parameters
            room_params[ParameterName.ROOM_POLYGON.value] = rotated_polygon

            GeometryRotator._update_window_coords(room_params, rotated_window, window_geom.wall_thickness)

            # Set direction_angle to 0 after rotation (window now points right)
            room_params[ParameterName.DIRECTION_ANGLE.value] = 0.0

        return rotated_params

    @staticmethod
    def _update_window_coords(param_dict: dict, window: WindowGeometry, thickness: float) -> dict:
        """Update parameter dict with window coordinates and thickness"""
        param_dict[ParameterName.X1.value] = window.x1
        param_dict[ParameterName.Y1.value] = window.y1
        param_dict[ParameterName.X2.value] = window.x2
        param_dict[ParameterName.Y2.value] = window.y2
        param_dict[ParameterName.WALL_THICKNESS.value] = thickness
        return param_dict
