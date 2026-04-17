"""
Geometry service.

SRP: owns geometric queries — direction angle, reference point, and external
reference point. These are coordinate calculations, not encoding operations.
"""
from typing import Any, Dict
import logging

from src.core.enums import ParameterName
from src.components.geometry import WindowGeometry, RoomPolygon
from src.models import ReferencePointResult
from src.validation import ValidatorManager, RequestType

logger = logging.getLogger(__name__)


class GeometryService:
    """Calculates geometric properties of windows relative to room polygons."""

    def calculate_direction_angle(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate direction_angle for each window from room polygon and window coordinates.

        Args:
            parameters: Must contain room_polygon and windows dict (x1, y1, x2, y2 per window)

        Returns:
            {window_id: direction_angle_in_radians}

        Raises:
            ValueError: If parameters are invalid or calculation fails
        """
        self._validate(RequestType.CALCULATE_DIRECTION, parameters)

        room_polygon = self._parse_room_polygon(parameters)
        results = {}
        for window_id, window_params in parameters[ParameterName.WINDOWS.value].items():
            try:
                window_geom = WindowGeometry(
                    x1=window_params["x1"],
                    y1=window_params["y1"],
                    z1=window_params.get("z1", 0.0),
                    x2=window_params["x2"],
                    y2=window_params["y2"],
                    z2=window_params.get("z2", 1.0),
                )
                angle = window_geom.calculate_direction_from_polygon(room_polygon)
                results[window_id] = angle
                logger.info(
                    "Calculated direction_angle for '%s': %.4f rad (%.2f°)",
                    window_id, angle, angle * 180 / 3.14159,
                )
            except Exception as exc:
                raise ValueError(f"Failed to calculate direction_angle for window '{window_id}': {exc}") from exc

        return results

    def calculate_reference_point(self, parameters: Dict[str, Any]) -> Dict[str, ReferencePointResult]:
        """
        Calculate the reference point (window edge on room boundary) for each window.

        Args:
            parameters: Must contain room_polygon and windows dict (x1, y1, z1, x2, y2, z2 per window)

        Returns:
            {window_id: ReferencePointResult}

        Raises:
            ValueError: If parameters are invalid or calculation fails
        """
        self._validate(RequestType.GET_REFERENCE_POINT, parameters)

        room_polygon = self._parse_room_polygon(parameters)
        results = {}
        for window_id, window_params in parameters[ParameterName.WINDOWS.value].items():
            try:
                window_geom = WindowGeometry.from_dict(window_params)
                ref_point = window_geom.calculate_reference_point_from_polygon(room_polygon)
                results[window_id] = ReferencePointResult.from_point(ref_point)
                logger.info(
                    "Calculated reference_point for '%s': (%.4f, %.4f, %.4f)",
                    window_id, ref_point.x, ref_point.y, ref_point.z,
                )
            except Exception as exc:
                raise ValueError(f"Failed to calculate reference_point for window '{window_id}': {exc}") from exc

        return results

    def calculate_external_reference_point(self, parameters: Dict[str, Any]) -> Dict[str, ReferencePointResult]:
        """
        Calculate the external reference point (opposite face of the window) for each window.

        Args:
            parameters: Must contain room_polygon and windows dict (x1, y1, z1, x2, y2, z2 per window)

        Returns:
            {window_id: ReferencePointResult}

        Raises:
            ValueError: If parameters are invalid or calculation fails
        """
        self._validate(RequestType.GET_EXTERNAL_REFERENCE_POINT, parameters)

        room_polygon = self._parse_room_polygon(parameters)
        results = {}
        for window_id, window_params in parameters[ParameterName.WINDOWS.value].items():
            try:
                window_geom = WindowGeometry.from_dict(window_params)
                ext_point = window_geom.calculate_external_reference_point_from_polygon(room_polygon)
                results[window_id] = ReferencePointResult.from_point(ext_point)
                logger.info(
                    "Calculated external_reference_point for '%s': (%.4f, %.4f, %.4f)",
                    window_id, ext_point.x, ext_point.y, ext_point.z,
                )
            except Exception as exc:
                raise ValueError(
                    f"Failed to calculate external_reference_point for window '{window_id}': {exc}"
                ) from exc

        return results

    @staticmethod
    def _validate(request_type: RequestType, parameters: Dict[str, Any]) -> None:
        result = ValidatorManager.validate(request_type, parameters)
        if not result.is_valid:
            raise ValueError("; ".join(str(e) for e in result.errors))

    @staticmethod
    def _parse_room_polygon(parameters: Dict[str, Any]) -> RoomPolygon:
        try:
            return RoomPolygon.from_dict(parameters[ParameterName.ROOM_POLYGON.value])
        except Exception as exc:
            raise ValueError(f"Invalid room_polygon: {exc}") from exc
