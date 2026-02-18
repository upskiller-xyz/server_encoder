from typing import Dict, Any, Optional
from src.core import ParameterName
from src.components.geometry import WindowGeometry, RoomPolygon


class ParameterNormalizer:
    """
    Normalizes parameters by converting dicts to proper geometry classes

    Ensures we always work with WindowGeometry and RoomPolygon classes, not dicts
    """

    @staticmethod
    def normalize_window_geometry(window_params: Dict[str, Any]) -> Optional[WindowGeometry]:
        """
        Extract or create WindowGeometry from parameters

        Args:
            window_params: Window parameter dictionary

        Returns:
            WindowGeometry object or None if no geometry found
        """
        # Check if window_geometry exists
        if ParameterName.WINDOW_GEOMETRY.value in window_params:
            geom = window_params[ParameterName.WINDOW_GEOMETRY.value]
            if isinstance(geom, WindowGeometry):
                return geom
            # Convert dict to WindowGeometry
            return WindowGeometry.from_dict(geom)

        # Check if individual coordinates exist
        if all(k in window_params for k in [
            ParameterName.X1.value, ParameterName.Y1.value, ParameterName.Z1.value,
            ParameterName.X2.value, ParameterName.Y2.value, ParameterName.Z2.value
        ]):
            return WindowGeometry(
                x1=window_params[ParameterName.X1.value],
                y1=window_params[ParameterName.Y1.value],
                z1=window_params[ParameterName.Z1.value],
                x2=window_params[ParameterName.X2.value],
                y2=window_params[ParameterName.Y2.value],
                z2=window_params[ParameterName.Z2.value],
                direction_angle=window_params.get(ParameterName.DIRECTION_ANGLE.value, None)
            )

        return None

    @staticmethod
    def normalize_room_polygon(room_params: Dict[str, Any]) -> Optional[RoomPolygon]:
        """
        Extract or create RoomPolygon from parameters

        Args:
            room_params: Room parameter dictionary

        Returns:
            RoomPolygon object or None if no polygon found
        """
        if ParameterName.ROOM_POLYGON.value not in room_params:
            return None

        polygon = room_params[ParameterName.ROOM_POLYGON.value]
        if isinstance(polygon, RoomPolygon):
            return polygon
        # RoomPolygon.from_dict() actually expects a list, not a dict (despite the name)
        if isinstance(polygon, (list, tuple)):
            return RoomPolygon.from_dict(polygon)  # type:ignore
        # If it's a dict, it might be a serialized RoomPolygon
        if isinstance(polygon, dict) and 'vertices' in polygon:
            return RoomPolygon.from_dict(polygon['vertices'])
        return RoomPolygon.from_dict(polygon)  # type:ignore
