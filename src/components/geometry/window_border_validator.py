from typing import List, Tuple, Union
from src.components.geometry.window_geometry import WindowGeometry
from src.components.geometry.room_polygon import RoomPolygon
from src.core import ParameterName
from src.core import GRAPHICS_CONSTANTS
from src.core import WindowNotOnPolygonError


class WindowBorderValidator:
    """
    Validator for checking if window is positioned on room polygon border

    The internal side of the window should lie on one of the polygon's edges.
    Uses Shapely geometry for precise calculations.
    """
    @classmethod
    def validate_window_on_border(
        cls,
        window_geometry: WindowGeometry,
        room_polygon: RoomPolygon
    ) -> Tuple[bool, str]:
        """
        Validate that window internal side lies on a polygon edge

        The window is defined by:
        - Bounding box (x1,y1) to (x2,y2) - gives spatial extent
        - direction_angle (window normal in radians) - gives orientation

        We create a rotated rectangle using two lines perpendicular to direction_angle,
        passing through (x1,y1) and (x2,y2), then check if one edge touches the polygon.

        Args:
            window_geometry: Window geometry with coordinates and direction_angle
            room_polygon: Room polygon vertices

        Returns:
            (is_valid, error_message) tuple
        """


        w_edges = window_geometry.get_candidate_edges()

        window_on_edge = any(
            room_polygon.boundary_contains(w_edge, GRAPHICS_CONSTANTS.WINDOW_PLACEMENT_TOLERANCE)
            for w_edge in w_edges
        )

        if not window_on_edge:
            raise WindowNotOnPolygonError(
            window_bbox=(window_geometry.x1, window_geometry.y1, window_geometry.x2, window_geometry.y2),
            direction_angle=window_geometry.direction_angle,
            tolerance=GRAPHICS_CONSTANTS.WINDOW_PLACEMENT_TOLERANCE,
            polygon_edges=room_polygon.get_edges(),
            window_edges=w_edges
        )

        return True, ""
    

    @classmethod
    def validate_from_dict(
        cls,
        window_data: dict,
        polygon_data: List[Union[dict, List[float]]]
    ) -> Tuple[bool, str]:
        """
        Validate window position from dictionary data

        Args:
            window_data: Window geometry dict with x1, y1, z1, x2, y2, z2
            polygon_data: Room polygon as list of points
            tolerance: Distance tolerance in meters

        Returns:
            (is_valid, error_message) tuple
        """
        try:
            # Parse window geometry
            if ParameterName.WINDOW_GEOMETRY.value in window_data:
                window_data = window_data[ParameterName.WINDOW_GEOMETRY.value]
            window_geom = WindowGeometry.from_dict(window_data)

            # Parse room polygon
            room_poly = RoomPolygon.from_dict(polygon_data)

            return cls.validate_window_on_border(window_geom, room_poly)

        except WindowNotOnPolygonError as e:
            return False, str(e)
        except (KeyError, ValueError, AttributeError) as e:
            return False, f"Error parsing geometry data: {type(e).__name__}: {str(e)}"
