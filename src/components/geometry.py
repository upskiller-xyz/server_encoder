from typing import List, Tuple, Any, Union, Dict, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import math
import numpy as np
import cv2
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint, LineString as ShapelyLine, box as ShapelyBox
from shapely.affinity import rotate as shapely_rotate
from src.components.enums import ImageDimensions, GeometryType, ParameterName
from src.components.graphics_constants import GRAPHICS_CONSTANTS

logger = logging.Logger("logger")

@dataclass
class Point2D:
    """Represents a 2D point in meters"""
    x: float
    y: float

    def to_pixel(self, resolution: float = 0.1) -> Tuple[int, int]:
        """
        Convert point from meters to pixels

        Args:
            resolution: Meters per pixel (default 0.1m = 10cm)

        Returns:
            (x_pixel, y_pixel) tuple
        """
        return (GRAPHICS_CONSTANTS.get_pixel_value(self.x), GRAPHICS_CONSTANTS.get_pixel_value(self.y))


@dataclass
class Point3D:
    """Represents a 3D point in meters"""
    x: float
    y: float
    z: float

    def to_point2d(self) -> Point2D:
        """Convert to 2D point by dropping z coordinate"""
        return Point2D(self.x, self.y)

    def to_pixel(self, resolution: float = 0.1) -> Tuple[int, int]:
        """
        Convert point from meters to pixels (x, y only)

        Args:
            resolution: Meters per pixel (default 0.1m = 10cm)

        Returns:
            (x_pixel, y_pixel) tuple
        """
        return self.to_point2d().to_pixel(resolution)


class IPolygonDataParser(ABC):
    """
    Abstract base class for polygon data parsers (Strategy Pattern)

    Each parser handles a specific input format and converts it to
    a list of (x, y) vertex tuples.
    """

    @abstractmethod
    def can_parse(self, data: Any) -> bool:
        """
        Check if this parser can handle the given data format

        Args:
            data: Input data to check

        Returns:
            True if parser can handle this format
        """
        pass

    @abstractmethod
    def parse(self, data: Any) -> List[Tuple[float, float]]:
        """
        Parse data into list of vertex tuples

        Args:
            data: Input data to parse

        Returns:
            List of (x, y) vertex tuples

        Raises:
            ValueError: If data format is invalid
        """
        pass


class DictPolygonParser(IPolygonDataParser):
    """
    Parser for dictionary-based polygon format: [{"x": 0, "y": 0}, ...]
    """

    def can_parse(self, data: Any) -> bool:
        """Check if data is a list of dictionaries"""
        if not isinstance(data, list) or not data:
            return False
        return isinstance(data[0], dict)

    def parse(self, data: List[dict]) -> List[Tuple[float, float]]:
        """
        Parse dictionary format polygon data

        Args:
            data: List of dicts like [{"x": 0, "y": 0}, ...]

        Returns:
            List of (x, y) vertex tuples

        Raises:
            ValueError: If dict format is invalid
        """
        vertices = []
        for i, point in enumerate(data):
            if not isinstance(point, dict):
                raise ValueError(
                    f"Parameter 'room_polygon' point at index {i} is not a dict. "
                    f"Got type: {type(point).__name__}, value: {point}"
                )

            if "x" not in point or "y" not in point:
                raise ValueError(
                    f"Parameter 'room_polygon' point at index {i} missing 'x' or 'y' key. "
                    f"Got: {point}. Expected format: {{'x': value, 'y': value}}"
                )

            try:
                x = float(point["x"])
                y = float(point["y"])
                vertices.append((x, y))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Parameter 'room_polygon' point at index {i} has invalid coordinate values. "
                    f"Error: {type(e).__name__}: {str(e)}. "
                    f"Point: {point}"
                )

        return vertices


class ListPolygonParser(IPolygonDataParser):
    """
    Parser for list-based polygon format: [[0, 0], [3, 0], ...]
    """

    def can_parse(self, data: Any) -> bool:
        """Check if data is a list of lists/tuples"""
        if not isinstance(data, list) or not data:
            return False
        return isinstance(data[0], (list, tuple))

    def parse(self, data: List[List[float]]) -> List[Tuple[float, float]]:
        """
        Parse list format polygon data

        Args:
            data: List of lists/tuples like [[0, 0], [3, 0], ...]

        Returns:
            List of (x, y) vertex tuples

        Raises:
            ValueError: If list format is invalid
        """
        vertices = []
        for i, point in enumerate(data):
            if not isinstance(point, (list, tuple)):
                raise ValueError(
                    f"Parameter 'room_polygon' point at index {i} is not a list or tuple. "
                    f"Got type: {type(point).__name__}, value: {point}"
                )

            if len(point) < 2:
                raise ValueError(
                    f"Parameter 'room_polygon' point at index {i} must have at least 2 elements. "
                    f"Got: {point}. Expected format: [x, y]"
                )

            try:
                x = float(point[0])
                y = float(point[1])
                vertices.append((x, y))
            except (TypeError, ValueError, IndexError) as e:
                raise ValueError(
                    f"Parameter 'room_polygon' point at index {i} has invalid coordinate values. "
                    f"Error: {type(e).__name__}: {str(e)}. "
                    f"Point: {point}"
                )

        return vertices


class PolygonParserFactory:
    """
    Factory for creating appropriate polygon parsers (Factory Pattern)

    Uses Strategy Pattern to select the right parser based on data format.
    """

    # Available parsers in priority order
    _PARSERS = [
        DictPolygonParser(),
        ListPolygonParser(),
    ]

    @classmethod
    def get_parser(cls, data: Any) -> IPolygonDataParser:
        """
        Get appropriate parser for the given data format

        Args:
            data: Input data to parse

        Returns:
            Parser instance that can handle this data

        Raises:
            ValueError: If no parser can handle the data format
        """
        # Strategy Pattern: Try each parser until one matches
        for parser in cls._PARSERS:
            if parser.can_parse(data):
                return parser

        # No parser found - provide helpful error
        raise ValueError(
            f"Parameter 'room_polygon' has unsupported format. "
            f"Got type: {type(data).__name__}, value: {data}. "
            f"Expected formats: [{'x': val, 'y': val}, ...] or [[x, y], ...]"
        )


class GeometryAdapter:
    """
    Adapter for converting Shapely geometry types to coordinate arrays (Adapter Pattern)

    Handles different geometry types that result from clipping operations and converts
    them to numpy arrays suitable for cv2.fillPoly.
    """

    @staticmethod
    def _extract_polygon_coords(geometry: Any) -> List[Tuple[float, float]]:
        """Extract coordinates from a Polygon geometry"""
        return list(geometry.exterior.coords)[:-1]  # Remove duplicate last point

    @staticmethod
    def _extract_multi_polygon_coords(geometry: Any) -> List[Tuple[float, float]]:
        """Extract coordinates from MultiPolygon by taking the largest polygon"""
        largest = max(geometry.geoms, key=lambda p: p.area)
        return list(largest.exterior.coords)[:-1]

    @staticmethod
    def _extract_geometry_collection_coords(geometry: Any) -> List[Tuple[float, float]]:
        """Extract coordinates from GeometryCollection by finding polygons"""
        polygons = [g for g in geometry.geoms if g.geom_type == GeometryType.POLYGON.value]
        if not polygons:
            return []
        largest = max(polygons, key=lambda p: p.area)
        return list(largest.exterior.coords)[:-1]

    # Strategy map: GeometryType -> extraction function (Strategy Pattern)
    GEOMETRY_HANDLERS: Dict[GeometryType, Callable] = {
        GeometryType.POLYGON: _extract_polygon_coords.__func__,
        GeometryType.MULTI_POLYGON: _extract_multi_polygon_coords.__func__,
        GeometryType.GEOMETRY_COLLECTION: _extract_geometry_collection_coords.__func__,
    }

    @classmethod
    def vertical_mirror(cls, poly:ShapelyPolygon)->ShapelyPolygon:
        crds = np.array([x for x in poly.exterior.coords])
        return ShapelyPolygon(crds.dot([[1,0],[0,-1]]))

    @classmethod
    def extract_coordinates(
        cls,
        geometry: Any,
        fallback_coords: List[List[int]] = []
    ) -> np.ndarray:
        """
        Extract coordinates from a Shapely geometry object

        Args:
            geometry: Shapely geometry object (result of clipping)
            fallback_coords: Fallback coordinates if extraction fails

        Returns:
            Numpy array of shape (1, N, 2) for cv2.fillPoly
        """
        # Handle empty geometry
        if geometry.is_empty:
            return np.array([[[0, 0]]], dtype=np.int32)

        # Get geometry type
        geom_type_str = geometry.geom_type

        # Try to find handler in map

        for geom_type, handler in cls.GEOMETRY_HANDLERS.items():
            if geom_type_str == geom_type.value:
                coords = handler(geometry)
                if coords:
                    return np.array([coords], dtype=np.int32)
                # If extraction returned empty, fall through to fallback
                break

        # Fallback: use provided fallback coordinates or empty polygon
        if fallback_coords:
            logger.info("[GEOMETRY ADAPTER]: Using fallback coordinates")
            return np.array([fallback_coords], dtype=np.int32)
        return np.array([[[0, 0]]], dtype=np.int32)


class RoomPolygon:
    """
    Represents a room's floor plan as a polygon

    Coordinate system:
    - Origin is at window center on the outer façade plane
    - X-axis points right (parallel to façade)
    - Y-axis points into the room (perpendicular to façade)
    - Window is on the right side of the image
    """

    def __init__(self, vertices: List[Tuple[float, ...]]):
        """
        Initialize room polygon

        Args:
            vertices: List of (x, y) coordinates in meters
        """
        if len(vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")

        self._vertices = [Point2D(x, y) for x, y in vertices]

    @property
    def vertices(self) -> List[Point2D]:
        """Get polygon vertices"""
        return self._vertices

    def rotate(self, angle_degrees: float, center: Point2D | None = None) -> 'RoomPolygon':
        """
        Rotate polygon around a center point using Shapely

        Args:
            angle_degrees: Rotation angle in degrees (positive = counter-clockwise)
            center: Center of rotation (default: origin (0,0))

        Returns:
            New RoomPolygon with rotated vertices
        """
        if center is None:
            center = Point2D(0, 0)

        # Create Shapely polygon from vertices
        
        shapely_poly = ShapelyPolygon(self.get_coords())
        
        rotated_poly = shapely_rotate(
            shapely_poly,
            angle_degrees,
            origin=(center.x, center.y)
        )

        # Extract rotated vertices
        rotated_vertices = list(rotated_poly.exterior.coords)[:-1]  # Remove duplicate last point
        return RoomPolygon(rotated_vertices)


    def get_edges(self):
        polygon_coords = self.get_coords()
        
        return [ShapelyLine([
            polygon_coords[i], 
            polygon_coords[(i + 1) % len(polygon_coords)]]) for i in range(len(polygon_coords))]
    
    def get_coords(self):
        return [(v.x, v.y) for v in self._vertices]

    def _build_edge(self, ind:int):
        v1 = self._vertices[ind]
        v2 = self._vertices[(ind + 1) % len(self._vertices)]

        return ShapelyLine([(v1.x, v1.y), (v2.x, v2.y)])

    def _window_edge_and_rotation(
        self,
        window_line: ShapelyLine,
        tolerance: float = 0.01
    ) -> Tuple[ShapelyLine, int, float]:
        """
        Find which polygon edge contains the window and calculate rotation needed.

        Args:
            window_x1, window_y1: First window endpoint
            window_x2, window_y2: Second window endpoint
            tolerance: Distance tolerance for edge matching

        Returns:
            Tuple of (edge_index, rotation_angle_degrees, needs_flip)
            - edge_index: Index of the edge containing the window
            - rotation_angle_degrees: Angle to rotate so window edge is horizontal (constant y)
            - needs_flip: Whether to flip direction so room extends in -y direction
        """
        

        edges = [self._build_edge(i) for i in range(len(self._vertices))]
        edges = [(i,edge) for i,edge in enumerate(edges) if edge.buffer(tolerance).contains(window_line)]

        if len(edges)<1:
            raise ValueError(
            f"Window at ({window_line.coords[0][0]:.2f}, {window_line.coords[0][1]:.2f}) to ({window_line.coords[1][0]:.2f}, {window_line.coords[1][1]:.2f}) "
            f"does not lie on any polygon edge")
        
        ind, edge = edges[0]
        v1 = self._vertices[ind]
        v2 = self._vertices[(ind + 1) % len(self._vertices)]
        edge_angle = math.atan2(v2.y - v1.y, v2.x - v1.x) * 180 / math.pi

        return edge, ind, edge_angle
        
    def to_pixel_array(
        self,
        window_x1: float | None,
        window_y1: float | None,
        window_x2: float | None,
        window_y2: float | None,
        image_size: int = 128,
        direction_angle: float | None = None
    ) -> np.ndarray:
        """
        Convert polygon to pixel coordinates for drawing on image

        The room polygon is positioned so that:
        - The edge containing the window is rotated to be horizontal (constant y)
        - The window edge aligns with the left edge of the window area on the image
        - The room extends to the left (negative x in image coordinates)
        - Window outer wall (left edge) is at: image_size - 12 - wall_thickness pixels from left
        - Resolution: 1 pixel = 0.1m (10cm) for 128x128 image, scales proportionally

        Args:
            image_size: Image dimension in pixels (default 128)
            window_x1: Window left x coordinate in meters (required)
            window_y1: Window front y coordinate in meters (required)
            window_x2: Window right x coordinate in meters (required)
            window_y2: Window back y coordinate in meters (required)
            direction_angle: Window direction angle in radians (optional, if not provided will be calculated from polygon edge)

        Returns:
            Numpy array of shape (N, 1, 2) for cv2.fillPoly
        """
        if window_x1 is None or window_y1 is None or window_x2 is None or window_y2 is None:
            raise ValueError("Window coordinates required for room positioning")
            
        rotated_polygon = self.get_coords()
        
        # Create WindowGeometry WITH direction_angle so get_candidate_edges()
        # returns rotated edges for diagonal walls (matches merger's logic)
        if direction_angle is not None:
            window = WindowGeometry(
                window_x1, window_y1, 0,
                window_x2, window_y2, 0,
                direction_angle=direction_angle
            )
        else:
            window = WindowGeometry.from_corners(window_x1, window_y1, 0, window_x2, window_y2, 0)
        w_edges = window.get_candidate_edges()

        # Create polygon from rotated coordinates to check which edge is on it
        rotated_room_poly = ShapelyPolygon(rotated_polygon)
        tolerance = GRAPHICS_CONSTANTS.WINDOW_PLACEMENT_TOLERANCE

        # Check which edge is on the polygon boundary
        edge_on_boundary = None

        res = [w_edge for w_edge in w_edges if rotated_room_poly.boundary.buffer(tolerance).contains(w_edge)]

        window_center_rotated = Point2D((window_x1 + window_x2) / 2, (window_y1 + window_y2) / 2)
        if len(res) > 0:
            edge_on_boundary = res[0]
            edge_coords = list(edge_on_boundary.coords)
            window_center_rotated = Point2D(
                (edge_coords[0][0] + edge_coords[1][0]) / 2,
                (edge_coords[0][1] + edge_coords[1][1]) / 2
            )
        else:
            # Fallback: project window center onto polygon boundary (matches merger's fallback)
            center = ShapelyPoint((window_x1 + window_x2) / 2, (window_y1 + window_y2) / 2)
            poly_boundary = ShapelyLine(list(rotated_polygon) + [rotated_polygon[0]])
            projected = poly_boundary.interpolate(poly_boundary.project(center))
            window_center_rotated = Point2D(projected.x, projected.y)

        rotated_polygon = RoomPolygon(rotated_polygon)


        wall_thickness_px = GRAPHICS_CONSTANTS.get_pixel_value(GRAPHICS_CONSTANTS.WALL_THICKNESS_M, image_size)
        if direction_angle is not None:
            wall_thickness_px = window.wall_thickness_px
                
        
        # Window's left edge position on image
        window_left_edge_x = image_size - GRAPHICS_CONSTANTS.WINDOW_OFFSET_PX - wall_thickness_px

        # Room should align 1 pixel to the left of window for perfect adjacency (C-frame)
        room_facade_x = window_left_edge_x - GRAPHICS_CONSTANTS.ROOM_FACADE_OFFSET_PX
        window_y_pixels = image_size // 2


        # First pass: calculate room extent to check for obstruction bar overlap
        dims = ImageDimensions(image_size)
        obs_bar_x_start, _, _, _ = dims.get_obstruction_bar_position()

        offsets = [GeometryOps.offset_coords(vertex, window_center_rotated) for vertex in rotated_polygon.vertices]

        offsets = [[GRAPHICS_CONSTANTS.get_pixel_value(i, image_size) for i in dd] for dd in offsets]

        # Flip y-axis: image coordinates have y increasing downward, geometric coordinates have y increasing upward
        pixel_coords = [[room_facade_x + dx,
                         window_y_pixels - dy] for [dx, dy] in offsets]
        
        # Clip room to avoid overlap with obstruction bar
        right_boundary = obs_bar_x_start - GRAPHICS_CONSTANTS.OBSTRUCTION_BAR_GAP_PX 
        # Create clipping rectangle: x from 0 to right_boundary, y from 0 to image_size
        room_poly = ShapelyPolygon(pixel_coords)
        clip_box = ShapelyBox(0, 0, right_boundary, image_size)

        # Clip the polygon
        clipped = room_poly.intersection(clip_box)
        
        # clipped = room_poly
        # Use GeometryAdapter to handle different geometry types (Adapter Pattern)
        return GeometryAdapter.extract_coordinates(clipped, fallback_coords=pixel_coords)

    @classmethod
    def from_dict(cls, data: List) -> 'RoomPolygon':
        """
        Create polygon from list of coordinate dictionaries or lists

        Args:
            data: List of dicts like [{"x": 0, "y": 0}, {"x": 3, "y": 0}, ...]
                  OR list of lists/tuples like [[0, 0], [3, 0], ...] or [(0, 0), (3, 0), ...]

        Returns:
            RoomPolygon instance

        Raises:
            ValueError: If data format is invalid
        """
        if not data:
            raise ValueError("Polygon data cannot be empty")

        # Check format of first element to determine data structure
        first_element = data[0]

        if isinstance(first_element, dict):
            # List of dictionaries format: [{"x": 0, "y": 0}, ...]
            vertices = [(point["x"], point["y"]) for point in data]
        elif isinstance(first_element, (list, tuple)):
            # List of lists/tuples format: [[0, 0], ...] or [(0, 0), ...]
            vertices = [(point[0], point[1]) for point in data]
        else:
            raise ValueError(
                f"Invalid polygon data format. Expected list of dicts or list of lists/tuples, "
                f"but got list of {type(first_element).__name__}"
            )

        return cls(vertices)

class GeometryOps:

    @classmethod 
    def project(cls, vv:Point2D | Point3D, sin_a:float, cos_a:float):
        return vv.x * cos_a + vv.y * sin_a
    
    @classmethod
    def offset_coords(cls, vv:Point2D|Point3D, vv1:Point2D|Point3D):
        dx = vv.x - vv1.x  # Along façade
        dy = vv.y - vv1.y  # Perpendicular to façade
        return [dx, dy]
    
    @classmethod
    def projection_dist(cls, vv:Point2D|Point3D, vv1:Point2D|Point3D, angle):
        # Unit vector perpendicular to direction_angle
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Project both points onto the perpendicular direction
        # Point 1 projection
        proj1 = GeometryOps.project(vv, sin_a, cos_a)
        proj2 = GeometryOps.project(vv1, sin_a, cos_a)
        
        # Window width is the distance between projections
        return abs(proj2 - proj1)
    
    @classmethod
    def rotate_vertex(cls, vv:Point2D|Point3D, sin_a:float, cos_a:float):
        rot_x = cls.rotate_coord(vv, sin_a, cos_a) 
        rot_y = cls.rotate_coord(vv, sin_a, cos_a, False)
        return (rot_x, rot_y)
    
    @classmethod
    def rotate_coord(cls, vv:Point2D|Point3D, sin_a:float, cos_a:float, x_axis=True):
        if x_axis:
            return vv.x * cos_a - vv.y * sin_a 
        return vv.x * sin_a + vv.y * cos_a
    
    @classmethod
    def perpendicular_dir_inside_polygon(cls, room_poly, edge_coords, perp)->bool:
        test_offset = 0.1
        edge_center_x = (edge_coords[0][0] + edge_coords[1][0]) / 2
        edge_center_y = (edge_coords[0][1] + edge_coords[1][1]) / 2
        test_x1 = edge_center_x + test_offset * math.cos(perp)
        test_y1 = edge_center_y + test_offset * math.sin(perp)
        test_point1 = ShapelyPoint(test_x1, test_y1)
        return room_poly.contains(test_point1)
    
    @classmethod
    def normalize_angle(cls, angle):
        # Normalize to [0, 2π)
        while angle < 0:
            angle += 2 * math.pi
        while angle >= 2 * math.pi:
            angle -= 2 * math.pi
        return angle

class WindowGeometry:
    """
    Represents window geometry from bounding box coordinates

    Window is viewed from top (plan view):
    - Appears as a vertical line/rectangle
    - Width (horizontal) = outer wall thickness (appears as horizontal width on image)
    - Height (vertical on image) = window width in 3D (x2 - x1)
    - Located 12 pixels from right edge (8 pixels from obstruction bar)

    Coordinate system:
    - X-axis: along façade (horizontal)
    - Y-axis: perpendicular to façade (into room, not used in top view)
    - Z-axis: vertical (height)
    """

    def __init__(
        self,
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
        direction_angle: float = 0
    ):
        """
        Initialize window geometry from bounding box

        Args:
            x1, y1, z1: Left-bottom corner coordinates in meters
            x2, y2, z2: Right-top corner coordinates in meters
            direction_angle: Window direction angle in radians (optional)
        """
        self._corner1 = Point3D(x1, y1, z1)
        self._corner2 = Point3D(x2, y2, z2)
        self._direction_angle = direction_angle

        # Ensure corner1 is bottom-left and corner2 is top-right
        self._x_min = min(x1, x2)
        self._x_max = max(x1, x2)
        self._z_min = min(z1, z2)
        self._z_max = max(z1, z2)
        self._y_min = min(y1,y2)
        self._y_max = max(y1, y2)

    @property
    def window_width_3d(self) -> float:
        """
        Get window width in 3D space (perpendicular to direction_angle)

        If direction_angle is provided, calculates the perpendicular distance
        between the two lines perpendicular to direction_angle passing through
        (x1,y1) and (x2,y2).

        Otherwise falls back to max of x-span and y-span.
        """
        if self._direction_angle is None:
            return max(self._x_max - self._x_min, self._y_max - self._y_min)
        edge_angle = self._direction_angle + math.pi / 2
        return GeometryOps.projection_dist(self._corner1, self._corner2, edge_angle)
            
    @property
    def wall_thickness(self) -> float:
        """
        Get wall thickness from window bounding box (distance in direction_angle direction)

        This is the distance between the two parallel edges of the window rectangle
        that are perpendicular to direction_angle. It represents the wall thickness
        where the window is installed.

        If direction_angle is not provided, falls back to min of bounding box dimensions.
        """
        if self._direction_angle is None:
            return min(self._y_max - self._y_min, self._x_max - self._x_min)
        
        return GeometryOps.projection_dist(self._corner1, self._corner2, self._direction_angle)
        
    @property
    def window_height_3d(self) -> float:
        """Get window height in 3D space (vertical, z-direction)"""
        return self._z_max - self._z_min

    @property
    def sill_height(self) -> float:
        """Get sill height (minimum z coordinate)"""
        return self._z_min

    @property
    def top_height(self) -> float:
        """Get top height (maximum z coordinate)"""
        return self._z_max

    @property
    def x1(self) -> float:
        """Get x1 coordinate"""
        return self._corner1.x

    @property
    def y1(self) -> float:
        """Get y1 coordinate"""
        return self._corner1.y

    @property
    def x2(self) -> float:
        """Get x2 coordinate"""
        return self._corner2.x

    @property
    def y2(self) -> float:
        """Get y2 coordinate"""
        return self._corner2.y

    @property
    def z1(self) -> float:
        """Get z1 coordinate"""
        return self._corner1.z

    @property
    def z2(self) -> float:
        """Get z2 coordinate"""
        return self._corner2.z
    
    @property
    def niche_center(self)->Point2D:
        return Point2D((self.x1 + self.x2) *0.5, 
                       (self.y1 + self.y2) *0.5)

    @property
    def direction_angle(self) -> float:
        """Get window direction angle in radians (None if not set)"""
        return self._direction_angle

    def rotate(self, angle_degrees: float, center: Point2D | None = None) -> 'WindowGeometry':
        """
        Rotate window geometry around a center point using Shapely

        Args:
            angle_degrees: Rotation angle in degrees (positive = counter-clockwise)
            center: Center of rotation in 2D (default: origin (0,0))

        Returns:
            New WindowGeometry with rotated coordinates
        """
        if center is None:
            center = Point2D(0, 0)

        # Create line segment from the two corners
        line = ShapelyLine([(self._corner1.x, self._corner1.y), (self._corner2.x, self._corner2.y)])
        rotated_line = shapely_rotate(line, angle_degrees, origin=(center.x, center.y))

        # Extract rotated coordinates
        coords = list(rotated_line.coords)
        new_x1, new_y1 = coords[0]
        new_x2, new_y2 = coords[1]

        return WindowGeometry(
            new_x1, new_y1, self._corner1.z,
            new_x2, new_y2, self._corner2.z
        )

    def get_pixel_bounds(
        self,
        image_size: int = 128,
        window_offset_px: int | None = None
    ) -> Tuple[int, int, int, int]:
        """
        Get window bounds in pixel coordinates for top view

        In top view:
        - Window appears as vertical line at fixed x position (default 12px from right)
        - Horizontal extent = wall thickness (approximately constant)
        - Vertical extent = window width in 3D converted to pixels

        Args:
            image_size: Image dimension in pixels (default 128)
            window_offset_px: Distance from right edge in pixels (uses GRAPHICS_CONSTANTS if None)

        Returns:
            (x_start, y_start, x_end, y_end) tuple in pixels
        """
        if window_offset_px is None:
            window_offset_px = GRAPHICS_CONSTANTS.WINDOW_OFFSET_PX

        window_x_end = image_size - window_offset_px
        window_x_start = window_x_end - self.wall_thickness_px
        
        window_height_px = GRAPHICS_CONSTANTS.get_pixel_value(self.window_width_3d)  

        # Center vertically
        
        window_y_start = image_size // 2 - window_height_px // 2
        window_y_end = window_y_start + window_height_px

        # Clamp to image bounds
        x_start = max(0, window_x_start)
        x_end = min(image_size, window_x_end)
        y_start = max(0, window_y_start)
        y_end = min(image_size, window_y_end)

        return (x_start, y_start, x_end, y_end)
    
    @property
    def wall_thickness_px(self):
        wall_thickness_m = self.wall_thickness
        # Fallback to default if calculated thickness is 0 (e.g., when y1==y2)
        if wall_thickness_m == 0:
            wall_thickness_m = GRAPHICS_CONSTANTS.WALL_THICKNESS_M
        return GRAPHICS_CONSTANTS.get_pixel_value(wall_thickness_m) 

    @classmethod
    def from_corners(
        cls,
        x1: float, y1: float, z1: float,
        x2: float, y2: float, z2: float
    ) -> 'WindowGeometry':
        """
        Create window geometry from corner coordinates

        Args:
            x1, y1, z1: First corner in meters
            x2, y2, z2: Second corner in meters

        Returns:
            WindowGeometry instance
        """
        return cls(x1, y1, z1, x2, y2, z2)

    @classmethod
    def from_dict(cls, data: dict) -> 'WindowGeometry':
        """
        Create window geometry from dictionary

        Args:
            data: Dict with keys x1, y1, z1, x2, y2, z2, and optionally direction_angle

        Returns:
            WindowGeometry instance
        """
        return cls(
            x1=data[ParameterName.X1.value],
            y1=data[ParameterName.Y1.value],
            z1=data[ParameterName.Z1.value],
            x2=data[ParameterName.X2.value],
            y2=data[ParameterName.Y2.value],
            z2=data[ParameterName.Z2.value],
            direction_angle=data.get(ParameterName.DIRECTION_ANGLE.value, 0)
        )

    def get_candidate_edges(self):
        # When direction_angle is available, compute the actual rotated rectangle
        # edges instead of axis-aligned bbox edges. This handles diagonal walls.
        if self._direction_angle is not None and self._direction_angle != 0:
            return self._compute_rotated_rectangle_edges()

        # Axis-aligned fallback when no direction_angle
        window_edge1 = ShapelyLine([(self.x1, self.y1), (self.x2, self.y1)])
        window_edge2 = ShapelyLine([(self.x1, self.y2), (self.x2, self.y2)])
        window_edge3 = ShapelyLine([(self.x1, self.y1), (self.x1, self.y2)])
        window_edge4 = ShapelyLine([(self.x2, self.y1), (self.x2, self.y2)])
        return [window_edge1, window_edge2, window_edge3, window_edge4]

    def _compute_rotated_rectangle_edges(self):
        """Compute edges of the rotated window rectangle using direction_angle."""
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2

        # Wall direction (perpendicular to window normal)
        wall_angle = self._direction_angle + math.pi / 2
        wall_dx = math.cos(wall_angle)
        wall_dy = math.sin(wall_angle)

        # Normal direction
        norm_dx = math.cos(self._direction_angle)
        norm_dy = math.sin(self._direction_angle)

        half_width = self.window_width_3d / 2
        half_thick = self.wall_thickness / 2

        # 4 corners of rotated rectangle
        c1 = (cx - half_width * wall_dx - half_thick * norm_dx,
              cy - half_width * wall_dy - half_thick * norm_dy)
        c2 = (cx + half_width * wall_dx - half_thick * norm_dx,
              cy + half_width * wall_dy - half_thick * norm_dy)
        c3 = (cx + half_width * wall_dx + half_thick * norm_dx,
              cy + half_width * wall_dy + half_thick * norm_dy)
        c4 = (cx - half_width * wall_dx + half_thick * norm_dx,
              cy - half_width * wall_dy + half_thick * norm_dy)

        # 4 edges: interior, exterior, and two sides
        return [
            ShapelyLine([c1, c2]),  # interior (away from normal)
            ShapelyLine([c3, c4]),  # exterior (toward normal)
            ShapelyLine([c1, c4]),  # side
            ShapelyLine([c2, c3]),  # side
        ]

    def get_room_edge(self, room_polygon:RoomPolygon, tolerance=GRAPHICS_CONSTANTS.WINDOW_PLACEMENT_TOLERANCE)->list:
        w_edges = self.get_candidate_edges()
        # Find which polygon edge contains one of the window edges

        poly_edges = room_polygon.get_edges()
        res = [(edge, i, j) for i,edge in enumerate(poly_edges) for j, w_edge in enumerate(w_edges) if edge.buffer(tolerance).contains(w_edge)]
        return res

    def _project_to_polygon_edge(
        self,
        room_polygon: 'RoomPolygon',
        tolerance: float = 0.01
    ) -> tuple:
        """
        Project window center onto the polygon boundary and find the containing edge.

        This is a shared helper for calculate_reference_point_from_polygon and
        calculate_direction_from_polygon.

        Args:
            room_polygon: The room polygon containing this window
            tolerance: Distance tolerance for edge matching (meters)

        Returns:
            Tuple of (projected_point: ShapelyPoint, edge_index: int, edge: ShapelyLine)

        Raises:
            ValueError: If projected point is not on any polygon edge
        """
        # Calculate center of the window bounding box
        window_center = ShapelyPoint(self.niche_center.x, self.niche_center.y)

        # Create polygon boundary as closed line
        poly_boundary = ShapelyLine(room_polygon.get_coords() + [room_polygon.get_coords()[0]])

        # Project window center onto the closest point on the boundary
        dist_along = poly_boundary.project(window_center)
        projected_point = poly_boundary.interpolate(dist_along)

        # Find which edge contains the projected point
        poly_edges = room_polygon.get_edges()
        for i, edge in enumerate(poly_edges):
            if edge.buffer(tolerance).contains(projected_point):
                return (projected_point, i, edge)

        # No edge found - raise error
        raise ValueError(
            f"Projected point ({projected_point.x:.2f}, {projected_point.y:.2f}) "
            f"does not lie on any polygon edge (tolerance: {tolerance}m)"
        )

    def calculate_reference_point_from_polygon(
        self,
        room_polygon: 'RoomPolygon',
        tolerance: float = 0.01
    ) -> Point3D:
        """
        Calculate window reference point from room polygon edge

        The reference point is the center of the window bounding box projected onto
        the room boundary. This ensures accurate positioning even for windows on
        diagonal or curved walls.

        Args:
            room_polygon: The room polygon containing this window
            tolerance: Distance tolerance for edge matching (meters)

        Returns:
            Point3D with (x, y, z) coordinates where z is the vertical center of the window
        """
        projected_point, _, _ = self._project_to_polygon_edge(room_polygon, tolerance)

        # Calculate Z center
        ref_z = (self.z1 + self.z2) * 0.5

        return Point3D(projected_point.x, projected_point.y, ref_z)

    def calculate_direction_from_polygon(
        self,
        room_polygon: 'RoomPolygon',
        tolerance: float = 0.01
    ) -> float:
        """
        EXPERIMENTAL: Calculate direction_angle from room polygon edge

        Finds which polygon edge contains the window and calculates the direction
        the window is facing (perpendicular to the edge, pointing away from room).

        Args:
            room_polygon: The room polygon containing this window
            tolerance: Distance tolerance for edge matching (meters)

        Returns:
            direction_angle in radians (0 = pointing right/east, π/2 = pointing up/north)
        """
        projected_point, edge_index, edge = self._project_to_polygon_edge(room_polygon, tolerance)

        polygon_coords = room_polygon.get_coords()
        v1 = polygon_coords[edge_index]
        v2 = polygon_coords[(edge_index + 1) % len(polygon_coords)]

        # Edge angle (direction along the edge)
        edge_angle = math.atan2(v2[1] - v1[1], v2[0] - v1[0])

        # Two possible perpendiculars
        perps = [edge_angle + math.pi / 2, edge_angle - math.pi / 2]

        # Get center point of the edge for inside/outside check
        edge_coords = list(edge.coords)
        room_poly = ShapelyPolygon(polygon_coords)

        # Select the perpendicular pointing OUTSIDE the room (window facing direction)
        calculated_angle = perps[0]
        for perp in perps:
            if not GeometryOps.perpendicular_dir_inside_polygon(room_poly, edge_coords, perp):
                calculated_angle = perp
                break

        return GeometryOps.normalize_angle(calculated_angle)
    

class WindowBorderValidator:
    """
    Validator for checking if window is positioned on room polygon border

    The internal side of the window should lie on one of the polygon's edges.
    Uses Shapely geometry for precise calculations.
    """

    def __init__(self, tolerance: float = 0.01):
        """
        Initialize validator

        Args:
            tolerance: Distance tolerance in meters (default 1cm)
        """
        self._tolerance = tolerance

    def validate_window_on_border(
        self,
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
        
        
        poly_edges = room_polygon.get_edges()
        
        w_edges = window_geometry.get_candidate_edges()   

        # Check if either edge1 or edge2 lies on a polygon edge
        window_on_edge = False
        
        for poly_edge in poly_edges:
            for w_edge in w_edges:
                # Check edge1
                if poly_edge.buffer(self._tolerance).contains(w_edge):
                    window_on_edge = True
                    break
            if window_on_edge==True:
                break
                
        if not window_on_edge:
            return False, (
                f"Window edge not on polygon border. "
                f"Window bounding box from ({window_geometry.x1:.3f}, {window_geometry.y1:.3f}) to "
                f"({window_geometry.x2:.3f}, {window_geometry.y2:.3f}) "
                f"with direction_angle={window_geometry.direction_angle:.3f} rad "
                f"({window_geometry.direction_angle * 180 / math.pi:.1f}°). "
                f"Expected edges: ({window_geometry.x1:.1f}, {window_geometry.y1:.1f})->({window_geometry.x2:.1f}, {window_geometry.y1:.1f}) "
                f"or ({window_geometry.x1:.1f}, {window_geometry.y2:.1f})->({window_geometry.x2:.1f}, {window_geometry.y2:.1f}) "
                f"to lie on polygon edge (tolerance: {self._tolerance}m)."
            )

        return True, ""

    @classmethod
    def validate_from_dict(
        cls,
        window_data: dict,
        polygon_data: List[Union[dict, List[float]]],
        tolerance: float = 0.01
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
            
            return cls(tolerance=tolerance).validate_window_on_border(window_geom, room_poly)

        except (KeyError, ValueError, AttributeError) as e:
            return False, f"Error parsing geometry data: {type(e).__name__}: {str(e)}"


class WindowHeightValidator:
    """Validates that window z-coordinates lie between room floor and roof."""

    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize validator.

        Args:
            tolerance: Numerical tolerance for height comparisons
        """
        self._tolerance = tolerance

    def validate_window_height_bounds(
        self,
        window_geometry: WindowGeometry,
        floor_height: float,
        roof_height: float
    ) -> Tuple[bool, str]:
        """
        Validate that window z-coordinates are within floor-roof bounds.

        Args:
            window_geometry: Window geometry with z1, z2 coordinates
            floor_height: Floor height (floor_height_above_terrain)
            roof_height: Roof height (floor_height_above_terrain + height_roof_over_floor)

        Returns:
            Tuple of (is_valid, error_message)
            - (True, "") if window is within bounds
            - (False, error_message) if window extends beyond floor or roof
        """
        try:
            z1 = window_geometry.z1
            z2 = window_geometry.z2

            window_bottom = min(z1, z2)
            window_top = max(z1, z2)

            # Check if window bottom is below floor
            if window_bottom < floor_height - self._tolerance:
                return False, (
                    f"Window bottom ({window_bottom:.2f}m) is below floor "
                    f"({floor_height:.2f}m). Window z-coordinates must be between "
                    f"floor and roof heights."
                )

            # Check if window top is above roof
            if window_top > roof_height + self._tolerance:
                return False, (
                    f"Window top ({window_top:.2f}m) is above roof "
                    f"({roof_height:.2f}m). Window z-coordinates must be between "
                    f"floor and roof heights."
                )

            return True, ""

        except (KeyError, ValueError, AttributeError) as e:
            return False, f"Error validating window height: {type(e).__name__}: {str(e)}"

    @classmethod
    def validate_from_parameters(
        cls,
        window_geometry_data: Dict[str, Any],
        floor_height: float,
        roof_height: float,
        tolerance: float = 1e-6
    ) -> Tuple[bool, str]:
        """
        Validate window height from parameter dictionaries.

        Args:
            window_geometry_data: Dict with x1, y1, z1, x2, y2, z2
            floor_height: Floor height above terrain
            roof_height: Roof height (floor + height_roof_over_floor)
            tolerance: Numerical tolerance for comparisons

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Parse window geometry
            window_geom = WindowGeometry.from_dict(window_geometry_data)

            # Validate
            validator = cls(tolerance=tolerance)
            return validator.validate_window_height_bounds(window_geom, floor_height, roof_height)

        except (KeyError, ValueError, AttributeError, TypeError) as e:
            return False, f"Error parsing height data: {type(e).__name__}: {str(e)}"
