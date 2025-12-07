from typing import List, Tuple, Any, Union, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import cv2
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint, LineString
from shapely.affinity import rotate as shapely_rotate


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
        return (int(self.x / resolution), int(self.y / resolution))


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


class RoomPolygon:
    """
    Represents a room's floor plan as a polygon

    Coordinate system:
    - Origin is at window center on the outer façade plane
    - X-axis points right (parallel to façade)
    - Y-axis points into the room (perpendicular to façade)
    - Window is on the right side of the image
    """

    def __init__(self, vertices: List[Tuple[float, float]]):
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

    def rotate(self, angle_degrees: float, center: Point2D = None) -> 'RoomPolygon':
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
        coords = [(v.x, v.y) for v in self._vertices]
        shapely_poly = ShapelyPolygon(coords)

        # Rotate using Shapely (angle in degrees, counter-clockwise, around origin by default)
        rotated_poly = shapely_rotate(
            shapely_poly,
            angle_degrees,
            origin=(center.x, center.y)
        )

        # Extract rotated vertices
        rotated_vertices = list(rotated_poly.exterior.coords)[:-1]  # Remove duplicate last point
        return RoomPolygon(rotated_vertices)

    def to_pixel_array(
        self,
        image_size: int = 128,
        window_x1: float = None,
        window_y1: float = None,
        window_x2: float = None,
        window_y2: float = None
    ) -> np.ndarray:
        """
        Convert polygon to pixel coordinates for drawing on image

        The room polygon is positioned so that:
        - Its rightmost side (where window is located) aligns with the left edge of window area
        - Window outer wall (left edge) is at: image_size - 12 - wall_thickness pixels from left
        - Resolution: 1 pixel = 0.1m (10cm) for 128x128 image, scales proportionally

        Args:
            image_size: Image dimension in pixels (default 128)
            window_x1: Window left x coordinate in meters (required)
            window_y1: Window front y coordinate in meters (required)
            window_x2: Window right x coordinate in meters (required)
            window_y2: Window back y coordinate in meters (required)

        Returns:
            Numpy array of shape (N, 1, 2) for cv2.fillPoly
        """
        if window_x1 is None or window_y1 is None or window_x2 is None or window_y2 is None:
            raise ValueError("Window coordinates required for room positioning")

        # Calculate resolution based on image size (scales proportionally)
        scale = image_size / 128.0
        resolution = 0.1 / scale  # Meters per pixel

        # Window center in 3D coordinates
        window_center_x = (window_x1 + window_x2) / 2.0
        window_center_y = (window_y1 + window_y2) / 2.0

        # Window position calculation:
        # - Window is 12px from right edge
        # - Wall thickness is ~0.3m = 3 pixels at base scale
        # - Room façade (y=0) aligns with window's left edge
        window_offset_px = 12
        wall_thickness_m = 0.3
        wall_thickness_px = round(wall_thickness_m / resolution)

        # Window's left edge position
        window_left_edge_x = image_size - window_offset_px - wall_thickness_px

        # Room should align 1 pixel to the left of window for perfect adjacency (C-frame)
        room_facade_x = window_left_edge_x - 1
        window_y_pixels = image_size // 2

        # First pass: calculate room extent to check for obstruction bar overlap
        # Find leftmost (minimum) x-coordinate of room polygon in pixels
        from src.components.enums import ImageDimensions
        dims = ImageDimensions(image_size)
        obs_bar_x_start, _, _, _ = dims.get_obstruction_bar_position()

        # Convert vertices to pixel coordinates
        # Coordinate transformation from 3D to 2D top-down view:
        #   3D x (along façade) -> image y (vertical, down is positive)
        #   3D y (into room) -> image x (horizontal, but REVERSED: into room = leftward)
        pixel_coords = []
        for vertex in self._vertices:
            # Offset from window center in meters
            dx = vertex.x - window_center_x  # Along façade
            dy = vertex.y - window_center_y  # Into room (perpendicular to façade)

            # Map to pixel coordinates:
            # - x_pixel: y=0 (façade) is at room_facade_x, larger y goes further LEFT (subtract)
            # - y_pixel: x=0 (window center) is at image center, positive x goes DOWN
            x_pixel = room_facade_x - round(dy / resolution)
            y_pixel = window_y_pixels + round(dx / resolution)

            pixel_coords.append([x_pixel, y_pixel])

        # Clip polygon to boundaries using Shapely
        # Room should end 2 pixels before obstruction bar (which itself is 4 pixels from right edge)
        # At 128x128: obs_bar at x=124, so room should clip at x<=122 (6 pixels from right edge)
        # Use obs_bar_x_start - 3 because Shapely box(minx, miny, maxx, maxy) max values are inclusive,
        # so box(0, 0, 121, 128) clips at x<=121, giving us 2-pixel gap before obs_bar at 124
        right_boundary = obs_bar_x_start - 3

        # Create clipping rectangle: x from 0 to right_boundary, y from 0 to image_size
        from shapely.geometry import Polygon as ShapelyPolygon, box

        room_poly = ShapelyPolygon(pixel_coords)
        clip_box = box(0, 0, right_boundary, image_size)

        # Clip the polygon
        clipped = room_poly.intersection(clip_box)

        # Handle different geometry types that might result from clipping
        if clipped.is_empty:
            # No intersection - return empty polygon
            return np.array([[[0, 0]]], dtype=np.int32)
        elif clipped.geom_type == 'Polygon':
            # Simple polygon - extract coordinates
            clipped_coords = list(clipped.exterior.coords)[:-1]  # Remove duplicate last point
        elif clipped.geom_type == 'MultiPolygon':
            # Multiple polygons - take the largest one
            largest = max(clipped.geoms, key=lambda p: p.area)
            clipped_coords = list(largest.exterior.coords)[:-1]
        elif clipped.geom_type == 'GeometryCollection':
            # Mixed geometry types - extract polygons
            polygons = [g for g in clipped.geoms if g.geom_type == 'Polygon']
            if polygons:
                largest = max(polygons, key=lambda p: p.area)
                clipped_coords = list(largest.exterior.coords)[:-1]
            else:
                return np.array([[[0, 0]]], dtype=np.int32)
        else:
            # Fallback - use original coordinates
            clipped_coords = pixel_coords

        return np.array([clipped_coords], dtype=np.int32)

    @classmethod
    def from_dict(cls, data: Union[List[dict], List[List[float]]]) -> 'RoomPolygon':
        """
        Create polygon from list of coordinate dictionaries or lists (Factory Method Pattern)

        Uses Strategy Pattern via PolygonParserFactory to select appropriate parser.

        Args:
            data: List of dicts like [{"x": 0, "y": 0}, {"x": 3, "y": 0}, ...]
                  OR list of lists like [[0, 0], [3, 0], ...]

        Returns:
            RoomPolygon instance

        Raises:
            ValueError: If data format is invalid
        """
        # Validate input type
        if not isinstance(data, list) or not data:
            raise ValueError(
                f"Parameter 'room_polygon' must be a non-empty list. "
                f"Got type: {type(data).__name__}, value: {data}"
            )

        # Factory Pattern: Get appropriate parser for data format
        parser = PolygonParserFactory.get_parser(data)

        # Strategy Pattern: Use selected parser to extract vertices
        vertices = parser.parse(data)

        # Create and return polygon instance
        return cls(vertices)


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
        z2: float
    ):
        """
        Initialize window geometry from bounding box

        Args:
            x1, y1, z1: Left-bottom corner coordinates in meters
            x2, y2, z2: Right-top corner coordinates in meters
        """
        self._corner1 = Point3D(x1, y1, z1)
        self._corner2 = Point3D(x2, y2, z2)

        # Ensure corner1 is bottom-left and corner2 is top-right
        self._x_min = min(x1, x2)
        self._x_max = max(x1, x2)
        self._z_min = min(z1, z2)
        self._z_max = max(z1, z2)

    @property
    def window_width_3d(self) -> float:
        """Get window width in 3D space (along façade, x-direction)"""
        return self._x_max - self._x_min

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

    def get_facade_orientation(self) -> float:
        """
        Determine which facade wall this window is on

        Returns angle in degrees representing the window's orientation:
        - 0°: South facade (default, y ≈ 0, window spans in x)
        - 90°: West facade (x < 0, window spans in y)
        - 180°: North facade (y > 0, window spans in x)
        - 270°: East facade (x > 0, window spans in y)

        Coordinate system:
        - Origin at window center
        - Y-axis: positive points into room (north), negative is outside (south)
        - X-axis: positive points right (east), negative points left (west)
        """
        tolerance = 0.01  # 1cm tolerance for "same" coordinate

        x_span = abs(self._corner2.x - self._corner1.x)
        y_span = abs(self._corner2.y - self._corner1.y)

        avg_x = (self._corner1.x + self._corner2.x) / 2
        avg_y = (self._corner1.y + self._corner2.y) / 2

        # Window spans in x direction (south or north facade)
        if x_span > tolerance and y_span <= tolerance:
            # South: y ≈ 0 or negative (at or outside building)
            # North: y significantly > 0 (inside building, far wall)
            if avg_y < 1.0:  # Within 1m of facade - treat as south
                return 0.0  # South facade
            else:
                return 180.0  # North facade

        # Window spans in y direction (east or west facade)
        elif y_span > tolerance and x_span <= tolerance:
            # West: x < 0 (left side)
            # East: x > 0 (right side)
            if avg_x < 0:
                return 90.0  # West facade
            else:
                return 270.0  # East facade

        # Default to south if unclear
        return 0.0

    def rotate(self, angle_degrees: float, center: Point2D = None) -> 'WindowGeometry':
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
        from shapely.geometry import LineString
        from shapely.affinity import rotate as shapely_rotate

        line = LineString([(self._corner1.x, self._corner1.y), (self._corner2.x, self._corner2.y)])
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
        window_offset_px: int = 12
    ) -> Tuple[int, int, int, int]:
        """
        Get window bounds in pixel coordinates for top view

        In top view:
        - Window appears as vertical line at fixed x position (12px from right)
        - Horizontal extent = wall thickness (approximately constant)
        - Vertical extent = window width in 3D (x2 - x1) converted to pixels

        Args:
            image_size: Image dimension in pixels (default 128)
            window_offset_px: Distance from right edge in pixels (default 12)

        Returns:
            (x_start, y_start, x_end, y_end) tuple in pixels
        """
        resolution = 0.1  # meters per pixel for 128x128 image
        scale = image_size / 128.0

        # Adjust resolution for scaled images
        actual_resolution = resolution / scale

        # Window position: fixed distance from right edge
        window_x_end = image_size - window_offset_px

        # Window thickness (wall thickness) - typically ~0.3m = 3 pixels at base scale
        wall_thickness_m = 0.3
        wall_thickness_px = round(wall_thickness_m / actual_resolution)
        window_x_start = window_x_end - wall_thickness_px

        # Vertical extent based on window width in 3D (appears as height in top view)
        window_width_m = self.window_width_3d
        window_height_px = round(window_width_m / actual_resolution)

        # Center vertically
        image_center_y = image_size // 2
        window_y_start = image_center_y - window_height_px // 2
        window_y_end = window_y_start + window_height_px

        # Clamp to image bounds
        x_start = max(0, window_x_start)
        x_end = min(image_size, window_x_end)
        y_start = max(0, window_y_start)
        y_end = min(image_size, window_y_end)

        return (x_start, y_start, x_end, y_end)

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
            data: Dict with keys x1, y1, z1, x2, y2, z2

        Returns:
            WindowGeometry instance
        """
        return cls(
            x1=data["x1"],
            y1=data["y1"],
            z1=data["z1"],
            x2=data["x2"],
            y2=data["y2"],
            z2=data["z2"]
        )


class WindowPosition:
    """
    DEPRECATED: Use WindowGeometry instead

    Represents window position and dimensions (legacy format)
    """

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        width: float,
        height: float
    ):
        """
        Initialize window position

        Args:
            x: X coordinate of window center in meters (along façade)
            y: Y coordinate of window center in meters (perpendicular to façade)
            z: Z coordinate of window top in meters (sill height + height)
            width: Window width in meters
            height: Window height in meters
        """
        self._position = Point3D(x, y, z)
        self._width = width
        self._height = height

    @property
    def position(self) -> Point3D:
        """Get window position"""
        return self._position

    @property
    def width(self) -> float:
        """Get window width"""
        return self._width

    @property
    def height(self) -> float:
        """Get window height"""
        return self._height

    @property
    def sill_height(self) -> float:
        """Calculate sill height from z coordinate"""
        return self._position.z - self._height

    def get_pixel_bounds(
        self,
        resolution: float = 0.1,
        image_size: int = 128,
        window_offset: float = 1.24
    ) -> Tuple[int, int, int, int]:
        """
        Get window bounds in pixel coordinates

        Args:
            resolution: Meters per pixel (default 0.1m = 10cm)
            image_size: Image dimension in pixels (default 128)
            window_offset: Window center offset from right edge in meters

        Returns:
            (x_start, y_start, x_end, y_end) tuple in pixels
        """
        # Window reference point
        window_x_pixels = image_size - int(window_offset / resolution)
        window_y_pixels = image_size // 2

        # Convert position to pixel offset
        x_center = window_x_pixels + int(self._position.x / resolution)
        y_center = window_y_pixels + int(self._position.y / resolution)

        # Calculate bounds
        width_pixels = int(self._width / resolution)
        height_pixels = int(self._height / resolution)

        x_start = max(0, x_center - width_pixels // 2)
        x_end = min(image_size, x_center + width_pixels // 2)
        y_start = max(0, y_center - height_pixels // 2)
        y_end = min(image_size, y_center + height_pixels // 2)

        return (x_start, y_start, x_end, y_end)

    @classmethod
    def from_dict(cls, data: dict) -> 'WindowPosition':
        """
        Create window position from dictionary

        Args:
            data: Dict with keys x, y, z, width, height

        Returns:
            WindowPosition instance
        """
        return cls(
            x=data["x"],
            y=data["y"],
            z=data["z"],
            width=data["width"],
            height=data["height"]
        )


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

        Args:
            window_geometry: Window geometry with x1, y1, x2, y2 coordinates
            room_polygon: Room polygon vertices

        Returns:
            (is_valid, error_message) tuple
        """
        # Get window line segment (the internal side facing the room)
        # Window spans from (x1, y1) to (x2, y2)
        window_line = LineString([
            (window_geometry.x1, window_geometry.y1),
            (window_geometry.x2, window_geometry.y2)
        ])

        # Get room polygon edges
        polygon_coords = [(v.x, v.y) for v in room_polygon._vertices]
        room_poly = ShapelyPolygon(polygon_coords)

        # Check if window line lies on the polygon boundary
        # Method 1: Check if both endpoints are on or very close to the boundary
        point1 = ShapelyPoint(window_geometry.x1, window_geometry.y1)
        point2 = ShapelyPoint(window_geometry.x2, window_geometry.y2)

        dist1 = point1.distance(room_poly.boundary)
        dist2 = point2.distance(room_poly.boundary)

        if dist1 > self._tolerance or dist2 > self._tolerance:
            return False, (
                f"Window endpoints not on polygon border. "
                f"Point 1 ({window_geometry.x1:.3f}, {window_geometry.y1:.3f}) distance: {dist1:.4f}m, "
                f"Point 2 ({window_geometry.x2:.3f}, {window_geometry.y2:.3f}) distance: {dist2:.4f}m. "
                f"Tolerance: {self._tolerance}m. "
                f"Window must lie on one of the room polygon edges."
            )

        # Method 2: Check if window line is part of one of the polygon edges
        # Get all polygon edges
        edges = []
        for i in range(len(polygon_coords)):
            p1 = polygon_coords[i]
            p2 = polygon_coords[(i + 1) % len(polygon_coords)]
            edges.append(LineString([p1, p2]))

        # Check if window line lies on any edge
        window_on_edge = False
        for edge in edges:
            # Check if window line is contained within this edge (with tolerance)
            if edge.buffer(self._tolerance).contains(window_line):
                window_on_edge = True
                break

        if not window_on_edge:
            return False, (
                f"Window line does not lie on any polygon edge. "
                f"Window spans from ({window_geometry.x1:.3f}, {window_geometry.y1:.3f}) "
                f"to ({window_geometry.x2:.3f}, {window_geometry.y2:.3f}). "
                f"It must be positioned along one of the room polygon sides."
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
            if "window_geometry" in window_data:
                geom_data = window_data["window_geometry"]
                window_geom = WindowGeometry.from_dict(geom_data)
            else:
                window_geom = WindowGeometry(
                    x1=window_data["x1"],
                    y1=window_data["y1"],
                    z1=window_data["z1"],
                    x2=window_data["x2"],
                    y2=window_data["y2"],
                    z2=window_data["z2"]
                )

            # Parse room polygon
            room_poly = RoomPolygon.from_dict(polygon_data)

            # Validate
            validator = cls(tolerance=tolerance)
            return validator.validate_window_on_border(window_geom, room_poly)

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
