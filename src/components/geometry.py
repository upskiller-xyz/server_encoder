from typing import List, Tuple, Any, Union, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
import numpy as np
import cv2
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint, LineString as ShapelyLine, box as ShapelyBox
from shapely.affinity import rotate as shapely_rotate
from src.components.enums import ImageDimensions
from src.components.graphics_constants import GRAPHICS_CONSTANTS


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
        print("ROTATING", angle_degrees, coords)
        rotated_poly = shapely_rotate(
            shapely_poly,
            angle_degrees,
            origin=(center.x, center.y)
        )

        # Extract rotated vertices
        rotated_vertices = list(rotated_poly.exterior.coords)[:-1]  # Remove duplicate last point
        return RoomPolygon(rotated_vertices)

    def _find_window_edge_and_rotation(
        self,
        window_x1: float,
        window_y1: float,
        window_x2: float,
        window_y2: float,
        tolerance: float = 0.01
    ) -> Tuple[int, float, bool]:
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
        

        window_line = ShapelyLine([(window_x1, window_y1), (window_x2, window_y2)])

        # Find which edge contains the window
        for i in range(len(self._vertices)):
            v1 = self._vertices[i]
            v2 = self._vertices[(i + 1) % len(self._vertices)]

            edge = ShapelyLine([(v1.x, v1.y), (v2.x, v2.y)])

            # Check if window line is contained in this edge (with buffer for tolerance)
            if edge.buffer(tolerance).contains(window_line):
                # Found the window edge
                # Calculate angle of this edge
                dx = v2.x - v1.x
                dy = v2.y - v1.y

                # Angle of edge from horizontal
                edge_angle = math.atan2(dy, dx) * 180 / math.pi

                # We want to rotate so this edge becomes horizontal (y = constant)
                # An edge at angle θ from horizontal needs rotation of -θ to become horizontal
                rotation_needed = -edge_angle

                # After rotation, check if room extends in +y or -y direction
                # We need to check the actual polygon area on each side of the window edge
                angle_rad = rotation_needed * math.pi / 180
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)

                # Rotate all vertices
                rotated_vertices = []
                for v in self._vertices:
                    rot_x = v.x * cos_a - v.y * sin_a
                    rot_y = v.x * sin_a + v.y * cos_a
                    rotated_vertices.append((rot_x, rot_y))

                # Rotate window y coordinate
                rot_w_y = (window_y1 * sin_a + window_y1 * cos_a +
                          window_y2 * sin_a + window_y2 * cos_a) / 2

                # Create polygon from rotated vertices
                room_poly = ShapelyPolygon(rotated_vertices)

                # Split at window edge
                upper_box = ShapelyBox(-10000, rot_w_y, 10000, 10000)
                lower_box = ShapelyBox(-10000, -10000, 10000, rot_w_y)

                upper_part = room_poly.intersection(upper_box)
                lower_part = room_poly.intersection(lower_box)

                # For correct positioning, room should extend in +y direction (above)
                # If more area is below, we need to flip
                needs_flip = lower_part.area > upper_part.area

                return i, rotation_needed, needs_flip

        raise ValueError(
            f"Window at ({window_x1:.2f}, {window_y1:.2f}) to ({window_x2:.2f}, {window_y2:.2f}) "
            f"does not lie on any polygon edge"
        )

    def to_pixel_array(
        self,
        image_size: int = 128,
        window_x1: float = None,
        window_y1: float = None,
        window_x2: float = None,
        window_y2: float = None,
        direction_angle: float = None
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
        print("WINDOW", window_x1, window_x2, window_y1, window_y2)
        # Calculate rotation angle to make window point to the right (0 radians)
        if direction_angle is not None:
            # direction_angle is in radians, shows where window is currently pointing
            # We want it to point right (0 radians), so rotation = -direction_angle
            rotation_angle = -direction_angle * 180 / math.pi  # Convert to degrees
        else:
            # Fallback: calculate from polygon edge
            _, rotation_angle, _ = self._find_window_edge_and_rotation(
                window_x1, window_y1, window_x2, window_y2
            )
            
        rotated_polygon =  [(v.x, v.y) for v in self._vertices] 

        window_center_rotated = Point2D((window_x1 + window_x2) / 2, (window_y1 + window_y2) / 2)
        print("WINDOW center rotated", window_center_rotated.x, window_center_rotated.y)
        
        
        print("polygon coords", rotated_polygon)
        
        rotated_polygon = RoomPolygon(rotated_polygon)

        # Calculate resolution based on image size (scales proportionally)
        resolution = GRAPHICS_CONSTANTS.get_resolution(image_size)

        wall_thickness_px = round(GRAPHICS_CONSTANTS.WALL_THICKNESS_M / resolution)
        # Window's left edge position on image
        window_left_edge_x = image_size - GRAPHICS_CONSTANTS.WINDOW_OFFSET_PX - wall_thickness_px

        # Room should align 1 pixel to the left of window for perfect adjacency (C-frame)
        room_facade_x = window_left_edge_x - GRAPHICS_CONSTANTS.ROOM_FACADE_OFFSET_PX
        window_y_pixels = image_size // 2

        # First pass: calculate room extent to check for obstruction bar overlap
        dims = ImageDimensions(image_size)
        obs_bar_x_start, _, _, _ = dims.get_obstruction_bar_position()


        pixel_coords = []
        for vertex in rotated_polygon.vertices:
            # Offset from rotated window center in meters
            dx = vertex.x - window_center_rotated.x  # Along façade
            dy = vertex.y - window_center_rotated.y  # Perpendicular to façade
            
            x_pixel = room_facade_x + round(dx / resolution)
            y_pixel = window_y_pixels + round(dy / resolution)

            pixel_coords.append([x_pixel, y_pixel])
        print("PIXEL coords", pixel_coords)

        # Clip room to avoid overlap with obstruction bar
        right_boundary = obs_bar_x_start - GRAPHICS_CONSTANTS.OBSTRUCTION_BAR_GAP_PX
        print("right_boundary", right_boundary)
        # Create clipping rectangle: x from 0 to right_boundary, y from 0 to image_size
        room_poly = ShapelyPolygon(pixel_coords)
        clip_box = ShapelyBox(0, 0, right_boundary, image_size)

        # Clip the polygon
        clipped = room_poly.intersection(clip_box)
        print("clipped", clipped)
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
        z2: float,
        direction_angle: float = None
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
        if self._direction_angle is not None:
            # direction_angle is the window normal (direction window faces)
            # Window width is perpendicular to this direction
            # Calculate perpendicular direction (edge direction)
            edge_angle = self._direction_angle + math.pi / 2

            # Unit vector perpendicular to direction_angle
            perp_dx = math.cos(edge_angle)
            perp_dy = math.sin(edge_angle)

            # Project both points onto the perpendicular direction
            # Point 1 projection
            proj1 = self._corner1.x * perp_dx + self._corner1.y * perp_dy
            # Point 2 projection
            proj2 = self._corner2.x * perp_dx + self._corner2.y * perp_dy

            # Window width is the distance between projections
            return abs(proj2 - proj1)
        else:
            # Fallback: use max of bounding box dimensions
            return max(self._x_max - self._x_min, self._y_max - self._y_min)

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
    def direction_angle(self) -> float:
        """Get window direction angle in radians (None if not set)"""
        return self._direction_angle

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
        window_offset_px: int = None
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

        # Get resolution for this image size
        actual_resolution = GRAPHICS_CONSTANTS.get_resolution(image_size)

        # Window position: fixed distance from right edge
        window_x_end = image_size - window_offset_px

        # Window thickness (wall thickness)
        wall_thickness_px = round(GRAPHICS_CONSTANTS.WALL_THICKNESS_M / actual_resolution)
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
            data: Dict with keys x1, y1, z1, x2, y2, z2, and optionally direction_angle

        Returns:
            WindowGeometry instance
        """
        return cls(
            x1=data["x1"],
            y1=data["y1"],
            z1=data["z1"],
            x2=data["x2"],
            y2=data["y2"],
            z2=data["z2"],
            direction_angle=data.get("direction_angle")
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
        # Get room polygon
        polygon_coords = [(v.x, v.y) for v in room_polygon._vertices]
        room_poly = ShapelyPolygon(polygon_coords)

        # If direction_angle is provided, use it to create proper window rectangle
        if window_geometry.direction_angle is not None:
            # direction_angle is the window normal (direction window faces)
            # We need perpendicular direction for window edge
            edge_angle = window_geometry.direction_angle + math.pi / 2

            # Calculate perpendicular unit vector (direction along window edge)
            edge_dx = math.cos(edge_angle)
            edge_dy = math.sin(edge_angle)

            # Create two lines perpendicular to direction_angle
            # Line 1: passes through (x1, y1), perpendicular to direction_angle
            # Line 2: passes through (x2, y2), perpendicular to direction_angle

            # For validation, we check if these two parallel lines both touch the polygon boundary
            # If they do, the window rectangle has one edge on the polygon

            # Calculate distance from each line to polygon boundary
            # We'll create line segments along the perpendicular direction
            line_length = 100  # Large enough to cover any reasonable polygon

            # Line 1 endpoints
            line1_p1 = (window_geometry.x1 - edge_dx * line_length,
                        window_geometry.y1 - edge_dy * line_length)
            line1_p2 = (window_geometry.x1 + edge_dx * line_length,
                        window_geometry.y1 + edge_dy * line_length)
            line1 = ShapelyLine([line1_p1, line1_p2])

            # Line 2 endpoints
            line2_p1 = (window_geometry.x2 - edge_dx * line_length,
                        window_geometry.y2 - edge_dy * line_length)
            line2_p2 = (window_geometry.x2 + edge_dx * line_length,
                        window_geometry.y2 + edge_dy * line_length)
            line2 = ShapelyLine([line2_p1, line2_p2])

            # Check if at least one line intersects the polygon boundary
            # The line that intersects is the window edge facing the room
            intersects1 = line1.distance(room_poly.boundary) < self._tolerance
            intersects2 = line2.distance(room_poly.boundary) < self._tolerance

            if not (intersects1 or intersects2):
                return False, (
                    f"Window does not lie on polygon border. "
                    f"Window at ({window_geometry.x1:.3f}, {window_geometry.y1:.3f}) to "
                    f"({window_geometry.x2:.3f}, {window_geometry.y2:.3f}) "
                    f"with direction_angle={window_geometry.direction_angle:.3f} rad "
                    f"({window_geometry.direction_angle * 180 / math.pi:.1f}°). "
                    f"Neither edge touches the room polygon boundary."
                )

            # Additionally verify the window edge actually lies on a polygon edge
            # Get the line that intersects (this is the internal edge)
            window_edge = line1 if intersects1 else line2
            window_x = window_geometry.x1 if intersects1 else window_geometry.x2
            window_y = window_geometry.y1 if intersects1 else window_geometry.y2

            # Find which polygon edge contains this line
            edges = []
            for i in range(len(polygon_coords)):
                p1 = polygon_coords[i]
                p2 = polygon_coords[(i + 1) % len(polygon_coords)]
                edges.append(ShapelyLine([p1, p2]))

            window_on_edge = False
            for edge in edges:
                # Check if the window point lies on this edge
                point = ShapelyPoint(window_x, window_y)
                if edge.distance(point) < self._tolerance:
                    window_on_edge = True
                    break

            if not window_on_edge:
                return False, (
                    f"Window edge not aligned with polygon edge. "
                    f"Window at ({window_x:.3f}, {window_y:.3f}) "
                    f"does not lie on any polygon edge (tolerance: {self._tolerance}m)."
                )

            return True, ""

        else:
            # Fallback: assume (x1,y1) to (x2,y2) defines the window edge directly
            # This is the old behavior when direction_angle is not provided
            window_line = ShapelyLine([
                (window_geometry.x1, window_geometry.y1),
                (window_geometry.x2, window_geometry.y2)
            ])

            # Check if both endpoints are on the boundary
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

            # Check if window line is part of one of the polygon edges
            edges = []
            for i in range(len(polygon_coords)):
                p1 = polygon_coords[i]
                p2 = polygon_coords[(i + 1) % len(polygon_coords)]
                edges.append(ShapelyLine([p1, p2]))

            window_on_edge = False
            for edge in edges:
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
