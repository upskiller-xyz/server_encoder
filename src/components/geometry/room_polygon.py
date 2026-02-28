from typing import List, Tuple
import math
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint, LineString as ShapelyLine, box as ShapelyBox
from shapely.affinity import rotate as shapely_rotate
from src.components.geometry.point_2d import Point2D
from src.components.geometry.geometry_adapter import GeometryAdapter
from src.core import ImageDimensions
from src.core import GRAPHICS_CONSTANTS
from src.core.enums import ParameterName
from src.components.geometry.window_geometry import WindowGeometry
from src.components.geometry.geometry_ops import GeometryOps


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

    def _build_edge(self, ind: int):
        v1 = self._vertices[ind]
        v2 = self._vertices[(ind + 1) % len(self._vertices)]

        return ShapelyLine([(v1.x, v1.y), (v2.x, v2.y)])

    def _window_edge_and_rotation(
        self,
        window_line: ShapelyLine,
        tolerance: float = GRAPHICS_CONSTANTS.WINDOW_PLACEMENT_TOLERANCE
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
        edges = [(i, edge) for i, edge in enumerate(edges) if edge.buffer(tolerance).contains(window_line)]

        if len(edges) < 1:
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

        window_center_rotated = Point2D((window_x1 + window_x2) * 0.5, (window_y1 + window_y2) * 0.5)
        if len(res) > 0:
            
            edge_on_boundary = res[0]
            edge_coords = list(edge_on_boundary.coords)
            window_center_rotated = Point2D(
                np.round((edge_coords[0][0] + edge_coords[1][0]) * 0.5, 2),
                np.round((edge_coords[0][1] + edge_coords[1][1]) * 0.5, 2)
            )
        else:
            # Fallback: project window center onto polygon boundary (matches merger's fallback)
            center = ShapelyPoint((window_x1 + window_x2) * 0.5, (window_y1 + window_y2) * 0.5)
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
        if isinstance(data, cls):
            return data

        # Check format of first element to determine data structure
        first_element = data[0]

        if isinstance(first_element, dict):
            # List of dictionaries format: [{"x": 0, "y": 0}, ...]
            vertices = [(point[ParameterName.X.value], point[ParameterName.Y.value]) for point in data]
        elif isinstance(first_element, (list, tuple)):
            # List of lists/tuples format: [[0, 0], ...] or [(0, 0), ...]
            vertices = [(point[0], point[1]) for point in data]
        else:
            raise ValueError(
                f"Invalid polygon data format. Expected list of dicts or list of lists/tuples, "
                f"but got list of {type(first_element).__name__}"
            )

        return cls(vertices)
