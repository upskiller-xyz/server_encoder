"""
Geometry module for window encoding operations.

This module provides classes for handling geometric operations related to
window encoding, including polygon parsing, coordinate transformations,
and window validation.
"""

from src.components.geometry.point_2d import Point2D
from src.components.geometry.point_3d import Point3D
from src.components.geometry.polygon_parser import (
    IPolygonDataParser,
    DictPolygonParser,
    ListPolygonParser,
    PolygonParserFactory
)
from src.components.geometry.geometry_adapter import GeometryAdapter
from src.components.geometry.room_polygon import RoomPolygon
from src.components.geometry.geometry_ops import GeometryOps
from src.components.geometry.window_geometry import WindowGeometry
from src.components.geometry.window_border_validator import WindowBorderValidator
from src.components.geometry.window_height_validator import WindowHeightValidator

__all__ = [
    'Point2D',
    'Point3D',
    'IPolygonDataParser',
    'DictPolygonParser',
    'ListPolygonParser',
    'PolygonParserFactory',
    'GeometryAdapter',
    'RoomPolygon',
    'GeometryOps',
    'WindowGeometry',
    'WindowBorderValidator',
    'WindowHeightValidator',
]
