from typing import List, Tuple, Any, Dict, Callable
import logging
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from src.core import GeometryType

logger = logging.Logger("logger")


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
        GeometryType.POLYGON: _extract_polygon_coords.__func__, # type: ignore
        GeometryType.MULTI_POLYGON: _extract_multi_polygon_coords.__func__, # type: ignore
        GeometryType.GEOMETRY_COLLECTION: _extract_geometry_collection_coords.__func__, # type: ignore
    }

    @classmethod
    def vertical_mirror(cls, poly: ShapelyPolygon) -> ShapelyPolygon:
        crds = np.array([x for x in poly.exterior.coords])
        return ShapelyPolygon(crds.dot([[1, 0], [0, -1]]))

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
