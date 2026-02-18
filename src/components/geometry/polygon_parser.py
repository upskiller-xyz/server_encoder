from typing import List, Tuple, Any
from abc import ABC, abstractmethod

from src.core import ParameterName


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

            if ParameterName.X.value not in point or ParameterName.Y.value not in point:
                raise ValueError(
                    f"Parameter 'room_polygon' point at index {i} missing 'x' or 'y' key. "
                    f"Got: {point}. Expected format: {{'x': value, 'y': value}}"
                )

            try:
                x = float(point[ParameterName.X.value])
                y = float(point[ParameterName.Y.value])
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
            f"Expected formats: [{ParameterName.X.value: val, ParameterName.Y.value: val}, ...] or [[x, y], ...]"
        )
