"""
Unit tests for WindowBorderValidator - validates window is positioned on room polygon border.
"""

import pytest
from src.components.geometry import WindowBorderValidator, WindowGeometry, RoomPolygon


class TestWindowBorderValidator:
    """Test WindowBorderValidator class."""

    def test_window_on_polygon_edge_simple_square(self):
        """Test window correctly positioned on edge of square room."""
        validator = WindowBorderValidator()

        # Square room: (0,0) -> (0,-1) -> (-1,-1) -> (-1,0) -> back to (0,0)
        room_polygon = RoomPolygon.from_dict([[0, 0], [0, -1], [-1, -1], [-1, 0]])

        # Window on the edge from (0,0) to (0,-1)
        window_geom = WindowGeometry(x1=0, y1=-0.2, z1=1.0, x2=0, y2=-0.8, z2=2.0)

        is_valid, error_msg = validator.validate_window_on_border(window_geom, room_polygon)

        assert is_valid is True
        assert error_msg == ""

    def test_window_outside_polygon_boundary(self):
        """Test window positioned outside polygon boundary (should fail)."""
        validator = WindowBorderValidator()

        # Square room: (0,0) -> (0,-1) -> (-1,-1) -> (-1,0)
        room_polygon = RoomPolygon.from_dict([[0, 0], [0, -1], [-1, -1], [-1, 0]])

        # Window OUTSIDE the polygon: y1=0.2, y2=1.8 (polygon y ranges from 0 to -1)
        window_geom = WindowGeometry(x1=0, y1=0.2, z1=1.0, x2=0, y2=1.8, z2=2.0)

        is_valid, error_msg = validator.validate_window_on_border(window_geom, room_polygon)

        assert is_valid is False
        assert "not on polygon border" in error_msg or "does not lie on any polygon edge" in error_msg

    def test_window_partially_outside_polygon(self):
        """Test window that partially extends outside polygon."""
        validator = WindowBorderValidator()

        # Square room
        room_polygon = RoomPolygon.from_dict([[0, 0], [0, -1], [-1, -1], [-1, 0]])

        # Window starts inside but extends outside
        window_geom = WindowGeometry(x1=0, y1=-0.5, z1=1.0, x2=0, y2=0.5, z2=2.0)

        is_valid, error_msg = validator.validate_window_on_border(window_geom, room_polygon)

        assert is_valid is False

    def test_window_on_different_edge(self):
        """Test window on different edge of polygon."""
        validator = WindowBorderValidator()

        # Square room
        room_polygon = RoomPolygon.from_dict([[0, 0], [0, -1], [-1, -1], [-1, 0]])

        # Window on bottom edge from (0,-1) to (-1,-1)
        window_geom = WindowGeometry(x1=-0.3, y1=-1, z1=1.0, x2=-0.7, y2=-1, z2=2.0)

        is_valid, error_msg = validator.validate_window_on_border(window_geom, room_polygon)

        assert is_valid is True
        assert error_msg == ""

    def test_validate_from_dict_flat_parameters(self):
        """Test validation from flat parameter dictionary."""
        # Square room
        room_polygon = [[0, 0], [0, -1], [-1, -1], [-1, 0]]

        # Window coordinates as flat dict (like in demo.ipynb)
        window_data = {
            "x1": 0, "y1": -0.2, "z1": 1.0,
            "x2": 0, "y2": -0.8, "z2": 2.0
        }

        is_valid, error_msg = WindowBorderValidator.validate_from_dict(
            window_data=window_data,
            polygon_data=room_polygon
        )

        assert is_valid is True
        assert error_msg == ""

    def test_validate_from_dict_outside_boundary(self):
        """Test validation catches window outside boundary with flat parameters."""
        # Square room
        room_polygon = [[0, 0], [0, -1], [-1, -1], [-1, 0]]

        # Window OUTSIDE polygon (matches user's example)
        window_data = {
            "x1": 0, "y1": 0.2, "z1": 18,
            "x2": 0, "y2": 1.8, "z2": 19.2
        }

        is_valid, error_msg = WindowBorderValidator.validate_from_dict(
            window_data=window_data,
            polygon_data=room_polygon
        )

        assert is_valid is False
        assert "not on polygon border" in error_msg or "does not lie on any polygon edge" in error_msg

    def test_validate_from_dict_nested_window_geometry(self):
        """Test validation with nested window_geometry object."""
        room_polygon = [[0, 0], [0, -1], [-1, -1], [-1, 0]]

        # Window as nested object
        window_data = {
            "window_geometry": {
                "x1": 0, "y1": -0.2, "z1": 1.0,
                "x2": 0, "y2": -0.8, "z2": 2.0
            }
        }

        is_valid, error_msg = WindowBorderValidator.validate_from_dict(
            window_data=window_data,
            polygon_data=room_polygon
        )

        assert is_valid is True
        assert error_msg == ""

    def test_window_diagonal_across_room(self):
        """Test window that goes diagonally (should fail - not on any edge)."""
        validator = WindowBorderValidator()

        room_polygon = RoomPolygon.from_dict([[0, 0], [0, -1], [-1, -1], [-1, 0]])

        # Diagonal window
        window_geom = WindowGeometry(x1=0, y1=0, z1=1.0, x2=-1, y2=-1, z2=2.0)

        is_valid, error_msg = validator.validate_window_on_border(window_geom, room_polygon)

        assert is_valid is False
        assert "does not lie on any polygon edge" in error_msg

    def test_window_at_corner(self):
        """Test window positioned at polygon corner."""
        validator = WindowBorderValidator()

        room_polygon = RoomPolygon.from_dict([[0, 0], [0, -1], [-1, -1], [-1, 0]])

        # Window at corner (0,0)
        window_geom = WindowGeometry(x1=0, y1=0, z1=1.0, x2=0, y2=0, z2=2.0)

        is_valid, error_msg = validator.validate_window_on_border(window_geom, room_polygon)

        # A zero-width window at a corner should be valid (it's on the boundary)
        assert is_valid is True

    def test_l_shaped_room_window_on_inner_edge(self):
        """Test window on inner edge of L-shaped room."""
        validator = WindowBorderValidator()

        # L-shaped room
        room_polygon = RoomPolygon.from_dict([
            [0, 0], [0, -2], [-2, -2], [-2, -1], [-1, -1], [-1, 0]
        ])

        # Window on the inner vertical edge
        window_geom = WindowGeometry(x1=-1, y1=-0.5, z1=1.0, x2=-1, y2=-0.9, z2=2.0)

        is_valid, error_msg = validator.validate_window_on_border(window_geom, room_polygon)

        assert is_valid is True

    def test_tolerance_parameter(self):
        """Test that tolerance parameter affects validation."""
        # Strict tolerance
        strict_validator = WindowBorderValidator(tolerance=1e-6)

        # Loose tolerance
        loose_validator = WindowBorderValidator(tolerance=0.1)

        room_polygon = RoomPolygon.from_dict([[0, 0], [0, -1], [-1, -1], [-1, 0]])

        # Window slightly off the edge (0.05m away)
        window_geom = WindowGeometry(x1=0.05, y1=-0.2, z1=1.0, x2=0.05, y2=-0.8, z2=2.0)

        # Should fail with strict tolerance
        is_valid_strict, _ = strict_validator.validate_window_on_border(window_geom, room_polygon)
        assert is_valid_strict is False

        # Should pass with loose tolerance
        is_valid_loose, _ = loose_validator.validate_window_on_border(window_geom, room_polygon)
        assert is_valid_loose is True

    def test_polygon_with_dict_coordinates(self):
        """Test validation with polygon defined using dict coordinates."""
        room_polygon = [
            {"x": 0, "y": 0},
            {"x": 0, "y": -1},
            {"x": -1, "y": -1},
            {"x": -1, "y": 0}
        ]

        window_data = {
            "x1": 0, "y1": -0.2, "z1": 1.0,
            "x2": 0, "y2": -0.8, "z2": 2.0
        }

        is_valid, error_msg = WindowBorderValidator.validate_from_dict(
            window_data=window_data,
            polygon_data=room_polygon
        )

        assert is_valid is True

    def test_error_handling_invalid_polygon(self):
        """Test error handling with invalid polygon data."""
        window_data = {"x1": 0, "y1": 0, "z1": 1.0, "x2": 0, "y2": 1, "z2": 2.0}

        # Invalid polygon (only 2 points)
        invalid_polygon = [[0, 0], [1, 1]]

        is_valid, error_msg = WindowBorderValidator.validate_from_dict(
            window_data=window_data,
            polygon_data=invalid_polygon
        )

        assert is_valid is False
        assert "Error parsing geometry data" in error_msg

    def test_error_handling_missing_window_coordinates(self):
        """Test error handling when window coordinates are missing."""
        room_polygon = [[0, 0], [0, -1], [-1, -1], [-1, 0]]

        # Missing z2
        incomplete_window = {"x1": 0, "y1": 0, "z1": 1.0, "x2": 0, "y2": 1}

        is_valid, error_msg = WindowBorderValidator.validate_from_dict(
            window_data=incomplete_window,
            polygon_data=room_polygon
        )

        assert is_valid is False
        assert "Error parsing geometry data" in error_msg
