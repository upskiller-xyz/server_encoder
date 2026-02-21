"""Unit tests for core exceptions"""

import pytest
from src.core.exceptions import (
    WindowEncoderException,
    GeometryValidationError,
    WindowNotOnPolygonError,
    WindowHeightValidationError,
    ParameterValidationError,
    EncodingError,
    RegionEncodingError,
)


class TestWindowEncoderException:
    """Tests for WindowEncoderException"""

    def test_base_exception_creation(self):
        """Test creating base exception"""
        exc = WindowEncoderException("Test error")
        assert str(exc) == "Test error"
        assert isinstance(exc, Exception)

    def test_base_exception_inheritance(self):
        """Test that WindowEncoderException is proper Exception"""
        exc = WindowEncoderException("message")
        assert isinstance(exc, WindowEncoderException)


class TestGeometryValidationError:
    """Tests for GeometryValidationError"""

    def test_geometry_validation_error_creation(self):
        """Test creating geometry validation error"""
        exc = GeometryValidationError("Invalid geometry")
        assert str(exc) == "Invalid geometry"
        assert isinstance(exc, WindowEncoderException)

    def test_geometry_validation_error_inheritance(self):
        """Test GeometryValidationError inheritance chain"""
        exc = GeometryValidationError("test")
        assert isinstance(exc, WindowEncoderException)
        assert isinstance(exc, Exception)


class TestWindowNotOnPolygonError:
    """Tests for WindowNotOnPolygonError"""

    def test_window_not_on_polygon_error_creation(self):
        """Test creating window not on polygon error"""
        window_bbox = (1.0, 2.0, 3.0, 4.0)
        direction_angle = 0.5
        tolerance = 0.1
        exc = WindowNotOnPolygonError(
            window_bbox=window_bbox,
            direction_angle=direction_angle,
            tolerance=tolerance
        )
        assert isinstance(exc, GeometryValidationError)
        assert exc.window_bbox == window_bbox

    def test_window_not_on_polygon_error_with_coordinates(self):
        """Test creating error with specific coordinates"""
        window_bbox = (1.0, 2.0, 3.0, 4.0)
        exc = WindowNotOnPolygonError(
            window_bbox=window_bbox,
            direction_angle=0.5,
            tolerance=0.1
        )
        assert "1" in str(exc) and "2" in str(exc)


class TestWindowHeightValidationError:
    """Tests for WindowHeightValidationError"""

    def test_window_height_validation_error_creation(self):
        """Test creating window height validation error"""
        exc = WindowHeightValidationError(
            window_bottom=0.5,
            window_top=2.5,
            floor_height=1.0,
            roof_height=3.0
        )
        assert isinstance(exc, GeometryValidationError)

    def test_window_height_validation_error_below_floor(self):
        """Test error when window extends below floor"""
        exc = WindowHeightValidationError(
            window_bottom=0.5,
            window_top=2.5,
            floor_height=1.0,
            roof_height=3.0,
            error_type="below_floor"
        )
        assert "0.50" in str(exc)
        assert "1.00" in str(exc)


class TestParameterValidationError:
    """Tests for ParameterValidationError"""

    def test_parameter_validation_error_creation(self):
        """Test creating parameter validation error"""
        exc = ParameterValidationError("frame_ratio")
        assert isinstance(exc, WindowEncoderException)

    def test_parameter_validation_error_message(self):
        """Test error message with parameter details"""
        exc = ParameterValidationError(
            "frame_ratio",
            details="must be between 0 and 1"
        )
        assert "frame_ratio" in str(exc)


class TestEncodingError:
    """Tests for EncodingError"""

    def test_encoding_error_creation(self):
        """Test creating encoding error"""
        exc = EncodingError("Encoding failed")
        assert isinstance(exc, WindowEncoderException)

    def test_encoding_error_message(self):
        """Test encoding error message"""
        msg = "Failed to encode window region"
        exc = EncodingError(msg)
        assert msg in str(exc)


class TestRegionEncodingError:
    """Tests for RegionEncodingError"""

    def test_region_encoding_error_creation(self):
        """Test creating region encoding error"""
        exc = RegionEncodingError(
            region_type="WINDOW",
            details="Failed to encode window region"
        )
        assert isinstance(exc, EncodingError)

    def test_region_encoding_error_with_region_info(self):
        """Test region encoding error with specific region"""
        exc = RegionEncodingError(
            region_type="WINDOW",
            details="Parameter validation failed"
        )
        assert "WINDOW" in str(exc)


class TestExceptionHierarchy:
    """Tests for exception hierarchy"""

    def test_exception_hierarchy_basic(self):
        """Test that WindowEncoderException is proper base"""
        exc = WindowEncoderException("test")
        assert isinstance(exc, Exception)

    def test_geometry_error_hierarchy(self):
        """Test that geometry errors extend WindowEncoderException"""
        exc = GeometryValidationError("test")
        assert isinstance(exc, WindowEncoderException)
        assert isinstance(exc, Exception)

    def test_can_catch_window_not_on_polygon_as_geometry(self):
        """Test catching WindowNotOnPolygonError as GeometryValidationError"""
        with pytest.raises(GeometryValidationError):
            raise WindowNotOnPolygonError(
                window_bbox=(1.0, 2.0, 3.0, 4.0),
                direction_angle=0.5,
                tolerance=0.1
            )

    def test_can_catch_encoding_errors_as_base(self):
        """Test catching encoding errors as EncodingError"""
        with pytest.raises(EncodingError):
            raise RegionEncodingError(
                region_type="WINDOW",
                details="test"
            )
