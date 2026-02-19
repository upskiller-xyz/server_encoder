"""
Tests for custom exception classes
"""

import unittest
from src.core import (
    WindowNotOnPolygonError,
    WindowHeightValidationError,
    ParameterValidationError,
    RegionEncodingError
)
from src.components.geometry import (
    WindowGeometry,
    RoomPolygon,
    WindowBorderValidator,
    WindowHeightValidator
)


class TestWindowNotOnPolygonError(unittest.TestCase):
    """Test WindowNotOnPolygonError exception"""

    def test_exception_raised_when_window_not_on_border(self):
        """Test that exception is raised when window is not on polygon border"""
        # Window outside the room polygon
        window = WindowGeometry(x1=10, y1=0, z1=1, x2=11, y2=0, z2=2)
        room = RoomPolygon([(0, 0), (5, 0), (5, 5), (0, 5)])

        with self.assertRaises(WindowNotOnPolygonError) as context:
            WindowBorderValidator.validate_window_on_border(window, room)

        exc = context.exception
        self.assertEqual(exc.window_bbox, (10, 0, 11, 0))
        self.assertEqual(exc.tolerance, 0.05)
        self.assertIsNotNone(exc.polygon_edges)
        self.assertIsNotNone(exc.window_edges)

    def test_exception_message_includes_details(self):
        """Test that exception message includes window and tolerance details"""
        window = WindowGeometry(x1=10, y1=0, z1=1, x2=11, y2=0, z2=2)
        room = RoomPolygon([(0, 0), (5, 0), (5, 5), (0, 5)])

        with self.assertRaises(WindowNotOnPolygonError) as context:
            WindowBorderValidator.validate_window_on_border(window, room)

        message = str(context.exception)
        self.assertIn("Window edge not on polygon border", message)
        self.assertIn("10.000", message)
        self.assertIn("0.05m", message)

    def test_no_exception_when_window_on_border(self):
        """Test that no exception is raised when window is on polygon border"""
        # Window on the right edge of the room
        window = WindowGeometry(x1=4.9, y1=0, z1=1, x2=5.0, y2=0, z2=2)
        room = RoomPolygon([(0, 0), (5, 0), (5, 5), (0, 5)])

        # Should not raise
        WindowBorderValidator.validate_window_on_border(window, room)


class TestWindowHeightValidationError(unittest.TestCase):
    """Test WindowHeightValidationError exception"""

    def test_exception_raised_when_window_below_floor(self):
        """Test that exception is raised when window is below floor"""
        window = WindowGeometry(x1=0, y1=0, z1=0.5, x2=1, y2=0, z2=2)

        with self.assertRaises(WindowHeightValidationError) as context:
            WindowHeightValidator.validate_window_height_bounds(
                window, floor_height=1.0, roof_height=3.0
            )

        exc = context.exception
        self.assertEqual(exc.error_type, "below_floor")
        self.assertEqual(exc.window_bottom, 0.5)
        self.assertEqual(exc.floor_height, 1.0)

    def test_exception_raised_when_window_above_roof(self):
        """Test that exception is raised when window is above roof"""
        window = WindowGeometry(x1=0, y1=0, z1=2.0, x2=1, y2=0, z2=3.5)

        with self.assertRaises(WindowHeightValidationError) as context:
            WindowHeightValidator.validate_window_height_bounds(
                window, floor_height=1.0, roof_height=3.0
            )

        exc = context.exception
        self.assertEqual(exc.error_type, "above_roof")
        self.assertEqual(exc.window_top, 3.5)
        self.assertEqual(exc.roof_height, 3.0)

    def test_no_exception_when_window_within_bounds(self):
        """Test that no exception is raised when window is within bounds"""
        window = WindowGeometry(x1=0, y1=0, z1=1.5, x2=1, y2=0, z2=2.5)

        # Should not raise
        WindowHeightValidator.validate_window_height_bounds(
            window, floor_height=1.0, roof_height=3.0
        )


class TestParameterValidationError(unittest.TestCase):
    """Test ParameterValidationError exception"""

    def test_exception_with_parameter_name(self):
        """Test exception with just parameter name"""
        exc = ParameterValidationError("window_height")
        self.assertEqual(exc.parameter_name, "window_height")
        self.assertIn("window_height", str(exc))

    def test_exception_with_details(self):
        """Test exception with parameter name and details"""
        exc = ParameterValidationError("window_height", "Value must be positive")
        self.assertEqual(exc.parameter_name, "window_height")
        self.assertEqual(exc.details, "Value must be positive")
        self.assertIn("window_height", str(exc))
        self.assertIn("Value must be positive", str(exc))


class TestRegionEncodingError(unittest.TestCase):
    """Test RegionEncodingError exception"""

    def test_exception_with_region_and_details(self):
        """Test exception with region type and details"""
        exc = RegionEncodingError("window", "Invalid pixel bounds")
        self.assertEqual(exc.region_type, "window")
        self.assertEqual(exc.details, "Invalid pixel bounds")
        self.assertIn("window", str(exc))
        self.assertIn("Invalid pixel bounds", str(exc))


if __name__ == '__main__':
    unittest.main()
