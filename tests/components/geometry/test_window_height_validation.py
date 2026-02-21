"""
Unit tests for WindowHeightValidator - validates window z-coordinates are between floor and roof.
"""

import pytest
from src.components.geometry import WindowHeightValidator, WindowGeometry


class TestWindowHeightValidator:
    """Test WindowHeightValidator class."""

    def test_window_within_bounds(self):
        """Test window that is properly within floor and roof bounds."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=1.0, x2=1.0, y2=0.0, z2=2.5)
        floor_height = 0.0
        roof_height = 3.0

        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height, roof_height
        )

        assert is_valid is True
        assert error_msg == ""

    def test_window_at_floor_level(self):
        """Test window that starts exactly at floor level."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=0.0, x2=1.0, y2=0.0, z2=2.0)
        floor_height = 0.0
        roof_height = 3.0

        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height, roof_height
        )

        assert is_valid is True
        assert error_msg == ""

    def test_window_at_roof_level(self):
        """Test window that ends exactly at roof level."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=1.0, x2=1.0, y2=0.0, z2=3.0)
        floor_height = 0.0
        roof_height = 3.0

        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height, roof_height
        )

        assert is_valid is True
        assert error_msg == ""

    def test_window_fills_entire_height(self):
        """Test window that spans from floor to roof."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=0.0, x2=1.0, y2=0.0, z2=3.0)
        floor_height = 0.0
        roof_height = 3.0

        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height, roof_height
        )

        assert is_valid is True
        assert error_msg == ""

    def test_window_below_floor(self):
        """Test window that extends below floor level."""
        from src.core.exceptions import WindowHeightValidationError

        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=-0.5, x2=1.0, y2=0.0, z2=2.0)
        floor_height = 0.0
        roof_height = 3.0

        with pytest.raises(WindowHeightValidationError) as exc_info:
            WindowHeightValidator.validate_window_height_bounds(
                window_geom, floor_height, roof_height
            )

        error_msg = str(exc_info.value)
        assert "below floor" in error_msg
        assert "-0.50m" in error_msg
        assert "0.00m" in error_msg

    def test_window_above_roof(self):
        """Test window that extends above roof level."""
        from src.core.exceptions import WindowHeightValidationError

        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=2.0, x2=1.0, y2=0.0, z2=3.5)
        floor_height = 0.0
        roof_height = 3.0

        with pytest.raises(WindowHeightValidationError) as exc_info:
            WindowHeightValidator.validate_window_height_bounds(
                window_geom, floor_height, roof_height
            )

        error_msg = str(exc_info.value)
        assert "above roof" in error_msg
        assert "3.50m" in error_msg
        assert "3.00m" in error_msg

    def test_window_both_below_and_above(self):
        """Test window that extends both below floor and above roof (should fail on floor check first)."""
        from src.core.exceptions import WindowHeightValidationError

        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=-1.0, x2=1.0, y2=0.0, z2=4.0)
        floor_height = 0.0
        roof_height = 3.0

        with pytest.raises(WindowHeightValidationError) as exc_info:
            WindowHeightValidator.validate_window_height_bounds(
                window_geom, floor_height, roof_height
            )

        error_msg = str(exc_info.value)
        assert "below floor" in error_msg  # Floor check happens first

    def test_window_with_elevated_floor(self):
        """Test window with non-zero floor height."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=2.5, x2=1.0, y2=0.0, z2=4.0)
        floor_height = 2.0
        roof_height = 5.0

        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height, roof_height
        )

        assert is_valid is True
        assert error_msg == ""

    def test_window_below_elevated_floor(self):
        """Test window below elevated floor."""
        from src.core.exceptions import WindowHeightValidationError

        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=1.5, x2=1.0, y2=0.0, z2=3.0)
        floor_height = 2.0
        roof_height = 5.0

        with pytest.raises(WindowHeightValidationError) as exc_info:
            WindowHeightValidator.validate_window_height_bounds(
                window_geom, floor_height, roof_height
            )

        error_msg = str(exc_info.value)
        assert "below floor" in error_msg
        assert "1.50m" in error_msg
        assert "2.00m" in error_msg

    def test_window_with_reversed_z_coordinates(self):
        """Test window where z2 < z1 (should handle both orderings)."""
        # z2 < z1, but window is still within bounds
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=2.5, x2=1.0, y2=0.0, z2=1.0)
        floor_height = 0.0
        roof_height = 3.0

        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height, roof_height
        )

        assert is_valid is True
        assert error_msg == ""

    def test_window_within_tolerance(self):
        """Test window that is within numerical tolerance of bounds."""
        # GRAPHICS_CONSTANTS.WINDOW_HEIGHT_TOLERANCE is used
        # Window bottom is slightly below floor but within tolerance
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=-1e-7, x2=1.0, y2=0.0, z2=2.0)
        floor_height = 0.0
        roof_height = 3.0

        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height, roof_height
        )

        assert is_valid is True
        assert error_msg == ""

    def test_window_outside_tolerance(self):
        """Test window that is outside numerical tolerance."""
        from src.core.exceptions import WindowHeightValidationError

        # GRAPHICS_CONSTANTS.WINDOW_HEIGHT_TOLERANCE is used
        # Window bottom is significantly below floor (outside tolerance)
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=-0.01, x2=1.0, y2=0.0, z2=2.0)
        floor_height = 0.0
        roof_height = 3.0

        with pytest.raises(WindowHeightValidationError) as exc_info:
            WindowHeightValidator.validate_window_height_bounds(
                window_geom, floor_height, roof_height
            )

        error_msg = str(exc_info.value)
        assert "below floor" in error_msg

    def test_validate_from_parameters_valid(self):
        """Test validation from parameter dictionaries with valid window."""
        window_data = {"x1": 0.0, "y1": 0.0, "z1": 1.0, "x2": 1.0, "y2": 0.0, "z2": 2.5}
        floor_height = 0.0
        roof_height = 3.0

        is_valid, error_msg = WindowHeightValidator.validate_from_parameters(
            window_data, floor_height, roof_height
        )

        assert is_valid is True
        assert error_msg == ""

    def test_validate_from_parameters_below_floor(self):
        """Test validation from parameters with window below floor."""
        window_data = {"x1": 0.0, "y1": 0.0, "z1": -0.5, "x2": 1.0, "y2": 0.0, "z2": 2.0}
        floor_height = 0.0
        roof_height = 3.0

        is_valid, error_msg = WindowHeightValidator.validate_from_parameters(
            window_data, floor_height, roof_height
        )

        assert is_valid is False
        assert "below floor" in error_msg

    def test_validate_from_parameters_above_roof(self):
        """Test validation from parameters with window above roof."""
        window_data = {"x1": 0.0, "y1": 0.0, "z1": 2.0, "x2": 1.0, "y2": 0.0, "z2": 3.5}
        floor_height = 0.0
        roof_height = 3.0

        is_valid, error_msg = WindowHeightValidator.validate_from_parameters(
            window_data, floor_height, roof_height
        )

        assert is_valid is False
        assert "above roof" in error_msg

    def test_validate_from_parameters_missing_z_coordinate(self):
        """Test validation with missing z coordinate in data."""
        window_data = {"x1": 0.0, "y1": 0.0, "z1": 1.0, "x2": 1.0, "y2": 0.0}  # Missing z2
        floor_height = 0.0
        roof_height = 3.0

        is_valid, error_msg = WindowHeightValidator.validate_from_parameters(
            window_data, floor_height, roof_height
        )

        assert is_valid is False
        assert "Error parsing height data" in error_msg

    def test_validate_from_parameters_invalid_data_type(self):
        """Test validation with invalid data type."""
        window_data = {"x1": "invalid", "y1": 0.0, "z1": 1.0, "x2": 1.0, "y2": 0.0, "z2": 2.0}
        floor_height = 0.0
        roof_height = 3.0

        is_valid, error_msg = WindowHeightValidator.validate_from_parameters(
            window_data, floor_height, roof_height
        )

        assert is_valid is False
        assert "Error parsing height data" in error_msg

    def test_realistic_scenario_valid(self):
        """Test realistic scenario with typical building dimensions."""
        # Floor at 0.3m above terrain, roof at 3.0m above floor
        # Window from 1.0m to 2.5m (valid)
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=1.0, x2=2.0, y2=0.0, z2=2.5)
        floor_height = 0.3
        roof_height = 0.3 + 3.0  # 3.3m

        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height, roof_height
        )

        assert is_valid is True
        assert error_msg == ""

    def test_realistic_scenario_window_too_low(self):
        """Test realistic scenario where window sill is below floor."""
        from src.core.exceptions import WindowHeightValidationError

        # Floor at 0.3m, but window starts at 0.2m (below floor)
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=0.2, x2=2.0, y2=0.0, z2=2.0)
        floor_height = 0.3
        roof_height = 3.3

        with pytest.raises(WindowHeightValidationError) as exc_info:
            WindowHeightValidator.validate_window_height_bounds(
                window_geom, floor_height, roof_height
            )

        error_msg = str(exc_info.value)
        assert "below floor" in error_msg

    def test_realistic_scenario_window_too_high(self):
        """Test realistic scenario where window extends above roof."""
        from src.core.exceptions import WindowHeightValidationError

        # Roof at 3.3m, but window goes to 3.5m
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=2.0, x2=2.0, y2=0.0, z2=3.5)
        floor_height = 0.3
        roof_height = 3.3

        with pytest.raises(WindowHeightValidationError) as exc_info:
            WindowHeightValidator.validate_window_height_bounds(
                window_geom, floor_height, roof_height
            )

        error_msg = str(exc_info.value)
        assert "above roof" in error_msg

    def test_zero_height_room(self):
        """Test edge case where floor and roof are at same height."""
        # Floor and roof at same height - window can only be a line
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=1.0, x2=2.0, y2=0.0, z2=1.0)
        floor_height = 1.0
        roof_height = 1.0

        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height, roof_height
        )

        assert is_valid is True
        assert error_msg == ""

    def test_zero_height_room_window_with_height(self):
        """Test window with height in zero-height room (should fail)."""
        from src.core.exceptions import WindowHeightValidationError

        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=1.0, x2=2.0, y2=0.0, z2=1.5)
        floor_height = 1.0
        roof_height = 1.0

        with pytest.raises(WindowHeightValidationError) as exc_info:
            WindowHeightValidator.validate_window_height_bounds(
                window_geom, floor_height, roof_height
            )

        error_msg = str(exc_info.value)
        assert "above roof" in error_msg
