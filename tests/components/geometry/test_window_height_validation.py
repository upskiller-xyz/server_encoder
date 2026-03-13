"""
Unit tests for WindowHeightValidator - validates window z-coordinates are between floor and roof.
Windows within 15cm tolerance are clamped, beyond that an error is raised.
"""

import pytest
from src.components.geometry import WindowHeightValidator, WindowGeometry
from src.core.exceptions import WindowHeightValidationError


class TestWindowHeightValidator:
    """Test WindowHeightValidator class."""

    # --- Valid windows (no clamping needed) ---

    def test_window_within_bounds(self):
        """Test window that is properly within floor and roof bounds."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=1.0, x2=1.0, y2=0.0, z2=2.5)
        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height=0.0, roof_height=3.0
        )
        assert is_valid is True
        assert error_msg == ""

    def test_window_at_floor_level(self):
        """Test window that starts exactly at floor level."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=0.0, x2=1.0, y2=0.0, z2=2.0)
        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height=0.0, roof_height=3.0
        )
        assert is_valid is True
        assert error_msg == ""

    def test_window_at_roof_level(self):
        """Test window that ends exactly at roof level."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=1.0, x2=1.0, y2=0.0, z2=3.0)
        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height=0.0, roof_height=3.0
        )
        assert is_valid is True
        assert error_msg == ""

    def test_window_fills_entire_height(self):
        """Test window that spans from floor to roof."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=0.0, x2=1.0, y2=0.0, z2=3.0)
        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height=0.0, roof_height=3.0
        )
        assert is_valid is True
        assert error_msg == ""

    def test_window_with_elevated_floor(self):
        """Test window with non-zero floor height."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=2.5, x2=1.0, y2=0.0, z2=4.0)
        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height=2.0, roof_height=5.0
        )
        assert is_valid is True
        assert error_msg == ""

    def test_window_with_reversed_z_coordinates(self):
        """Test window where z2 < z1 (should handle both orderings)."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=2.5, x2=1.0, y2=0.0, z2=1.0)
        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height=0.0, roof_height=3.0
        )
        assert is_valid is True
        assert error_msg == ""

    def test_realistic_scenario_valid(self):
        """Test realistic scenario with typical building dimensions."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=1.0, x2=2.0, y2=0.0, z2=2.5)
        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height=0.3, roof_height=3.3
        )
        assert is_valid is True
        assert error_msg == ""

    def test_zero_height_room(self):
        """Test edge case where floor and roof are at same height."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=1.0, x2=2.0, y2=0.0, z2=1.0)
        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height=1.0, roof_height=1.0
        )
        assert is_valid is True
        assert error_msg == ""

    # --- Clamping: within 15cm tolerance ---

    def test_window_bottom_clamped_to_floor(self):
        """Test BIM window frame extending 10cm below floor is clamped."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=-0.10, x2=1.0, y2=0.0, z2=2.0)
        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height=0.0, roof_height=3.0
        )
        assert is_valid is True
        assert window_geom.z1 == 0.0

    def test_window_top_clamped_to_roof(self):
        """Test BIM window frame extending 10cm above roof is clamped."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=1.0, x2=1.0, y2=0.0, z2=3.10)
        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height=0.0, roof_height=3.0
        )
        assert is_valid is True
        assert window_geom.z2 == 3.0

    def test_window_clamped_at_exact_tolerance_boundary(self):
        """Test window at exactly 15cm deviation is still clamped."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=-0.15, x2=1.0, y2=0.0, z2=2.0)
        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height=0.0, roof_height=3.0
        )
        assert is_valid is True
        assert window_geom.z1 == 0.0

    def test_window_clamped_with_reversed_z(self):
        """Test clamping works when z1 > z2 (reversed coordinates)."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=2.0, x2=1.0, y2=0.0, z2=-0.10)
        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height=0.0, roof_height=3.0
        )
        assert is_valid is True
        assert window_geom.z2 == 0.0

    def test_window_clamped_with_elevated_floor(self):
        """Test clamping with non-zero floor height (BIM realistic)."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=0.2, x2=2.0, y2=0.0, z2=2.0)
        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height=0.3, roof_height=3.3
        )
        assert is_valid is True
        assert window_geom.z1 == 0.3

    def test_window_tiny_deviation_clamped(self):
        """Test very small deviation (1mm) is clamped, not rejected."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=-0.001, x2=1.0, y2=0.0, z2=2.0)
        is_valid, error_msg = WindowHeightValidator.validate_window_height_bounds(
            window_geom, floor_height=0.0, roof_height=3.0
        )
        assert is_valid is True
        assert window_geom.z1 == 0.0

    # --- Errors: beyond 15cm tolerance ---

    def test_window_below_floor_beyond_tolerance(self):
        """Test window 50cm below floor raises error."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=-0.5, x2=1.0, y2=0.0, z2=2.0)
        with pytest.raises(WindowHeightValidationError) as exc_info:
            WindowHeightValidator.validate_window_height_bounds(
                window_geom, floor_height=0.0, roof_height=3.0
            )
        assert "below floor" in str(exc_info.value)

    def test_window_above_roof_beyond_tolerance(self):
        """Test window 50cm above roof raises error."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=2.0, x2=1.0, y2=0.0, z2=3.5)
        with pytest.raises(WindowHeightValidationError) as exc_info:
            WindowHeightValidator.validate_window_height_bounds(
                window_geom, floor_height=0.0, roof_height=3.0
            )
        assert "above roof" in str(exc_info.value)

    def test_window_both_below_and_above_beyond_tolerance(self):
        """Test window extending far beyond both bounds fails on floor check first."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=-1.0, x2=1.0, y2=0.0, z2=4.0)
        with pytest.raises(WindowHeightValidationError) as exc_info:
            WindowHeightValidator.validate_window_height_bounds(
                window_geom, floor_height=0.0, roof_height=3.0
            )
        assert "below floor" in str(exc_info.value)

    def test_window_just_beyond_tolerance(self):
        """Test window at 16cm deviation (just over 15cm) raises error."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=-0.16, x2=1.0, y2=0.0, z2=2.0)
        with pytest.raises(WindowHeightValidationError):
            WindowHeightValidator.validate_window_height_bounds(
                window_geom, floor_height=0.0, roof_height=3.0
            )

    def test_window_below_elevated_floor_beyond_tolerance(self):
        """Test window far below elevated floor raises error."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=1.5, x2=1.0, y2=0.0, z2=3.0)
        with pytest.raises(WindowHeightValidationError) as exc_info:
            WindowHeightValidator.validate_window_height_bounds(
                window_geom, floor_height=2.0, roof_height=5.0
            )
        assert "below floor" in str(exc_info.value)

    def test_realistic_scenario_window_too_low(self):
        """Test realistic scenario where window is far below floor."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=-0.5, x2=2.0, y2=0.0, z2=2.0)
        with pytest.raises(WindowHeightValidationError) as exc_info:
            WindowHeightValidator.validate_window_height_bounds(
                window_geom, floor_height=0.3, roof_height=3.3
            )
        assert "below floor" in str(exc_info.value)

    def test_realistic_scenario_window_too_high(self):
        """Test realistic scenario where window extends far above roof."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=2.0, x2=2.0, y2=0.0, z2=3.5)
        with pytest.raises(WindowHeightValidationError) as exc_info:
            WindowHeightValidator.validate_window_height_bounds(
                window_geom, floor_height=0.3, roof_height=3.3
            )
        assert "above roof" in str(exc_info.value)

    def test_zero_height_room_window_with_height(self):
        """Test window with height in zero-height room (should fail)."""
        window_geom = WindowGeometry(x1=0.0, y1=0.0, z1=1.0, x2=2.0, y2=0.0, z2=1.5)
        with pytest.raises(WindowHeightValidationError) as exc_info:
            WindowHeightValidator.validate_window_height_bounds(
                window_geom, floor_height=1.0, roof_height=1.0
            )
        assert "above roof" in str(exc_info.value)

    # --- validate_from_parameters ---

    def test_validate_from_parameters_valid(self):
        """Test validation from parameter dictionaries with valid window."""
        window_data = {"x1": 0.0, "y1": 0.0, "z1": 1.0, "x2": 1.0, "y2": 0.0, "z2": 2.5}
        is_valid, error_msg = WindowHeightValidator.validate_from_parameters(
            window_data, floor_height=0.0, roof_height=3.0
        )
        assert is_valid is True
        assert error_msg == ""

    def test_validate_from_parameters_below_floor(self):
        """Test validation from parameters with window far below floor."""
        window_data = {"x1": 0.0, "y1": 0.0, "z1": -0.5, "x2": 1.0, "y2": 0.0, "z2": 2.0}
        is_valid, error_msg = WindowHeightValidator.validate_from_parameters(
            window_data, floor_height=0.0, roof_height=3.0
        )
        assert is_valid is False
        assert "below floor" in error_msg

    def test_validate_from_parameters_above_roof(self):
        """Test validation from parameters with window far above roof."""
        window_data = {"x1": 0.0, "y1": 0.0, "z1": 2.0, "x2": 1.0, "y2": 0.0, "z2": 3.5}
        is_valid, error_msg = WindowHeightValidator.validate_from_parameters(
            window_data, floor_height=0.0, roof_height=3.0
        )
        assert is_valid is False
        assert "above roof" in error_msg

    def test_validate_from_parameters_missing_z_coordinate(self):
        """Test validation with missing z coordinate in data."""
        window_data = {"x1": 0.0, "y1": 0.0, "z1": 1.0, "x2": 1.0, "y2": 0.0}
        is_valid, error_msg = WindowHeightValidator.validate_from_parameters(
            window_data, floor_height=0.0, roof_height=3.0
        )
        assert is_valid is False
        assert "Error parsing height data" in error_msg

    def test_validate_from_parameters_invalid_data_type(self):
        """Test validation with invalid data type."""
        window_data = {"x1": "invalid", "y1": 0.0, "z1": 1.0, "x2": 1.0, "y2": 0.0, "z2": 2.0}
        is_valid, error_msg = WindowHeightValidator.validate_from_parameters(
            window_data, floor_height=0.0, roof_height=3.0
        )
        assert is_valid is False
        assert "Error parsing height data" in error_msg
