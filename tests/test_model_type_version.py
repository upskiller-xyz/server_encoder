"""
Unit tests for model type version suffix handling.
"""

import pytest
from src.main import ServerApplication


class TestModelTypeVersionParsing:
    """Test model type version suffix extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.server = ServerApplication()

    def test_extract_version_suffix_major_minor_patch(self):
        """Test extraction with major.minor.patch version."""
        result = self.server._extract_model_type_prefix("df_default_2.0.1")
        assert result == "df_default"

    def test_extract_version_suffix_major_minor(self):
        """Test extraction with major.minor version."""
        result = self.server._extract_model_type_prefix("da_custom_1.5")
        assert result == "da_custom"

    def test_no_version_suffix(self):
        """Test model type without version suffix."""
        result = self.server._extract_model_type_prefix("df_default")
        assert result == "df_default"

    def test_all_model_types_with_version(self):
        """Test all valid model types with version suffixes."""
        test_cases = [
            ("df_default_2.0.1", "df_default"),
            ("da_default_2.0.1", "da_default"),
            ("df_custom_1.5.2", "df_custom"),
            ("da_custom_3.0", "da_custom"),
        ]

        for input_str, expected in test_cases:
            result = self.server._extract_model_type_prefix(input_str)
            assert result == expected, f"Failed for input: {input_str}"

    def test_version_with_double_digits(self):
        """Test version with double-digit version numbers."""
        result = self.server._extract_model_type_prefix("df_default_12.34.56")
        assert result == "df_default"

    def test_version_in_middle_not_removed(self):
        """Test that version-like patterns in the middle are not removed."""
        # This shouldn't match because version pattern requires it at the end
        result = self.server._extract_model_type_prefix("df_2.0_default")
        assert result == "df_2.0_default"

    def test_single_digit_versions(self):
        """Test with single digit versions."""
        result = self.server._extract_model_type_prefix("df_default_1.0")
        assert result == "df_default"

    def test_multiple_underscores_preserved(self):
        """Test that underscores in model name are preserved."""
        # Hypothetical future model type
        result = self.server._extract_model_type_prefix("df_ultra_custom_2.0.1")
        assert result == "df_ultra_custom"

    def test_empty_string(self):
        """Test with empty string."""
        result = self.server._extract_model_type_prefix("")
        assert result == ""

    def test_version_only(self):
        """Test with version-like string only."""
        result = self.server._extract_model_type_prefix("_1.0")
        assert result == ""


class TestModelTypeIntegration:
    """Test model type parsing in the actual endpoint."""

    def setup_method(self):
        """Set up test fixtures."""
        self.server = ServerApplication()
        self.client = self.server._app.test_client()

    def test_encode_with_versioned_model_type(self):
        """Test encoding request with versioned model type."""
        payload = {
            "model_type": "df_default_2.0.1",
            "parameters": {
                "height_roof_over_floor": 3.0,
                "floor_height_above_terrain": 0.3,
                "room_polygon": [[0, 0], [0, -5], [-3, -5], [-3, 0]],
                "obstruction_angle_horizon": [30, 45],
                "obstruction_angle_zenith": [20, 35],
                "windows": {
                    "window_1": {
                        "window_frame_ratio": 0.8,
                        "x1": -2.5, "y1": 0.0, "z1": 0.9,
                        "x2": -0.5, "y2": 0.0, "z2": 2.4
                    }
                }
            }
        }

        response = self.client.post("/encode", json=payload)

        # Should succeed (200 OK)
        assert response.status_code == 200
        assert response.content_type == "image/png"

    def test_encode_with_invalid_versioned_model_type(self):
        """Test encoding request with invalid versioned model type."""
        payload = {
            "model_type": "invalid_type_2.0.1",
            "parameters": {}  # Empty parameters - will fail but at model type validation first
        }

        response = self.client.post("/encode", json=payload)

        # Should fail (400 Bad Request) with model type error
        assert response.status_code == 400
        # Response might be HTML or JSON depending on error handling
        response_text = response.get_data(as_text=True)
        assert "Invalid model_type" in response_text or "invalid_type" in response_text

    def test_encode_without_version_suffix(self):
        """Test that model types without version still work."""
        payload = {
            "model_type": "df_default",
            "parameters": {
                "height_roof_over_floor": 3.0,
                "floor_height_above_terrain": 0.3,
                "room_polygon": [[0, 0], [0, -5], [-3, -5], [-3, 0]],
                "obstruction_angle_horizon": [30, 45],
                "obstruction_angle_zenith": [20, 35],
                "windows": {
                    "window_1": {
                        "window_frame_ratio": 0.8,
                        "x1": -2.5, "y1": 0.0, "z1": 0.9,
                        "x2": -0.5, "y2": 0.0, "z2": 2.4
                    }
                }
            }
        }

        response = self.client.post("/encode", json=payload)

        # Should succeed (200 OK)
        assert response.status_code == 200
        assert response.content_type == "image/png"

    def test_all_model_types_with_versions(self):
        """Test all DF model types work with version suffixes (DA models require window_orientation)."""
        base_payload = {
            "parameters": {
                "height_roof_over_floor": 3.0,
                "floor_height_above_terrain": 0.3,
                "room_polygon": [[0, 0], [0, -5], [-3, -5], [-3, 0]],
                "obstruction_angle_horizon": [30, 45],
                "obstruction_angle_zenith": [20, 35],
                "windows": {
                    "window_1": {
                        "window_frame_ratio": 0.8,
                        "x1": -2.5, "y1": 0.0, "z1": 0.9,
                        "x2": -0.5, "y2": 0.0, "z2": 2.4
                    }
                }
            }
        }

        # Only test DF models (DA models need additional window_orientation parameter)
        model_types = [
            "df_default_2.0.1",
            "df_custom_3.0.0",
        ]

        for model_type in model_types:
            payload = {"model_type": model_type, **base_payload}
            response = self.client.post("/encode", json=payload)
            assert response.status_code == 200, f"Failed for model_type: {model_type}"
