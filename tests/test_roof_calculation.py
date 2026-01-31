"""Test roof height calculation with high floor_height_above_terrain values."""

import pytest
from src.components.encoding_service import EncodingService
from src.components.enums import ModelType
from src.server.services.logging import StructuredLogger
from src.server.enums import LogLevel


class TestRoofHeightCalculation:
    """Test that roof height is calculated correctly before clipping floor_height_above_terrain."""

    @pytest.fixture
    def encoding_service(self):
        """Create encoding service instance."""
        logger = StructuredLogger("test", LogLevel.INFO)
        return EncodingService(logger)

    def test_window_validation_uses_original_floor_height(self, encoding_service):
        """
        Test that window height validation uses original floor_height_above_terrain,
        not the clipped value.

        floor_height_above_terrain = 17.1m (will be clipped to 10.0m during encoding)
        height_roof_over_floor = 2.7m
        Roof height should be calculated as 17.1 + 2.7 = 19.8m for validation
        NOT 10.0 + 2.7 = 12.7m

        Window z-coordinates: z1=18.0m, z2=19.2m
        Window should be valid since 18.0-19.2m is within 17.1-19.8m
        """
        parameters = {
            "height_roof_over_floor": 2.7,
            "floor_height_above_terrain": 17.1,
            "room_polygon": [[0, 2], [0, -7], [-3, -7], [-3, 2]],
            "windows": {
                "test_window": {
                    "x1": 0, "y1": 0.2, "z1": 18.0,
                    "x2": 0, "y2": 1.8, "z2": 19.2,
                    "window_frame_ratio": 0.2,
                    "horizon": 0,
                    "zenith": 0,
                }
            }
        }

        # Should pass validation - window is within original roof bounds
        is_valid, error_msg = encoding_service.validate_parameters(
            parameters, ModelType.DF_DEFAULT
        )

        assert is_valid, (
            f"Window validation should pass with high floor_height_above_terrain. "
            f"Error: {error_msg}"
        )

    def test_clipping_happens_during_encoding(self, encoding_service):
        """
        Test that floor_height_above_terrain is clipped during encoding,
        not during validation.
        """
        parameters = {
            "height_roof_over_floor": 2.7,
            "floor_height_above_terrain": 17.1,  # Will be clipped to 10.0
            "room_polygon": [[0, 2], [0, -7], [-3, -7], [-3, 2]],
            "windows": {
                "test_window": {
                    "x1": 0, "y1": 0.2, "z1": 18.0,
                    "x2": 0, "y2": 1.8, "z2": 19.2,
                    "window_frame_ratio": 0.2,
                    "horizon": 0,
                    "zenith": 0,
                }
            }
        }

        # Validation should pass
        is_valid, error_msg = encoding_service.validate_parameters(
            parameters, ModelType.DF_DEFAULT
        )
        assert is_valid, f"Validation failed: {error_msg}"

        # Encoding should also work (with clipped values)
        try:
            image_bytes = encoding_service.encode_room_image(
                parameters, ModelType.DF_DEFAULT
            )
            assert image_bytes is not None
        except ValueError as e:
            pytest.fail(f"Encoding failed: {e}")

    def test_window_above_original_roof_fails_validation(self, encoding_service):
        """
        Test that window above the original roof height fails validation.

        floor_height = 17.1m, height_roof_over_floor = 2.7m
        Roof = 19.8m
        Window z2 = 20.0m (above roof)
        Should fail validation
        """
        parameters = {
            "height_roof_over_floor": 2.7,
            "floor_height_above_terrain": 17.1,
            "room_polygon": [[0, 2], [0, -7], [-3, -7], [-3, 2]],
            "windows": {
                "test_window": {
                    "x1": 0, "y1": 0.2, "z1": 18.0,
                    "x2": 0, "y2": 1.8, "z2": 20.0,  # Above roof
                    "window_frame_ratio": 0.2,
                    "horizon": 0,
                    "zenith": 0,
                }
            }
        }

        is_valid, error_msg = encoding_service.validate_parameters(
            parameters, ModelType.DF_DEFAULT
        )

        assert not is_valid, "Window above roof should fail validation"
        assert "above roof" in error_msg.lower()
        assert "19.8" in error_msg  # Should show correct roof height
