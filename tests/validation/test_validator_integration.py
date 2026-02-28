"""Integration tests for validator usage in encoding service"""
import pytest
from src.server.services.encoding_service import EncodingService
from src.core.enums import EncodingScheme


class TestValidatorIntegration:
    """Test that encoding service properly uses validators"""

    @pytest.fixture
    def encoding_service(self):
        """Create encoding service instance"""
        return EncodingService(encoding_scheme=EncodingScheme.HSV)

    def test_calculate_direction_missing_room_polygon(self, encoding_service):
        """Test that validator catches missing room_polygon"""
        parameters = {
            "windows": {
                "window1": {
                    "x1": -0.6, "y1": 0.0,
                    "x2": 0.6, "y2": 0.0
                }
            }
        }

        with pytest.raises(ValueError) as exc_info:
            encoding_service.calculate_direction_angle(parameters)

        assert "room_polygon" in str(exc_info.value).lower()

    def test_calculate_direction_missing_windows(self, encoding_service):
        """Test that validator catches missing windows"""
        parameters = {
            "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]
        }

        with pytest.raises(ValueError) as exc_info:
            encoding_service.calculate_direction_angle(parameters)

        assert "windows" in str(exc_info.value).lower()

    def test_calculate_direction_invalid_window_coordinates(self, encoding_service):
        """Test that validator catches missing window coordinates"""
        parameters = {
            "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "windows": {
                "window1": {
                    "x1": -0.6  # Missing y1, x2, y2
                }
            }
        }

        with pytest.raises(ValueError) as exc_info:
            encoding_service.calculate_direction_angle(parameters)

        assert "coordinates" in str(exc_info.value).lower()

    def test_calculate_reference_point_missing_3d_coordinates(self, encoding_service):
        """Test that validator catches missing 3D coordinates for reference point"""
        parameters = {
            "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "windows": {
                "window1": {
                    "x1": -0.6, "y1": 0.0,  # Missing z1, x2, y2, z2
                    "x2": 0.6, "y2": 0.0
                }
            }
        }

        with pytest.raises(ValueError) as exc_info:
            encoding_service.calculate_reference_point(parameters)

        assert "coordinates" in str(exc_info.value).lower()

    def test_calculate_external_reference_point_missing_3d_coordinates(self, encoding_service):
        """Test that validator catches missing 3D coordinates for external reference point"""
        parameters = {
            "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "windows": {
                "window1": {
                    "x1": -0.6, "y1": 0.0,  # Missing z1, x2, y2, z2
                    "x2": 0.6, "y2": 0.0
                }
            }
        }

        with pytest.raises(ValueError) as exc_info:
            encoding_service.calculate_external_reference_point(parameters)

        assert "coordinates" in str(exc_info.value).lower()

    def test_calculate_direction_valid_request(self, encoding_service):
        """Test that validator allows valid direction calculation request"""
        parameters = {
            "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "windows": {
                "window1": {
                    "x1": -0.6, "y1": 0.0,
                    "x2": 0.6, "y2": 0.0
                }
            }
        }

        # Should not raise
        result = encoding_service.calculate_direction_angle(parameters)
        assert isinstance(result, dict)
        assert "window1" in result

    def test_polygon_validator_minimum_vertices(self, encoding_service):
        """Test that validator catches polygons with < 3 vertices"""
        parameters = {
            "room_polygon": [[0, 0], [10, 0]],  # Only 2 vertices
            "windows": {
                "window1": {
                    "x1": -0.6, "y1": 0.0,
                    "x2": 0.6, "y2": 0.0
                }
            }
        }

        with pytest.raises(ValueError) as exc_info:
            encoding_service.calculate_direction_angle(parameters)

        assert "vertices" in str(exc_info.value).lower() or "length" in str(exc_info.value).lower()
