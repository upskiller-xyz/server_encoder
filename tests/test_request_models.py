"""
Tests for request models

Tests the WindowRequest and RoomEncodingRequest models
to ensure proper parsing, validation, and conversion.
"""
import unittest
from src.models import WindowRequest, RoomEncodingRequest
from src.core import ModelType


class TestWindowRequest(unittest.TestCase):
    """Test WindowRequest model"""

    def test_from_dict_minimal(self):
        """Test parsing minimal window request"""
        data = {
            'x1': 0.0, 'y1': 1.0, 'z1': 2.0,
            'x2': 1.0, 'y2': 2.0, 'z2': 4.0,
            'window_frame_ratio': 0.8
        }

        window = WindowRequest.from_dict(data)

        self.assertEqual(window.x1, 0.0)
        self.assertEqual(window.y1, 1.0)
        self.assertEqual(window.z1, 2.0)
        self.assertEqual(window.x2, 1.0)
        self.assertEqual(window.y2, 2.0)
        self.assertEqual(window.z2, 4.0)
        self.assertEqual(window.window_frame_ratio, 0.8)
        self.assertIsNone(window.direction_angle)

    def test_from_dict_with_optional(self):
        """Test parsing window with optional parameters"""
        data = {
            'x1': 0.0, 'y1': 1.0, 'z1': 2.0,
            'x2': 1.0, 'y2': 2.0, 'z2': 4.0,
            'window_frame_ratio': 0.8,
            'direction_angle': 1.5708,
            'horizon': 45.0,
            'zenith': 30.0
        }

        window = WindowRequest.from_dict(data)

        self.assertEqual(window.direction_angle, 1.5708)
        self.assertEqual(window.horizon, 45.0)
        self.assertEqual(window.zenith, 30.0)

    def test_validate_valid_window(self):
        """Test validation passes for valid window"""
        window = WindowRequest(
            x1=0.0, y1=1.0, z1=2.0,
            x2=1.0, y2=2.0, z2=4.0,
            window_frame_ratio=0.8
        )

        is_valid, error = window.validate()
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

    def test_validate_invalid_frame_ratio(self):
        """Test validation fails for invalid frame ratio"""
        window = WindowRequest(
            x1=0.0, y1=1.0, z1=2.0,
            x2=1.0, y2=2.0, z2=4.0,
            window_frame_ratio=1.5  # Invalid: > 1
        )

        is_valid, error = window.validate()
        self.assertFalse(is_valid)
        self.assertIn("window_frame_ratio", error)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        window = WindowRequest(
            x1=0.0, y1=1.0, z1=2.0,
            x2=1.0, y2=2.0, z2=4.0,
            window_frame_ratio=0.8,
            direction_angle=1.5708
        )

        result = window.to_dict()

        self.assertEqual(result['x1'], 0.0)
        self.assertEqual(result['window_frame_ratio'], 0.8)
        self.assertEqual(result['direction_angle'], 1.5708)


class TestRoomEncodingRequest(unittest.TestCase):
    """Test RoomEncodingRequest model"""

    def test_from_dict_single_window(self):
        """Test parsing request with single window (flat structure)"""
        data = {
            'model_type': 'df_default',
            'parameters': {
                'height_roof_over_floor': 3.0,
                'floor_height_above_terrain': 1.0,
                'room_polygon': [[0, 0], [5, 0], [5, 5], [0, 5]],
                'x1': 0.0, 'y1': 1.0, 'z1': 2.0,
                'x2': 1.0, 'y2': 2.0, 'z2': 4.0,
                'window_frame_ratio': 0.8
            }
        }

        request = RoomEncodingRequest.from_dict(data)

        self.assertEqual(request.model_type, ModelType.DF_DEFAULT)
        self.assertEqual(request.height_roof_over_floor, 3.0)
        self.assertEqual(request.floor_height_above_terrain, 1.0)
        self.assertEqual(len(request.windows), 1)
        self.assertIn('window_0', request.windows)

    def test_from_dict_multiple_windows(self):
        """Test parsing request with multiple windows"""
        data = {
            'model_type': 'df_default',
            'parameters': {
                'height_roof_over_floor': 3.0,
                'floor_height_above_terrain': 1.0,
                'room_polygon': [[0, 0], [5, 0], [5, 5], [0, 5]],
                'windows': {
                    'window_1': {
                        'x1': 0.0, 'y1': 1.0, 'z1': 2.0,
                        'x2': 1.0, 'y2': 2.0, 'z2': 4.0,
                        'window_frame_ratio': 0.8
                    },
                    'window_2': {
                        'x1': 3.0, 'y1': 1.0, 'z1': 2.0,
                        'x2': 4.0, 'y2': 2.0, 'z2': 4.0,
                        'window_frame_ratio': 0.7
                    }
                }
            }
        }

        request = RoomEncodingRequest.from_dict(data)

        self.assertEqual(len(request.windows), 2)
        self.assertIn('window_1', request.windows)
        self.assertIn('window_2', request.windows)
        self.assertEqual(request.windows['window_1'].window_frame_ratio, 0.8)
        self.assertEqual(request.windows['window_2'].window_frame_ratio, 0.7)

    def test_validate_valid_request(self):
        """Test validation passes for valid request"""
        request = RoomEncodingRequest(
            model_type=ModelType.DF_DEFAULT,
            height_roof_over_floor=3.0,
            floor_height_above_terrain=1.0,
            windows={
                'window_1': WindowRequest(
                    x1=0.0, y1=1.0, z1=2.0,
                    x2=1.0, y2=2.0, z2=4.0,
                    window_frame_ratio=0.8
                )
            }
        )

        is_valid, error = request.validate()
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

    def test_validate_negative_roof_height(self):
        """Test validation fails for negative roof height"""
        request = RoomEncodingRequest(
            model_type=ModelType.DF_DEFAULT,
            height_roof_over_floor=-1.0,  # Invalid
            floor_height_above_terrain=1.0,
            windows={
                'window_1': WindowRequest(
                    x1=0.0, y1=1.0, z1=2.0,
                    x2=1.0, y2=2.0, z2=4.0,
                    window_frame_ratio=0.8
                )
            }
        )

        is_valid, error = request.validate()
        self.assertFalse(is_valid)
        self.assertIn("height_roof_over_floor", error)

    def test_validate_no_windows(self):
        """Test validation fails when no windows provided"""
        request = RoomEncodingRequest(
            model_type=ModelType.DF_DEFAULT,
            height_roof_over_floor=3.0,
            floor_height_above_terrain=1.0,
            windows={}  # No windows
        )

        is_valid, error = request.validate()
        self.assertFalse(is_valid)
        self.assertIn("window", error.lower())

    def test_to_flat_dict_single_window(self):
        """Test conversion to flat dictionary for single window"""
        request = RoomEncodingRequest(
            model_type=ModelType.DF_DEFAULT,
            height_roof_over_floor=3.0,
            floor_height_above_terrain=1.0,
            room_polygon=[[0, 0], [5, 0], [5, 5], [0, 5]],
            windows={
                'window_1': WindowRequest(
                    x1=0.0, y1=1.0, z1=2.0,
                    x2=1.0, y2=2.0, z2=4.0,
                    window_frame_ratio=0.8
                )
            }
        )

        result = request.to_flat_dict()

        # Should have flat structure for single window
        self.assertEqual(result['height_roof_over_floor'], 3.0)
        self.assertEqual(result['x1'], 0.0)
        self.assertEqual(result['window_frame_ratio'], 0.8)
        self.assertNotIn('windows', result)  # Should be flat

    def test_to_flat_dict_multiple_windows(self):
        """Test conversion to flat dictionary for multiple windows"""
        request = RoomEncodingRequest(
            model_type=ModelType.DF_DEFAULT,
            height_roof_over_floor=3.0,
            floor_height_above_terrain=1.0,
            windows={
                'window_1': WindowRequest(
                    x1=0.0, y1=1.0, z1=2.0,
                    x2=1.0, y2=2.0, z2=4.0,
                    window_frame_ratio=0.8
                ),
                'window_2': WindowRequest(
                    x1=3.0, y1=1.0, z1=2.0,
                    x2=4.0, y2=2.0, z2=4.0,
                    window_frame_ratio=0.7
                )
            }
        )

        result = request.to_flat_dict()

        # Should have nested structure for multiple windows
        self.assertIn('windows', result)
        self.assertEqual(len(result['windows']), 2)
        self.assertIn('window_1', result['windows'])
        self.assertIn('window_2', result['windows'])


if __name__ == '__main__':
    unittest.main()
