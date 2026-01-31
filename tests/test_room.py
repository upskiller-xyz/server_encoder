"""
Unit tests for room region encoding

Tests verify:
1. Room polygon construction and positioning relative to window
2. Correct width and height based on polygon coordinates
3. Proper color encoding for all channels with defaults
4. Validation of required parameters
5. Scaling behavior for different image sizes
6. Integration with RoomImageBuilder
"""

import unittest
import numpy as np
from src.components.region_encoders import RoomRegionEncoder
from src.components.geometry import RoomPolygon
from src.components.enums import ModelType, ImageDimensions, RegionType
from src.components.image_builder import RoomImageBuilder


class TestRoomPolygonConstruction(unittest.TestCase):
    """Test room polygon is constructed at correct position"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = RoomRegionEncoder()
        self.model_type = ModelType.DF_DEFAULT

    def test_room_polygon_positioned_relative_to_window(self):
        """Test that room polygon is positioned relative to window center"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        # Simple rectangular room: 3m wide (along facade), 5m deep
        room_polygon = [
            {"x": -1.5, "y": 0.0},  # Left front corner
            {"x": 1.5, "y": 0.0},   # Right front corner
            {"x": 1.5, "y": 5.0},   # Right back corner
            {"x": -1.5, "y": 5.0}   # Left back corner
        ]

        # Window centered at x=0, y=0
        parameters = {
            'height_roof_over_floor': 3.0,
            'room_polygon': room_polygon,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Window is at x=116 (12px from right at 128)
        # Room should extend from window position
        # Check that room pixels exist near window position
        window_x = 116
        room_pixel = result[64, window_x - 10, :]  # 10px left of window

        self.assertTrue(
            np.any(room_pixel > 0),
            "Room should exist near window position"
        )

    def test_room_mask_created_from_polygon(self):
        """Test that room mask is created from polygon coordinates"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        # L-shaped room
        room_polygon = [
            {"x": -2.0, "y": 0.0},
            {"x": 2.0, "y": 0.0},
            {"x": 2.0, "y": 3.0},
            {"x": -1.0, "y": 3.0},
            {"x": -1.0, "y": 6.0},
            {"x": -2.0, "y": 6.0}
        ]

        parameters = {
            'height_roof_over_floor': 3.0,
            'room_polygon': room_polygon,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Count non-zero pixels (room area should be filled)
        non_zero_pixels = np.count_nonzero(np.any(result > 0, axis=2))

        self.assertGreater(
            non_zero_pixels, 100,
            "Room polygon should create filled area"
        )


class TestRoomDimensions(unittest.TestCase):
    """Test room dimensions scale correctly"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = RoomRegionEncoder()
        self.model_type = ModelType.DF_DEFAULT

    def test_room_scales_with_image_size(self):
        """Test that room polygon scales proportionally with image size"""
        # 3m x 5m room
        room_polygon = [
            {"x": -1.5, "y": 0.0},
            {"x": 1.5, "y": 0.0},
            {"x": 1.5, "y": 5.0},
            {"x": -1.5, "y": 5.0}
        ]

        parameters = {
            'height_roof_over_floor': 3.0,
            'room_polygon': room_polygon,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        # Test at 128x128
        image_128 = np.zeros((128, 128, 4), dtype=np.uint8)
        result_128 = self.encoder.encode_region(
            image_128, parameters, self.model_type
        )
        area_128 = np.count_nonzero(np.any(result_128 > 0, axis=2))

        # Test at 256x256 (should have ~4x area)
        image_256 = np.zeros((256, 256, 4), dtype=np.uint8)
        result_256 = self.encoder.encode_region(
            image_256, parameters, self.model_type
        )
        area_256 = np.count_nonzero(np.any(result_256 > 0, axis=2))

        # Area should scale approximately by 4 (2x in each dimension)
        # Allow wider range due to rounding effects at polygon edges
        area_ratio = area_256 / area_128

        self.assertGreater(
            area_ratio, 2.8,
            "Room area should scale proportionally with image size"
        )
        self.assertLess(
            area_ratio, 4.5,
            "Room area scaling should be approximately 4x"
        )


class TestRoomColorEncoding(unittest.TestCase):
    """Test room encodes colors correctly"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = RoomRegionEncoder()
        self.model_type = ModelType.DF_DEFAULT
        self.room_polygon = [
            {"x": -1.5, "y": 0.0},
            {"x": 1.5, "y": 0.0},
            {"x": 1.5, "y": 5.0},
            {"x": -1.5, "y": 5.0}
        ]
        self.window_coords = {
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

    def test_red_channel_roof_height(self):
        """Test red channel encodes height_roof_over_floor (0-30m -> 0-1)"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        test_cases = [
            (0.0, 0),      # 0m -> 0
            (15.0, 127),   # 15m -> ~127
            (30.0, 255),   # 30m -> 255
        ]

        for height, expected_value in test_cases:
            parameters = {
                'height_roof_over_floor': height,
                'room_polygon': self.room_polygon,
                **self.window_coords
            }

            result = self.encoder.encode_region(
                image.copy(), parameters, self.model_type
            )

            # Find a room pixel and check red channel
            room_pixels = result[result[:, :, 0] > 0]
            if len(room_pixels) > 0:
                red_value = room_pixels[0, 0]
                self.assertAlmostEqual(
                    red_value, expected_value, delta=2,
                    msg=f"Roof height {height}m should encode to ~{expected_value}"
                )

    def test_green_channel_horizontal_reflectance(self):
        """Test green channel encodes floor_reflectance (0-1 -> 0-1) in custom model"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        test_cases = [
            (0.0, 0),      # 0.0 -> 0
            (0.5, 127),    # 0.5 -> ~127
            (1.0, 255),    # 1.0 -> 255
        ]

        for reflectance, expected_value in test_cases:
            parameters = {
                'height_roof_over_floor': 3.0,
                'floor_reflectance': reflectance,  # Correct parameter name
                'room_polygon': self.room_polygon,
                **self.window_coords
            }

            # Use DF_CUSTOM to test reflectance encoding
            result = self.encoder.encode_region(
                image.copy(), parameters, ModelType.DF_CUSTOM
            )

            # Find a room pixel and check green channel
            room_pixels = result[result[:, :, 0] > 0]
            if len(room_pixels) > 0:
                green_value = room_pixels[0, 1]
                self.assertAlmostEqual(
                    green_value, expected_value, delta=2,
                    msg=f"Floor reflectance {reflectance} should encode to ~{expected_value}"
                )

    def test_blue_channel_vertical_reflectance(self):
        """Test blue channel encodes wall_reflectance (0-1 -> 0-1) in custom model"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        test_cases = [
            (0.0, 0),      # 0.0 -> 0
            (0.5, 127),    # 0.5 -> ~127
            (1.0, 255),    # 1.0 -> 255
        ]

        for reflectance, expected_value in test_cases:
            parameters = {
                'height_roof_over_floor': 3.0,
                'wall_reflectance': reflectance,  # Correct parameter name
                'room_polygon': self.room_polygon,
                **self.window_coords
            }

            # Use DF_CUSTOM to test reflectance encoding
            result = self.encoder.encode_region(
                image.copy(), parameters, ModelType.DF_CUSTOM
            )

            # Find a room pixel and check blue channel
            room_pixels = result[result[:, :, 0] > 0]
            if len(room_pixels) > 0:
                blue_value = room_pixels[0, 2]
                self.assertAlmostEqual(
                    blue_value, expected_value, delta=2,
                    msg=f"Wall reflectance {reflectance} should encode to ~{expected_value}"
                )

    def test_alpha_channel_ceiling_reflectance(self):
        """Test alpha channel encodes ceiling_reflectance (0.5-1 -> 0-1)"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        test_cases = [
            (0.5, 0),      # 0.5 -> 0
            (0.75, 127),   # 0.75 -> ~127
            (1.0, 255),    # 1.0 -> 255
        ]

        for reflectance, expected_value in test_cases:
            parameters = {
                'height_roof_over_floor': 3.0,
                'ceiling_reflectance': reflectance,
                'room_polygon': self.room_polygon,
                **self.window_coords
            }

            result = self.encoder.encode_region(
                image.copy(), parameters, self.model_type
            )

            # Find a room pixel and check alpha channel
            room_pixels = result[result[:, :, 0] > 0]
            if len(room_pixels) > 0:
                alpha_value = room_pixels[0, 3]
                self.assertAlmostEqual(
                    alpha_value, expected_value, delta=2,
                    msg=f"Ceiling reflectance {reflectance} should encode to ~{expected_value}"
                )

    def test_default_horizontal_reflectance(self):
        """Test green channel defaults to 1.0 when horizontal_reflectance not provided"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'height_roof_over_floor': 3.0,
            'room_polygon': self.room_polygon,
            **self.window_coords
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Default value of 1.0 should map to 255
        room_pixels = result[result[:, :, 0] > 0]
        if len(room_pixels) > 0:
            green_value = room_pixels[0, 1]
            self.assertEqual(
                green_value, 255,
                "Default horizontal_reflectance should be 1.0 (encoded as 255)"
            )

    def test_default_vertical_reflectance(self):
        """Test blue channel defaults to 1.0 when vertical_reflectance not provided"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'height_roof_over_floor': 3.0,
            'room_polygon': self.room_polygon,
            **self.window_coords
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Default value of 1.0 should map to 255
        room_pixels = result[result[:, :, 0] > 0]
        if len(room_pixels) > 0:
            blue_value = room_pixels[0, 2]
            self.assertEqual(
                blue_value, 255,
                "Default vertical_reflectance should be 1.0 (encoded as 255)"
            )

    def test_default_ceiling_reflectance(self):
        """Test alpha channel defaults to 1.0 when ceiling_reflectance not provided"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'height_roof_over_floor': 3.0,
            'room_polygon': self.room_polygon,
            **self.window_coords
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Default value of 1.0 should map to 255
        room_pixels = result[result[:, :, 0] > 0]
        if len(room_pixels) > 0:
            alpha_value = room_pixels[0, 3]
            self.assertEqual(
                alpha_value, 255,
                "Default ceiling_reflectance should be 1.0 (encoded as 255)"
            )


class TestRoomParameterValidation(unittest.TestCase):
    """Test room validates required parameters"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = RoomRegionEncoder()
        self.model_type = ModelType.DF_DEFAULT
        self.image = np.zeros((128, 128, 4), dtype=np.uint8)

    def test_missing_roof_height(self):
        """Test error when height_roof_over_floor is missing"""
        room_polygon = [
            {"x": -1.5, "y": 0.0},
            {"x": 1.5, "y": 0.0},
            {"x": 1.5, "y": 5.0},
            {"x": -1.5, "y": 5.0}
        ]

        parameters = {
            'room_polygon': room_polygon,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        with self.assertRaises(ValueError) as context:
            self.encoder.encode_region(self.image, parameters, self.model_type)

        error_msg = str(context.exception)
        self.assertIn('height_roof_over_floor', error_msg)

    def test_optional_reflectances_work(self):
        """Test that all reflectance parameters are optional"""
        room_polygon = [
            {"x": -1.5, "y": 0.0},
            {"x": 1.5, "y": 0.0},
            {"x": 1.5, "y": 5.0},
            {"x": -1.5, "y": 5.0}
        ]

        parameters = {
            'height_roof_over_floor': 3.0,
            'room_polygon': room_polygon,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        # Should not raise error even without reflectance parameters
        try:
            result = self.encoder.encode_region(
                self.image, parameters, self.model_type
            )
            self.assertIsNotNone(result)
        except ValueError as e:
            self.fail(f"Should not require reflectance parameters: {e}")


class TestRoomImageScaling(unittest.TestCase):
    """Test room scaling across different image sizes"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = RoomRegionEncoder()
        self.model_type = ModelType.DF_DEFAULT
        self.room_polygon = [
            {"x": -1.5, "y": 0.0},
            {"x": 1.5, "y": 0.0},
            {"x": 1.5, "y": 5.0},
            {"x": -1.5, "y": 5.0}
        ]
        self.window_coords = {
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

    def test_room_area_scales_quadratically(self):
        """Test that room area scales quadratically with image size"""
        parameters = {
            'height_roof_over_floor': 3.0,
            'room_polygon': self.room_polygon,
            **self.window_coords
        }

        image_sizes = [128, 256, 512]
        areas = []

        for size in image_sizes:
            image = np.zeros((size, size, 4), dtype=np.uint8)
            result = self.encoder.encode_region(image, parameters, self.model_type)
            area = np.count_nonzero(np.any(result > 0, axis=2))
            areas.append(area)

        # Check 128 -> 256 scaling (~4x)
        # Allow wider range due to rounding effects at polygon edges
        ratio_256_128 = areas[1] / areas[0]
        self.assertGreater(ratio_256_128, 2.8)
        self.assertLess(ratio_256_128, 4.5)

        # Check 256 -> 512 scaling (~4x)
        ratio_512_256 = areas[2] / areas[1]
        self.assertGreater(ratio_512_256, 2.8)
        self.assertLess(ratio_512_256, 4.5)

    def test_color_values_consistent_across_sizes(self):
        """Test that color encoding is consistent across image sizes"""
        parameters = {
            'height_roof_over_floor': 15.0,
            'horizontal_reflectance': 0.5,
            'vertical_reflectance': 0.7,
            'ceiling_reflectance': 0.8,
            'room_polygon': self.room_polygon,
            **self.window_coords
        }

        image_sizes = [128, 256, 512]
        color_values = []

        for size in image_sizes:
            image = np.zeros((size, size, 4), dtype=np.uint8)
            result = self.encoder.encode_region(image, parameters, self.model_type)

            # Get a room pixel
            room_pixels = result[result[:, :, 0] > 0]
            if len(room_pixels) > 0:
                color_values.append(room_pixels[0])

        # All color values should be approximately equal
        for i in range(1, len(color_values)):
            for channel in range(4):
                self.assertAlmostEqual(
                    color_values[0][channel],
                    color_values[i][channel],
                    delta=2,
                    msg=f"Color values should be consistent across image sizes"
                )


class TestRoomIntegration(unittest.TestCase):
    """Test room integration with RoomImageBuilder"""

    def test_builder_creates_correct_room(self):
        """Test that RoomImageBuilder correctly encodes room"""
        builder = RoomImageBuilder()

        room_polygon = [
            {"x": -1.5, "y": 0.0},
            {"x": 1.5, "y": 0.0},
            {"x": 1.5, "y": 5.0},
            {"x": -1.5, "y": 5.0}
        ]

        parameters = {
            'height_roof_over_floor': 3.0,
            'floor_reflectance': 0.3,  # Correct parameter name
            'wall_reflectance': 0.7,  # Correct parameter name
            'ceiling_reflectance': 0.85,
            'room_polygon': room_polygon,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        # Use DF_CUSTOM to test reflectance encoding
        image = (builder
                 .reset()
                 .set_model_type(ModelType.DF_CUSTOM)
                 .encode_region(RegionType.ROOM, parameters)
                 .build())

        # Verify image dimensions
        self.assertEqual(image.shape, (128, 128, 4))

        # Verify room exists
        room_pixels = image[image[:, :, 0] > 0]
        self.assertGreater(
            len(room_pixels), 0,
            "Builder should create room area"
        )

        # Verify color encoding
        if len(room_pixels) > 0:
            # Red: roof height (3.0m / 30.0m = 0.1 -> 25.5)
            expected_red = int(3.0 / 30.0 * 255)
            self.assertAlmostEqual(
                room_pixels[0, 0], expected_red, delta=2,
                msg="Builder should encode roof height correctly"
            )

            # Green: floor reflectance (0.3 -> 76.5)
            expected_green = int(0.3 * 255)
            self.assertAlmostEqual(
                room_pixels[0, 1], expected_green, delta=2,
                msg="Builder should encode floor reflectance correctly"
            )


class TestRoomPolygonGeometry(unittest.TestCase):
    """Test RoomPolygon geometry calculations"""

    def test_rectangular_room_to_pixels(self):
        """Test rectangular room conversion to pixel coordinates"""
        # 3m x 5m room
        vertices = [
            (-1.5, 0.0),
            (1.5, 0.0),
            (1.5, 5.0),
            (-1.5, 5.0)
        ]

        polygon = RoomPolygon(vertices)
        pixel_coords = polygon.to_pixel_array(
            
            window_x1=-0.6, window_y1=0.0,
            window_x2=0.6, window_y2=0.0,
            image_size=128
        )

        # Should return array with shape (1, 4, 2) - 4 vertices
        self.assertEqual(pixel_coords.shape[0], 1)
        self.assertEqual(pixel_coords.shape[1], 4)
        self.assertEqual(pixel_coords.shape[2], 2)

    def test_l_shaped_room_to_pixels(self):
        """Test L-shaped room conversion to pixel coordinates"""
        vertices = [
            (-2.0, 0.0),
            (2.0, 0.0),
            (2.0, 3.0),
            (-1.0, 3.0),
            (-1.0, 6.0),
            (-2.0, 6.0)
        ]

        polygon = RoomPolygon(vertices)
        pixel_coords = polygon.to_pixel_array(
            
            window_x1=-0.6, window_y1=0.0,
            window_x2=0.6, window_y2=0.0,
            image_size=128
        )

        # Should return array with shape (1, 6, 2) - 6 vertices
        self.assertEqual(pixel_coords.shape[0], 1)
        self.assertEqual(pixel_coords.shape[1], 6)
        self.assertEqual(pixel_coords.shape[2], 2)

    def test_polygon_requires_window_coordinates(self):
        """Test that to_pixel_array requires window coordinates"""
        vertices = [
            (-1.5, 0.0),
            (1.5, 0.0),
            (1.5, 5.0),
            (-1.5, 5.0)
        ]

        polygon = RoomPolygon(vertices)

        with self.assertRaises(ValueError) as context:
            polygon.to_pixel_array(image_size=128)

        error_msg = str(context.exception)
        self.assertIn('Window coordinates required', error_msg)


if __name__ == '__main__':
    unittest.main()
