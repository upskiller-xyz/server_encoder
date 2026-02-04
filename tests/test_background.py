"""
Unit tests for background region encoding

Tests verify:
1. Background fills correct area (excludes obstruction bar)
2. Proper color encoding for all channels with defaults
3. Validation of required parameters
4. Unique output range for floor_height_above_terrain (0.1-1)
5. Default values for optional parameters
6. Integration with RoomImageBuilder
"""

import math
import unittest

import numpy as np

from src.components.region_encoders import BackgroundRegionEncoder
from src.components.enums import ModelType, ImageDimensions, RegionType
from src.components.image_builder import RoomImageBuilder


class TestBackgroundConstruction(unittest.TestCase):
    """Test background is constructed in correct area"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = BackgroundRegionEncoder()
        self.model_type = ModelType.DF_DEFAULT

    def test_background_fills_except_obstruction_bar(self):
        """Test that background fills entire image (obstruction bar can be overwritten later)"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'floor_height_above_terrain': 2.0
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Background should exist at left side
        bg_pixel = result[64, 60, :]
        self.assertTrue(
            np.any(bg_pixel > 0),
            "Background should exist on left side of image"
        )

        # Background should fill entire image (including where obstruction bar will be)
        # The obstruction bar region can be overwritten by obstruction bar encoder later
        dims = ImageDimensions(128)
        bar_x_start, _, _, _ = dims.get_obstruction_bar_position()
        bar_pixel = result[64, bar_x_start, :]

        # Background should exist here too (will be overwritten by obstruction bar later)
        self.assertTrue(
            np.any(bar_pixel > 0),
            "Background should fill entire image including obstruction bar area"
        )


class TestBackgroundColorEncoding(unittest.TestCase):
    """Test background encodes colors correctly"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = BackgroundRegionEncoder()
        self.model_type = ModelType.DF_DEFAULT

    def test_green_channel_floor_height_above_terrain(self):
        """Test green channel encodes floor_height_above_terrain (0-10m -> 0.1-1)"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        test_cases = [
            (0.0, 25),     # 0m -> 0.1*255 = 25.5
            (5.0, 140),    # 5m -> 0.55*255 = 140.25
            (10.0, 255),   # 10m -> 1.0*255 = 255
        ]

        for height, expected_value in test_cases:
            parameters = {
                'floor_height_above_terrain': height
            }

            result = self.encoder.encode_region(
                image.copy(), parameters, self.model_type
            )

            # Check green channel value at background pixel
            bg_pixel = result[64, 60, 1]  # Green channel

            self.assertAlmostEqual(
                bg_pixel, expected_value, delta=2,
                msg=f"Floor height {height}m should encode to ~{expected_value}"
            )

    def test_red_channel_facade_reflectance(self):
        """Test red channel encodes facade_reflectance (0-1 -> 0-1)"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        test_cases = [
            (0.0, 0),      # 0.0 -> 0
            (0.5, 127),    # 0.5 -> ~127
            (1.0, 255),    # 1.0 -> 255
        ]

        for reflectance, expected_value in test_cases:
            parameters = {
                'floor_height_above_terrain': 2.0,
                'facade_reflectance': reflectance
            }

            result = self.encoder.encode_region(
                image.copy(), parameters, self.model_type
            )

            # Check red channel value at background pixel
            bg_pixel = result[64, 60, 0]  # Red channel

            self.assertAlmostEqual(
                bg_pixel, expected_value, delta=2,
                msg=f"Facade reflectance {reflectance} should encode to ~{expected_value}"
            )

    def test_blue_channel_terrain_reflectance(self):
        """Test blue channel encodes terrain_reflectance (0-1 -> 0-1)"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        test_cases = [
            (0.0, 0),      # 0.0 -> 0
            (0.5, 127),    # 0.5 -> ~127
            (1.0, 255),    # 1.0 -> 255
        ]

        for reflectance, expected_value in test_cases:
            parameters = {
                'floor_height_above_terrain': 2.0,
                'terrain_reflectance': reflectance
            }

            result = self.encoder.encode_region(
                image.copy(), parameters, self.model_type
            )

            # Check blue channel value at background pixel
            bg_pixel = result[64, 60, 2]  # Blue channel

            self.assertAlmostEqual(
                bg_pixel, expected_value, delta=2,
                msg=f"Terrain reflectance {reflectance} should encode to ~{expected_value}"
            )

    def test_alpha_channel_direction_angle(self):
        """Test alpha channel encodes window_direction_angle (0-2pi rad -> 0-1)"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        test_cases = [
            (0.0, 0),                    # 0 rad (East) -> 0
            (math.pi, 127),              # pi rad -> ~127
            (2 * math.pi, 255),          # 2pi rad -> 255
        ]

        for angle, expected_value in test_cases:
            parameters = {
                'floor_height_above_terrain': 2.0,
                'window_direction_angle': angle
            }

            result = self.encoder.encode_region(
                image.copy(), parameters, self.model_type
            )

            # Check alpha channel value at background pixel
            bg_pixel = result[64, 60, 3]  # Alpha channel

            self.assertAlmostEqual(
                bg_pixel, expected_value, delta=2,
                msg=f"Direction angle {angle:.2f} rad should encode to ~{expected_value}"
            )

    def test_default_facade_reflectance(self):
        """Test red channel defaults to 1.0 when facade_reflectance not provided"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'floor_height_above_terrain': 2.0
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Default value of 1.0 should map to 255
        bg_pixel = result[64, 60, 0]  # Red channel

        self.assertEqual(
            bg_pixel, 255,
            "Default facade_reflectance should be 1.0 (encoded as 255)"
        )

    def test_default_terrain_reflectance(self):
        """Test blue channel defaults to 1.0 when terrain_reflectance not provided"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'floor_height_above_terrain': 2.0
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Default value of 1.0 should map to 255
        bg_pixel = result[64, 60, 2]  # Blue channel

        self.assertEqual(
            bg_pixel, 255,
            "Default terrain_reflectance should be 1.0 (encoded as 255)"
        )

    def test_default_direction_angle(self):
        """Test alpha channel defaults to pi when window_direction_angle not provided"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'floor_height_above_terrain': 2.0
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Default value of pi should map to ~127 (pi / 2pi * 255)
        bg_pixel = result[64, 60, 3]  # Alpha channel

        self.assertAlmostEqual(
            bg_pixel, 127, delta=2,
            msg="Default window_direction_angle should be pi (encoded as ~127)"
        )


class TestBackgroundParameterValidation(unittest.TestCase):
    """Test background validates required parameters"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = BackgroundRegionEncoder()
        self.model_type = ModelType.DF_DEFAULT
        self.image = np.zeros((128, 128, 4), dtype=np.uint8)

    def test_missing_floor_height_above_terrain(self):
        """Test error when floor_height_above_terrain is missing"""
        parameters = {
            'facade_reflectance': 0.6
        }

        with self.assertRaises(ValueError) as context:
            self.encoder.encode_region(self.image, parameters, self.model_type)

        error_msg = str(context.exception)
        self.assertIn('floor_height_above_terrain', error_msg)

    def test_optional_reflectances_work(self):
        """Test that all reflectance parameters are optional"""
        parameters = {
            'floor_height_above_terrain': 2.0
        }

        # Should not raise error even without reflectance parameters
        try:
            result = self.encoder.encode_region(
                self.image, parameters, self.model_type
            )
            self.assertIsNotNone(result)
        except ValueError as e:
            self.fail(f"Should not require reflectance parameters: {e}")

    def test_all_parameters_provided(self):
        """Test encoding with all parameters provided"""
        parameters = {
            'floor_height_above_terrain': 2.5,
            'facade_reflectance': 0.6,
            'terrain_reflectance': 0.25,
            'window_direction_angle': math.pi
        }

        try:
            result = self.encoder.encode_region(
                self.image, parameters, self.model_type
            )
            self.assertIsNotNone(result)

            # Verify all channels encoded
            bg_pixel = result[64, 60, :]
            self.assertTrue(np.all(bg_pixel > 0))
        except Exception as e:
            self.fail(f"Should work with all parameters: {e}")


class TestBackgroundUniformEncoding(unittest.TestCase):
    """Test that background encoding is uniform across region"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = BackgroundRegionEncoder()
        self.model_type = ModelType.DF_DEFAULT

    def test_background_uniform_across_region(self):
        """Test that all background pixels have same values"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'floor_height_above_terrain': 3.0,
            'facade_reflectance': 0.5,
            'terrain_reflectance': 0.3,
            'window_direction_angle': math.pi / 2  # 90 degrees in radians
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Sample multiple background pixels
        sample_pixels = [
            result[10, 10, :],
            result[50, 50, :],
            result[100, 10, :],
            result[10, 100, :]
        ]

        # All should be identical (or very close due to floating point)
        reference = sample_pixels[0]
        for pixel in sample_pixels[1:]:
            for channel in range(4):
                self.assertEqual(
                    pixel[channel], reference[channel],
                    "All background pixels should have same values"
                )


class TestBackgroundSpecialOutputRange(unittest.TestCase):
    """Test special output range for floor_height_above_terrain (0.1-1)"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = BackgroundRegionEncoder()
        self.model_type = ModelType.DF_DEFAULT

    def test_floor_height_minimum_maps_to_0_1(self):
        """Test that 0m maps to 0.1 normalized (25.5 in pixel value)"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'floor_height_above_terrain': 0.0
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        bg_pixel = result[64, 60, 1]  # Green channel

        # 0m should map to 0.1 normalized = 0.1 * 255 = 25.5
        self.assertAlmostEqual(
            bg_pixel, 25, delta=2,
            msg="0m floor height should map to 0.1 normalized (~25)"
        )

    def test_floor_height_maximum_maps_to_1_0(self):
        """Test that 10m maps to 1.0 normalized (255 in pixel value)"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'floor_height_above_terrain': 10.0
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        bg_pixel = result[64, 60, 1]  # Green channel

        # 10m should map to 1.0 normalized = 255
        self.assertEqual(
            bg_pixel, 255,
            "10m floor height should map to 1.0 normalized (255)"
        )

    def test_floor_height_never_below_minimum(self):
        """Test that floor height never encodes below 0.1 normalized"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'floor_height_above_terrain': 0.0
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        bg_pixel = result[64, 60, 1]  # Green channel

        # Should never be less than 0.1 * 255 = 25.5
        self.assertGreaterEqual(
            bg_pixel, 25,
            "Floor height should never encode below 0.1 normalized"
        )


class TestBackgroundIntegration(unittest.TestCase):
    """Test background integration with RoomImageBuilder"""

    def test_builder_creates_correct_background(self):
        """Test that RoomImageBuilder correctly encodes background"""
        builder = RoomImageBuilder()

        parameters = {
            'floor_height_above_terrain': 2.0,
            'facade_reflectance': 0.6,
            'terrain_reflectance': 0.3,
            'window_direction_angle': math.pi
        }

        image = (builder
                 .reset()
                 .set_model_type(ModelType.DF_DEFAULT)
                 .encode_region(RegionType.BACKGROUND, parameters)
                 .build())

        # Verify image dimensions
        self.assertEqual(image.shape, (128, 128, 4))

        # Verify background exists
        bg_pixel = image[64, 60, :]
        self.assertTrue(
            np.any(bg_pixel > 0),
            "Builder should create background area"
        )

        # Verify color encoding
        # Green: floor height (2.0m / 10.0m = 0.2, maps to 0.1 + 0.2*0.9 = 0.28 -> 71.4)
        expected_green = int((0.1 + (2.0/10.0)*0.9) * 255)
        self.assertAlmostEqual(
            bg_pixel[1], expected_green, delta=2,
            msg="Builder should encode floor height correctly"
        )

        # Red: facade reflectance (0.6 -> 153)
        expected_red = int(0.6 * 255)
        self.assertAlmostEqual(
            bg_pixel[0], expected_red, delta=2,
            msg="Builder should encode facade reflectance correctly"
        )


if __name__ == '__main__':
    unittest.main()
