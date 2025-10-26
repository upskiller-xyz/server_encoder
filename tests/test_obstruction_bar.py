"""
Unit tests for obstruction bar encoding

Tests verify:
1. Bar construction at correct position and dimensions
2. Correct width and height based on image size
3. Proper color encoding for all channels
4. Validation of required parameters
5. Scaling behavior for different image sizes
6. Balcony underside reflectance encoding in alpha channel
"""

import unittest
import numpy as np
from src.components.region_encoders import ObstructionBarEncoder
from src.components.enums import ModelType, ImageDimensions, RegionType
from src.components.image_builder import RoomImageBuilder


class TestObstructionBarConstruction(unittest.TestCase):
    """Test obstruction bar is constructed at correct position"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = ObstructionBarEncoder()
        self.model_type = ModelType.DF_DEFAULT

    def test_bar_exists_at_right_edge(self):
        """Test that obstruction bar is located at right edge of image"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'obstruction_angle_zenith': 35.0,
            'obstruction_angle_horizon': 45.0
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Bar should be at x coordinates 124-127 (4 pixels wide at right edge)
        bar_x_start = 124
        bar_y_center = 64

        # Check that bar region has non-zero values
        bar_pixel = result[bar_y_center, bar_x_start, :]
        self.assertTrue(
            np.any(bar_pixel > 0),
            "Obstruction bar should have non-zero pixel values"
        )

        # Check that area outside bar is still zero (background)
        non_bar_pixel = result[bar_y_center, bar_x_start - 5, :]
        self.assertTrue(
            np.all(non_bar_pixel == 0),
            "Area outside obstruction bar should remain zero"
        )

    def test_bar_centered_vertically(self):
        """Test that obstruction bar is centered vertically"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'obstruction_angle_zenith': 35.0,
            'obstruction_angle_horizon': 45.0
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Bar should be 64 pixels tall, centered vertically
        # Center at y=64, so bar spans y=32 to y=95 (32 up, 32 down from center)
        bar_y_start = 32
        bar_y_end = 96
        bar_x = 126  # Middle of bar

        # Check that bar exists in expected vertical range
        bar_top_pixel = result[bar_y_start, bar_x, :]
        bar_bottom_pixel = result[bar_y_end - 1, bar_x, :]

        self.assertTrue(
            np.any(bar_top_pixel > 0),
            "Top of obstruction bar should have values"
        )
        self.assertTrue(
            np.any(bar_bottom_pixel > 0),
            "Bottom of obstruction bar should have values"
        )

        # Check that area above and below bar is zero
        above_bar = result[bar_y_start - 1, bar_x, :]
        below_bar = result[bar_y_end, bar_x, :]

        self.assertTrue(
            np.all(above_bar == 0),
            "Area above obstruction bar should be zero"
        )
        self.assertTrue(
            np.all(below_bar == 0),
            "Area below obstruction bar should be zero"
        )


class TestObstructionBarDimensions(unittest.TestCase):
    """Test obstruction bar has correct dimensions"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = ObstructionBarEncoder()
        self.model_type = ModelType.DF_DEFAULT

    def test_bar_width_128px_image(self):
        """Test bar is 4 pixels wide for 128x128 image"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'obstruction_angle_zenith': 35.0,
            'obstruction_angle_horizon': 45.0
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Bar should span x=124 to x=127 (4 pixels)
        bar_y = 64  # Middle row

        # Check bar exists at expected x positions
        for x in range(124, 128):
            pixel = result[bar_y, x, :]
            self.assertTrue(
                np.any(pixel > 0),
                f"Bar should exist at x={x}"
            )

        # Check bar doesn't extend beyond
        pixel_before = result[bar_y, 123, :]
        self.assertTrue(
            np.all(pixel_before == 0),
            "Bar should not extend before x=124"
        )

    def test_bar_height_128px_image(self):
        """Test bar is 64 pixels tall for 128x128 image"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'obstruction_angle_zenith': 35.0,
            'obstruction_angle_horizon': 45.0
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Bar should span y=32 to y=95 (64 pixels)
        bar_x = 126  # Middle of bar

        # Check bar exists at expected y positions
        for y in range(32, 96):
            pixel = result[y, bar_x, :]
            self.assertTrue(
                np.any(pixel > 0),
                f"Bar should exist at y={y}"
            )

        # Check bar doesn't extend beyond
        pixel_above = result[31, bar_x, :]
        pixel_below = result[96, bar_x, :]

        self.assertTrue(
            np.all(pixel_above == 0),
            "Bar should not extend above y=32"
        )
        self.assertTrue(
            np.all(pixel_below == 0),
            "Bar should not extend below y=95"
        )


class TestObstructionBarColorEncoding(unittest.TestCase):
    """Test obstruction bar encodes colors correctly"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = ObstructionBarEncoder()
        self.model_type = ModelType.DF_DEFAULT

    def test_red_channel_horizon_angle(self):
        """Test red channel encodes obstruction_angle_horizon (0-90° -> 0-1)"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        test_cases = [
            (0.0, 0),      # 0° -> 0
            (45.0, 127),   # 45° -> ~127
            (90.0, 255),   # 90° -> 255
        ]

        for angle, expected_value in test_cases:
            parameters = {
                'obstruction_angle_zenith': 35.0,
                'obstruction_angle_horizon': angle
            }

            result = self.encoder.encode_region(
                image.copy(), parameters, self.model_type
            )

            # Check red channel value at bar center
            bar_pixel = result[64, 126, 0]  # Red channel

            self.assertAlmostEqual(
                bar_pixel, expected_value, delta=2,
                msg=f"Horizon angle {angle}° should encode to ~{expected_value} in red channel"
            )

    def test_blue_channel_zenith_angle(self):
        """Test blue channel encodes obstruction_angle_zenith (0-70° -> 0.2-0.8)"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        test_cases = [
            (0.0, 51),     # 0° -> 0.2*255 = 51
            (35.0, 127),   # 35° -> 0.5*255 = 127
            (70.0, 204),   # 70° -> 0.8*255 = 204
        ]

        for angle, expected_value in test_cases:
            parameters = {
                'obstruction_angle_zenith': angle,
                'obstruction_angle_horizon': 45.0
            }

            result = self.encoder.encode_region(
                image.copy(), parameters, self.model_type
            )

            # Check blue channel value at bar center
            bar_pixel = result[64, 126, 2]  # Blue channel

            self.assertAlmostEqual(
                bar_pixel, expected_value, delta=2,
                msg=f"Zenith angle {angle}° should encode to ~{expected_value} in blue channel"
            )

    def test_green_channel_context_reflectance(self):
        """Test green channel encodes context_reflectance (0.1-0.6 -> 0-1)"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        test_cases = [
            (0.1, 0),      # 0.1 -> 0
            (0.35, 127),   # 0.35 -> ~127
            (0.6, 255),    # 0.6 -> 255
        ]

        for reflectance, expected_value in test_cases:
            parameters = {
                'obstruction_angle_zenith': 35.0,
                'obstruction_angle_horizon': 45.0,
                'context_reflectance': reflectance
            }

            result = self.encoder.encode_region(
                image.copy(), parameters, self.model_type
            )

            # Check green channel value at bar center
            bar_pixel = result[64, 126, 1]  # Green channel

            self.assertAlmostEqual(
                bar_pixel, expected_value, delta=2,
                msg=f"Context reflectance {reflectance} should encode to ~{expected_value} in green channel"
            )

    def test_green_channel_default_context_reflectance(self):
        """Test green channel defaults to 1.0 for unobstructed areas"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        # Don't provide context_reflectance parameter
        parameters = {
            'obstruction_angle_zenith': 35.0,
            'obstruction_angle_horizon': 45.0
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Default value of 1.0 should map to 255 in green channel
        # (1.0 is outside the 0.1-0.6 range, so it clamps to max)
        bar_pixel = result[64, 126, 1]  # Green channel

        self.assertEqual(
            bar_pixel, 255,
            "Default context_reflectance should be 1.0 (encoded as 255)"
        )

    def test_array_values_distributed_over_rows(self):
        """Test that array values are distributed across bar rows"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        # Create gradient arrays for 64 rows
        zenith_angles = np.linspace(0, 70, 64)
        horizon_angles = np.linspace(0, 90, 64)

        parameters = {
            'obstruction_angle_zenith': zenith_angles.tolist(),
            'obstruction_angle_horizon': horizon_angles.tolist()
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Check that first row has different values than last row
        bar_x = 126
        first_row_blue = result[32, bar_x, 2]   # First row, blue channel
        last_row_blue = result[95, bar_x, 2]    # Last row, blue channel

        self.assertNotEqual(
            first_row_blue, last_row_blue,
            "Array values should create gradient across rows"
        )

        # First row should be closer to 0° encoding
        self.assertLess(
            first_row_blue, 100,
            "First row should encode low angle value"
        )

        # Last row should be closer to 70° encoding
        self.assertGreater(
            last_row_blue, 150,
            "Last row should encode high angle value"
        )


class TestObstructionBarParameterValidation(unittest.TestCase):
    """Test obstruction bar validates required parameters"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = ObstructionBarEncoder()
        self.model_type = ModelType.DF_DEFAULT
        self.image = np.zeros((128, 128, 4), dtype=np.uint8)

    def test_missing_both_required_parameters(self):
        """Test error when both required parameters are missing"""
        with self.assertRaises(ValueError) as context:
            self.encoder.encode_region(self.image, {}, self.model_type)

        error_msg = str(context.exception)
        self.assertIn('obstruction_angle_zenith', error_msg)
        self.assertIn('obstruction_angle_horizon', error_msg)

    def test_missing_zenith_angle(self):
        """Test error when obstruction_angle_zenith is missing"""
        parameters = {'obstruction_angle_horizon': 45.0}

        with self.assertRaises(ValueError) as context:
            self.encoder.encode_region(self.image, parameters, self.model_type)

        error_msg = str(context.exception)
        self.assertIn('obstruction_angle_zenith', error_msg)

    def test_missing_horizon_angle(self):
        """Test error when obstruction_angle_horizon is missing"""
        parameters = {'obstruction_angle_zenith': 35.0}

        with self.assertRaises(ValueError) as context:
            self.encoder.encode_region(self.image, parameters, self.model_type)

        error_msg = str(context.exception)
        self.assertIn('obstruction_angle_horizon', error_msg)

    def test_context_reflectance_optional(self):
        """Test that context_reflectance is optional"""
        parameters = {
            'obstruction_angle_zenith': 35.0,
            'obstruction_angle_horizon': 45.0
            # context_reflectance not provided
        }

        # Should not raise an error
        try:
            result = self.encoder.encode_region(
                self.image, parameters, self.model_type
            )
            self.assertIsNotNone(result)
        except ValueError:
            self.fail("context_reflectance should be optional")

    def test_all_parameters_provided(self):
        """Test successful encoding when all parameters provided"""
        parameters = {
            'obstruction_angle_zenith': 35.0,
            'obstruction_angle_horizon': 45.0,
            'context_reflectance': 0.3
        }

        result = self.encoder.encode_region(
            self.image, parameters, self.model_type
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (128, 128, 4))


class TestObstructionBarImageScaling(unittest.TestCase):
    """Test obstruction bar scales correctly for different image sizes"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = ObstructionBarEncoder()
        self.model_type = ModelType.DF_DEFAULT

    def test_bar_dimensions_256px_image(self):
        """Test bar dimensions scale correctly for 256x256 image"""
        # For 256x256 image (scale=2.0):
        # Bar width: 4 * 2 = 8 pixels
        # Bar height: 64 * 2 = 128 pixels

        dims = ImageDimensions(256)
        image = np.zeros((256, 256, 4), dtype=np.uint8)

        parameters = {
            'obstruction_angle_zenith': 35.0,
            'obstruction_angle_horizon': 45.0
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Calculate expected dimensions
        self.assertEqual(dims.obstruction_bar_width, 8)
        self.assertEqual(dims.obstruction_bar_height, 128)

        # Bar should be at right edge
        bar_x_start = 256 - 8  # = 248
        bar_y_center = 128

        # Check bar exists at expected position
        bar_pixel = result[bar_y_center, bar_x_start, :]
        self.assertTrue(
            np.any(bar_pixel > 0),
            "Bar should exist at scaled position"
        )

        # Check width: bar should span 8 pixels
        for x in range(bar_x_start, 256):
            pixel = result[bar_y_center, x, :]
            self.assertTrue(
                np.any(pixel > 0),
                f"Bar should exist at x={x} for 256px image"
            )

    def test_bar_dimensions_1024px_image(self):
        """Test bar dimensions scale correctly for 1024x1024 image"""
        # For 1024x1024 image (scale=8.0):
        # Bar width: 4 * 8 = 32 pixels
        # Bar height: 64 * 8 = 512 pixels

        dims = ImageDimensions(1024)
        image = np.zeros((1024, 1024, 4), dtype=np.uint8)

        parameters = {
            'obstruction_angle_zenith': 35.0,
            'obstruction_angle_horizon': 45.0
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Calculate expected dimensions
        self.assertEqual(dims.obstruction_bar_width, 32)
        self.assertEqual(dims.obstruction_bar_height, 512)

        # Bar should be at right edge
        bar_x_start = 1024 - 32  # = 992
        bar_y_center = 512

        # Check bar exists at expected position
        bar_pixel = result[bar_y_center, bar_x_start, :]
        self.assertTrue(
            np.any(bar_pixel > 0),
            "Bar should exist at scaled position"
        )

    def test_bar_vertical_centering_scales(self):
        """Test bar remains vertically centered for different image sizes"""
        test_sizes = [128, 256, 512, 1024]

        for size in test_sizes:
            dims = ImageDimensions(size)
            image = np.zeros((size, size, 4), dtype=np.uint8)

            parameters = {
                'obstruction_angle_zenith': 35.0,
                'obstruction_angle_horizon': 45.0
            }

            result = self.encoder.encode_region(
                image, parameters, self.model_type
            )

            # Get bar position
            x_start, y_start, x_end, y_end = dims.get_obstruction_bar_position()

            # Check bar is centered
            expected_y_center = size // 2
            actual_y_center = (y_start + y_end) // 2

            self.assertAlmostEqual(
                actual_y_center, expected_y_center, delta=1,
                msg=f"Bar should be vertically centered for {size}px image"
            )

            # Check bar exists at center
            bar_x = (x_start + x_end) // 2
            bar_pixel = result[expected_y_center, bar_x, :]

            self.assertTrue(
                np.any(bar_pixel > 0),
                f"Bar should exist at center for {size}px image"
            )

    def test_proportional_scaling(self):
        """Test bar dimensions maintain correct proportions across sizes"""
        base_dims = ImageDimensions(128)

        test_cases = [
            (256, 2.0),
            (512, 4.0),
            (1024, 8.0)
        ]

        for size, scale_factor in test_cases:
            scaled_dims = ImageDimensions(size)

            expected_width = int(base_dims.obstruction_bar_width * scale_factor)
            expected_height = int(base_dims.obstruction_bar_height * scale_factor)

            self.assertEqual(
                scaled_dims.obstruction_bar_width, expected_width,
                f"Width should scale by {scale_factor} for {size}px image"
            )
            self.assertEqual(
                scaled_dims.obstruction_bar_height, expected_height,
                f"Height should scale by {scale_factor} for {size}px image"
            )


class TestObstructionBarIntegration(unittest.TestCase):
    """Integration tests for obstruction bar with RoomImageBuilder"""

    def test_builder_creates_correct_bar(self):
        """Test that RoomImageBuilder creates obstruction bar correctly"""
        builder = RoomImageBuilder()

        parameters = {
            'obstruction_angle_zenith': 35.0,
            'obstruction_angle_horizon': 45.0,
            'context_reflectance': 0.3
        }

        image = (builder
                 .reset()
                 .set_model_type(ModelType.DF_DEFAULT)
                 .encode_region(RegionType.OBSTRUCTION_BAR, parameters)
                 .build())

        # Verify image dimensions
        self.assertEqual(image.shape, (128, 128, 4))

        # Verify bar exists at expected location
        bar_pixel = image[64, 126, :]
        self.assertTrue(
            np.any(bar_pixel > 0),
            "Builder should create obstruction bar"
        )


class TestBalconyUndersideReflectance(unittest.TestCase):
    """Test balcony_reflectance parameter encoding"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = ObstructionBarEncoder()
        self.model_type = ModelType.DF_CUSTOM
        self.image = np.zeros((128, 128, 4), dtype=np.uint8)

    def test_alpha_channel_default_value(self):
        """Test alpha channel defaults to 0.8 when balcony_reflectance not provided"""
        parameters = {
            'obstruction_angle_zenith': 35.0,
            'obstruction_angle_horizon': 45.0
        }

        result = self.encoder.encode_region(
            self.image.copy(), parameters, self.model_type
        )

        # Default value of 0.8 should encode to ~204 (0.8 * 255)
        bar_pixel_alpha = result[64, 126, 3]  # Alpha channel

        self.assertAlmostEqual(
            bar_pixel_alpha, 204, delta=2,
            msg="Default balcony_reflectance (0.8) should encode to ~204"
        )

    def test_alpha_channel_custom_values(self):
        """Test alpha channel encodes custom balcony_reflectance values"""
        test_cases = [
            (0.0, 0),      # 0.0 -> 0
            (0.5, 127),    # 0.5 -> ~127
            (0.7, 178),    # 0.7 -> ~178
            (1.0, 255),    # 1.0 -> 255
        ]

        for reflectance, expected_value in test_cases:
            parameters = {
                'obstruction_angle_zenith': 35.0,
                'obstruction_angle_horizon': 45.0,
                'balcony_reflectance': reflectance
            }

            result = self.encoder.encode_region(
                self.image.copy(), parameters, self.model_type
            )

            # Check alpha channel value at bar center
            bar_pixel_alpha = result[64, 126, 3]  # Alpha channel

            self.assertAlmostEqual(
                bar_pixel_alpha, expected_value, delta=2,
                msg=f"balcony_reflectance {reflectance} should encode to ~{expected_value}"
            )

    def test_alpha_channel_uniform_across_bar(self):
        """Test alpha channel has uniform value across entire bar"""
        parameters = {
            'obstruction_angle_zenith': 35.0,
            'obstruction_angle_horizon': 45.0,
            'balcony_reflectance': 0.6
        }

        result = self.encoder.encode_region(
            self.image.copy(), parameters, self.model_type
        )

        # Get alpha values from multiple positions in bar
        bar_x = 126
        alpha_values = []
        for y in range(32, 96):  # All 64 rows of bar
            alpha_values.append(result[y, bar_x, 3])

        # All alpha values should be the same
        expected_alpha = int(0.6 * 255)
        for alpha in alpha_values:
            self.assertAlmostEqual(
                alpha, expected_alpha, delta=2,
                msg="Alpha channel should be uniform across all bar rows"
            )

    def test_alpha_channel_all_columns_in_bar(self):
        """Test alpha channel is encoded across all 4 columns of bar"""
        parameters = {
            'obstruction_angle_zenith': 35.0,
            'obstruction_angle_horizon': 45.0,
            'balcony_reflectance': 0.75
        }

        result = self.encoder.encode_region(
            self.image.copy(), parameters, self.model_type
        )

        expected_alpha = int(0.75 * 255)
        bar_y = 64  # Center row

        # Check all 4 columns of bar (x=124-127)
        for x in range(124, 128):
            alpha_value = result[bar_y, x, 3]
            self.assertAlmostEqual(
                alpha_value, expected_alpha, delta=2,
                msg=f"Alpha should be encoded at x={x}"
            )

    def test_alpha_does_not_affect_other_channels(self):
        """Test balcony_reflectance doesn't interfere with other channels"""
        parameters = {
            'obstruction_angle_zenith': 35.0,
            'obstruction_angle_horizon': 45.0,
            'balcony_reflectance': 0.9
        }

        result = self.encoder.encode_region(
            self.image.copy(), parameters, self.model_type
        )

        bar_pixel = result[64, 126, :]

        # Alpha channel should be 0.9
        self.assertAlmostEqual(
            bar_pixel[3], int(0.9 * 255), delta=2,
            msg="Alpha channel should encode balcony reflectance"
        )

        # Red channel should still encode horizon angle (45° -> ~127)
        self.assertAlmostEqual(
            bar_pixel[0], 127, delta=2,
            msg="Red channel should still encode horizon angle"
        )

        # Blue channel should still encode zenith angle (35° -> ~127)
        self.assertAlmostEqual(
            bar_pixel[2], 127, delta=2,
            msg="Blue channel should still encode zenith angle"
        )

    def test_works_with_all_model_types(self):
        """Test balcony_reflectance works with all model types"""
        parameters = {
            'obstruction_angle_zenith': 35.0,
            'obstruction_angle_horizon': 45.0,
            'balcony_reflectance': 0.85
        }

        model_types = [
            ModelType.DF_DEFAULT,
            ModelType.DF_CUSTOM,
            ModelType.DA_DEFAULT,
            ModelType.DA_CUSTOM
        ]

        expected_alpha = int(0.85 * 255)

        for model_type in model_types:
            result = self.encoder.encode_region(
                self.image.copy(), parameters, model_type
            )

            bar_pixel_alpha = result[64, 126, 3]
            self.assertAlmostEqual(
                bar_pixel_alpha, expected_alpha, delta=2,
                msg=f"Should work with {model_type.value}"
            )

    def test_integration_with_builder(self):
        """Test balcony_reflectance works through RoomImageBuilder"""
        builder = RoomImageBuilder()

        parameters = {
            'obstruction_angle_zenith': 35.0,
            'obstruction_angle_horizon': 45.0,
            'balcony_reflectance': 0.65
        }

        image = (builder
                 .reset()
                 .set_model_type(ModelType.DF_CUSTOM)
                 .encode_region(RegionType.OBSTRUCTION_BAR, parameters)
                 .build())

        # Verify alpha channel encoded correctly
        bar_pixel_alpha = image[64, 126, 3]
        expected_alpha = int(0.65 * 255)

        self.assertAlmostEqual(
            bar_pixel_alpha, expected_alpha, delta=2,
            msg="Builder should encode balcony_reflectance to alpha channel"
        )


if __name__ == '__main__':
    unittest.main()
