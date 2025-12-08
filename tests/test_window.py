"""
Unit tests for window encoding

Tests verify:
1. Window construction at correct position (12px from right, centered vertically)
2. Correct width (wall thickness ~3px) and height (based on window width in 3D)
3. Proper color encoding for all channels (including reversed mappings)
4. Validation of required parameters
5. Default values for optional parameters
6. Scaling behavior for different image sizes
7. Window geometry calculation from bounding box
"""

import unittest
import numpy as np
from src.components.region_encoders import WindowRegionEncoder
from src.components.geometry import WindowGeometry
from src.components.enums import ModelType, RegionType
from src.components.image_builder import RoomImageBuilder


class TestWindowGeometry(unittest.TestCase):
    """Test WindowGeometry class for bounding box calculations"""

    def test_window_dimensions_from_bbox(self):
        """Test window dimensions calculated from bounding box"""
        # Window: 1.2m wide (x), 1.5m tall (z)
        window = WindowGeometry(
            x1=-0.6, y1=0.0, z1=0.9,
            x2=0.6, y2=0.0, z2=2.4
        )

        self.assertEqual(window.window_width_3d, 1.2)
        self.assertEqual(window.window_height_3d, 1.5)
        self.assertEqual(window.sill_height, 0.9)
        self.assertEqual(window.top_height, 2.4)

    def test_window_dimensions_order_independent(self):
        """Test that corner order doesn't matter"""
        # Reversed corners
        window = WindowGeometry(
            x1=0.6, y1=0.0, z1=2.4,
            x2=-0.6, y2=0.0, z2=0.9
        )

        self.assertEqual(window.window_width_3d, 1.2)
        self.assertEqual(window.window_height_3d, 1.5)
        self.assertEqual(window.sill_height, 0.9)
        self.assertEqual(window.top_height, 2.4)

    def test_window_pixel_bounds_128(self):
        """Test window pixel bounds for 128x128 image"""
        window = WindowGeometry(
            x1=-0.6, y1=0.0, z1=0.9,
            x2=0.6, y2=0.0, z2=2.4
        )

        x_start, y_start, x_end, y_end = window.get_pixel_bounds(image_size=128)

        # Window should be 12px from right edge
        self.assertEqual(x_end, 128 - 12)  # 116
        # Width should be ~3px (wall thickness 0.3m / 0.1m per pixel)
        self.assertAlmostEqual(x_end - x_start, 3, delta=1)
        # Height should be window width in 3D: 1.2m = 12 pixels
        self.assertAlmostEqual(y_end - y_start, 12, delta=1)
        # Should be centered vertically
        y_center = (y_start + y_end) // 2
        self.assertAlmostEqual(y_center, 64, delta=1)

    def test_window_pixel_bounds_256(self):
        """Test window pixel bounds scale for 256x256 image"""
        window = WindowGeometry(
            x1=-0.6, y1=0.0, z1=0.9,
            x2=0.6, y2=0.0, z2=2.4
        )

        x_start, y_start, x_end, y_end = window.get_pixel_bounds(image_size=256)

        # Window should be 12px from right edge (doesn't scale)
        self.assertEqual(x_end, 256 - 12)  # 244
        # Width should be ~6px (scaled 2x)
        self.assertAlmostEqual(x_end - x_start, 6, delta=1)
        # Height should be 24 pixels (scaled 2x)
        self.assertAlmostEqual(y_end - y_start, 24, delta=2)


class TestWindowConstruction(unittest.TestCase):
    """Test window is constructed at correct position"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = WindowRegionEncoder()
        self.model_type = ModelType.DF_DEFAULT

    def test_window_exists_at_correct_position(self):
        """Test that window is located 12px from right edge"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Window should be at x=114-116 (12px from right, ~3px wide)
        window_x = 115
        window_y = 64  # Center

        # Check that window region has non-zero values
        window_pixel = result[window_y, window_x, :]
        self.assertTrue(
            np.any(window_pixel > 0),
            "Window should have non-zero pixel values"
        )

        # Check that area far from window is still zero
        far_pixel = result[window_y, 50, :]
        self.assertTrue(
            np.all(far_pixel == 0),
            "Area far from window should remain zero"
        )

    def test_window_centered_vertically(self):
        """Test that window is centered vertically based on its width"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        # Window with 1.2m width = 12 pixels tall in top view
        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        window_x = 115

        # Window should be centered around y=64
        # Check a range of y values to find the window extent
        window_ys = []
        for y in range(128):
            if np.any(result[y, window_x, :] > 0):
                window_ys.append(y)

        if window_ys:
            y_min = min(window_ys)
            y_max = max(window_ys)
            y_center = (y_min + y_max) // 2

            self.assertAlmostEqual(
                y_center, 64, delta=2,
                msg="Window should be centered vertically"
            )


class TestWindowDimensions(unittest.TestCase):
    """Test window has correct dimensions"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = WindowRegionEncoder()
        self.model_type = ModelType.DF_DEFAULT

    def test_window_width_wall_thickness(self):
        """Test window width equals wall thickness (~3 pixels)"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        window_y = 64

        # Count pixels with window values along horizontal line
        window_xs = []
        for x in range(128):
            if np.any(result[window_y, x, :] > 0):
                window_xs.append(x)

        if window_xs:
            window_width = len(window_xs)
            # Should be approximately 3 pixels (0.3m wall thickness / 0.1m per pixel)
            self.assertAlmostEqual(
                window_width, 3, delta=1,
                msg="Window width should be ~3 pixels (wall thickness)"
            )

    def test_window_height_from_3d_width(self):
        """Test window height in image equals window width in 3D space"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        # Window with 1.2m width in 3D = 12 pixels in top view
        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        window_x = 115

        # Count pixels with window values along vertical line
        window_ys = []
        for y in range(128):
            if np.any(result[y, window_x, :] > 0):
                window_ys.append(y)

        if window_ys:
            window_height = len(window_ys)
            # 1.2m = 12 pixels at 0.1m/pixel
            self.assertAlmostEqual(
                window_height, 12, delta=2,
                msg="Window height should match window width in 3D (12 pixels for 1.2m)"
            )

    def test_different_window_sizes(self):
        """Test different window sizes produce different pixel heights"""
        image1 = np.zeros((128, 128, 4), dtype=np.uint8)
        image2 = np.zeros((128, 128, 4), dtype=np.uint8)

        # Small window: 0.8m wide
        params_small = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -0.4, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.4, 'y2': 0.0, 'z2': 2.4
        }

        # Large window: 2.0m wide
        params_large = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -1.0, 'y1': 0.0, 'z1': 0.9,
            'x2': 1.0, 'y2': 0.0, 'z2': 2.4
        }

        result1 = self.encoder.encode_region(image1, params_small, self.model_type)
        result2 = self.encoder.encode_region(image2, params_large, self.model_type)

        window_x = 115

        # Count heights
        height1 = sum(1 for y in range(128) if np.any(result1[y, window_x, :] > 0))
        height2 = sum(1 for y in range(128) if np.any(result2[y, window_x, :] > 0))

        # Large window should have greater height in pixels
        self.assertGreater(height2, height1, "Larger window should have more pixels")


class TestWindowColorEncoding(unittest.TestCase):
    """Test window encodes colors correctly"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = WindowRegionEncoder()
        self.model_type = ModelType.DF_DEFAULT

    def test_red_channel_sill_height(self):
        """Test red channel encodes sill_height (0-5m → 0-1)"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        test_cases = [
            (0.0, 0),      # 0m → 0
            (2.5, 127),    # 2.5m → ~127
            (5.0, 255),    # 5m → 255
        ]

        for sill_height, expected_value in test_cases:
            parameters = {
                'window_sill_height': sill_height,
                'window_frame_ratio': 0.8,
                'window_height': 1.5,
                'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
                'x2': 0.6, 'y2': 0.0, 'z2': 2.4
            }

            result = self.encoder.encode_region(
                image.copy(), parameters, self.model_type
            )

            # Check red channel value at window center
            window_pixel = result[64, 115, 0]  # Red channel

            self.assertAlmostEqual(
                window_pixel, expected_value, delta=2,
                msg=f"Sill height {sill_height}m should encode to ~{expected_value} in red channel"
            )

    def test_green_channel_frame_ratio_reversed(self):
        """Test green channel encodes frame_ratio (1-0 → 0-1, REVERSED)"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        test_cases = [
            (1.0, 0),      # 1.0 → 0 (reversed)
            (0.5, 127),    # 0.5 → ~127
            (0.0, 255),    # 0.0 → 255 (reversed)
        ]

        for frame_ratio, expected_value in test_cases:
            parameters = {
                'window_sill_height': 0.9,
                'window_frame_ratio': frame_ratio,
                'window_height': 1.5,
                'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
                'x2': 0.6, 'y2': 0.0, 'z2': 2.4
            }

            result = self.encoder.encode_region(
                image.copy(), parameters, self.model_type
            )

            # Check green channel value at window center
            window_pixel = result[64, 115, 1]  # Green channel

            self.assertAlmostEqual(
                window_pixel, expected_value, delta=2,
                msg=f"Frame ratio {frame_ratio} should encode to ~{expected_value} in green channel (reversed)"
            )

    def test_blue_channel_window_height_reversed(self):
        """Test blue channel encodes window_height (0.2-5m → 0.99-0.01, REVERSED)"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        test_cases = [
            (0.2, 252),    # 0.2m → 0.99*255 = 252 (reversed, min height)
            (2.6, 127),    # 2.6m → 0.5*255 = 127 (middle)
            (5.0, 2),      # 5.0m → 0.01*255 = 2 (reversed, max height)
        ]

        for window_height, expected_value in test_cases:
            parameters = {
                'window_sill_height': 0.9,
                'window_frame_ratio': 0.8,
                'window_height': window_height,
                'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
                'x2': 0.6, 'y2': 0.0, 'z2': 2.4
            }

            result = self.encoder.encode_region(
                image.copy(), parameters, self.model_type
            )

            # Check blue channel value at window center
            window_pixel = result[64, 115, 2]  # Blue channel

            self.assertAlmostEqual(
                window_pixel, expected_value, delta=2,
                msg=f"Window height {window_height}m should encode to ~{expected_value} in blue channel (reversed)"
            )

    def test_alpha_channel_default_frame_reflectance(self):
        """Test alpha channel defaults to 0.8 for window_frame_reflectance"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        # Don't provide window_frame_reflectance parameter
        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Default value of 0.8 should map to 204 in alpha channel (0.8*255)
        window_pixel = result[64, 115, 3]  # Alpha channel

        # For DF_DEFAULT model, alpha might not be set
        # But if provided or custom model, it should be 204
        # Let's test with custom model
        result_custom = self.encoder.encode_region(
            image.copy(), parameters, ModelType.DF_CUSTOM
        )
        window_pixel_custom = result_custom[64, 115, 3]

        self.assertAlmostEqual(
            window_pixel_custom, 204, delta=2, msg="Default window_frame_reflectance should be 0.8 (encoded as 204)"
        )

    def test_alpha_channel_custom_frame_reflectance(self):
        """Test alpha channel encodes custom window_frame_reflectance"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'window_frame_reflectance': 0.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        result = self.encoder.encode_region(
            image, parameters, ModelType.DF_CUSTOM
        )

        # 0.5 should map to 127 in alpha channel (0.5*255)
        window_pixel = result[64, 115, 3]  # Alpha channel

        self.assertAlmostEqual(
            window_pixel, 127, delta=2, msg="Custom window_frame_reflectance 0.5 should encode to ~127 in alpha channel"
        )


class TestWindowParameterValidation(unittest.TestCase):
    """Test window validates required parameters"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = WindowRegionEncoder()
        self.model_type = ModelType.DF_DEFAULT
        self.image = np.zeros((128, 128, 4), dtype=np.uint8)

    def test_missing_all_required_parameters(self):
        """Test error when all required parameters are missing"""
        with self.assertRaises(ValueError) as context:
            self.encoder.encode_region(self.image, {}, self.model_type)

        error_msg = str(context.exception)
        # window_sill_height and window_height are now auto-calculated from window geometry
        # Only window_frame_ratio and window geometry are required
        self.assertIn('window_frame_ratio', error_msg)
        self.assertIn('window geometry', error_msg.lower())

    def test_missing_sill_height(self):
        """Test error when sill_height is missing"""
        parameters = {
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        with self.assertRaises(ValueError) as context:
            self.encoder.encode_region(self.image, parameters, self.model_type)

        error_msg = str(context.exception)
        self.assertIn('window_sill_height', error_msg)

    def test_missing_frame_ratio(self):
        """Test error when frame_ratio is missing"""
        parameters = {
            'window_sill_height': 0.9,
            'window_height': 1.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        with self.assertRaises(ValueError) as context:
            self.encoder.encode_region(self.image, parameters, self.model_type)

        error_msg = str(context.exception)
        self.assertIn('window_frame_ratio', error_msg)

    def test_missing_window_height(self):
        """Test that window_height is auto-calculated when missing (no error expected)"""
        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4,
            'floor_height_above_terrain': 0.0  # Required for auto-calculation
        }

        # Should NOT raise an error - window_height is auto-calculated
        result_image = self.encoder.encode_region(self.image, parameters, self.model_type)

        # Verify encoding succeeded
        self.assertIsNotNone(result_image)

    def test_missing_geometry(self):
        """Test error when window geometry is missing"""
        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5
        }

        with self.assertRaises(ValueError) as context:
            self.encoder.encode_region(self.image, parameters, self.model_type)

        error_msg = str(context.exception)
        self.assertIn('geometry', error_msg.lower())

    def test_window_frame_reflectance_optional(self):
        """Test that window_frame_reflectance is optional"""
        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
            # window_frame_reflectance not provided
        }

        # Should not raise an error
        try:
            result = self.encoder.encode_region(
                self.image, parameters, self.model_type
            )
            self.assertIsNotNone(result)
        except ValueError:
            self.fail("window_frame_reflectance should be optional")

    def test_all_parameters_provided(self):
        """Test successful encoding when all parameters provided"""
        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'window_frame_reflectance': 0.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        result = self.encoder.encode_region(
            self.image, parameters, self.model_type
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (128, 128, 4))

    def test_geometry_as_object(self):
        """Test geometry can be provided as WindowGeometry object"""
        geometry = WindowGeometry(
            x1=-0.6, y1=0.0, z1=0.9,
            x2=0.6, y2=0.0, z2=2.4
        )

        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'window_geometry': geometry
        }

        result = self.encoder.encode_region(
            self.image, parameters, self.model_type
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (128, 128, 4))


class TestWindowImageScaling(unittest.TestCase):
    """Test window area scales correctly for different image sizes"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = WindowRegionEncoder()
        self.model_type = ModelType.DF_DEFAULT

    def test_window_position_128px(self):
        """Test window position for 128x128 image"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Window should be 12px from right edge
        # Right edge is at x=116 (128 - 12)
        window_x_end = 128 - 12

        # Check window exists at expected position
        window_pixel = result[64, window_x_end - 1, :]
        self.assertTrue(
            np.any(window_pixel > 0),
            "Window should exist 12px from right edge in 128px image"
        )

        # Check outside window area is zero
        outside_pixel = result[64, window_x_end - 10, :]
        self.assertTrue(
            np.all(outside_pixel == 0),
            "Area outside window should be zero"
        )

    def test_window_position_256px(self):
        """Test window position for 256x256 image"""
        image = np.zeros((256, 256, 4), dtype=np.uint8)

        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Window should still be 12px from right edge (doesn't scale)
        window_x_end = 256 - 12

        # Check window exists at expected position
        window_pixel = result[128, window_x_end - 1, :]
        self.assertTrue(
            np.any(window_pixel > 0),
            "Window should exist 12px from right edge in 256px image"
        )

    def test_window_position_512px(self):
        """Test window position for 512x512 image"""
        image = np.zeros((512, 512, 4), dtype=np.uint8)

        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        result = self.encoder.encode_region(image, parameters, self.model_type)

        # Window should still be 12px from right edge
        window_x_end = 512 - 12

        # Check window exists at expected position
        window_pixel = result[256, window_x_end - 1, :]
        self.assertTrue(
            np.any(window_pixel > 0),
            "Window should exist 12px from right edge in 512px image"
        )

    def test_window_width_scales_with_image_size(self):
        """Test window width (wall thickness) scales proportionally"""
        test_sizes = [128, 256, 512]
        expected_widths = {
            128: 3,   # Base: 0.3m / 0.1m per pixel = 3px
            256: 6,   # Scaled 2x
            512: 12   # Scaled 4x
        }

        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        for size in test_sizes:
            image = np.zeros((size, size, 4), dtype=np.uint8)
            result = self.encoder.encode_region(
                image, parameters, self.model_type
            )

            # Count pixels with window values along center horizontal line
            center_y = size // 2
            window_xs = []
            for x in range(size):
                if np.any(result[center_y, x, :] > 0):
                    window_xs.append(x)

            if window_xs:
                window_width = len(window_xs)
                expected = expected_widths[size]

                self.assertAlmostEqual(
                    window_width, expected, delta=2,
                    msg=f"Window width should be ~{expected}px for {size}x{size} image"
                )

    def test_window_height_scales_with_image_size(self):
        """Test window height (from 3D width) scales proportionally"""
        # Window width in 3D: 1.2m
        # Base resolution: 0.1m per pixel
        test_sizes = [128, 256, 512]
        expected_heights = {
            128: 12,  # Base: 1.2m / 0.1m = 12px
            256: 24,  # Scaled 2x
            512: 48   # Scaled 4x
        }

        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        for size in test_sizes:
            image = np.zeros((size, size, 4), dtype=np.uint8)
            result = self.encoder.encode_region(
                image, parameters, self.model_type
            )

            # Count pixels with window values along vertical line near window
            window_x = size - 14  # Just left of window edge
            window_ys = []
            for y in range(size):
                if np.any(result[y, window_x, :] > 0):
                    window_ys.append(y)

            if window_ys:
                window_height = len(window_ys)
                expected = expected_heights[size]

                self.assertAlmostEqual(
                    window_height, expected, delta=3,
                    msg=f"Window height should be ~{expected}px for {size}x{size} image"
                )

    def test_window_centered_vertically_all_sizes(self):
        """Test window remains centered vertically for all image sizes"""
        test_sizes = [128, 256, 512, 1024]

        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        for size in test_sizes:
            image = np.zeros((size, size, 4), dtype=np.uint8)
            result = self.encoder.encode_region(
                image, parameters, self.model_type
            )

            # Find window extent vertically
            window_x = size - 14
            window_ys = []
            for y in range(size):
                if np.any(result[y, window_x, :] > 0):
                    window_ys.append(y)

            if window_ys:
                y_min = min(window_ys)
                y_max = max(window_ys)
                y_center = (y_min + y_max) // 2
                expected_center = size // 2

                self.assertAlmostEqual(
                    y_center, expected_center, delta=2,
                    msg=f"Window should be vertically centered in {size}x{size} image"
                )

    def test_window_area_proportional_to_image_size(self):
        """Test total window area scales proportionally with image size squared"""
        test_sizes = [128, 256]

        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        areas = {}
        for size in test_sizes:
            image = np.zeros((size, size, 4), dtype=np.uint8)
            result = self.encoder.encode_region(
                image, parameters, self.model_type
            )

            # Count total pixels with window values
            window_pixels = 0
            for y in range(size):
                for x in range(size):
                    if np.any(result[y, x, :] > 0):
                        window_pixels += 1

            areas[size] = window_pixels

        # Area should scale with square of size ratio
        # 256/128 = 2, but since window position doesn't scale (stays 12px from right),
        # the actual area ratio will be slightly more than 4x
        size_ratio = 256 / 128
        expected_area_ratio = size_ratio ** 2  # ~4.0
        actual_area_ratio = areas[256] / areas[128] if areas[128] > 0 else 0

        # Window area should be close to 4x (allowing for fixed position offset)
        self.assertGreater(
            actual_area_ratio, expected_area_ratio - 0.5,
            msg=f"Window area should scale by at least {expected_area_ratio}x when image doubles"
        )
        self.assertLess(
            actual_area_ratio, expected_area_ratio + 2.0,
            msg=f"Window area should scale by at most {expected_area_ratio + 2.0}x when image doubles"
        )

    def test_different_window_sizes_scale_correctly(self):
        """Test different window sizes scale correctly across image sizes"""
        # Test with a larger window
        parameters_large = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -1.0, 'y1': 0.0, 'z1': 0.9,  # 2.0m wide
            'x2': 1.0, 'y2': 0.0, 'z2': 2.4
        }

        test_sizes = [128, 256]

        for size in test_sizes:
            image = np.zeros((size, size, 4), dtype=np.uint8)
            result = self.encoder.encode_region(
                image, parameters_large, self.model_type
            )

            # Window should still exist at correct position
            window_x = size - 14
            center_y = size // 2

            window_pixel = result[center_y, window_x, :]
            self.assertTrue(
                np.any(window_pixel > 0),
                f"Large window should exist in {size}x{size} image"
            )

            # Count height
            window_ys = []
            for y in range(size):
                if np.any(result[y, window_x, :] > 0):
                    window_ys.append(y)

            if window_ys:
                window_height = len(window_ys)
                # 2.0m window should be ~20px at 128, ~40px at 256
                scale_factor = size / 128
                expected_height = int(20 * scale_factor)

                self.assertAlmostEqual(
                    window_height, expected_height, delta=4,
                    msg=f"Large window (2.0m) height should be ~{expected_height}px in {size}x{size} image"
                )

    def test_window_color_values_consistent_across_sizes(self):
        """Test that encoded color values are consistent regardless of image size"""
        test_sizes = [128, 256, 512]

        parameters = {
            'window_sill_height': 2.5,    # Should encode to ~127
            'window_frame_ratio': 0.5,    # Should encode to ~127 (reversed)
            'window_height': 2.6,  # Should encode to ~127 (reversed)
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        reference_values = None

        for size in test_sizes:
            image = np.zeros((size, size, 4), dtype=np.uint8)
            result = self.encoder.encode_region(
                image, parameters, self.model_type
            )

            # Get window pixel values
            center_y = size // 2
            window_x = size - 14

            window_pixel = result[center_y, window_x, :]

            if reference_values is None:
                reference_values = window_pixel
            else:
                # Values should be the same across all image sizes
                np.testing.assert_allclose(
                    window_pixel[:3],  # RGB channels
                    reference_values[:3],
                    atol=2,
                    err_msg=f"Window color values should be consistent in {size}x{size} image"
                )


class TestWindowIntegration(unittest.TestCase):
    """Integration tests for window encoding with RoomImageBuilder"""

    def test_builder_creates_correct_window(self):
        """Test that RoomImageBuilder creates window correctly"""
        builder = RoomImageBuilder()

        parameters = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }

        image = (builder
                 .reset()
                 .set_model_type(ModelType.DF_DEFAULT)
                 .encode_region(RegionType.WINDOW, parameters)
                 .build())

        # Verify image dimensions
        self.assertEqual(image.shape, (128, 128, 4))

        # Verify window exists at expected location
        window_pixel = image[64, 115, :]
        self.assertTrue(
            np.any(window_pixel > 0),
            "Builder should create window"
        )

    def test_builder_with_all_regions(self):
        """Test building image with window and other regions"""
        builder = RoomImageBuilder()

        all_params = {
            'obstruction_bar': {
                'obstruction_angle_horizon': 45.0,
                'obstruction_angle_zenith': 30.0
            },
            'window': {
                'window_sill_height': 0.9,
                'window_frame_ratio': 0.8,
                'window_height': 1.5,
                'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
                'x2': 0.6, 'y2': 0.0, 'z2': 2.4
            }
        }

        image = (builder
                 .reset()
                 .set_model_type(ModelType.DF_DEFAULT)
                 .encode_region(RegionType.OBSTRUCTION_BAR, all_params['obstruction_bar'])
                 .encode_region(RegionType.WINDOW, all_params['window'])
                 .build())

        # Verify both regions exist
        # Window
        window_pixel = image[64, 115, :]
        self.assertTrue(np.any(window_pixel > 0), "Window should exist")

        # Obstruction bar
        bar_pixel = image[64, 126, :]
        self.assertTrue(np.any(bar_pixel > 0), "Obstruction bar should exist")


if __name__ == '__main__':
    unittest.main()
