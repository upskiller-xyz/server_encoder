"""
Unit tests for 2-pixel background border enforcement

Tests verify:
1. Top 2 rows remain background
2. Bottom 2 rows remain background
3. Left 2 columns remain background
4. Room, window, and obstruction bar respect the border
5. Border enforcement works across different image sizes
"""

import unittest
import numpy as np
from src.components.region_encoders import (
    BackgroundRegionEncoder,
    RoomRegionEncoder,
    WindowRegionEncoder,
    ObstructionBarEncoder
)
from src.components.enums import ModelType, RegionType
from src.components.image_builder import RoomImageBuilder


class TestBorderEnforcement(unittest.TestCase):
    """Test that 2-pixel border is always background"""

    def setUp(self):
        """Set up test fixtures"""
        self.model_type = ModelType.DF_DEFAULT
        self.bg_encoder = BackgroundRegionEncoder()
        self.room_encoder = RoomRegionEncoder()
        self.window_encoder = WindowRegionEncoder()
        self.obstruction_encoder = ObstructionBarEncoder()

    def test_top_border_remains_background(self):
        """Test that top 2 rows remain background after all encodings"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        # Encode background
        bg_params = {'floor_height_above_terrain': 2.0}
        image = self.bg_encoder.encode_region(image, bg_params, self.model_type)

        # Store background values
        bg_top = image[0:2, :, :].copy()

        # Encode room (large polygon that might extend to edges)
        room_params = {
            'height_roof_over_floor': 3.0,
            'room_polygon': [
                {"x": -5.0, "y": 0.0},
                {"x": 5.0, "y": 0.0},
                {"x": 5.0, "y": 10.0},
                {"x": -5.0, "y": 10.0}
            ],
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }
        image = self.room_encoder.encode_region(image, room_params, self.model_type)

        # Top 2 rows should still match background
        np.testing.assert_array_equal(
            image[0:2, :, :],
            bg_top,
            err_msg="Top 2 rows should remain background"
        )

    def test_bottom_border_remains_background(self):
        """Test that bottom 2 rows remain background after all encodings"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        # Encode background
        bg_params = {'floor_height_above_terrain': 2.0}
        image = self.bg_encoder.encode_region(image, bg_params, self.model_type)

        # Store background values
        bg_bottom = image[126:128, :, :].copy()

        # Encode room
        room_params = {
            'height_roof_over_floor': 3.0,
            'room_polygon': [
                {"x": -5.0, "y": 0.0},
                {"x": 5.0, "y": 0.0},
                {"x": 5.0, "y": 10.0},
                {"x": -5.0, "y": 10.0}
            ],
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }
        image = self.room_encoder.encode_region(image, room_params, self.model_type)

        # Bottom 2 rows should still match background
        np.testing.assert_array_equal(
            image[126:128, :, :],
            bg_bottom,
            err_msg="Bottom 2 rows should remain background"
        )

    def test_left_border_remains_background(self):
        """Test that left 2 columns remain background after all encodings"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        # Encode background
        bg_params = {'floor_height_above_terrain': 2.0}
        image = self.bg_encoder.encode_region(image, bg_params, self.model_type)

        # Store background values
        bg_left = image[:, 0:2, :].copy()

        # Encode room
        room_params = {
            'height_roof_over_floor': 3.0,
            'room_polygon': [
                {"x": -5.0, "y": 0.0},
                {"x": 5.0, "y": 0.0},
                {"x": 5.0, "y": 10.0},
                {"x": -5.0, "y": 10.0}
            ],
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }
        image = self.room_encoder.encode_region(image, room_params, self.model_type)

        # Left 2 columns should still match background
        np.testing.assert_array_equal(
            image[:, 0:2, :],
            bg_left,
            err_msg="Left 2 columns should remain background"
        )

    def test_window_respects_border(self):
        """Test that window encoding respects 2-pixel border"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        # Encode background
        bg_params = {'floor_height_above_terrain': 2.0}
        image = self.bg_encoder.encode_region(image, bg_params, self.model_type)

        bg_values = image.copy()

        # Encode window (centered, should not touch borders)
        window_params = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }
        image = self.window_encoder.encode_region(image, window_params, self.model_type)

        # Check borders haven't changed
        np.testing.assert_array_equal(
            image[0:2, :, :],
            bg_values[0:2, :, :],
            err_msg="Window should not affect top border"
        )
        np.testing.assert_array_equal(
            image[126:128, :, :],
            bg_values[126:128, :, :],
            err_msg="Window should not affect bottom border"
        )

    def test_obstruction_bar_respects_vertical_border(self):
        """Test that obstruction bar respects top/bottom 2-pixel border"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)

        # Encode background
        bg_params = {'floor_height_above_terrain': 2.0}
        image = self.bg_encoder.encode_region(image, bg_params, self.model_type)

        bg_values = image.copy()

        # Encode obstruction bar
        obstruction_params = {
            'obstruction_angle_zenith': 35.0,
            'obstruction_angle_horizon': 45.0
        }
        image = self.obstruction_encoder.encode_region(
            image, obstruction_params, self.model_type
        )

        # Check top/bottom borders haven't changed
        np.testing.assert_array_equal(
            image[0:2, :, :],
            bg_values[0:2, :, :],
            err_msg="Obstruction bar should not affect top border"
        )
        np.testing.assert_array_equal(
            image[126:128, :, :],
            bg_values[126:128, :, :],
            err_msg="Obstruction bar should not affect bottom border"
        )


class TestBorderEnforcementAllRegions(unittest.TestCase):
    """Test border enforcement with all regions combined"""

    def test_all_regions_respect_border(self):
        """Test that all regions together respect the 2-pixel border"""
        image = np.zeros((128, 128, 4), dtype=np.uint8)
        model_type = ModelType.DF_DEFAULT

        # Encode background
        bg_encoder = BackgroundRegionEncoder()
        bg_params = {'floor_height_above_terrain': 2.0}
        image = bg_encoder.encode_region(image, bg_params, model_type)

        bg_border = {
            'top': image[0:2, :, :].copy(),
            'bottom': image[126:128, :, :].copy(),
            'left': image[:, 0:2, :].copy()
        }

        # Encode room
        room_encoder = RoomRegionEncoder()
        room_params = {
            'height_roof_over_floor': 3.0,
            'room_polygon': [
                {"x": -5.0, "y": 0.0},
                {"x": 5.0, "y": 0.0},
                {"x": 5.0, "y": 10.0},
                {"x": -5.0, "y": 10.0}
            ],
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }
        image = room_encoder.encode_region(image, room_params, model_type)

        # Encode window
        window_encoder = WindowRegionEncoder()
        window_params = {
            'window_sill_height': 0.9,
            'window_frame_ratio': 0.8,
            'window_height': 1.5,
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }
        image = window_encoder.encode_region(image, window_params, model_type)

        # Encode obstruction bar
        obstruction_encoder = ObstructionBarEncoder()
        obstruction_params = {
            'obstruction_angle_zenith': 35.0,
            'obstruction_angle_horizon': 45.0
        }
        image = obstruction_encoder.encode_region(image, obstruction_params, model_type)

        # Verify all borders remain background
        np.testing.assert_array_equal(
            image[0:2, :, :],
            bg_border['top'],
            err_msg="Top border should remain background after all encodings"
        )
        np.testing.assert_array_equal(
            image[126:128, :, :],
            bg_border['bottom'],
            err_msg="Bottom border should remain background after all encodings"
        )
        np.testing.assert_array_equal(
            image[:, 0:2, :],
            bg_border['left'],
            err_msg="Left border should remain background after all encodings"
        )


class TestBorderEnforcementScaling(unittest.TestCase):
    """Test border enforcement scales with image size"""

    def test_border_scales_with_256px_image(self):
        """Test that 2-pixel border is enforced in 256x256 images"""
        image = np.zeros((256, 256, 4), dtype=np.uint8)
        model_type = ModelType.DF_DEFAULT

        # Encode background
        bg_encoder = BackgroundRegionEncoder()
        bg_params = {'floor_height_above_terrain': 2.0}
        image = bg_encoder.encode_region(image, bg_params, model_type)

        bg_values = image.copy()

        # Encode room with large polygon
        room_encoder = RoomRegionEncoder()
        room_params = {
            'height_roof_over_floor': 3.0,
            'room_polygon': [
                {"x": -10.0, "y": 0.0},
                {"x": 10.0, "y": 0.0},
                {"x": 10.0, "y": 20.0},
                {"x": -10.0, "y": 20.0}
            ],
            'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
            'x2': 0.6, 'y2': 0.0, 'z2': 2.4
        }
        image = room_encoder.encode_region(image, room_params, model_type)

        # Check borders (still 2 pixels, not scaled)
        np.testing.assert_array_equal(
            image[0:2, :, :],
            bg_values[0:2, :, :],
            err_msg="Top 2-pixel border should remain in 256px image"
        )
        np.testing.assert_array_equal(
            image[254:256, :, :],
            bg_values[254:256, :, :],
            err_msg="Bottom 2-pixel border should remain in 256px image"
        )
        np.testing.assert_array_equal(
            image[:, 0:2, :],
            bg_values[:, 0:2, :],
            err_msg="Left 2-pixel border should remain in 256px image"
        )


class TestBorderEnforcementIntegration(unittest.TestCase):
    """Test border enforcement through RoomImageBuilder"""

    def test_builder_respects_border(self):
        """Test that RoomImageBuilder ensures 2-pixel border"""
        builder = RoomImageBuilder()

        # Create image with all regions
        image = (builder
                 .reset()
                 .set_model_type(ModelType.DF_DEFAULT)
                 .encode_region(RegionType.BACKGROUND, {'floor_height_above_terrain': 2.0})
                 .encode_region(RegionType.ROOM, {
                     'height_roof_over_floor': 3.0,
                     'room_polygon': [
                         {"x": -5.0, "y": 0.0},
                         {"x": 5.0, "y": 0.0},
                         {"x": 5.0, "y": 10.0},
                         {"x": -5.0, "y": 10.0}
                     ],
                     'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
                     'x2': 0.6, 'y2': 0.0, 'z2': 2.4
                 })
                 .encode_region(RegionType.WINDOW, {
                     'window_sill_height': 0.9,
                     'window_frame_ratio': 0.8,
                     'window_height': 1.5,
                     'x1': -0.6, 'y1': 0.0, 'z1': 0.9,
                     'x2': 0.6, 'y2': 0.0, 'z2': 2.4
                 })
                 .encode_region(RegionType.OBSTRUCTION_BAR, {
                     'obstruction_angle_zenith': 35.0,
                     'obstruction_angle_horizon': 45.0
                 })
                 .build())

        # Get background values from border
        bg_top = image[0, 64, :]
        bg_bottom = image[127, 64, :]
        bg_left = image[64, 0, :]

        # All border pixels should match these background values
        for x in range(128):
            for row in range(2):
                np.testing.assert_array_equal(
                    image[row, x, :],
                    bg_top,
                    err_msg=f"Top border pixel at ({row}, {x}) should be background"
                )
                np.testing.assert_array_equal(
                    image[126 + row, x, :],
                    bg_bottom,
                    err_msg=f"Bottom border pixel at ({126+row}, {x}) should be background"
                )

        for y in range(128):
            for col in range(2):
                np.testing.assert_array_equal(
                    image[y, col, :],
                    bg_left,
                    err_msg=f"Left border pixel at ({y}, {col}) should be background"
                )


if __name__ == '__main__':
    unittest.main()
