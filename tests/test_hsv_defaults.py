"""
Test HSV encoding default pixel overrides
"""
import pytest
import numpy as np
from src.components.enums import EncodingScheme, ModelType, RegionType, ChannelType
from src.components.region_encoders import (
    BackgroundRegionEncoder,
    RoomRegionEncoder,
    WindowRegionEncoder,
    ObstructionBarEncoder
)


class TestHSVDefaultPixelOverrides:
    """Test that HSV encoding uses correct default pixel values per model type"""

    def test_background_df_default_hsv_defaults(self):
        """Test background region with DF_DEFAULT model uses HSV pixel overrides"""
        encoder = BackgroundRegionEncoder(encoding_scheme=EncodingScheme.HSV)

        # Minimal parameters - only required, no optional reflectances/orientation
        params = {
            "floor_height_above_terrain": 2.0,
            "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.2,
            "horizon": 45.0,
            "zenith": 30.0,
            "x1": 5.0, "y1": 0.0, "z1": 2.5, "x2": 7.0, "y2": 0.0, "z2": 4.5
        }

        image = np.zeros((128, 128, 4), dtype=np.uint8)
        result = encoder.encode_region(image, params, ModelType.DF_DEFAULT)

        # Check background pixels (away from borders and obstruction bar)
        bg_pixel = result[64, 64]  # Middle of background

        # HSV encoding for DF_DEFAULT should use overrides:
        # Alpha (window_orientation): 190
        # Red (facade_reflectance): 190
        # Blue (terrain_reflectance): 190
        assert bg_pixel[3] == 190, f"Alpha should be 190 for DF_DEFAULT HSV, got {bg_pixel[3]}"
        assert bg_pixel[0] == 190, f"Red should be 190 for DF_DEFAULT HSV, got {bg_pixel[0]}"
        assert bg_pixel[2] == 190, f"Blue should be 190 for DF_DEFAULT HSV, got {bg_pixel[2]}"

    def test_background_da_default_hsv_defaults(self):
        """Test background region with DA_DEFAULT model uses different HSV pixel overrides"""
        encoder = BackgroundRegionEncoder(encoding_scheme=EncodingScheme.HSV)

        params = {
            "floor_height_above_terrain": 2.0,
            "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.2,
            "horizon": 45.0,
            "zenith": 30.0,
            "x1": 5.0, "y1": 0.0, "z1": 2.5, "x2": 7.0, "y2": 0.0, "z2": 4.5,
            "window_orientation": 3.14159  # DA models: direction in radians
        }

        image = np.zeros((128, 128, 4), dtype=np.uint8)
        result = encoder.encode_region(image, params, ModelType.DA_DEFAULT)

        bg_pixel = result[64, 64]

        # DA_DEFAULT should have different defaults than DF_DEFAULT:
        # Red (facade_reflectance): 200 (not 190)
        # Blue (terrain_reflectance): 200 (not 190)
        # Alpha uses actual orientation encoding (not override since it's provided)
        assert bg_pixel[0] == 200, f"Red should be 200 for DA_DEFAULT HSV, got {bg_pixel[0]}"
        assert bg_pixel[2] == 200, f"Blue should be 200 for DA_DEFAULT HSV, got {bg_pixel[2]}"

    def test_room_hsv_defaults_all_models(self):
        """Test room region HSV defaults are consistent across all model types"""
        encoder = RoomRegionEncoder(encoding_scheme=EncodingScheme.HSV)

        for model_type in [ModelType.DF_DEFAULT, ModelType.DA_DEFAULT, ModelType.DF_CUSTOM, ModelType.DA_CUSTOM]:
            params = {
                "floor_height_above_terrain": 2.0,
                "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "height_roof_over_floor": 3.0,
                "window_frame_ratio": 0.2,
                "horizon": 45.0,
                "zenith": 30.0,
                "x1": 5.0, "y1": 0.0, "z1": 2.5, "x2": 7.0, "y2": 0.0, "z2": 4.5
            }
            if model_type in [ModelType.DA_DEFAULT, ModelType.DA_CUSTOM]:
                params["window_orientation"] = 3.14159

            image = np.zeros((128, 128, 4), dtype=np.uint8)
            result = encoder.encode_region(image, params, model_type)

            # Find a room pixel (need to check where room is)
            # Room should be in the left portion of the image
            room_pixel = result[64, 50]

            # All model types should have same room defaults in HSV:
            # Alpha (ceiling_reflectance): 220
            # Green (floor_reflectance): 220
            # Blue (wall_reflectance): 220
            if not np.array_equal(room_pixel, [0, 0, 0, 0]):  # If this is actually room
                assert room_pixel[3] == 220, f"Room alpha should be 220 for {model_type}, got {room_pixel[3]}"
                assert room_pixel[1] == 220, f"Room green should be 220 for {model_type}, got {room_pixel[1]}"
                assert room_pixel[2] == 220, f"Room blue should be 220 for {model_type}, got {room_pixel[2]}"

    def test_obstruction_bar_hsv_defaults(self):
        """Test obstruction bar HSV defaults"""
        encoder = ObstructionBarEncoder(encoding_scheme=EncodingScheme.HSV)

        params = {
            "floor_height_above_terrain": 2.0,
            "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.2,
            "horizon": 45.0,
            "zenith": 30.0,
            "x1": 5.0, "y1": 0.0, "z1": 2.5, "x2": 7.0, "y2": 0.0, "z2": 4.5
        }

        image = np.zeros((128, 128, 4), dtype=np.uint8)
        result = encoder.encode_region(image, params, ModelType.DF_DEFAULT)

        # Obstruction bar is on the right edge
        obs_pixel = result[64, 125]

        # HSV defaults for obstruction bar (all model types):
        # Alpha (balcony_reflectance): 210
        # Green (context_reflectance): 210
        assert obs_pixel[3] == 210, f"Obstruction bar alpha should be 210, got {obs_pixel[3]}"
        assert obs_pixel[1] == 210, f"Obstruction bar green should be 210, got {obs_pixel[1]}"

    def test_window_hsv_defaults(self):
        """Test window HSV defaults"""
        encoder = WindowRegionEncoder(encoding_scheme=EncodingScheme.HSV)

        params = {
            "floor_height_above_terrain": 2.0,
            "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.2,
            "horizon": 45.0,
            "zenith": 30.0,
            "x1": 5.0, "y1": 0.0, "z1": 2.5, "x2": 7.0, "y2": 0.0, "z2": 4.5
        }

        image = np.zeros((128, 128, 4), dtype=np.uint8)
        result = encoder.encode_region(image, params, ModelType.DF_DEFAULT)

        # Window should be around x=116 (12px from right, before obstruction bar)
        window_pixel = result[64, 116]

        # Window HSV defaults (all model types):
        # Alpha (window_frame_reflectance): 230
        if not np.array_equal(window_pixel, [0, 0, 0, 0]):  # If this is actually window
            assert window_pixel[3] == 230, f"Window alpha should be 230, got {window_pixel[3]}"

    def test_rgb_encoding_no_overrides(self):
        """Test that RGB encoding does NOT use HSV overrides"""
        encoder = BackgroundRegionEncoder(encoding_scheme=EncodingScheme.RGB)

        params = {
            "floor_height_above_terrain": 2.0,
            "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.2,
            "horizon": 45.0,
            "zenith": 30.0,
            "x1": 5.0, "y1": 0.0, "z1": 2.5, "x2": 7.0, "y2": 0.0, "z2": 4.5
        }

        image = np.zeros((128, 128, 4), dtype=np.uint8)
        result = encoder.encode_region(image, params, ModelType.DF_DEFAULT)

        bg_pixel = result[64, 64]

        # RGB encoding should use normal encoding, NOT the HSV overrides (190)
        # facade_reflectance default is 1.0, which encodes to 255 in RGB
        assert bg_pixel[0] == 255, f"RGB encoding should not use HSV override, got {bg_pixel[0]}"
        assert bg_pixel[2] == 255, f"RGB encoding should not use HSV override, got {bg_pixel[2]}"

    def test_custom_model_uses_actual_encoding_for_custom_params(self):
        """Test that CUSTOM models use actual encoding when ✅ is shown in CSV"""
        encoder = BackgroundRegionEncoder(encoding_scheme=EncodingScheme.HSV)

        params = {
            "floor_height_above_terrain": 2.0,
            "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.2,
            "horizon": 45.0,
            "zenith": 30.0,
            "x1": 5.0, "y1": 0.0, "z1": 2.5, "x2": 7.0, "y2": 0.0, "z2": 4.5
        }

        image = np.zeros((128, 128, 4), dtype=np.uint8)
        result = encoder.encode_region(image, params, ModelType.DF_CUSTOM)

        bg_pixel = result[64, 64]

        # DF_CUSTOM: facade_reflectance and terrain_reflectance are ✅ (use actual encoding)
        # Default facade_reflectance = 1.0 encodes to 255
        # Default terrain_reflectance = 1.0 encodes to 255
        # But window_orientation still uses override (190)
        assert bg_pixel[0] == 255, f"DF_CUSTOM should encode facade_reflectance normally, got {bg_pixel[0]}"
        assert bg_pixel[2] == 255, f"DF_CUSTOM should encode terrain_reflectance normally, got {bg_pixel[2]}"
        assert bg_pixel[3] == 190, f"DF_CUSTOM should still use window_orientation override, got {bg_pixel[3]}"
