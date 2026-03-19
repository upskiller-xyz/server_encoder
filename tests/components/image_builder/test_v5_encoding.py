"""
Tests for V5 encoding: single-channel float32 geometric mask.

V5 produces a (H, W, 1) float32 image where:
  Background → 0.0
  Room       → 1.0
  Window     → 0.6
"""
import numpy as np
import pytest

from src.core import EncodingScheme, ModelType, RegionType
from src.core.enums import V5_MASK_VALUES
from src.components.image_builder.v5_image_director import V5ImageDirector
from src.server.services.encoding_service_v5 import V5EncodingService


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
_PARAMS = {
    "floor_height_above_terrain": 2.0,
    "height_roof_over_floor": 3.0,
    "room_polygon": [[-3, 0], [3, 0], [3, 5], [-3, 5]],
    "window_frame_ratio": 0.3,
    "x1": -0.6, "y1": 0.0, "z1": 1.5,
    "x2": 0.6, "y2": 0.0, "z2": 3.0,
}

_MODEL = ModelType.DF_DEFAULT


def _encode() -> np.ndarray:
    director = V5ImageDirector()
    image, _ = director.construct_from_flat_parameters(_MODEL, dict(_PARAMS))
    return image


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------
class TestV5OutputFormat:
    """V5 image must be single-channel float32."""

    def test_image_shape_is_h_w_1(self):
        image = _encode()
        assert image.ndim == 3
        assert image.shape[2] == 1, f"Expected 1 channel, got {image.shape[2]}"

    def test_image_dtype_is_float32(self):
        image = _encode()
        assert image.dtype == np.float32

    def test_image_values_in_range_0_to_1(self):
        image = _encode()
        assert image.min() >= 0.0
        assert image.max() <= 1.0


# ---------------------------------------------------------------------------
# Value semantics
# ---------------------------------------------------------------------------
class TestV5MaskValues:
    """Correct intensity values for each region."""

    def test_background_pixels_are_zero(self):
        image = _encode()
        # Top-left corner is always background (2-pixel border)
        assert image[0, 0, 0] == V5_MASK_VALUES[RegionType.BACKGROUND]
        assert image[0, 0, 0] == 0.0

    def test_room_pixels_are_one(self):
        image = _encode()
        # Centre of image should be inside the room
        cx, cy = image.shape[1] // 2, image.shape[0] // 2
        assert image[cy, cx, 0] == V5_MASK_VALUES[RegionType.ROOM]
        assert image[cy, cx, 0] == pytest.approx(1.0)

    def test_window_pixels_are_0_6(self):
        image = _encode()
        # At least some pixels should have the window value
        window_pixels = image[image == V5_MASK_VALUES[RegionType.WINDOW]]
        assert len(window_pixels) > 0, "No window pixels found"
        assert all(p == pytest.approx(0.6) for p in window_pixels)

    def test_only_three_distinct_values(self):
        image = _encode()
        unique_values = np.unique(image)
        allowed = [0.0, 0.6, 1.0]
        for val in unique_values:
            assert any(abs(float(val) - a) < 1e-5 for a in allowed), (
                f"Unexpected pixel value: {float(val)}"
            )


# ---------------------------------------------------------------------------
# Comparison with V2 (different type/shape)
# ---------------------------------------------------------------------------
class TestV5VsV2:
    """V5 output is fundamentally different from V2 (RGBA uint8)."""

    def test_v5_is_single_channel_vs_v2_four_channels(self):
        from src.components.image_builder.room_image_director import RoomImageDirector
        from src.components.image_builder.room_image_builder import RoomImageBuilder

        builder = RoomImageBuilder(encoding_scheme=EncodingScheme.V2)
        v2_director = RoomImageDirector(builder, encoding_scheme=EncodingScheme.V2)
        v2_image, _ = v2_director.construct_from_flat_parameters(_MODEL, dict(_PARAMS))

        v5_image = _encode()
        assert v2_image.shape[2] == 4
        assert v5_image.shape[2] == 1

    def test_v5_dtype_differs_from_v2(self):
        from src.components.image_builder.room_image_director import RoomImageDirector
        from src.components.image_builder.room_image_builder import RoomImageBuilder

        builder = RoomImageBuilder(encoding_scheme=EncodingScheme.V2)
        v2_director = RoomImageDirector(builder, encoding_scheme=EncodingScheme.V2)
        v2_image, _ = v2_director.construct_from_flat_parameters(_MODEL, dict(_PARAMS))

        v5_image = _encode()
        assert v2_image.dtype == np.uint8
        assert v5_image.dtype == np.float32


# ---------------------------------------------------------------------------
# Room mask
# ---------------------------------------------------------------------------
class TestV5RoomMask:
    """V5 director returns a binary room mask consistent with the image."""

    def test_mask_is_returned(self):
        director = V5ImageDirector()
        _, mask = director.construct_from_flat_parameters(_MODEL, dict(_PARAMS))
        assert mask is not None

    def test_mask_shape_matches_image_spatial_dims(self):
        director = V5ImageDirector()
        image, mask = director.construct_from_flat_parameters(_MODEL, dict(_PARAMS))
        assert mask.shape[:2] == image.shape[:2]

    def test_mask_pixels_align_with_room_value(self):
        director = V5ImageDirector()
        image, mask = director.construct_from_flat_parameters(_MODEL, dict(_PARAMS))
        # Every pixel set to 1.0 in the image should be inside the room mask
        room_value_positions = (image[:, :, 0] == 1.0)
        # Room mask area should cover at least the room-value pixels
        # (window may overlap, but room pixels must be within the mask)
        assert np.all(mask[room_value_positions] > 0), (
            "Room-value pixels not covered by room mask"
        )


# ---------------------------------------------------------------------------
# V5EncodingService
# ---------------------------------------------------------------------------
class TestV5EncodingService:
    """Test the V5EncodingService entry point."""

    def test_encode_returns_float32_single_channel(self):
        service = V5EncodingService()
        image, mask = service.encode_room_image_arrays(dict(_PARAMS), _MODEL)
        assert image.dtype == np.float32
        assert image.shape[2] == 1

    def test_encode_validates_missing_room_polygon(self):
        service = V5EncodingService()
        params = {k: v for k, v in _PARAMS.items() if k != "room_polygon"}
        with pytest.raises(ValueError, match="room_polygon"):
            service.encode_room_image_arrays(params, _MODEL)

    def test_encode_validates_missing_window_geometry(self):
        service = V5EncodingService()
        window_keys = {"x1", "y1", "z1", "x2", "y2", "z2", "window_geometry"}
        params = {k: v for k, v in _PARAMS.items() if k not in window_keys}
        with pytest.raises(ValueError, match="window geometry"):
            service.encode_room_image_arrays(params, _MODEL)

    def test_encode_does_not_require_horizon_or_zenith(self):
        """V5 should succeed without obstruction parameters."""
        service = V5EncodingService()
        params = {k: v for k, v in _PARAMS.items() if k not in {"horizon", "zenith"}}
        image, _ = service.encode_room_image_arrays(params, _MODEL)
        assert image is not None

    def test_encode_does_not_require_reflectances(self):
        """V5 should succeed without any reflectance parameters."""
        service = V5EncodingService()
        reflectance_keys = {
            "facade_reflectance", "terrain_reflectance",
            "ceiling_reflectance", "floor_reflectance", "wall_reflectance",
            "window_frame_reflectance",
        }
        params = {k: v for k, v in _PARAMS.items() if k not in reflectance_keys}
        image, _ = service.encode_room_image_arrays(params, _MODEL)
        assert image is not None

    def test_parse_request_does_not_require_height_fields(self):
        """V5 parse_request should not fail when height_roof_over_floor or
        floor_height_above_terrain are absent (they are unused by V5)."""
        service = V5EncodingService()
        geometry_only = {k: v for k, v in _PARAMS.items()
                         if k not in {"height_roof_over_floor", "floor_height_above_terrain"}}
        request_data = {"model_type": "df_default", "parameters": geometry_only}
        req = service.parse_request(request_data)
        assert req.model_type.value == "df_default"


# ---------------------------------------------------------------------------
# V5_MASK_VALUES constant
# ---------------------------------------------------------------------------
class TestV5MaskValuesConstant:
    def test_background_is_zero(self):
        assert V5_MASK_VALUES[RegionType.BACKGROUND] == 0.0

    def test_room_is_one(self):
        assert V5_MASK_VALUES[RegionType.ROOM] == 1.0

    def test_window_is_0_6(self):
        assert V5_MASK_VALUES[RegionType.WINDOW] == pytest.approx(0.6)
