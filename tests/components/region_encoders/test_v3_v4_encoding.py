"""
Tests for V3 and V4 encoding schemes.

V3: Same channel mapping as V2 (HSV) but the obstruction bar is removed entirely.
V4: Same channel mapping as V2 (HSV) but instead of a bar, the obstruction vector
    is applied to the floor-plan bounding box via element-wise multiplication.
"""
import numpy as np
import pytest

from src.core import EncodingScheme, ModelType, RegionType, ImageDimensions
from src.core.enums import REGION_CHANNEL_MAPPING_V2
from src.components.image_builder.room_image_director import RoomImageDirector
from src.components.image_builder.room_image_builder import RoomImageBuilder
from src.components.region_encoders.obstruction_strategies import (
    ObstructionStrategyFactory,
    ObstructionBarStrategy,
    NoObstructionStrategy,
    BoundingBoxObstructionStrategy,
)


# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------
_BASE_PARAMS = {
    "floor_height_above_terrain": 2.0,
    "height_roof_over_floor": 3.0,
    "room_polygon": [[-3, 0], [3, 0], [3, 5], [-3, 5]],
    "window_frame_ratio": 0.3,
    "horizon": 45.0,
    "zenith": 30.0,
    "x1": -0.6, "y1": 0.0, "z1": 1.5,
    "x2": 0.6, "y2": 0.0, "z2": 3.0,
}


def _make_director(encoding_scheme: EncodingScheme) -> RoomImageDirector:
    builder = RoomImageBuilder(encoding_scheme=encoding_scheme)
    return RoomImageDirector(builder, encoding_scheme=encoding_scheme)


def _encode(encoding_scheme: EncodingScheme) -> np.ndarray:
    director = _make_director(encoding_scheme)
    image, _ = director.construct_from_flat_parameters(ModelType.DF_DEFAULT, dict(_BASE_PARAMS))
    return image


# ---------------------------------------------------------------------------
# Strategy factory tests
# ---------------------------------------------------------------------------
class TestObstructionStrategyFactory:
    """Test that the factory creates the correct strategy for each scheme."""

    def test_v1_creates_bar_strategy(self):
        strategy = ObstructionStrategyFactory.create(EncodingScheme.V1)
        assert isinstance(strategy, ObstructionBarStrategy)

    def test_v2_creates_bar_strategy(self):
        strategy = ObstructionStrategyFactory.create(EncodingScheme.V2)
        assert isinstance(strategy, ObstructionBarStrategy)

    def test_v3_creates_no_obstruction_strategy(self):
        strategy = ObstructionStrategyFactory.create(EncodingScheme.V3)
        assert isinstance(strategy, NoObstructionStrategy)

    def test_v4_creates_bounding_box_strategy(self):
        strategy = ObstructionStrategyFactory.create(EncodingScheme.V4)
        assert isinstance(strategy, BoundingBoxObstructionStrategy)

    def test_unknown_scheme_raises(self):
        with pytest.raises((ValueError, KeyError)):
            ObstructionStrategyFactory.create("unknown")  # type: ignore


# ---------------------------------------------------------------------------
# V3 encoding – obstruction bar absent
# ---------------------------------------------------------------------------
class TestV3NoObstructionBar:
    """V3 should produce an image without any obstruction bar pixels."""

    def _get_bar_region(self, image: np.ndarray) -> np.ndarray:
        dims = ImageDimensions(image.shape[1])
        x_start, y_start, x_end, y_end = dims.get_obstruction_bar_position()
        return image[y_start:y_end, x_start:x_end, :]

    def test_obstruction_bar_area_is_background_only(self):
        """In V3 the bar position on the right edge should contain background data, not bar data."""
        image_v2 = _encode(EncodingScheme.V2)
        image_v3 = _encode(EncodingScheme.V3)

        bar_v2 = self._get_bar_region(image_v2)
        bar_v3 = self._get_bar_region(image_v3)

        # V2 and V3 must differ at the bar position: V2 places bar data there, V3 leaves background
        assert not np.array_equal(bar_v2, bar_v3), (
            "V3 should NOT render the obstruction bar; bar region should differ from V2"
        )

    def test_non_bar_region_equals_v2(self):
        """All pixels outside the bar area should be identical between V2 and V3."""
        image_v2 = _encode(EncodingScheme.V2)
        image_v3 = _encode(EncodingScheme.V3)

        dims = ImageDimensions(128)
        x_start, _, _, _ = dims.get_obstruction_bar_position()

        # Compare only the region to the LEFT of the bar
        assert np.array_equal(image_v2[:, :x_start, :], image_v3[:, :x_start, :]), (
            "V3 non-bar pixels should match V2"
        )

    def test_v3_image_is_not_all_zeros(self):
        """V3 should still encode background/room/window data."""
        image_v3 = _encode(EncodingScheme.V3)
        assert np.any(image_v3 > 0), "V3 image should not be all zeros"


# ---------------------------------------------------------------------------
# V4 encoding – bounding-box obstruction (no bar)
# ---------------------------------------------------------------------------
class TestV4BoundingBoxObstruction:
    """V4 removes the bar and applies the obstruction vector to the room bounding box."""

    def test_no_obstruction_bar_in_bar_position(self):
        """V4 should not render the obstruction bar in the usual bar position."""
        image_v2 = _encode(EncodingScheme.V2)
        image_v4 = _encode(EncodingScheme.V4)

        dims = ImageDimensions(128)
        x_start, y_start, x_end, y_end = dims.get_obstruction_bar_position()

        bar_v2 = image_v2[y_start:y_end, x_start:x_end, :]
        bar_v4 = image_v4[y_start:y_end, x_start:x_end, :]

        assert not np.array_equal(bar_v2, bar_v4), (
            "V4 should not place obstruction bar data at the standard bar position"
        )

    def test_bounding_box_region_differs_from_v3(self):
        """V4 modulates the room bounding box; the result should differ from V3."""
        image_v3 = _encode(EncodingScheme.V3)
        image_v4 = _encode(EncodingScheme.V4)

        assert not np.array_equal(image_v3, image_v4), (
            "V4 bounding-box obstruction should produce different pixels from V3"
        )

    def test_bounding_box_pixels_are_scaled_down(self):
        """After multiplying by obstruction values ≤ 1, bounding-box pixels should be ≤ V3 pixels."""
        image_v3 = _encode(EncodingScheme.V3)
        image_v4 = _encode(EncodingScheme.V4)

        # Find the approximate room bounding box from V3 (where room pixels exist)
        # Room region has non-zero pixels; compare a central column
        # Any non-zero channel in v4 should be <= the corresponding v3 channel
        nonzero_mask = np.any(image_v3 > 0, axis=2)
        if not np.any(nonzero_mask):
            pytest.skip("No non-zero pixels found in V3 image")

        rows, cols = np.where(nonzero_mask)
        # Check that all V4 pixels are <= V3 pixels (multiplication can only reduce values)
        assert np.all(image_v4[rows, cols, :] <= image_v3[rows, cols, :]), (
            "V4 obstruction multiplication should only reduce (or maintain) pixel values"
        )

    def test_v4_image_is_not_all_zeros(self):
        """V4 should still produce a non-zero image."""
        image_v4 = _encode(EncodingScheme.V4)
        assert np.any(image_v4 > 0), "V4 image should not be all zeros"


# ---------------------------------------------------------------------------
# V3 / V4 share V2 channel mapping
# ---------------------------------------------------------------------------
class TestV3V4ChannelMapping:
    """V3 and V4 use the same channel mapping as V2 (not V1)."""

    def test_v3_background_uses_v2_channel_order(self):
        """V3 background region should encode floor_height in green (V2 style)."""
        director = _make_director(EncodingScheme.V3)
        params = dict(_BASE_PARAMS)
        params["floor_height_above_terrain"] = 5.0  # 5m -> ~0.55 normalized -> ~140
        image, _ = director.construct_from_flat_parameters(ModelType.DF_DEFAULT, params)

        # Green channel (index 1) should encode floor height (same as V2)
        bg_green = image[64, 50, 1]  # Background pixel, green channel
        assert bg_green > 0, "V3 background green channel should encode floor height like V2"

    def test_v4_background_uses_v2_channel_order(self):
        """V4 background region should encode floor_height in green (V2 style)."""
        director = _make_director(EncodingScheme.V4)
        params = dict(_BASE_PARAMS)
        params["floor_height_above_terrain"] = 5.0
        image, _ = director.construct_from_flat_parameters(ModelType.DF_DEFAULT, params)

        bg_green = image[64, 50, 1]
        assert bg_green > 0, "V4 background green channel should encode floor height like V2"


# ---------------------------------------------------------------------------
# NoObstructionStrategy unit test
# ---------------------------------------------------------------------------
class TestNoObstructionStrategy:
    """Unit tests for the NoObstructionStrategy class."""

    def test_apply_returns_image_unchanged(self):
        strategy = NoObstructionStrategy()
        image = np.ones((128, 128, 4), dtype=np.uint8) * 100
        original = image.copy()
        room_mask = np.ones((128, 128), dtype=np.uint8)
        result = strategy.apply(image, room_mask, {}, ModelType.DF_DEFAULT)
        assert np.array_equal(result, original)

    def test_apply_with_none_mask(self):
        strategy = NoObstructionStrategy()
        image = np.ones((128, 128, 4), dtype=np.uint8) * 50
        original = image.copy()
        result = strategy.apply(image, None, {}, ModelType.DF_DEFAULT)
        assert np.array_equal(result, original)


# ---------------------------------------------------------------------------
# BoundingBoxObstructionStrategy unit test
# ---------------------------------------------------------------------------
class TestBoundingBoxObstructionStrategy:
    """Unit tests for the BoundingBoxObstructionStrategy class."""

    def test_apply_with_empty_mask_returns_image_unchanged(self):
        strategy = BoundingBoxObstructionStrategy()
        image = np.ones((128, 128, 4), dtype=np.uint8) * 200
        original = image.copy()
        room_mask = np.zeros((128, 128), dtype=np.uint8)  # No room
        params = {"horizon": 45.0, "zenith": 30.0}
        result = strategy.apply(image, room_mask, params, ModelType.DF_DEFAULT)
        assert np.array_equal(result, original)

    def test_apply_with_none_mask_returns_image_unchanged(self):
        strategy = BoundingBoxObstructionStrategy()
        image = np.ones((128, 128, 4), dtype=np.uint8) * 200
        original = image.copy()
        params = {"horizon": 45.0, "zenith": 30.0}
        result = strategy.apply(image, None, params, ModelType.DF_DEFAULT)
        assert np.array_equal(result, original)

    def test_apply_scales_pixels_in_bbox(self):
        """Multiplying by [0, 1] obstruction values reduces pixel intensities."""
        strategy = BoundingBoxObstructionStrategy()
        image = np.ones((128, 128, 4), dtype=np.uint8) * 200

        # Small room mask in the centre
        room_mask = np.zeros((128, 128), dtype=np.uint8)
        room_mask[30:80, 10:70] = 1

        params = {"horizon": 45.0, "zenith": 30.0}
        result = strategy.apply(image.copy(), room_mask, params, ModelType.DF_DEFAULT)

        # Pixels inside the bounding box should be ≤ original (obstruction ≤ 1)
        assert np.all(result[30:80, 10:70, :] <= 200), (
            "BoundingBoxObstructionStrategy should only reduce pixel values"
        )

    def test_bounding_box_derived_from_mask(self):
        """Strategy should touch only the bounding box of the room mask."""
        strategy = BoundingBoxObstructionStrategy()
        image = np.ones((128, 128, 4), dtype=np.uint8) * 200

        # Room mask in top-left corner only
        room_mask = np.zeros((128, 128), dtype=np.uint8)
        room_mask[5:20, 5:30] = 1

        params = {"horizon": 45.0, "zenith": 30.0}
        result = strategy.apply(image.copy(), room_mask, params, ModelType.DF_DEFAULT)

        # Pixels clearly outside the bounding box must be unchanged
        assert np.all(result[50:, 50:, :] == 200), (
            "Pixels outside bounding box must not be modified"
        )
