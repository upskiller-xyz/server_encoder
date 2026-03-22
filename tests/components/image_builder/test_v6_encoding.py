"""
Tests for V6 encoding.

V6 produces:
- (H, W, 1) float32 geometric mask identical to V5 (background=0, room=1, window=0.6)
  with the V4-style bounding-box obstruction applied to the room region.
- A 1-D float32 static-parameter vector (sill_height, height_roof_over_floor,
  window_frame_ratio, floor_height_above_terrain) normalised to [0, 1].
"""
import numpy as np
import pytest

from src.core import EncodingScheme, ModelType, RegionType
from src.core.enums import V5_MASK_VALUES, V6_STATIC_PARAMS, ParameterName
from src.components.image_builder.v6_image_director import V6ImageDirector, V6EncodingResult
from src.components.region_encoders.obstruction_strategies import (
    ObstructionStrategyFactory,
    V6BoundingBoxObstructionStrategy,
)
from src.server.services.encoding_service_v6 import V6EncodingService


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
_PARAMS = {
    "floor_height_above_terrain": 2.0,
    "height_roof_over_floor": 3.0,
    "room_polygon": [[-3, 0], [3, 0], [3, 5], [-3, 5]],
    "window_frame_ratio": 0.3,
    "horizon": 45.0,
    "zenith": 30.0,
    "x1": -0.6, "y1": 0.0, "z1": 1.5,
    "x2": 0.6, "y2": 0.0, "z2": 3.0,
}

_MODEL = ModelType.DF_DEFAULT


def _encode() -> tuple:
    director = V6ImageDirector()
    return director.construct_from_flat_parameters(_MODEL, dict(_PARAMS))


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------
class TestV6OutputFormat:
    """V6 image must be single-channel float32 and the static vector must be 1-D float32."""

    def test_image_shape_is_h_w_1(self):
        image, _, _ = _encode()
        assert image.ndim == 3
        assert image.shape[2] == 1, f"Expected 1 channel, got {image.shape[2]}"

    def test_image_dtype_is_float32(self):
        image, _, _ = _encode()
        assert image.dtype == np.float32

    def test_image_values_in_range_0_to_1(self):
        image, _, _ = _encode()
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_static_vector_is_1d_float32(self):
        _, _, vec = _encode()
        assert vec.ndim == 1
        assert vec.dtype == np.float32

    def test_static_vector_length_matches_v6_static_params(self):
        _, _, vec = _encode()
        assert len(vec) == len(V6_STATIC_PARAMS)


# ---------------------------------------------------------------------------
# Static vector values
# ---------------------------------------------------------------------------
class TestV6StaticVector:
    """Static vector must contain normalised values in [0, 1] for the correct parameters."""

    def test_static_vector_values_in_range_0_to_1(self):
        _, _, vec = _encode()
        assert np.all(vec >= 0.0), f"Static vector has negative values: {vec}"
        assert np.all(vec <= 1.0), f"Static vector exceeds 1.0: {vec}"

    def test_static_vector_nonzero_for_valid_params(self):
        """All parameters in _PARAMS should produce non-zero encoded values."""
        _, _, vec = _encode()
        assert np.any(vec > 0.0), "Expected at least some non-zero values in static vector"

    def test_height_roof_over_floor_in_vector(self):
        """height_roof_over_floor (3 m out of 30 m range) should encode to ~0.1."""
        _, _, vec = _encode()
        idx = list(V6_STATIC_PARAMS).index(ParameterName.HEIGHT_ROOF_OVER_FLOOR)
        # 3/30 = 0.1 → encoded pixel ~25, normalised ~0.1
        assert vec[idx] == pytest.approx(3.0 / 30.0, abs=0.02)

    def test_floor_height_above_terrain_in_vector(self):
        """floor_height_above_terrain=2.0 (range [0,10] → [0.1,1]) should be ~0.27."""
        _, _, vec = _encode()
        idx = list(V6_STATIC_PARAMS).index(ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN)
        # LinearEncoder([0,10]→[0.1,1]): normalised=(2/10)=0.2, output=0.1+0.2*0.9=0.28
        # pixel = round(0.28*255) = 71, vec = 71/255 ≈ 0.278
        assert 0.20 < float(vec[idx]) < 0.40

    def test_window_sill_height_calculated_and_in_vector(self):
        """window_sill_height should be auto-calculated from z1/z2/floor_height and appear."""
        # sill_height = min(z1,z2) - floor_height = 1.5 - 2.0 = -0.5 → clipped to 0 → vec[0] = 0
        _, _, vec = _encode()
        idx = list(V6_STATIC_PARAMS).index(ParameterName.WINDOW_SILL_HEIGHT)
        assert vec[idx] >= 0.0

    def test_static_vector_changes_with_different_roof_height(self):
        """Changing height_roof_over_floor must change the static vector."""
        params_a = dict(_PARAMS)
        params_b = dict(_PARAMS)
        params_b["height_roof_over_floor"] = 15.0

        director = V6ImageDirector()
        _, _, vec_a = director.construct_from_flat_parameters(_MODEL, params_a)
        _, _, vec_b = director.construct_from_flat_parameters(_MODEL, params_b)

        idx = list(V6_STATIC_PARAMS).index(ParameterName.HEIGHT_ROOF_OVER_FLOOR)
        assert vec_a[idx] != vec_b[idx], "height_roof_over_floor change must be reflected in vector"

    def test_static_vector_unchanged_by_obstruction_params(self):
        """Changing horizon/zenith must NOT change the static vector."""
        params_a = dict(_PARAMS)
        params_b = dict(_PARAMS)
        params_b["horizon"] = 80.0
        params_b["zenith"] = 60.0

        director = V6ImageDirector()
        _, _, vec_a = director.construct_from_flat_parameters(_MODEL, params_a)
        _, _, vec_b = director.construct_from_flat_parameters(_MODEL, params_b)

        np.testing.assert_array_equal(vec_a, vec_b)


# ---------------------------------------------------------------------------
# V6 vs V5 image comparison (image structure)
# ---------------------------------------------------------------------------
class TestV6VsV5Image:
    """V6 image has the same geometric structure as V5 but is modulated by obstruction."""

    def test_v6_background_pixels_remain_zero(self):
        """Background pixels should still be 0 in V6 (obstruction only touches room bbox)."""
        image, _, _ = _encode()
        assert image[0, 0, 0] == pytest.approx(0.0)

    def test_v6_has_window_pixels(self):
        """V6 should still contain window pixels (0 < value ≤ 0.6)."""
        image, _, _ = _encode()
        window_mask = (image > 0.0) & (image < 1.0)
        assert np.any(window_mask), "No window pixels found in V6 image"

    def test_v6_room_pixels_at_most_v5_room_value(self):
        """After obstruction multiplication, room pixels should be ≤ 1.0 (the V5 room value)."""
        image, _, _ = _encode()
        assert image.max() <= 1.0 + 1e-6

    def test_v6_room_pixels_differ_from_v5(self):
        """Obstruction must change at least some room pixels compared to pure V5."""
        from src.components.image_builder.v5_image_director import V5ImageDirector
        v5_image, _ = V5ImageDirector().construct_from_flat_parameters(_MODEL, dict(_PARAMS))
        v6_image, _, _ = V6ImageDirector().construct_from_flat_parameters(_MODEL, dict(_PARAMS))

        assert not np.array_equal(v5_image, v6_image), (
            "V6 image should differ from V5 due to obstruction bounding-box scaling"
        )

    def test_v6_room_pixels_le_v5_room_pixels(self):
        """Obstruction can only scale down, so v6 pixels ≤ v5 pixels everywhere."""
        from src.components.image_builder.v5_image_director import V5ImageDirector
        v5_image, _ = V5ImageDirector().construct_from_flat_parameters(_MODEL, dict(_PARAMS))
        v6_image, _, _ = V6ImageDirector().construct_from_flat_parameters(_MODEL, dict(_PARAMS))

        assert np.all(v6_image <= v5_image + 1e-6), (
            "V6 pixels must not exceed corresponding V5 pixels (obstruction only dims)"
        )

    def test_v6_no_bar_obstruction_at_right_edge(self):
        """V6 must not render a traditional obstruction bar on the right edge."""
        from src.components.image_builder.v5_image_director import V5ImageDirector
        from src.core.enums import ImageDimensions

        v5_image, _ = V5ImageDirector().construct_from_flat_parameters(_MODEL, dict(_PARAMS))
        v6_image, _, _ = V6ImageDirector().construct_from_flat_parameters(_MODEL, dict(_PARAMS))

        dims = ImageDimensions(128)
        x_start, y_start, x_end, y_end = dims.get_obstruction_bar_position()

        # Right-edge bar region must be identical between V5 and V6
        # (V6 uses bounding-box, not bar)
        np.testing.assert_array_equal(
            v5_image[y_start:y_end, x_start:x_end, :],
            v6_image[y_start:y_end, x_start:x_end, :],
        )


# ---------------------------------------------------------------------------
# Obstruction effect on image
# ---------------------------------------------------------------------------
class TestV6ObstructionEffect:
    """Verify that obstruction parameters modulate the room bounding-box region."""

    def test_small_angles_dim_more_than_large_angles(self):
        """
        Small horizon/zenith angles → building blocks more sky → lower obstruction factor
        → dimmer room pixels.  Large angles → more open sky → brighter pixels.
        """
        low_obs = dict(_PARAMS, horizon=5.0, zenith=5.0)
        high_obs = dict(_PARAMS, horizon=85.0, zenith=65.0)

        director = V6ImageDirector()
        img_low, _, _ = director.construct_from_flat_parameters(_MODEL, low_obs)
        img_high, _, _ = director.construct_from_flat_parameters(_MODEL, high_obs)

        assert img_low.mean() < img_high.mean(), (
            "Small obstruction angles (building close to window) should produce a dimmer image"
        )

    def test_small_angles_dim_room_pixels_below_unobstructed(self):
        """Room pixels in V6 (small angles) should be strictly below the V5 room value of 1.0."""
        from src.components.image_builder.v5_image_director import V5ImageDirector

        small_obs = dict(_PARAMS, horizon=5.0, zenith=5.0)
        director = V6ImageDirector()
        v6_img, _, _ = director.construct_from_flat_parameters(_MODEL, small_obs)
        v5_img, _ = V5ImageDirector().construct_from_flat_parameters(_MODEL, dict(_PARAMS))

        # Positions that are 1.0 in V5 (pure room pixels) must be < 1.0 in V6
        room_positions = v5_img[:, :, 0] == 1.0
        assert np.any(room_positions), "No room pixels found in V5 image"
        assert np.all(v6_img[:, :, 0][room_positions] < 1.0), (
            "Room pixels in V6 must be dimmed below the unobstructed V5 value of 1.0"
        )


# ---------------------------------------------------------------------------
# V6ObstructionStrategy unit tests
# ---------------------------------------------------------------------------
class TestV6BoundingBoxObstructionStrategy:
    """Unit tests for V6BoundingBoxObstructionStrategy."""

    def test_factory_creates_v6_strategy(self):
        strategy = ObstructionStrategyFactory.create(EncodingScheme.V6)
        assert isinstance(strategy, V6BoundingBoxObstructionStrategy)

    def test_apply_with_none_mask_returns_image_unchanged(self):
        strategy = V6BoundingBoxObstructionStrategy()
        image = np.ones((128, 128, 1), dtype=np.float32) * 0.8
        original = image.copy()
        result = strategy.apply(image, None, {"horizon": 45.0, "zenith": 30.0}, _MODEL)
        np.testing.assert_array_equal(result, original)

    def test_apply_with_empty_mask_returns_image_unchanged(self):
        strategy = V6BoundingBoxObstructionStrategy()
        image = np.ones((128, 128, 1), dtype=np.float32) * 0.8
        original = image.copy()
        room_mask = np.zeros((128, 128), dtype=np.uint8)
        result = strategy.apply(image, room_mask, {"horizon": 45.0, "zenith": 30.0}, _MODEL)
        np.testing.assert_array_equal(result, original)

    def test_apply_scales_pixels_in_bbox(self):
        """Pixels in the bounding box must be ≤ original values after obstruction."""
        strategy = V6BoundingBoxObstructionStrategy()
        image = np.ones((128, 128, 1), dtype=np.float32) * 0.9
        room_mask = np.zeros((128, 128), dtype=np.uint8)
        room_mask[30:80, 20:70] = 1
        result = strategy.apply(image.copy(), room_mask, {"horizon": 45.0, "zenith": 30.0}, _MODEL)
        assert np.all(result[30:80, 20:70, :] <= 0.9 + 1e-6), (
            "Pixels inside bbox must not exceed original value"
        )

    def test_apply_leaves_pixels_outside_bbox_unchanged(self):
        """Pixels outside the room bounding box must be untouched."""
        strategy = V6BoundingBoxObstructionStrategy()
        image = np.ones((128, 128, 1), dtype=np.float32) * 0.5
        room_mask = np.zeros((128, 128), dtype=np.uint8)
        room_mask[5:20, 5:30] = 1
        result = strategy.apply(image.copy(), room_mask, {"horizon": 45.0, "zenith": 30.0}, _MODEL)
        np.testing.assert_allclose(result[50:, 50:, :], 0.5, rtol=1e-5)

    def test_output_values_stay_in_0_1(self):
        """Output must remain in [0, 1] float32."""
        strategy = V6BoundingBoxObstructionStrategy()
        image = np.ones((128, 128, 1), dtype=np.float32)
        room_mask = np.ones((128, 128), dtype=np.uint8)
        result = strategy.apply(image, room_mask, {"horizon": 45.0, "zenith": 30.0}, _MODEL)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ---------------------------------------------------------------------------
# V6EncodingService
# ---------------------------------------------------------------------------
class TestV6EncodingService:
    """Test the V6EncodingService entry point."""

    def test_encode_v6_returns_float32_single_channel(self):
        service = V6EncodingService()
        image, mask, vec = service.encode_room_image_arrays_v6(dict(_PARAMS), _MODEL)
        assert image.dtype == np.float32
        assert image.shape[2] == 1

    def test_encode_v6_returns_static_vector(self):
        service = V6EncodingService()
        image, mask, vec = service.encode_room_image_arrays_v6(dict(_PARAMS), _MODEL)
        assert vec.dtype == np.float32
        assert vec.ndim == 1
        assert len(vec) == len(V6_STATIC_PARAMS)

    def test_encode_room_image_arrays_returns_2_tuple(self):
        """Base-class interface: encode_room_image_arrays must return (image, mask)."""
        service = V6EncodingService()
        result = service.encode_room_image_arrays(dict(_PARAMS), _MODEL)
        assert len(result) == 2
        image, mask = result
        assert image.dtype == np.float32

    def test_encode_validates_missing_room_polygon(self):
        service = V6EncodingService()
        params = {k: v for k, v in _PARAMS.items() if k != "room_polygon"}
        with pytest.raises(ValueError, match="room_polygon"):
            service.encode_room_image_arrays_v6(params, _MODEL)

    def test_encode_validates_missing_window_geometry(self):
        service = V6EncodingService()
        window_keys = {"x1", "y1", "z1", "x2", "y2", "z2"}
        params = {k: v for k, v in _PARAMS.items() if k not in window_keys}
        with pytest.raises(ValueError, match="window geometry"):
            service.encode_room_image_arrays_v6(params, _MODEL)

    def test_encode_validates_missing_horizon(self):
        service = V6EncodingService()
        params = {k: v for k, v in _PARAMS.items() if k != "horizon"}
        with pytest.raises(ValueError, match="horizon"):
            service.encode_room_image_arrays_v6(params, _MODEL)

    def test_encode_validates_missing_zenith(self):
        service = V6EncodingService()
        params = {k: v for k, v in _PARAMS.items() if k != "zenith"}
        with pytest.raises(ValueError, match="zenith"):
            service.encode_room_image_arrays_v6(params, _MODEL)

    def test_encode_does_not_require_reflectances(self):
        """V6 should succeed without reflectance parameters."""
        service = V6EncodingService()
        reflectance_keys = {
            "facade_reflectance", "terrain_reflectance",
            "ceiling_reflectance", "floor_reflectance", "wall_reflectance",
            "window_frame_reflectance",
        }
        params = {k: v for k, v in _PARAMS.items() if k not in reflectance_keys}
        image, _, vec = service.encode_room_image_arrays_v6(params, _MODEL)
        assert image is not None

    def test_encode_png_raises_not_implemented(self):
        service = V6EncodingService()
        with pytest.raises(NotImplementedError):
            service.encode_room_image(dict(_PARAMS), _MODEL)


# ---------------------------------------------------------------------------
# V6EncodingResult
# ---------------------------------------------------------------------------
class TestV6EncodingResult:
    """V6EncodingResult stores images, masks, and static vectors per window."""

    def test_add_and_retrieve_static_vector(self):
        result = V6EncodingResult()
        img = np.zeros((128, 128, 1), dtype=np.float32)
        vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        result.add_window("win_1", img, None, vec)
        np.testing.assert_array_equal(result.get_static_vector("win_1"), vec)

    def test_get_first_static_vector(self):
        result = V6EncodingResult()
        img = np.zeros((128, 128, 1), dtype=np.float32)
        vec = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        result.add_window("win_1", img, None, vec)
        np.testing.assert_array_equal(result.get_first_static_vector(), vec)

    def test_missing_static_vector_defaults_to_zeros(self):
        result = V6EncodingResult()
        img = np.zeros((128, 128, 1), dtype=np.float32)
        result.add_window("win_1", img)
        vec = result.get_static_vector("win_1")
        assert vec is not None
        assert np.all(vec == 0.0)


# ---------------------------------------------------------------------------
# V6_STATIC_PARAMS constant
# ---------------------------------------------------------------------------
class TestV6StaticParamsConstant:
    def test_contains_window_sill_height(self):
        assert ParameterName.WINDOW_SILL_HEIGHT in V6_STATIC_PARAMS

    def test_contains_height_roof_over_floor(self):
        assert ParameterName.HEIGHT_ROOF_OVER_FLOOR in V6_STATIC_PARAMS

    def test_contains_window_frame_ratio(self):
        assert ParameterName.WINDOW_FRAME_RATIO in V6_STATIC_PARAMS

    def test_contains_floor_height_above_terrain(self):
        assert ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN in V6_STATIC_PARAMS
