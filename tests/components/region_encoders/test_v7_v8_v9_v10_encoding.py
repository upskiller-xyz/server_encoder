"""
Tests for V7, V8, V9, V10 encoding schemes.

V7: Like V4 (bounding-box obstruction, HSVA) but height_roof_over_floor and
    floor_height_above_terrain are injected as fixed defaults when absent.

V8: Like V7 but heights are supplied via height_vector=[roof, floor] rather
    than as individual parameters.

V9: Like V7 but the alpha channel is dropped — output is 3-channel (H, S, V).
    The room mask comes from the director, not the alpha channel.

V10: Like V8 but 3-channel (alpha dropped).
"""
import numpy as np
import pytest

from src.core import EncodingScheme, ModelType
from src.core.enums import DEFAULT_PARAMETER_VALUES, ParameterName
from src.components.image_builder.room_image_director import RoomImageDirector
from src.components.image_builder.room_image_builder import RoomImageBuilder
from src.components.region_encoders.obstruction_strategies import (
    ObstructionStrategyFactory,
    BoundingBoxObstructionStrategy,
)
from src.server.services.encoding_service import EncodingService
from src.validation.parameter_validators.encoding_parameter_validator import EncodingParameterValidator
from src.components.parameter_encoders import EncoderFactory


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_BASE_PARAMS = {
    "floor_height_above_terrain": 2.0,
    "height_roof_over_floor": 16.0,
    "room_polygon": [[-3, 0], [3, 0], [3, 5], [-3, 5]],
    "window_frame_ratio": 0.3,
    "horizon": 45.0,
    "zenith": 30.0,
    "x1": -0.6, "y1": 0.0, "z1": 2.5,
    "x2": 0.6, "y2": 0.0, "z2": 4.0,
}

_MODEL = ModelType.DF_DEFAULT

_HEIGHT_DEFAULTS = {
    "height_roof_over_floor": DEFAULT_PARAMETER_VALUES[ParameterName.HEIGHT_ROOF_OVER_FLOOR],
    "floor_height_above_terrain": DEFAULT_PARAMETER_VALUES[ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN],
}


def _make_director(scheme: EncodingScheme) -> RoomImageDirector:
    builder = RoomImageBuilder(encoding_scheme=scheme)
    return RoomImageDirector(builder, encoding_scheme=scheme)


def _encode(scheme: EncodingScheme, params: dict | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    director = _make_director(scheme)
    p = dict(_BASE_PARAMS) if params is None else dict(params)
    return director.construct_from_flat_parameters(_MODEL, p)


def _make_validator(scheme: EncodingScheme) -> EncodingParameterValidator:
    return EncodingParameterValidator(scheme, EncoderFactory())


# ---------------------------------------------------------------------------
# Obstruction strategy factory — V7/V8/V9/V10
# ---------------------------------------------------------------------------

class TestObstructionStrategyV7V8V9V10:
    def test_v7_creates_bounding_box_strategy(self):
        assert isinstance(ObstructionStrategyFactory.create(EncodingScheme.V7), BoundingBoxObstructionStrategy)

    def test_v8_creates_bounding_box_strategy(self):
        assert isinstance(ObstructionStrategyFactory.create(EncodingScheme.V8), BoundingBoxObstructionStrategy)

    def test_v9_creates_bounding_box_strategy(self):
        assert isinstance(ObstructionStrategyFactory.create(EncodingScheme.V9), BoundingBoxObstructionStrategy)

    def test_v10_creates_bounding_box_strategy(self):
        assert isinstance(ObstructionStrategyFactory.create(EncodingScheme.V10), BoundingBoxObstructionStrategy)


# ---------------------------------------------------------------------------
# V7 — fixed height defaults
# ---------------------------------------------------------------------------

class TestV7HeightDefaults:
    """V7 injects height defaults when the caller omits them."""

    def _preprocess(self, params: dict) -> dict:
        validator = _make_validator(EncodingScheme.V7)
        p = dict(params)
        validator._preprocess_scheme_params(p)
        return p

    def test_missing_roof_height_is_injected(self):
        params = {k: v for k, v in _BASE_PARAMS.items() if k != "height_roof_over_floor"}
        result = self._preprocess(params)
        assert result["height_roof_over_floor"] == _HEIGHT_DEFAULTS["height_roof_over_floor"]

    def test_missing_floor_height_is_injected(self):
        params = {k: v for k, v in _BASE_PARAMS.items() if k != "floor_height_above_terrain"}
        result = self._preprocess(params)
        assert result["floor_height_above_terrain"] == _HEIGHT_DEFAULTS["floor_height_above_terrain"]

    def test_both_defaults_injected_when_both_absent(self):
        params = {k: v for k, v in _BASE_PARAMS.items()
                  if k not in ("height_roof_over_floor", "floor_height_above_terrain")}
        result = self._preprocess(params)
        assert result["height_roof_over_floor"] == _HEIGHT_DEFAULTS["height_roof_over_floor"]
        assert result["floor_height_above_terrain"] == _HEIGHT_DEFAULTS["floor_height_above_terrain"]

    def test_provided_heights_are_not_overridden(self):
        params = dict(_BASE_PARAMS)
        params["height_roof_over_floor"] = 20.0
        params["floor_height_above_terrain"] = 3.0
        result = self._preprocess(params)
        assert result["height_roof_over_floor"] == 20.0
        assert result["floor_height_above_terrain"] == 3.0

    def test_v7_encodes_without_height_params(self):
        service = EncodingService(EncodingScheme.V7)
        params = {k: v for k, v in _BASE_PARAMS.items()
                  if k not in ("height_roof_over_floor", "floor_height_above_terrain")}
        image, mask = service.encode_room_image_arrays(params, _MODEL)
        assert image is not None
        assert image.shape[2] == 4
        assert np.any(image > 0)

    def test_v7_image_matches_v4_with_same_heights(self):
        params = dict(_BASE_PARAMS)
        params["height_roof_over_floor"] = _HEIGHT_DEFAULTS["height_roof_over_floor"]
        params["floor_height_above_terrain"] = _HEIGHT_DEFAULTS["floor_height_above_terrain"]

        image_v4, _ = _encode(EncodingScheme.V4, params)
        image_v7, _ = _encode(EncodingScheme.V7, params)

        np.testing.assert_array_equal(image_v4, image_v7)


# ---------------------------------------------------------------------------
# V8 — height_vector unpacking
# ---------------------------------------------------------------------------

class TestV8HeightVector:
    """V8 unpacks height_vector=[roof, floor] before validation."""

    def _preprocess(self, params: dict) -> dict:
        validator = _make_validator(EncodingScheme.V8)
        p = dict(params)
        validator._preprocess_scheme_params(p)
        return p

    def test_height_vector_is_unpacked(self):
        params = {k: v for k, v in _BASE_PARAMS.items()
                  if k not in ("height_roof_over_floor", "floor_height_above_terrain")}
        params["height_vector"] = [18.0, 1.5]
        result = self._preprocess(params)
        assert result["height_roof_over_floor"] == 18.0
        assert result["floor_height_above_terrain"] == 1.5

    def test_height_vector_overrides_individual_params(self):
        params = dict(_BASE_PARAMS)
        params["height_vector"] = [22.0, 0.5]
        result = self._preprocess(params)
        assert result["height_roof_over_floor"] == 22.0
        assert result["floor_height_above_terrain"] == 0.5

    def test_height_vector_wrong_length_raises(self):
        validator = _make_validator(EncodingScheme.V8)
        params = dict(_BASE_PARAMS)
        params["height_vector"] = [18.0]  # too short
        with pytest.raises(ValueError, match="height_vector"):
            validator._preprocess_scheme_params(params)

    def test_height_vector_missing_injects_defaults(self):
        params = {k: v for k, v in _BASE_PARAMS.items()
                  if k not in ("height_roof_over_floor", "floor_height_above_terrain")}
        result = self._preprocess(params)
        assert result["height_roof_over_floor"] == _HEIGHT_DEFAULTS["height_roof_over_floor"]
        assert result["floor_height_above_terrain"] == _HEIGHT_DEFAULTS["floor_height_above_terrain"]

    def test_v8_encodes_with_height_vector(self):
        service = EncodingService(EncodingScheme.V8)
        params = {k: v for k, v in _BASE_PARAMS.items()
                  if k not in ("height_roof_over_floor", "floor_height_above_terrain")}
        params["height_vector"] = [18.0, 1.0]
        image, mask = service.encode_room_image_arrays(params, _MODEL)
        assert image is not None
        assert image.shape[2] == 4
        assert np.any(image > 0)

    def test_v8_with_height_vector_matches_v7_with_same_heights(self):
        roof, floor = 18.0, 1.0

        params_v7 = dict(_BASE_PARAMS)
        params_v7["height_roof_over_floor"] = roof
        params_v7["floor_height_above_terrain"] = floor

        params_v8 = {k: v for k, v in _BASE_PARAMS.items()
                     if k not in ("height_roof_over_floor", "floor_height_above_terrain")}
        params_v8["height_vector"] = [roof, floor]

        service_v7 = EncodingService(EncodingScheme.V7)
        service_v8 = EncodingService(EncodingScheme.V8)

        img_v7, _ = service_v7.encode_room_image_arrays(params_v7, _MODEL)
        img_v8, _ = service_v8.encode_room_image_arrays(params_v8, _MODEL)

        np.testing.assert_array_equal(img_v7, img_v8)


# ---------------------------------------------------------------------------
# V9 — 3-channel output (alpha dropped from V7)
# ---------------------------------------------------------------------------

class TestV9ThreeChannel:
    def test_output_is_3_channel(self):
        service = EncodingService(EncodingScheme.V9)
        image, _ = service.encode_room_image_arrays(dict(_BASE_PARAMS), _MODEL)
        assert image.shape[2] == 3, f"Expected 3 channels, got {image.shape[2]}"

    def test_output_dtype_is_uint8(self):
        service = EncodingService(EncodingScheme.V9)
        image, _ = service.encode_room_image_arrays(dict(_BASE_PARAMS), _MODEL)
        assert image.dtype == np.uint8

    def test_rgb_channels_match_v7(self):
        """First 3 channels of V7 must equal the V9 output."""
        params = dict(_BASE_PARAMS)
        service_v7 = EncodingService(EncodingScheme.V7)
        service_v9 = EncodingService(EncodingScheme.V9)

        img_v7, _ = service_v7.encode_room_image_arrays(params, _MODEL)
        img_v9, _ = service_v9.encode_room_image_arrays(params, _MODEL)

        np.testing.assert_array_equal(img_v7[:, :, :3], img_v9)

    def test_mask_is_not_none(self):
        service = EncodingService(EncodingScheme.V9)
        _, mask = service.encode_room_image_arrays(dict(_BASE_PARAMS), _MODEL)
        assert mask is not None

    def test_mask_does_not_cover_full_image(self):
        service = EncodingService(EncodingScheme.V9)
        _, mask = service.encode_room_image_arrays(dict(_BASE_PARAMS), _MODEL)
        coverage = (mask > 0).mean()
        assert coverage < 1.0, f"Mask covers 100% of pixels (coverage={coverage:.2f})"

    def test_v9_without_height_params_uses_defaults(self):
        service = EncodingService(EncodingScheme.V9)
        params = {k: v for k, v in _BASE_PARAMS.items()
                  if k not in ("height_roof_over_floor", "floor_height_above_terrain")}
        image, mask = service.encode_room_image_arrays(params, _MODEL)
        assert image.shape[2] == 3
        assert mask is not None

    def test_multi_window_output_is_3_channel(self):
        service = EncodingService(EncodingScheme.V9)
        params = {
            "room_polygon": _BASE_PARAMS["room_polygon"],
            "height_roof_over_floor": _BASE_PARAMS["height_roof_over_floor"],
            "floor_height_above_terrain": _BASE_PARAMS["floor_height_above_terrain"],
            "horizon": _BASE_PARAMS["horizon"],
            "zenith": _BASE_PARAMS["zenith"],
            "windows": {
                "w1": {k: _BASE_PARAMS[k] for k in ("x1", "y1", "z1", "x2", "y2", "z2", "window_frame_ratio")},
            },
        }
        result = service.encode_multi_window_images_arrays(params, _MODEL)
        for wid in result.window_ids():
            assert result.images[wid].shape[2] == 3, f"Window {wid}: expected 3 channels"


# ---------------------------------------------------------------------------
# V10 — 3-channel output (alpha dropped from V8)
# ---------------------------------------------------------------------------

class TestV10ThreeChannel:
    def test_output_is_3_channel(self):
        service = EncodingService(EncodingScheme.V10)
        params = {k: v for k, v in _BASE_PARAMS.items()
                  if k not in ("height_roof_over_floor", "floor_height_above_terrain")}
        params["height_vector"] = [18.0, 1.0]
        image, _ = service.encode_room_image_arrays(params, _MODEL)
        assert image.shape[2] == 3

    def test_rgb_channels_match_v9_with_same_heights(self):
        roof, floor = 18.0, 1.0

        params_v9 = dict(_BASE_PARAMS)
        params_v9["height_roof_over_floor"] = roof
        params_v9["floor_height_above_terrain"] = floor

        params_v10 = {k: v for k, v in _BASE_PARAMS.items()
                      if k not in ("height_roof_over_floor", "floor_height_above_terrain")}
        params_v10["height_vector"] = [roof, floor]

        service_v9 = EncodingService(EncodingScheme.V9)
        service_v10 = EncodingService(EncodingScheme.V10)

        img_v9, _ = service_v9.encode_room_image_arrays(params_v9, _MODEL)
        img_v10, _ = service_v10.encode_room_image_arrays(params_v10, _MODEL)

        np.testing.assert_array_equal(img_v9, img_v10)

    def test_mask_does_not_cover_full_image(self):
        service = EncodingService(EncodingScheme.V10)
        params = {k: v for k, v in _BASE_PARAMS.items()
                  if k not in ("height_roof_over_floor", "floor_height_above_terrain")}
        params["height_vector"] = [18.0, 1.0]
        _, mask = service.encode_room_image_arrays(params, _MODEL)
        assert mask is not None
        coverage = (mask > 0).mean()
        assert coverage < 1.0, f"Mask covers 100% of pixels (coverage={coverage:.2f})"

    def test_height_vector_wrong_length_raises(self):
        service = EncodingService(EncodingScheme.V10)
        params = {k: v for k, v in _BASE_PARAMS.items()
                  if k not in ("height_roof_over_floor", "floor_height_above_terrain")}
        params["height_vector"] = [18.0]  # too short
        with pytest.raises(ValueError, match="height_vector"):
            service.encode_room_image_arrays(params, _MODEL)
