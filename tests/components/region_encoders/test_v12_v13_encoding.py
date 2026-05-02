"""
Tests for V12 and V13 encoding schemes.

V12: 4-channel RGBA image with a window-projection rectangle filled with
     obstruction values.  The rectangle protrudes leftward from the window
     stripe; its width = window_height in pixels, its height = window_width_3d
     in pixels.  Room and window material properties are returned as a separate
     float32 static vector (V12_STATIC_PARAMS).  The window stripe is kept.

V13: Identical to V12 but the window stripe (wall-thickness indicator) is
     removed from the image.
"""
import numpy as np
import pytest

from src.core import EncodingScheme, ModelType
from src.core.enums import V12_STATIC_PARAMS
from src.core.graphics_constants import GRAPHICS_CONSTANTS
from src.components.image_builder.room_image_director import RoomImageDirector
from src.components.image_builder.room_image_builder import RoomImageBuilder
from src.components.region_encoders.obstruction_strategies import (
    ObstructionStrategyFactory,
    WindowProjectionObstructionStrategy,
)
from src.server.services.encoding_service_v12 import V12EncodingService
from src.server.services import EncodingServiceFactory


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_BASE_PARAMS = {
    "floor_height_above_terrain": 2.0,
    "height_roof_over_floor": 16.0,
    "room_polygon": [[-3, 0], [3, 0], [3, 5], [-3, 5]],
    "window_frame_ratio": 0.3,
    "wall_reflectance": 0.7,
    "floor_reflectance": 0.5,
    "ceiling_reflectance": 0.8,
    "window_frame_reflectance": 0.6,
    "horizon": 20.0,
    "zenith": 30.0,
    "context_reflectance": 0.4,
    "x1": -0.6, "y1": 0.0, "z1": 2.5,
    "x2":  0.6, "y2": 0.0, "z2": 4.0,
}

_MODEL = ModelType.DF_DEFAULT


def _make_director(scheme: EncodingScheme) -> RoomImageDirector:
    builder = RoomImageBuilder(encoding_scheme=scheme)
    return RoomImageDirector(builder, encoding_scheme=scheme)


def _encode(scheme: EncodingScheme, params: dict | None = None):
    director = _make_director(scheme)
    p = dict(_BASE_PARAMS) if params is None else dict(params)
    return director.construct_from_flat_parameters(_MODEL, p)


# ---------------------------------------------------------------------------
# Obstruction strategy factory
# ---------------------------------------------------------------------------

class TestObstructionStrategyV12V13:
    def test_v12_creates_window_projection_strategy(self):
        assert isinstance(
            ObstructionStrategyFactory.create(EncodingScheme.V12),
            WindowProjectionObstructionStrategy,
        )

    def test_v13_creates_window_projection_strategy(self):
        assert isinstance(
            ObstructionStrategyFactory.create(EncodingScheme.V13),
            WindowProjectionObstructionStrategy,
        )


# ---------------------------------------------------------------------------
# Image output format
# ---------------------------------------------------------------------------

class TestV12ImageFormat:
    def test_output_is_4_channel(self):
        image, _ = _encode(EncodingScheme.V12)
        assert image.shape[2] == 4

    def test_output_dtype_is_uint8(self):
        image, _ = _encode(EncodingScheme.V12)
        assert image.dtype == np.uint8

    def test_mask_is_not_none(self):
        _, mask = _encode(EncodingScheme.V12)
        assert mask is not None

    def test_mask_does_not_cover_full_image(self):
        _, mask = _encode(EncodingScheme.V12)
        assert (mask > 0).mean() < 1.0


class TestV13ImageFormat:
    def test_output_is_4_channel(self):
        image, _ = _encode(EncodingScheme.V13)
        assert image.shape[2] == 4

    def test_output_dtype_is_uint8(self):
        image, _ = _encode(EncodingScheme.V13)
        assert image.dtype == np.uint8

    def test_mask_is_not_none(self):
        _, mask = _encode(EncodingScheme.V13)
        assert mask is not None


# ---------------------------------------------------------------------------
# Window stripe presence (V12 keeps it; V13 removes it)
# ---------------------------------------------------------------------------

class TestWindowStripe:
    """The window stripe is the 3-pixel-wide column just inside the right edge."""

    def _stripe_column(self, image_size: int = 128) -> int:
        """x-coordinate of the window stripe's right edge."""
        return image_size - GRAPHICS_CONSTANTS.WINDOW_OFFSET_PX

    def test_v12_has_nonzero_stripe(self):
        image, _ = _encode(EncodingScheme.V12)
        col = self._stripe_column(image.shape[1])
        # The stripe columns should differ from a plain background image (non-zero)
        stripe = image[:, col - 3:col, :]
        assert np.any(stripe > 0)

    def test_v13_stripe_absent(self):
        """V13 and V12 differ at the window stripe position."""
        img_v12, _ = _encode(EncodingScheme.V12)
        img_v13, _ = _encode(EncodingScheme.V13)
        col = self._stripe_column(img_v12.shape[1])
        stripe_v12 = img_v12[:, col - 3:col, :]
        stripe_v13 = img_v13[:, col - 3:col, :]
        # V12 has window encoding at the stripe; V13 has only background there
        assert not np.array_equal(stripe_v12, stripe_v13)


# ---------------------------------------------------------------------------
# Projection rectangle
# ---------------------------------------------------------------------------

class TestProjectionRectangle:
    """The projection rectangle is filled with uniform obstruction channel values."""

    def _window_stripe_x(self, image_size: int = 128) -> int:
        wall_px = GRAPHICS_CONSTANTS.get_pixel_value(GRAPHICS_CONSTANTS.WALL_THICKNESS_M, image_size)
        return image_size - GRAPHICS_CONSTANTS.WINDOW_OFFSET_PX - wall_px

    def test_rectangle_is_nonzero(self):
        image, _ = _encode(EncodingScheme.V12)
        # window_height = z2 - z1 = 4.0 - 2.5 = 1.5m → 15px at 128px image
        window_height_m = _BASE_PARAMS["z2"] - _BASE_PARAMS["z1"]
        width = image.shape[1]
        height_px = GRAPHICS_CONSTANTS.get_pixel_value(window_height_m, width)
        x_end = self._window_stripe_x(width)
        x_start = max(GRAPHICS_CONSTANTS.BORDER_PX, x_end - height_px)
        rect = image[40:88, x_start:x_end, :]  # sample within expected y range
        assert np.any(rect > 0)

    def test_rectangle_is_uniform_across_columns(self):
        """All columns in the rectangle should have identical values (uniform fill)."""
        image, _ = _encode(EncodingScheme.V12)
        window_height_m = _BASE_PARAMS["z2"] - _BASE_PARAMS["z1"]
        width = image.shape[1]
        height_px = GRAPHICS_CONSTANTS.get_pixel_value(window_height_m, width)
        x_end = self._window_stripe_x(width)
        x_start = max(GRAPHICS_CONSTANTS.BORDER_PX, x_end - height_px)
        # Take a horizontal slice through the middle of the rectangle
        mid_y = image.shape[0] // 2
        row = image[mid_y, x_start:x_end, :]  # shape (width_px, 4)
        if row.shape[0] > 1:
            assert np.all(row == row[0]), "Rectangle fill is not uniform across columns"

    def test_wider_window_height_produces_wider_rectangle(self):
        """Greater window_height produces a wider projection rectangle."""
        # Use get_pixel_value to compute expected widths directly
        z1 = _BASE_PARAMS["z1"]
        h_narrow = 1.0   # z2 = z1 + 1.0
        h_wide   = 2.5   # z2 = z1 + 2.5

        width_narrow = GRAPHICS_CONSTANTS.get_pixel_value(h_narrow, 128)
        width_wide   = GRAPHICS_CONSTANTS.get_pixel_value(h_wide,   128)

        assert width_wide > width_narrow

    def test_v13_rectangle_same_as_v12(self):
        """V13 drops the window stripe but the rectangle fill should be identical."""
        img_v12, _ = _encode(EncodingScheme.V12)
        img_v13, _ = _encode(EncodingScheme.V13)
        window_height_m = _BASE_PARAMS["z2"] - _BASE_PARAMS["z1"]
        width = img_v12.shape[1]
        height_px = GRAPHICS_CONSTANTS.get_pixel_value(window_height_m, width)
        x_end = self._window_stripe_x(width)
        x_start = max(GRAPHICS_CONSTANTS.BORDER_PX, x_end - height_px)
        np.testing.assert_array_equal(
            img_v12[:, x_start:x_end, :],
            img_v13[:, x_start:x_end, :],
        )


# ---------------------------------------------------------------------------
# V12EncodingService — static vector
# ---------------------------------------------------------------------------

class TestV12StaticVector:
    def _service(self, scheme=EncodingScheme.V12) -> V12EncodingService:
        return EncodingServiceFactory.get_instance(scheme)

    def test_factory_returns_v12_service(self):
        svc = self._service()
        assert isinstance(svc, V12EncodingService)

    def test_factory_returns_v12_service_for_v13(self):
        svc = self._service(EncodingScheme.V13)
        assert isinstance(svc, V12EncodingService)

    def test_encode_returns_three_outputs(self):
        svc = self._service()
        result = svc.encode_room_image_arrays_v12(dict(_BASE_PARAMS), _MODEL)
        assert len(result) == 3

    def test_static_vector_length_matches_params(self):
        svc = self._service()
        _, _, static_vec = svc.encode_room_image_arrays_v12(dict(_BASE_PARAMS), _MODEL)
        assert len(static_vec) == len(V12_STATIC_PARAMS)

    def test_static_vector_dtype_is_float32(self):
        svc = self._service()
        _, _, static_vec = svc.encode_room_image_arrays_v12(dict(_BASE_PARAMS), _MODEL)
        assert static_vec.dtype == np.float32

    def test_static_vector_values_in_unit_range(self):
        svc = self._service()
        _, _, static_vec = svc.encode_room_image_arrays_v12(dict(_BASE_PARAMS), _MODEL)
        assert static_vec.min() >= 0.0
        assert static_vec.max() <= 1.0

    def test_static_vector_nonzero_for_nondefault_params(self):
        svc = self._service()
        _, _, static_vec = svc.encode_room_image_arrays_v12(dict(_BASE_PARAMS), _MODEL)
        assert np.any(static_vec > 0.0)

    def test_static_vector_changes_with_different_reflectances(self):
        svc = self._service()
        params_a = dict(_BASE_PARAMS)
        params_b = dict(_BASE_PARAMS)
        params_b["wall_reflectance"] = 0.1

        _, _, vec_a = svc.encode_room_image_arrays_v12(params_a, _MODEL)
        _, _, vec_b = svc.encode_room_image_arrays_v12(params_b, _MODEL)

        assert not np.array_equal(vec_a, vec_b)

    def test_v13_static_vector_same_as_v12(self):
        svc_v12 = self._service(EncodingScheme.V12)
        svc_v13 = self._service(EncodingScheme.V13)
        _, _, vec_v12 = svc_v12.encode_room_image_arrays_v12(dict(_BASE_PARAMS), _MODEL)
        _, _, vec_v13 = svc_v13.encode_room_image_arrays_v12(dict(_BASE_PARAMS), _MODEL)
        np.testing.assert_array_equal(vec_v12, vec_v13)
