"""
V5 encoding service.

V5 produces a single-channel float32 image (no parameter encoding).
Validation is geometry-only: room_polygon + window geometry are required;
obstruction and reflectance parameters are not.
"""
from typing import Dict, Any, Tuple, Optional

import numpy as np
import logging

from src.core import ModelType, ParameterName, EncodingScheme
from src.components.image_builder.v5_image_director import V5ImageDirector
from src.components.parameter_encoders import EncoderFactory
from src.models import EncodingResult, RoomEncodingRequest
from src.validation import ValidationUtils
from src.server.services.encoding_service import EncodingService

logger = logging.getLogger(__name__)


class V5EncodingService(EncodingService):
    """
    Encoding service for V5: geometric mask, single-channel float32 output.

    Inherits request parsing and direction-angle calculation from EncodingService.
    Overrides:
    - __init__: uses V5ImageDirector instead of RoomImageBuilder + RoomImageDirector
    - validate_parameters: only geometry is required (no reflectances/obstruction)
    - encode_room_image_arrays: skips uint8 cast; returns float32 (H, W, 1)
    - encode_room_image: raises NotImplementedError (PNG export of float32 single-channel
      is not supported; callers should use encode_room_image_arrays instead)
    - encode_multi_window_images: raises NotImplementedError for the same reason
    """

    def __init__(self) -> None:
        # Intentionally bypass EncodingService.__init__ to avoid creating
        # RoomImageBuilder / RoomImageDirector which are not needed for V5.
        self._encoding_scheme = EncodingScheme.V5
        self._director = V5ImageDirector()
        self._encoder_factory = EncoderFactory()

    # ------------------------------------------------------------------
    # Request parsing (inject defaults for fields required by RoomEncodingRequest
    # but not used by V5; avoids validation errors for missing height parameters)
    # ------------------------------------------------------------------

    def parse_request(self, data: dict) -> RoomEncodingRequest:
        """
        Parse a V5 encode request.

        V5 only requires room geometry.  Inject neutral defaults for fields
        that WindowRequest.from_dict() and RoomEncodingRequest.from_dict()
        require but that V5 never uses:
        - height_roof_over_floor, floor_height_above_terrain (room-level)
        - window_frame_ratio (window-level; only required window field beyond coordinates)
        """
        params = dict(data.get(ParameterName.PARAMETERS.value, {}))

        # Room-level defaults
        if ParameterName.HEIGHT_ROOF_OVER_FLOOR.value not in params:
            params[ParameterName.HEIGHT_ROOF_OVER_FLOOR.value] = 1.0
        if ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value not in params:
            params[ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value] = 0.0

        # Window-level defaults: window_frame_ratio is the only required window
        # field (beyond x1/y1/z1/x2/y2/z2) checked by WindowRequest.from_dict()
        windows = params.get(ParameterName.WINDOWS.value)
        if isinstance(windows, dict):
            patched_windows = {}
            for window_id, window_params in windows.items():
                wp = dict(window_params)
                if ParameterName.WINDOW_FRAME_RATIO.value not in wp:
                    wp[ParameterName.WINDOW_FRAME_RATIO.value] = 0.0
                patched_windows[window_id] = wp
            params[ParameterName.WINDOWS.value] = patched_windows
        elif ParameterName.WINDOW_FRAME_RATIO.value not in params:
            # Flat (single-window) structure
            params[ParameterName.WINDOW_FRAME_RATIO.value] = 0.0

        patched_data = {**data, ParameterName.PARAMETERS.value: params}
        return super().parse_request(patched_data)

    # ------------------------------------------------------------------
    # Validation (geometry-only)
    # ------------------------------------------------------------------

    def validate_parameters(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType,
    ) -> Tuple[bool, str]:
        """V5 only requires room_polygon and window geometry."""
        if ParameterName.WINDOWS.value in parameters:
            if not isinstance(parameters[ParameterName.WINDOWS.value], dict):
                return (
                    False,
                    "Parameter 'windows' must be a dictionary mapping window_id to window parameters",
                )
            for window_id, window_params in parameters[ParameterName.WINDOWS.value].items():
                if not isinstance(window_params, dict):
                    return (
                        False,
                        f"Window '{window_id}' parameters must be a dictionary",
                    )
                merged = {**parameters, **window_params}
                merged.pop(ParameterName.WINDOWS.value, None)
                is_valid, error_msg = self._validate_geometry(merged)
                if not is_valid:
                    return False, f"Window '{window_id}': {error_msg}"
            return True, ""

        return self._validate_geometry(parameters)

    def _validate_geometry(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Check that room_polygon and window geometry are present."""
        missing = []
        if ParameterName.ROOM_POLYGON.value not in parameters:
            missing.append(ParameterName.ROOM_POLYGON.value)

        has_window_coords = ValidationUtils.has_window_coordinates(
            parameters, require_3d=True
        )
        has_window_geom = ParameterName.WINDOW_GEOMETRY.value in parameters
        if not has_window_coords and not has_window_geom:
            missing.append("window geometry (x1,y1,z1,x2,y2,z2 or window_geometry)")

        if missing:
            return False, f"Missing required parameters: {', '.join(missing)}"
        return True, ""

    # ------------------------------------------------------------------
    # Encoding (returns float32, not uint8)
    # ------------------------------------------------------------------

    def encode_room_image_arrays(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Encode room as a V5 geometric mask.

        Returns:
            Tuple of (image_array, mask_array)
            - image_array: (H, W, 1) float32 in range [0, 1]
            - mask_array:  (H, W) uint8 binary room mask or None
        """
        parameters = self._ensure_direction_angle(parameters)

        is_valid, error_msg = self.validate_parameters(parameters, model_type)
        if not is_valid:
            logger.error(f"V5 parameter validation failed: {error_msg}")
            raise ValueError(error_msg)

        logger.info(
            f"Encoding V5 mask - model_type: {model_type.value}, "
            f"param_count: {len(parameters)}"
        )

        image_array, mask_array = self._director.construct_from_flat_parameters(
            model_type, parameters
        )

        logger.info(f"V5 mask encoded successfully - shape: {image_array.shape}")
        return image_array, mask_array

    def encode_multi_window_images_arrays(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType,
    ) -> EncodingResult:
        """Encode multiple V5 masks, one per window."""
        if ParameterName.WINDOWS.value not in parameters:
            single_image, single_mask = self.encode_room_image_arrays(parameters, model_type)
            result = EncodingResult()
            result.add_window("window_1", single_image, single_mask)
            return result

        logger.info(
            f"Encoding multi-window V5 masks - model_type: {model_type.value}, "
            f"window_count: {len(parameters[ParameterName.WINDOWS.value])}"
        )

        result = self._director.construct_multi_window_images(model_type, parameters)

        logger.info(f"Multi-window V5 masks encoded - count: {len(result.images)}")
        return result

    def encode_room_image(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType,
    ):
        raise NotImplementedError(
            "V5 encoding produces a float32 single-channel array and cannot be exported "
            "as a PNG byte stream. Use encode_room_image_arrays() instead."
        )

    def encode_multi_window_images(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType,
    ):
        raise NotImplementedError(
            "V5 encoding produces float32 single-channel arrays and cannot be exported "
            "as PNG byte streams. Use encode_multi_window_images_arrays() instead."
        )
