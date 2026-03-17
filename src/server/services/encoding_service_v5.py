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
from src.components.geometry import WindowGeometry, RoomPolygon
from src.components.calculators import ParameterCalculatorRegistry
from src.models import EncodingResult, RoomEncodingRequest
from src.validation import ValidationUtils
from src.server.services.encoding_service import EncodingService

logger = logging.getLogger(__name__)


class V5EncodingService(EncodingService):
    """
    Encoding service for V5: geometric mask, single-channel float32 output.

    Inherits request parsing, direction-angle calculation, and multi-window
    orchestration from EncodingService.  Overrides:
    - __init__: uses V5ImageDirector instead of RoomImageBuilder + RoomImageDirector
    - validate_parameters: only geometry is required (no reflectances/obstruction)
    - encode_room_image_arrays: skips uint8 cast; returns float32
    - encode_room_image: not supported for V5 (PNG encoding of float single-channel is uncommon)
    """

    def __init__(self) -> None:
        # Intentionally bypass EncodingService.__init__ to avoid creating
        # RoomImageBuilder / RoomImageDirector which are not needed for V5.
        self._encoding_scheme = EncodingScheme.V5
        self._director = V5ImageDirector()
        self._encoder_factory = EncoderFactory()

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
