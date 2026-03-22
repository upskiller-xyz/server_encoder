"""
V6 encoding service.

V6 produces:
- A single-channel float32 geometric mask (same as V5) with the bounding-box
  obstruction applied (V4-style, single-channel).
- A companion 1-D float32 static-parameter vector whose values (sill_height,
  height_roof_over_floor, window_frame_ratio, floor_height_above_terrain) are
  normalised using the same per-parameter encoding ranges as V1–V4.

The encode_room_image_arrays() method (base-class interface) returns (image, mask)
for callers that do not need the static vector.

The V6-specific encode_room_image_arrays_v6() method returns the full
(image, mask, static_vector) triple.
"""
from typing import Any, Dict, Optional, Tuple
import logging

import numpy as np

from src.core import ModelType, ParameterName, EncodingScheme
from src.components.image_builder.v6_image_director import V6ImageDirector, V6EncodingResult
from src.components.parameter_encoders import EncoderFactory
from src.models import RoomEncodingRequest
from src.validation import ValidationUtils
from src.server.services.encoding_service_v5 import V5EncodingService

logger = logging.getLogger(__name__)


class V6EncodingService(V5EncodingService):
    """
    Encoding service for V6.

    Inherits geometry-only validation and request parsing from V5EncodingService.
    V6 also requires obstruction parameters (horizon, zenith) because they are
    applied to the bounding-box region of the image.

    Overrides:
    - __init__: uses V6ImageDirector
    - validate_parameters: adds horizon/zenith requirement on top of V5 geometry check
    - encode_room_image_arrays: returns (image, mask) — interface parity with base class
    - encode_room_image_arrays_v6: V6-specific — returns (image, mask, static_vector)
    - encode_multi_window_images_arrays_v6: multi-window variant returning V6EncodingResult
    """

    def __init__(self) -> None:
        # Bypass both parent __init__s; we only need V6ImageDirector + EncoderFactory
        self._encoding_scheme = EncodingScheme.V6
        self._director = V6ImageDirector()
        self._encoder_factory = EncoderFactory()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_parameters(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType,
    ) -> Tuple[bool, str]:
        """
        V6 requires room_polygon, window geometry, horizon, and zenith.

        Horizon and zenith are required because they drive the bounding-box
        obstruction that is baked into the image.
        """
        if ParameterName.WINDOWS.value in parameters:
            if not isinstance(parameters[ParameterName.WINDOWS.value], dict):
                return (
                    False,
                    "Parameter 'windows' must be a dictionary mapping window_id to window parameters",
                )
            for window_id, window_params in parameters[ParameterName.WINDOWS.value].items():
                if not isinstance(window_params, dict):
                    return False, f"Window '{window_id}' parameters must be a dictionary"
                merged = {**parameters, **window_params}
                merged.pop(ParameterName.WINDOWS.value, None)
                is_valid, error_msg = self._validate_v6_parameters(merged)
                if not is_valid:
                    return False, f"Window '{window_id}': {error_msg}"
            return True, ""

        return self._validate_v6_parameters(parameters)

    def _validate_v6_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Check geometry + obstruction parameters."""
        # Geometry check (inherited from V5)
        is_valid, error_msg = self._validate_geometry(parameters)
        if not is_valid:
            return False, error_msg

        # Obstruction check
        missing = []
        if ParameterName.HORIZON.value not in parameters:
            missing.append(ParameterName.HORIZON.value)
        if ParameterName.ZENITH.value not in parameters:
            missing.append(ParameterName.ZENITH.value)
        if missing:
            return False, f"Missing required parameters: {', '.join(missing)}"

        return True, ""

    # ------------------------------------------------------------------
    # Encoding — base-class interface (image + mask only)
    # ------------------------------------------------------------------

    def encode_room_image_arrays(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Encode V6 image and mask (drops the static vector).

        Returns:
            Tuple of (image_array, mask_array)
            - image_array: (H, W, 1) float32 in [0, 1]
            - mask_array:  (H, W) uint8 binary room mask or None
        """
        image, mask, _ = self.encode_room_image_arrays_v6(parameters, model_type)
        return image, mask

    # ------------------------------------------------------------------
    # Encoding — V6-specific (image + mask + static vector)
    # ------------------------------------------------------------------

    def encode_room_image_arrays_v6(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Encode V6 image, mask, and static parameter vector.

        Returns:
            Tuple of (image_array, mask_array, static_vector)
            - image_array:   (H, W, 1) float32 in [0, 1]
            - mask_array:    (H, W) uint8 binary room mask or None
            - static_vector: 1-D float32 array (length = len(V6_STATIC_PARAMS))
        """
        parameters = self._ensure_direction_angle(parameters)

        is_valid, error_msg = self.validate_parameters(parameters, model_type)
        if not is_valid:
            logger.error("V6 parameter validation failed: %s", error_msg)
            raise ValueError(error_msg)

        logger.info(
            "Encoding V6 image - model_type: %s, param_count: %d",
            model_type.value, len(parameters),
        )

        image_array, mask_array, static_vector = self._director.construct_from_flat_parameters(
            model_type, parameters
        )

        logger.info(
            "V6 image encoded - shape: %s, static_vector: %s",
            image_array.shape, static_vector,
        )
        return image_array, mask_array, static_vector

    def encode_multi_window_images_arrays_v6(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType,
    ) -> V6EncodingResult:
        """Encode multiple V6 images (one per window), each with a static vector."""
        if ParameterName.WINDOWS.value not in parameters:
            image, mask, static_vector = self.encode_room_image_arrays_v6(parameters, model_type)
            result = V6EncodingResult()
            result.add_window("window_1", image, mask, static_vector)
            return result

        logger.info(
            "Encoding multi-window V6 - model_type: %s, window_count: %d",
            model_type.value, len(parameters[ParameterName.WINDOWS.value]),
        )

        result = self._director.construct_multi_window_images(model_type, parameters)

        logger.info("Multi-window V6 encoded - count: %d", len(result.images))
        return result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Unsupported PNG export (same rationale as V5)
    # ------------------------------------------------------------------

    def encode_room_image(self, parameters: Dict[str, Any], model_type: ModelType):
        raise NotImplementedError(
            "V6 encoding produces a float32 single-channel array and cannot be exported "
            "as a PNG byte stream. Use encode_room_image_arrays_v6() instead."
        )

    def encode_multi_window_images(self, parameters: Dict[str, Any], model_type: ModelType):
        raise NotImplementedError(
            "V6 encoding produces float32 single-channel arrays and cannot be exported "
            "as PNG byte streams. Use encode_multi_window_images_arrays_v6() instead."
        )
