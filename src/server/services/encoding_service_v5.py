"""
Geometric mask encoding service (V5 and V6).

Both schemes produce a single-channel float32 image (background=0, room=1, window=0.6).
V6 additionally applies bounding-box obstruction to the image and returns a companion
static-parameter vector alongside the image.

  V5: geometry-only input; returns (image, mask)
  V6: geometry + obstruction input; returns (image, mask, static_vector) via
      encode_room_image_arrays_v6() / encode_multi_window_images_arrays_v6()
"""
from typing import Any, Dict, Optional, Tuple
import logging

import numpy as np

from src.core import ModelType, ParameterName, EncodingScheme
from src.components.image_builder.v5_image_director import V5ImageDirector
from src.components.image_builder.v6_image_director import V6ImageDirector, V6EncodingResult
from src.components.parameter_encoders import EncoderFactory
from src.models import EncodingResult, RoomEncodingRequest
from src.validation import ValidationUtils
from src.validation.parameter_validators.encoding_parameter_validator import EncodingParameterValidator
from src.server.services.encoding_service import EncodingService

logger = logging.getLogger(__name__)


class V5EncodingService(EncodingService):
    """
    Encoding service for V5 and V6 (geometric mask family).

    Inherits request parsing from EncodingService.
    Overrides:
    - __init__: selects V5ImageDirector or V6ImageDirector; skips RGBA components
    - validate_parameters: geometry-only for V5; adds horizon/zenith for V6
    - encode_room_image_arrays: returns float32 (H, W, 1); no uint8 cast
    - encode_room_image / encode_multi_window_images: raise NotImplementedError
    - encode_room_image_arrays_v6: V6-only; returns (image, mask, static_vector)
    - encode_multi_window_images_arrays_v6: V6-only multi-window variant
    """

    def __init__(self, encoding_scheme: EncodingScheme = EncodingScheme.V5) -> None:
        # Bypass EncodingService.__init__: RGBA builder/director not needed here
        self._encoding_scheme = encoding_scheme
        self._director = V6ImageDirector() if encoding_scheme == EncodingScheme.V6 else V5ImageDirector()
        self._encoder_factory = EncoderFactory()
        self._validator = EncodingParameterValidator(encoding_scheme, self._encoder_factory)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_parameters(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType,
    ) -> Tuple[bool, str]:
        """
        V5: geometry only (room_polygon + window geometry).
        V6: geometry + horizon + zenith.
        """
        if ParameterName.WINDOWS.value in parameters:
            if not isinstance(parameters[ParameterName.WINDOWS.value], dict):
                return False, "Parameter 'windows' must be a dictionary mapping window_id to window parameters"
            for window_id, window_params in parameters[ParameterName.WINDOWS.value].items():
                if not isinstance(window_params, dict):
                    return False, f"Window '{window_id}' parameters must be a dictionary"
                merged = {**parameters, **window_params}
                merged.pop(ParameterName.WINDOWS.value, None)
                is_valid, error_msg = self._validate_flat(merged)
                if not is_valid:
                    return False, f"Window '{window_id}': {error_msg}"
            return True, ""

        return self._validate_flat(parameters)

    def _validate_flat(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate a single flat parameter dict."""
        missing = []
        if ParameterName.ROOM_POLYGON.value not in parameters:
            missing.append(ParameterName.ROOM_POLYGON.value)

        has_coords = ValidationUtils.has_window_coordinates(parameters, require_3d=True)
        has_geom = ParameterName.WINDOW_GEOMETRY.value in parameters
        if not has_coords and not has_geom:
            missing.append("window geometry (x1,y1,z1,x2,y2,z2 or window_geometry)")

        if self._encoding_scheme == EncodingScheme.V6:
            if ParameterName.HORIZON.value not in parameters:
                missing.append(ParameterName.HORIZON.value)
            if ParameterName.ZENITH.value not in parameters:
                missing.append(ParameterName.ZENITH.value)

        if missing:
            return False, f"Missing required parameters: {', '.join(missing)}"
        return True, ""

    # ------------------------------------------------------------------
    # Encoding — base interface (image + mask)
    # ------------------------------------------------------------------

    def encode_room_image_arrays(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Encode geometric mask.

        Returns:
            (image_array, mask_array)
            - image_array: (H, W, 1) float32 in [0, 1]
            - mask_array:  (H, W) uint8 binary room mask or None
        """
        self._validator.ensure_direction_angle(parameters)

        is_valid, error_msg = self.validate_parameters(parameters, model_type)
        if not is_valid:
            logger.error("V5/V6 parameter validation failed: %s", error_msg)
            raise ValueError(error_msg)

        logger.info("Encoding V%s mask - model_type: %s", self._encoding_scheme.value, model_type.value)

        result = self._director.construct_from_flat_parameters(model_type, parameters)
        image_array = result[0]
        mask_array = result[1]

        logger.info("V%s mask encoded - shape: %s", self._encoding_scheme.value, image_array.shape)
        return image_array, mask_array

    def encode_multi_window_images_arrays(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType,
    ) -> EncodingResult:
        """Encode multiple geometric masks, one per window."""
        if ParameterName.WINDOWS.value not in parameters:
            image, mask = self.encode_room_image_arrays(parameters, model_type)
            result = EncodingResult()
            result.add_window("window_1", image, mask)
            return result

        logger.info(
            "Encoding multi-window V%s masks - model_type: %s, window_count: %d",
            self._encoding_scheme.value, model_type.value,
            len(parameters[ParameterName.WINDOWS.value]),
        )
        result = self._director.construct_multi_window_images(model_type, parameters)
        logger.info("Multi-window V%s masks encoded - count: %d", self._encoding_scheme.value, len(result.images))
        return result

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

        Raises:
            NotImplementedError: When called on a V5 instance
        """
        if self._encoding_scheme != EncodingScheme.V6:
            raise NotImplementedError("encode_room_image_arrays_v6 is only available for V6 encoding.")

        self._validator.ensure_direction_angle(parameters)

        is_valid, error_msg = self.validate_parameters(parameters, model_type)
        if not is_valid:
            logger.error("V6 parameter validation failed: %s", error_msg)
            raise ValueError(error_msg)

        logger.info("Encoding V6 image - model_type: %s, param_count: %d", model_type.value, len(parameters))

        image_array, mask_array, static_vector = self._director.construct_from_flat_parameters(
            model_type, parameters
        )

        logger.info("V6 image encoded - shape: %s, static_vector: %s", image_array.shape, static_vector)
        return image_array, mask_array, static_vector

    def encode_multi_window_images_arrays_v6(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType,
    ) -> V6EncodingResult:
        """
        Encode multiple V6 images (one per window), each with a static vector.

        Raises:
            NotImplementedError: When called on a V5 instance
        """
        if self._encoding_scheme != EncodingScheme.V6:
            raise NotImplementedError("encode_multi_window_images_arrays_v6 is only available for V6 encoding.")

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
    # PNG export not supported
    # ------------------------------------------------------------------

    def encode_room_image(self, parameters: Dict[str, Any], model_type: ModelType):
        raise NotImplementedError(
            "Geometric mask encoding produces float32 single-channel arrays and cannot be "
            "exported as PNG byte streams. Use encode_room_image_arrays() instead."
        )

    def encode_multi_window_images(self, parameters: Dict[str, Any], model_type: ModelType):
        raise NotImplementedError(
            "Geometric mask encoding produces float32 single-channel arrays and cannot be "
            "exported as PNG byte streams. Use encode_multi_window_images_arrays() instead."
        )
