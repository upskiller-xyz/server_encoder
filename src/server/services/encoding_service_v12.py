"""
Window-projection encoding service (V12 and V13).

Both schemes produce a 4-channel RGBA uint8 image where obstruction data is
encoded as a rectangle that protrudes into the room from the window position:

  - Rectangle width  (x, into room) = window_height in metres → pixels
  - Rectangle height (y)            = window_width_3d → same as window stripe

Room and window material properties are NOT encoded in the image; they are
returned as a separate 1-D float32 static vector (V12_STATIC_PARAMS order).

  V12: background + room + window stripe + projection rectangle + static_vector
  V13: background + room + projection rectangle + static_vector  (no window stripe)
"""
from typing import Any, Dict, Optional, Tuple
import logging

import numpy as np

from src.core import ModelType, ParameterName, EncodingScheme
from src.core.enums import V12_STATIC_PARAMS
from src.components.calculators.parameter_calculator_registry import ParameterCalculatorRegistry
from src.components.parameter_encoders import EncoderFactory
from src.server.services.encoding_service import EncodingService

logger = logging.getLogger(__name__)


class V12EncodingService(EncodingService):
    """
    Encoding service for V12 and V13.

    Extends EncodingService with a static-parameter vector output alongside
    the standard (image, mask) pair.  Use encode_room_image_arrays_v12() to
    retrieve all three outputs.
    """

    def __init__(self, encoding_scheme: EncodingScheme) -> None:
        super().__init__(encoding_scheme)

    # ------------------------------------------------------------------
    # V12-specific encode entry points
    # ------------------------------------------------------------------

    def encode_room_image_arrays_v12(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Encode parameters into (image, room_mask, static_vector).

        Args:
            parameters: Flat parameter dictionary.
            model_type: Model type for HSV override lookup.

        Returns:
            - image:         (H, W, 4) uint8 encoded image
            - room_mask:     (H, W) uint8 binary mask, or None
            - static_vector: 1-D float32 array (length = len(V12_STATIC_PARAMS))
        """
        image, mask = self._director.construct_from_flat_parameters(model_type, parameters)
        static_vector = self._build_static_vector(parameters)
        return image, mask, static_vector

    def encode_multi_window_images_arrays_v12(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType,
    ) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]]:
        """
        Encode one (image, mask, static_vector) triple per window.

        Returns:
            Dict mapping window_id → (image, mask, static_vector).
        """
        windows_key = ParameterName.WINDOWS.value
        if windows_key not in parameters:
            image, mask, static_vector = self.encode_room_image_arrays_v12(parameters, model_type)
            return {"window_1": (image, mask, static_vector)}

        shared = {k: v for k, v in parameters.items() if k != windows_key}
        result = {}
        for window_id, window_params in parameters[windows_key].items():
            merged = {**shared, **window_params}
            image, mask, static_vector = self.encode_room_image_arrays_v12(merged, model_type)
            result[window_id] = (image, mask, static_vector)
        return result

    # ------------------------------------------------------------------
    # Static vector
    # ------------------------------------------------------------------

    def _build_static_vector(self, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Encode V12_STATIC_PARAMS into a normalised float32 vector.

        Derived parameters (window_sill_height, window_height) are calculated
        from the raw parameter dict before lookup.  Missing parameters are
        encoded as 0.0.
        """
        derived = ParameterCalculatorRegistry.calculate_derived_parameters(parameters)
        flat = {**parameters, **derived}

        vector = np.zeros(len(V12_STATIC_PARAMS), dtype=np.float32)
        for idx, param_name in enumerate(V12_STATIC_PARAMS):
            key = param_name.value
            if key not in flat:
                logger.debug("V12 static vector: '%s' not found, using 0.0", key)
                continue
            try:
                encoder = self._encoder_factory.create_encoder(key)
                pixel = encoder.encode(float(flat[key]))
                vector[idx] = float(pixel) / 255.0
            except (ValueError, TypeError) as exc:
                logger.warning("V12 static vector: could not encode '%s': %s", key, exc)

        return vector
