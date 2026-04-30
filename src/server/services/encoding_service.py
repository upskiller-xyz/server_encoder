from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import cv2
import logging

from src.core.enums import FileFormat
from src.models import EncodingResult, EncodedBytesResult, RoomEncodingRequest
from src.core import ModelType, ParameterName, EncodingScheme
from src.components.image_builder import RoomImageBuilder, RoomImageDirector
from src.components.parameter_encoders import EncoderFactory
from src.validation.parameter_validators.encoding_parameter_validator import EncodingParameterValidator

logger = logging.getLogger(__name__)


class EncodingService:
    """
    Service for encoding room parameters into images.

    SRP: orchestrates image construction only. Parameter validation,
    clipping, and direction-angle preprocessing are delegated to
    EncodingParameterValidator.
    """

    def __init__(self, encoding_scheme: EncodingScheme = EncodingScheme.V2):
        self._encoding_scheme = encoding_scheme
        self._builder = RoomImageBuilder(encoding_scheme=encoding_scheme)
        self._director = RoomImageDirector(self._builder, encoding_scheme=encoding_scheme)
        self._encoder_factory = EncoderFactory()
        self._validator = EncodingParameterValidator(encoding_scheme, self._encoder_factory)

    def parse_request(self, data: Dict[str, Any]) -> RoomEncodingRequest:
        try:
            request = RoomEncodingRequest.from_dict(data)
            logger.info(f"Parsed request: model_type={request.model_type.value}, windows={len(request.windows)}")
            return request
        except Exception as e:
            logger.error(f"Failed to parse request: {str(e)}")
            raise ValueError(f"Invalid request format: {str(e)}")

    def validate_request(self, request: RoomEncodingRequest) -> Tuple[bool, str]:
        is_valid, error_msg = request.validate()
        if not is_valid:
            logger.error(f"Request validation failed: {error_msg}")
            return False, error_msg

        parameters = request.to_flat_dict()
        return self._validator.validate(parameters, request.model_type)

    def encode_from_request(
        self,
        request: RoomEncodingRequest,
        return_format: FileFormat = FileFormat.ARRAYS
    ) -> Union[Tuple[np.ndarray, Optional[np.ndarray]], Tuple[bytes, Optional[bytes]]]:
        is_valid, error_msg = self.validate_request(request)
        if not is_valid:
            raise ValueError(error_msg)

        parameters = request.to_flat_dict()
        if return_format == FileFormat.ARRAYS:
            return self.encode_room_image_arrays(parameters, request.model_type)
        return self.encode_room_image(parameters, request.model_type)

    def encode_room_image_arrays(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self._validator.ensure_direction_angle(parameters)

        is_valid, error_msg = self._validator.validate(parameters, model_type)
        if not is_valid:
            logger.error(f"Parameter validation failed: {error_msg}")
            raise ValueError(error_msg)

        logger.info(f"Encoding room image arrays - model_type: {model_type.value}, param_count: {len(parameters)}")

        image_array, mask_array = self._director.construct_from_flat_parameters(model_type, parameters)
        image_array = image_array.astype(np.uint8)

        if self._encoding_scheme in (EncodingScheme.V9, EncodingScheme.V10, EncodingScheme.V11):
            # Alpha channel encodes reflectances that are always at their defaults — drop it.
            # The binary room mask is already returned separately by the director.
            image_array = image_array[:, :, :3]

        logger.info(f"Room image arrays encoded successfully - shape: {image_array.shape}")
        if mask_array is not None:
            logger.info(f"Room mask array encoded successfully - shape: {mask_array.shape}")

        return image_array, mask_array

    def encode_room_image(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> Tuple[bytes, Optional[bytes]]:
        image_array, mask_array = self.encode_room_image_arrays(parameters, model_type)
        if image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGRA)
        success, buffer = cv2.imencode(FileFormat.PNG.value, image_array)
        if not success:
            raise RuntimeError("Failed to encode image to PNG")

        logger.info(f"Room image encoded successfully - size: {len(buffer)} bytes")

        mask_bytes = None
        if mask_array is not None:
            success_mask, mask_buffer = cv2.imencode(FileFormat.PNG.value, mask_array)
            if success_mask:
                mask_bytes = mask_buffer.tobytes()
                logger.info(f"Room mask encoded successfully - size: {len(mask_buffer)} bytes")

        return buffer.tobytes(), mask_bytes

    def encode_multi_window_images_arrays(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> EncodingResult:
        if ParameterName.WINDOWS.value not in parameters:
            single_image, single_mask = self.encode_room_image_arrays(parameters, model_type)
            result = EncodingResult()
            result.add_window("window_1", single_image, single_mask)
            return result

        logger.info(
            f"Encoding multi-window image arrays - model_type: {model_type.value}, "
            f"window_count: {len(parameters[ParameterName.WINDOWS.value])}"
        )

        self._validator.ensure_direction_angle(parameters)
        is_valid, error_msg = self._validator.validate(parameters, model_type)
        if not is_valid:
            logger.error(f"Parameter validation failed: {error_msg}")
            raise ValueError(error_msg)

        result = self._director.construct_multi_window_images(model_type, parameters)

        for window_id in result.window_ids():
            img = result.images[window_id].astype(np.uint8)
            if self._encoding_scheme in (EncodingScheme.V9, EncodingScheme.V10, EncodingScheme.V11):
                img = img[:, :, :3]
            result.images[window_id] = img

        logger.info(f"Multi-window image arrays encoded successfully - count: {len(result.images)}")
        return result

    def encode_multi_window_images(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> EncodedBytesResult:
        if ParameterName.WINDOWS.value not in parameters:
            single_image, single_mask = self.encode_room_image(parameters, model_type)
            result = EncodedBytesResult()
            result.add_window("window_1", single_image, single_mask)
            return result

        logger.info(
            f"Encoding multi-window images - model_type: {model_type.value}, "
            f"window_count: {len(parameters[ParameterName.WINDOWS.value])}"
        )

        array_result = self._director.construct_multi_window_images(model_type, parameters)

        result = EncodedBytesResult()
        for window_id in array_result.window_ids():
            image_array = array_result.get_image(window_id).astype(np.uint8)  # type: ignore
            if self._encoding_scheme in (EncodingScheme.V9, EncodingScheme.V10, EncodingScheme.V11):
                image_array = image_array[:, :, :3]
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGRA)

            success, buffer = cv2.imencode(FileFormat.PNG.value, image_array)
            if not success:
                raise RuntimeError(f"Failed to encode image to PNG for window {window_id}")

            mask_bytes = None
            mask_array = array_result.get_mask(window_id)
            if mask_array is not None:
                success_mask, mask_buffer = cv2.imencode(FileFormat.PNG.value, mask_array)
                if success_mask:
                    mask_bytes = mask_buffer.tobytes()

            result.add_window(window_id, buffer.tobytes(), mask_bytes)

        logger.info(f"Multi-window images encoded successfully - count: {len(result.images)}")
        return result

    def validate_parameters(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> Tuple[bool, str]:
        """Delegate to EncodingParameterValidator. Kept for backwards compatibility."""
        return self._validator.validate(parameters, model_type)
