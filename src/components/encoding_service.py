from typing import Dict, Any, Tuple, Union
import numpy as np
import cv2
from src.components.interfaces import IEncodingService
from src.components.enums import ModelType
from src.components.image_builder import RoomImageBuilder, RoomImageDirector
from src.components.encoders import EncoderFactory
from src.server.services.logging import StructuredLogger


class EncodingService(IEncodingService):
    """
    Service for encoding room parameters into images

    Follows Dependency Injection and Single Responsibility principles
    """

    def __init__(self, logger: StructuredLogger):
        """
        Initialize encoding service

        Args:
            logger: Logger instance for structured logging
        """
        self._logger = logger
        self._builder = RoomImageBuilder()
        self._director = RoomImageDirector(self._builder)
        self._encoder_factory = EncoderFactory()

    def encode_room_image(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> bytes:
        """
        Encode room parameters into PNG image bytes

        Args:
            parameters: Dictionary of encoding parameters
            model_type: The model type to use

        Returns:
            PNG image as bytes

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        is_valid, error_msg = self.validate_parameters(parameters, model_type)
        if not is_valid:
            self._logger.error(f"Parameter validation failed: {error_msg}")
            raise ValueError(error_msg)

        self._logger.info(
            f"Encoding room image - model_type: {model_type.value}, "
            f"param_count: {len(parameters)}"
        )

        # Build image using director
        image_array = self._director.construct_from_flat_parameters(
            model_type,
            parameters
        )
        
        image_array = image_array.astype(np.uint8)
        # Convert RGBA to BGRA for OpenCV
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGRA)
        # Encode to PNG
        success, buffer = cv2.imencode('.png', image_array)
        if not success:
            raise RuntimeError("Failed to encode image to PNG")

        self._logger.info(f"Room image encoded successfully - size: {len(buffer)} bytes")

        return buffer.tobytes()

    def encode_multi_window_images(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> Dict[str, bytes]:
        """
        Encode multiple room images (one per window) into PNG image bytes

        Args:
            parameters: Dictionary of encoding parameters including 'windows' dict
            model_type: The model type to use

        Returns:
            Dictionary mapping window_id to PNG image bytes
            Example: {"window_1": bytes1, "window_2": bytes2}

        Raises:
            ValueError: If parameters are invalid
        """
        # Check if multiple windows are provided
        if "windows" not in parameters:
            # Single window case - return as dict for consistency
            single_image = self.encode_room_image(parameters, model_type)
            return {"window_1": single_image}

        self._logger.info(
            f"Encoding multi-window images - model_type: {model_type.value}, "
            f"window_count: {len(parameters['windows'])}"
        )

        # Build multiple images using director
        image_arrays = self._director.construct_multi_window_images(
            model_type,
            parameters
        )

        # Convert each image to PNG bytes
        result = {}
        for window_id, image_array in image_arrays.items():
            # Convert RGBA to BGRA for OpenCV
            image_array = image_array.astype(np.uint8)
            # Convert RGBA to BGRA for OpenCV
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGRA)

            # Encode to PNG
            success, buffer = cv2.imencode('.png', image_array)
            if not success:
                raise RuntimeError(f"Failed to encode image to PNG for window {window_id}")

            result[window_id] = buffer.tobytes()

        self._logger.info(
            f"Multi-window images encoded successfully - count: {len(result)}"
        )

        return result

    def validate_parameters(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> Tuple[bool, str]:
        """
        Validate encoding parameters

        Args:
            parameters: Parameters to validate
            model_type: Model type being used

        Returns:
            (is_valid, error_message)
        """
        # Check if using unified structure with windows
        if "windows" in parameters and isinstance(parameters["windows"], dict):
            # Validate each window separately
            for window_id, window_params in parameters["windows"].items():
                # Merge shared params with window params for validation
                merged_params = {**parameters, **window_params}
                # Remove windows key from merged params to avoid recursion
                merged_params.pop("windows", None)

                is_valid, error_msg = self._validate_flat_parameters(
                    merged_params, model_type, window_id
                )
                if not is_valid:
                    return False, f"Window '{window_id}': {error_msg}"

            return True, ""
        else:
            # Legacy flat structure - validate directly
            return self._validate_flat_parameters(parameters, model_type)

    def _validate_flat_parameters(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType,
        window_id: str = None
    ) -> Tuple[bool, str]:
        """
        Validate flat parameter structure

        Args:
            parameters: Flat parameters to validate
            model_type: Model type being used
            window_id: Optional window ID for error messages

        Returns:
            (is_valid, error_message)
        """
        # Helper to check if parameter exists (supports both new and legacy names)
        def has_param(new_name: str, legacy_name: str = None) -> bool:
            if new_name in parameters:
                return True
            if legacy_name and legacy_name in parameters:
                return True
            return False

        # Check required parameters (support both new and legacy names)
        missing = []

        # All models need base parameters
        if not has_param("height_roof_over_floor", "height_roof_over_floor"):
            missing.append("height_roof_over_floor")
        if not has_param("window_sill_height", "window_sill_height"):
            missing.append("window_sill_height")
        if not has_param("window_frame_ratio", "window_frame_ratio"):
            missing.append("window_frame_ratio")
        if not has_param("window_height"):
            missing.append("window_height")
        if not has_param("floor_height_above_terrain", "floor_height_above_terrain"):
            missing.append("floor_height_above_terrain")
        if not has_param("obstruction_angle_horizon", "obstruction_angle_horizon"):
            missing.append("obstruction_angle_horizon")
        if not has_param("obstruction_angle_zenith", "obstruction_angle_zenith"):
            missing.append("obstruction_angle_zenith")

        # DA models need orientation
        if model_type in [ModelType.DA_DEFAULT, ModelType.DA_CUSTOM]:
            if not has_param("window_orientation"):
                missing.append("window_orientation")

        # Room polygon is required for all models
        if not has_param("room_polygon"):
            missing.append("room_polygon")

        # Custom models need reflectance parameters (all optional with defaults)
        # No validation needed since they have defaults

        if missing:
            return False, f"Missing required parameters: {', '.join(missing)}"

        # Validate parameter ranges
        for param_name, value in parameters.items():
            try:
                # Skip array parameters (obstruction angles)
                if isinstance(value, (list, np.ndarray)):
                    continue

                # Get valid range
                min_val, max_val = self._encoder_factory.get_parameter_range(param_name)

                # Validate range (handle reversed ranges)
                actual_min = min(min_val, max_val)
                actual_max = max(min_val, max_val)

                if not (actual_min <= float(value) <= actual_max):
                    return False, (
                        f"Parameter '{param_name}' value {value} "
                        f"outside valid range [{min_val}, {max_val}]"
                    )
            except ValueError:
                # Unknown parameter - skip (might be for future use)
                continue

        return True, ""


class EncodingServiceFactory:
    """Factory for creating encoding service instances (Singleton Pattern)"""

    _instance: EncodingService = None

    @classmethod
    def get_instance(cls, logger: StructuredLogger) -> EncodingService:
        """
        Get singleton instance of encoding service

        Args:
            logger: Logger instance

        Returns:
            EncodingService instance
        """
        if cls._instance is None:
            cls._instance = EncodingService(logger)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful for testing)"""
        cls._instance = None
