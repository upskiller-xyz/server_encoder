from typing import Dict, Any, Tuple, Union, Optional
import numpy as np
import cv2
from src.components.interfaces import IEncodingService, EncodingResult, EncodedBytesResult
from src.components.enums import ModelType, ParameterName, EncodingScheme
from src.components.image_builder import RoomImageBuilder, RoomImageDirector
from src.components.encoders import EncoderFactory
from src.components.geometry import WindowBorderValidator, WindowHeightValidator, WindowGeometry, RoomPolygon
from src.components.parameter_calculators import ParameterCalculatorRegistry
from src.server.services.logging import StructuredLogger


class EncodingService(IEncodingService):
    """
    Service for encoding room parameters into images

    Follows Dependency Injection and Single Responsibility principles
    """

    def __init__(self, logger: StructuredLogger, encoding_scheme: EncodingScheme = EncodingScheme.RGB):
        """
        Initialize encoding service

        Args:
            logger: Logger instance for structured logging
            encoding_scheme: Encoding scheme to use (default: HSV)
        """
        self._logger = logger
        self._encoding_scheme = encoding_scheme
        self._builder = RoomImageBuilder(encoding_scheme=encoding_scheme)
        self._director = RoomImageDirector(self._builder)
        self._encoder_factory = EncoderFactory()

    def encode_room_image_arrays(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Encode room parameters into numpy arrays

        Args:
            parameters: Dictionary of encoding parameters
            model_type: The model type to use

        Returns:
            Tuple of (image_array, mask_array)
            - image_array: 128×128 RGBA numpy array (uint8)
            - mask_array: 128×128 binary mask array (uint8) or None

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        is_valid, error_msg = self.validate_parameters(parameters, model_type)
        if not is_valid:
            self._logger.error(f"Parameter validation failed: {error_msg}")
            raise ValueError(error_msg)

        self._logger.info(
            f"Encoding room image arrays - model_type: {model_type.value}, "
            f"param_count: {len(parameters)}"
        )

        # Build image using director
        image_array, mask_array = self._director.construct_from_flat_parameters(
            model_type,
            parameters
        )

        image_array = image_array.astype(np.uint8)

        self._logger.info(f"Room image arrays encoded successfully - shape: {image_array.shape}")
        if mask_array is not None:
            self._logger.info(f"Room mask array encoded successfully - shape: {mask_array.shape}")

        return image_array, mask_array

    def encode_room_image(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> Tuple[bytes, Optional[bytes]]:
        """
        Encode room parameters into PNG image bytes

        Args:
            parameters: Dictionary of encoding parameters
            model_type: The model type to use

        Returns:
            Tuple of (image_bytes, mask_bytes)
            - image_bytes: PNG image as bytes
            - mask_bytes: PNG mask as bytes or None

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
        image_array, mask_array = self._director.construct_from_flat_parameters(
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

        # Encode mask if available
        mask_bytes = None
        if mask_array is not None:
            success_mask, mask_buffer = cv2.imencode('.png', mask_array)
            if success_mask:
                mask_bytes = mask_buffer.tobytes()
                self._logger.info(f"Room mask encoded successfully - size: {len(mask_buffer)} bytes")

        return buffer.tobytes(), mask_bytes

    def encode_multi_window_images_arrays(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> EncodingResult:
        """
        Encode multiple room images (one per window) into numpy arrays

        Args:
            parameters: Dictionary of encoding parameters including 'windows' dict
            model_type: The model type to use

        Returns:
            EncodingResult containing images and masks for all windows

        Raises:
            ValueError: If parameters are invalid
        """
        # Check if multiple windows are provided
        if ParameterName.WINDOWS.value not in parameters:
            # Single window case
            single_image, single_mask = self.encode_room_image_arrays(parameters, model_type)
            result = EncodingResult()
            result.add_window("window_1", single_image, single_mask)
            return result

        self._logger.info(
            f"Encoding multi-window image arrays - model_type: {model_type.value}, "
            f"window_count: {len(parameters[ParameterName.WINDOWS.value])}"
        )

        # Build multiple images using director
        result = self._director.construct_multi_window_images(
            model_type,
            parameters
        )

        # Convert to uint8
        for window_id in result.window_ids():
            result.images[window_id] = result.images[window_id].astype(np.uint8)

        self._logger.info(
            f"Multi-window image arrays encoded successfully - count: {len(result.images)}"
        )

        return result

    def encode_multi_window_images(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> EncodedBytesResult:
        """
        Encode multiple room images (one per window) into PNG image bytes

        Args:
            parameters: Dictionary of encoding parameters including 'windows' dict
            model_type: The model type to use

        Returns:
            EncodedBytesResult containing PNG image bytes and mask bytes for all windows

        Raises:
            ValueError: If parameters are invalid
        """
        # Check if multiple windows are provided
        if ParameterName.WINDOWS.value not in parameters:
            # Single window case
            single_image, single_mask = self.encode_room_image(parameters, model_type)
            result = EncodedBytesResult()
            result.add_window("window_1", single_image, single_mask)
            return result

        self._logger.info(
            f"Encoding multi-window images - model_type: {model_type.value}, "
            f"window_count: {len(parameters[ParameterName.WINDOWS.value])}"
        )

        # Build multiple images using director
        array_result = self._director.construct_multi_window_images(
            model_type,
            parameters
        )

        # Convert each image and mask to PNG bytes
        result = EncodedBytesResult()
        for window_id in array_result.window_ids():
            image_array = array_result.get_image(window_id)
            # Convert RGBA to BGRA for OpenCV
            image_array = image_array.astype(np.uint8)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGRA)

            # Encode to PNG
            success, buffer = cv2.imencode('.png', image_array)
            if not success:
                raise RuntimeError(f"Failed to encode image to PNG for window {window_id}")

            image_bytes = buffer.tobytes()

            # Encode mask if available
            mask_array = array_result.get_mask(window_id)
            mask_bytes = None
            if mask_array is not None:
                success_mask, mask_buffer = cv2.imencode('.png', mask_array)
                if success_mask:
                    mask_bytes = mask_buffer.tobytes()

            result.add_window(window_id, image_bytes, mask_bytes)

        self._logger.info(
            f"Multi-window images encoded successfully - count: {len(result.images)}"
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
        if ParameterName.WINDOWS.value in parameters:
            # Check if windows is a dict (expected format)
            if not isinstance(parameters[ParameterName.WINDOWS.value], dict):
                return False, "Parameter 'windows' must be a dictionary mapping window_id to window parameters"

            # Validate each window separately
            for window_id, window_params in parameters[ParameterName.WINDOWS.value].items():
                # Check if window_params is a dict
                if not isinstance(window_params, dict):
                    return False, f"Window '{window_id}' parameters must be a dictionary, got {type(window_params).__name__}"

                # Merge shared params with window params for validation
                merged_params = {**parameters, **window_params}
                # Remove windows key from merged params to avoid recursion
                merged_params.pop(ParameterName.WINDOWS.value, None)

                is_valid, error_msg = self._validate_flat_parameters(
                    merged_params, model_type, window_id
                )
                if not is_valid:
                    return False, f"Window '{window_id}': {error_msg}"

                # Write back clipped values to window_params
                # Only write back parameters that were in window_params originally
                for param_name in list(window_params.keys()):
                    if param_name in merged_params:
                        window_params[param_name] = merged_params[param_name]

                # Write back clipped shared parameters to main parameters dict
                # This ensures clipped values are available when constructing images
                for param_name in self._CLIPPING_CONFIG.keys():
                    if param_name in merged_params and param_name not in window_params:
                        # This is a shared parameter that was clipped
                        parameters[param_name] = merged_params[param_name]

            return True, ""
        else:
            # Legacy flat structure - validate directly
            return self._validate_flat_parameters(parameters, model_type)

    # Parameters that support clipping (Strategy Pattern)
    # Format: {param_name: (clip_min, clip_max, reject_below_min)}
    # - clip_min/max: values to clip to
    # - reject_below_min: if True, reject values < min instead of clipping
    _CLIPPING_CONFIG = {
        ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value: (0.0, 10.0, True),        # Reject < 0, clip > 10
        ParameterName.HEIGHT_ROOF_OVER_FLOOR.value: (15.0, 30.0, True),           # Reject <= 0, clip < 15 or > 30
        ParameterName.OBSTRUCTION_ANGLE_HORIZON.value: (0.0, 90.0, False),        # Clip both min and max
        ParameterName.OBSTRUCTION_ANGLE_ZENITH.value: (0.0, 70.0, False),         # Clip both min and max
    }

    def _clip_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Clip parameters that support clipping instead of validation errors.

        Uses Strategy Pattern with _CLIPPING_CONFIG map.

        Clipping rules:
        - floor_height_above_terrain: values > 10.0 clipped to 10.0, values < 0.0 rejected
        - height_roof_over_floor: values > 30.0 clipped to 30.0, values < 15.0 clipped to 15.0, values <= 0.0 rejected
        - obstruction_angle_horizon: values clipped to [0.0, 90.0] range
        - obstruction_angle_zenith: values clipped to [0.0, 70.0] range

        Args:
            parameters: Parameters to clip (modified in place)

        Raises:
            ValueError: If parameter value cannot be processed or is rejected
        """
        for param_name, (min_val, max_val, reject_below_min) in self._CLIPPING_CONFIG.items():
            if param_name not in parameters:
                continue

            try:
                param_value = parameters[param_name]

                # Skip arrays (obstruction angles can be arrays)
                # These are handled separately in validation
                if isinstance(param_value, (list, np.ndarray)):
                    continue

                value = float(param_value)
                original_value = value
                clipped = False

                # Handle minimum bound
                # Special case for height_roof_over_floor: must be > 0, clip to min if < min
                if param_name == ParameterName.HEIGHT_ROOF_OVER_FLOOR.value:
                    if value <= 0.0:
                        raise ValueError(
                            f"Parameter '{param_name}' value {value} not supported. "
                            f"Must be greater than 0. "
                            f"Values <= 0 are not supported."
                        )
                    elif value < min_val:
                        value = min_val
                        clipped = True
                elif value < min_val:
                    if reject_below_min:
                        raise ValueError(
                            f"Parameter '{param_name}' value {value} not supported. "
                            f"Valid range is [{min_val}, {max_val}]. "
                            f"Values below {min_val} are not supported."
                        )
                    else:
                        value = min_val
                        clipped = True

                # Handle maximum bound
                if value > max_val:
                    value = max_val
                    clipped = True

                # Update parameter if clipped
                if clipped:
                    self._logger.warning(
                        f"Parameter '{param_name}' value {original_value} outside range [{min_val}, {max_val}]. "
                        f"Value will be clipped to {value}."
                    )
                    parameters[param_name] = value

            except (TypeError, ValueError) as e:
                if "not supported" in str(e):
                    raise
                raise ValueError(
                    f"Parameter '{param_name}' has invalid value: {parameters.get(param_name)}. "
                    f"Error: {str(e)}"
                )

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
        # Calculate derived parameters first
        # This modifies parameters dict in place by adding calculated values
        # Pass logger to log warnings instead of failing
        calculated_params = ParameterCalculatorRegistry.calculate_derived_parameters(
            parameters,
            logger=self._logger
        )
        parameters.update(calculated_params)

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
        if not has_param(ParameterName.HEIGHT_ROOF_OVER_FLOOR.value):
            missing.append(ParameterName.HEIGHT_ROOF_OVER_FLOOR.value)
        if not has_param(ParameterName.WINDOW_FRAME_RATIO.value):
            missing.append(ParameterName.WINDOW_FRAME_RATIO.value)
        if not has_param(ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value):
            missing.append(ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value)
        if not has_param(ParameterName.OBSTRUCTION_ANGLE_HORIZON.value):
            missing.append(ParameterName.OBSTRUCTION_ANGLE_HORIZON.value)
        if not has_param(ParameterName.OBSTRUCTION_ANGLE_ZENITH.value):
            missing.append(ParameterName.OBSTRUCTION_ANGLE_ZENITH.value)

        # DA models need orientation
        if model_type in [ModelType.DA_DEFAULT, ModelType.DA_CUSTOM]:
            if not has_param(ParameterName.WINDOW_ORIENTATION.value):
                missing.append(ParameterName.WINDOW_ORIENTATION.value)

        # Room polygon is required for all models
        if not has_param(ParameterName.ROOM_POLYGON.value):
            missing.append(ParameterName.ROOM_POLYGON.value)

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

                # Skip validation for parameters with clipping enabled
                # They are already validated and clipped in _clip_parameters
                if param_name in self._CLIPPING_CONFIG:
                    continue

                # For other parameters, validate normally
                try:
                    float_value = float(value)
                except (TypeError, ValueError) as e:
                    return False, (
                        f"Parameter '{param_name}' has invalid value type: {type(value).__name__}. "
                        f"Expected numeric value, got: {value}. Error: {str(e)}"
                    )

                if not (actual_min <= float_value <= actual_max):
                    return False, (
                        f"Parameter '{param_name}' value {value} "
                        f"outside valid range [{min_val}, {max_val}]"
                    )
            except ValueError as e:
                # Unknown parameter - skip (might be for future use)
                if "Unknown parameter" in str(e):
                    continue
                # Re-raise if it's a different ValueError
                return False, f"Error validating parameter '{param_name}': {str(e)}"
            except Exception as e:
                # Catch any unexpected errors and report them with context
                return False, (
                    f"Unexpected error validating parameter '{param_name}' with value {value}: "
                    f"{type(e).__name__}: {str(e)}"
                )

        # Validate window geometry placement (if window coordinates and room_polygon are present)
        room_polygon = parameters.get(ParameterName.ROOM_POLYGON.value)

        # Check if we have window coordinates (either as nested object or flat)
        has_window_coords = (
            ParameterName.WINDOW_GEOMETRY.value in parameters or
            all(k in parameters for k in [
                ParameterName.X1.value, ParameterName.Y1.value, ParameterName.Z1.value,
                ParameterName.X2.value, ParameterName.Y2.value, ParameterName.Z2.value
            ])
        )

        if has_window_coords and room_polygon:
            # Validate window is on polygon border
            is_valid, error_msg = WindowBorderValidator.validate_from_dict(
                window_data=parameters,
                polygon_data=room_polygon
            )
            if not is_valid:
                return False, f"Window geometry validation failed: {error_msg}"

            # Validate window height is between floor and roof
            floor_height = parameters.get(ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value)
            height_roof_over_floor = parameters.get(ParameterName.HEIGHT_ROOF_OVER_FLOOR.value)

            if floor_height is not None and height_roof_over_floor is not None:
                roof_height = floor_height + height_roof_over_floor

                is_valid, error_msg = WindowHeightValidator.validate_from_parameters(
                    window_geometry_data=parameters.get(ParameterName.WINDOW_GEOMETRY.value) or parameters,
                    floor_height=floor_height,
                    roof_height=roof_height
                )
                if not is_valid:
                    return False, f"Window height validation failed: {error_msg}"

        # Clip parameters AFTER validation so validation uses original values
        # This ensures window height validation uses the actual floor/roof heights
        # before they are clipped for encoding
        try:
            self._clip_parameters(parameters)
        except ValueError as e:
            # Clipping rejected a parameter (e.g., negative value where not allowed)
            return False, str(e)

        return True, ""

    def calculate_direction_angle(
        self,
        parameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate direction_angle for window(s) from room polygon and window coordinates

        Args:
            parameters: Dictionary containing:
                - room_polygon: List of [x, y] coordinates defining the room
                - windows: Dictionary of window_id -> window parameters
                    Each window must have: x1, y1, x2, y2

        Returns:
            Dictionary mapping window_id -> direction_angle (in radians)

        Raises:
            ValueError: If required parameters are missing or calculation fails
        """
        # Validate required parameters
        if ParameterName.ROOM_POLYGON.value not in parameters:
            raise ValueError(f"Missing required parameter: '{ParameterName.ROOM_POLYGON.value}'")

        if ParameterName.WINDOWS.value not in parameters:
            raise ValueError(f"Missing required parameter: '{ParameterName.WINDOWS.value}'")

        if not isinstance(parameters[ParameterName.WINDOWS.value], dict):
            raise ValueError(f"'{ParameterName.WINDOWS.value}' must be a dictionary")

        # Create room polygon
        try:
            room_polygon = RoomPolygon.from_dict(parameters[ParameterName.ROOM_POLYGON.value])
        except Exception as e:
            raise ValueError(f"Invalid room_polygon: {str(e)}")

        # Calculate direction_angle for each window
        results = {}
        for window_id, window_params in parameters[ParameterName.WINDOWS.value].items():
            try:
                # Validate window has required coordinates
                required_coords = [
                    ParameterName.X1.value,
                    ParameterName.Y1.value,
                    ParameterName.X2.value,
                    ParameterName.Y2.value
                ]
                missing = [coord for coord in required_coords if coord not in window_params]
                if missing:
                    raise ValueError(f"Window '{window_id}' missing required coordinates: {', '.join(missing)}")

                # Create window geometry (z coordinates not needed for direction calculation)
                window_geom = WindowGeometry(
                    x1=window_params["x1"],
                    y1=window_params["y1"],
                    z1=window_params.get("z1", 0.0),  # Default z values if not provided
                    x2=window_params["x2"],
                    y2=window_params["y2"],
                    z2=window_params.get("z2", 1.0)
                )

                # Calculate direction angle
                direction_angle = window_geom.calculate_direction_from_polygon(room_polygon)
                results[window_id] = direction_angle

                self._logger.info(
                    f"Calculated direction_angle for '{window_id}': {direction_angle:.4f} rad "
                    f"({direction_angle * 180 / 3.14159:.2f}°)"
                )

            except Exception as e:
                error_msg = f"Failed to calculate direction_angle for window '{window_id}': {str(e)}"
                self._logger.error(error_msg)
                raise ValueError(error_msg)

        return results

    def calculate_reference_point(
        self,
        parameters: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate reference point for window(s) from room polygon and window coordinates

        The reference point is the center of the window edge that lies on the room boundary.

        Args:
            parameters: Dictionary containing:
                - room_polygon: List of [x, y] coordinates defining the room
                - windows: Dictionary of window_id -> window parameters
                    Each window must have: x1, y1, z1, x2, y2, z2

        Returns:
            Dictionary mapping window_id -> {"x": float, "y": float, "z": float}

        Raises:
            ValueError: If required parameters are missing or calculation fails
        """
        # Validate required parameters
        if ParameterName.ROOM_POLYGON.value not in parameters:
            raise ValueError(f"Missing required parameter: '{ParameterName.ROOM_POLYGON.value}'")

        if ParameterName.WINDOWS.value not in parameters:
            raise ValueError(f"Missing required parameter: '{ParameterName.WINDOWS.value}'")

        if not isinstance(parameters[ParameterName.WINDOWS.value], dict):
            raise ValueError(f"'{ParameterName.WINDOWS.value}' must be a dictionary")

        # Create room polygon
        try:
            room_polygon = RoomPolygon.from_dict(parameters[ParameterName.ROOM_POLYGON.value])
        except Exception as e:
            raise ValueError(f"Invalid room_polygon: {str(e)}")

        # Calculate reference point for each window
        results = {}
        for window_id, window_params in parameters[ParameterName.WINDOWS.value].items():
            try:
                # Validate window has required coordinates
                required_coords = [
                    ParameterName.X1.value,
                    ParameterName.Y1.value,
                    ParameterName.Z1.value,
                    ParameterName.X2.value,
                    ParameterName.Y2.value,
                    ParameterName.Z2.value
                ]
                missing = [coord for coord in required_coords if coord not in window_params]
                if missing:
                    raise ValueError(f"Window '{window_id}' missing required coordinates: {', '.join(missing)}")

                # Create window geometry
                window_geom = WindowGeometry(
                    x1=window_params[ParameterName.X1.value],
                    y1=window_params[ParameterName.Y1.value],
                    z1=window_params[ParameterName.Z1.value],
                    x2=window_params[ParameterName.X2.value],
                    y2=window_params[ParameterName.Y2.value],
                    z2=window_params[ParameterName.Z2.value]
                )

                # Calculate reference point
                ref_point = window_geom.calculate_reference_point_from_polygon(room_polygon)
                results[window_id] = {
                    "x": ref_point.x,
                    "y": ref_point.y,
                    "z": ref_point.z
                }

                self._logger.info(
                    f"Calculated reference_point for '{window_id}': "
                    f"({ref_point.x:.4f}, {ref_point.y:.4f}, {ref_point.z:.4f})"
                )

            except Exception as e:
                error_msg = f"Failed to calculate reference_point for window '{window_id}': {str(e)}"
                self._logger.error(error_msg)
                raise ValueError(error_msg)

        return results


class EncodingServiceFactory:
    """Factory for creating encoding service instances (Singleton Pattern per encoding scheme)"""

    _instances: Dict[EncodingScheme, EncodingService] = {}

    @classmethod
    def get_instance(cls, logger: StructuredLogger, encoding_scheme: EncodingScheme = EncodingScheme.RGB) -> EncodingService:
        """
        Get singleton instance of encoding service for specified encoding scheme

        Args:
            logger: Logger instance
            encoding_scheme: Encoding scheme to use (default: HSV)

        Returns:
            EncodingService instance
        """
        if encoding_scheme not in cls._instances:
            cls._instances[encoding_scheme] = EncodingService(logger, encoding_scheme)
        return cls._instances[encoding_scheme]

    @classmethod
    def reset_instance(cls, encoding_scheme: EncodingScheme = None) -> None:
        """
        Reset singleton instance(s) (useful for testing)

        Args:
            encoding_scheme: If specified, reset only that scheme's instance. Otherwise, reset all.
        """
        if encoding_scheme is None:
            cls._instances = {}
        else:
            cls._instances.pop(encoding_scheme, None)
