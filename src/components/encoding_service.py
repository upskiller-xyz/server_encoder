from typing import Dict, Any, Tuple, Union
import numpy as np
import cv2
from abc import ABC, abstractmethod
from src.components.interfaces import IEncodingService
from src.components.enums import ModelType
from src.components.image_builder import RoomImageBuilder, RoomImageDirector
from src.components.encoders import EncoderFactory
from src.components.geometry import WindowBorderValidator, WindowHeightValidator
from src.server.services.logging import StructuredLogger


class IParameterCalculator(ABC):
    """
    Abstract base class for parameter calculators (Strategy Pattern)

    Calculates derived parameters from input data.
    """

    @abstractmethod
    def can_calculate(self, parameters: Dict[str, Any]) -> bool:
        """
        Check if this calculator can compute from given parameters

        Args:
            parameters: Input parameters

        Returns:
            True if calculator has required inputs
        """
        pass

    @abstractmethod
    def calculate(self, parameters: Dict[str, Any]) -> Any:
        """
        Calculate the parameter value

        Args:
            parameters: Input parameters

        Returns:
            Calculated value

        Raises:
            ValueError: If calculation cannot be performed
        """
        pass

    @abstractmethod
    def get_parameter_name(self) -> str:
        """Get the name of the parameter this calculator produces"""
        pass


class WindowSillHeightCalculator(IParameterCalculator):
    """
    Calculator for window_sill_height parameter

    Formula:
    - window_sill_height = max(0, min(z1, z2) - floor_height_above_terrain)
    - Capped to 0 if window bottom is below floor level

    Where:
    - z1, z2: Window bottom and top Z coordinates
    - floor_height_above_terrain: Height of floor above terrain
    """

    def can_calculate(self, parameters: Dict[str, Any]) -> bool:
        """Check if we have window geometry and floor height"""
        has_window_geometry = "window_geometry" in parameters
        has_z_coords = "z1" in parameters and "z2" in parameters
        has_floor_height = "floor_height_above_terrain" in parameters

        return (has_window_geometry or has_z_coords) and has_floor_height

    def calculate(self, parameters: Dict[str, Any]) -> float:
        """
        Calculate window sill height

        Args:
            parameters: Must contain z1, z2 (or window_geometry) and floor_height_above_terrain

        Returns:
            Calculated window sill height in meters (minimum 0)

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        try:
            # Extract Z coordinates
            if "window_geometry" in parameters:
                geometry = parameters["window_geometry"]
                z1 = float(geometry.get("z1"))
                z2 = float(geometry.get("z2"))
            else:
                z1 = float(parameters["z1"])
                z2 = float(parameters["z2"])

            floor_height = float(parameters["floor_height_above_terrain"])

            # Calculate: min(z1, z2) - floor_height, capped at 0
            window_bottom = min(z1, z2)
            window_sill_height = max(0.0, window_bottom - floor_height)

            return window_sill_height

        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"Cannot calculate window_sill_height. "
                f"Required: z1, z2, floor_height_above_terrain. "
                f"Error: {type(e).__name__}: {str(e)}"
            )

    def get_parameter_name(self) -> str:
        """Get parameter name"""
        return "window_sill_height"


class WindowHeightCalculator(IParameterCalculator):
    """
    Calculator for window_height parameter

    Formula:
    - If min(z1, z2) >= floor_height_above_terrain:
        window_height = abs(z2 - z1)  (normal case)
    - If min(z1, z2) < floor_height_above_terrain:
        window_height = max(z1, z2) - floor_height_above_terrain  (window starts at floor)

    Where:
    - z1, z2: Window bottom and top Z coordinates
    - floor_height_above_terrain: Height of floor above terrain (optional for this calc)
    """

    def can_calculate(self, parameters: Dict[str, Any]) -> bool:
        """Check if we have window Z coordinates"""
        has_window_geometry = "window_geometry" in parameters
        has_z_coords = "z1" in parameters and "z2" in parameters

        return has_window_geometry or has_z_coords

    def calculate(self, parameters: Dict[str, Any]) -> float:
        """
        Calculate window height

        Args:
            parameters: Must contain z1, z2 (or window_geometry)
                       Optional: floor_height_above_terrain for floor adjustment

        Returns:
            Calculated window height in meters

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        try:
            # Extract Z coordinates
            if "window_geometry" in parameters:
                geometry = parameters["window_geometry"]
                z1 = float(geometry.get("z1"))
                z2 = float(geometry.get("z2"))
            else:
                z1 = float(parameters["z1"])
                z2 = float(parameters["z2"])

            window_bottom = min(z1, z2)
            window_top = max(z1, z2)

            # Check if floor height is available
            if "floor_height_above_terrain" in parameters:
                floor_height = float(parameters["floor_height_above_terrain"])

                # If window bottom is below floor, calculate height from floor
                if window_bottom < floor_height:
                    window_height = window_top - floor_height
                else:
                    # Normal case: full window height
                    window_height = window_top - window_bottom
            else:
                # No floor height available, use full window height
                window_height = abs(z2 - z1)

            return window_height

        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"Cannot calculate window_height. "
                f"Required: z1, z2. "
                f"Error: {type(e).__name__}: {str(e)}"
            )

    def get_parameter_name(self) -> str:
        """Get parameter name"""
        return "window_height"


class ParameterCalculatorRegistry:
    """
    Registry of parameter calculators (Factory + Registry Pattern)

    Manages automatic calculation of derived parameters.
    """

    # Available calculators (order matters: dependencies should be calculated first)
    _CALCULATORS = [
        WindowHeightCalculator(),         # No dependencies
        WindowSillHeightCalculator(),     # Depends on floor_height_above_terrain
    ]

    @classmethod
    def calculate_derived_parameters(
        cls,
        parameters: Dict[str, Any],
        logger=None
    ) -> Dict[str, Any]:
        """
        Calculate all derived parameters that can be computed

        Args:
            parameters: Input parameters
            logger: Optional logger for warnings

        Returns:
            Updated parameters with calculated values added
        """
        # Create copy to avoid modifying original
        result = parameters.copy()

        # Try each calculator
        for calculator in cls._CALCULATORS:
            param_name = calculator.get_parameter_name()

            # Skip if parameter already provided by user
            if param_name in result:
                continue

            # Calculate if possible
            if calculator.can_calculate(result):
                try:
                    calculated_value = calculator.calculate(result)
                    result[param_name] = calculated_value
                except ValueError as e:
                    # Calculator couldn't compute
                    # Log warning if logger available, otherwise re-raise
                    if logger:
                        logger.warning(
                            f"Failed to calculate '{param_name}': {str(e)}. "
                            f"This parameter must be provided manually."
                        )
                    else:
                        # Re-raise if no logger (strict mode)
                        raise

        return result


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
        if "windows" in parameters:
            # Check if windows is a dict (expected format)
            if not isinstance(parameters["windows"], dict):
                return False, "Parameter 'windows' must be a dictionary mapping window_id to window parameters"

            # Validate each window separately
            for window_id, window_params in parameters["windows"].items():
                # Check if window_params is a dict
                if not isinstance(window_params, dict):
                    return False, f"Window '{window_id}' parameters must be a dictionary, got {type(window_params).__name__}"

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

    # Parameters that support clipping (Strategy Pattern)
    # Format: {param_name: (clip_min, clip_max, reject_below_min)}
    # - clip_min/max: values to clip to
    # - reject_below_min: if True, reject values < min instead of clipping
    _CLIPPING_CONFIG = {
        "floor_height_above_terrain": (0.0, 10.0, True),        # Reject < 0, clip > 10
        "height_roof_over_floor": (0.0, 30.0, True),            # Reject <= 0, clip > 30
        "obstruction_angle_horizon": (0.0, 90.0, False),        # Clip both min and max
        "obstruction_angle_zenith": (0.0, 70.0, False),         # Clip both min and max
        # window_sill_height is auto-calculated, no clipping needed
    }

    def _clip_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Clip parameters that support clipping instead of validation errors.

        Uses Strategy Pattern with _CLIPPING_CONFIG map.

        Clipping rules:
        - floor_height_above_terrain: values > 10.0 clipped to 10.0, values < 0.0 rejected
        - height_roof_over_floor: values > 30.0 clipped to 30.0, values <= 0.0 rejected
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
                # Special case for height_roof_over_floor: must be > 0, not >= 0
                if param_name == "height_roof_over_floor":
                    if value <= min_val:
                        raise ValueError(
                            f"Parameter '{param_name}' value {value} not supported. "
                            f"Must be greater than {min_val}. "
                            f"Values <= {min_val} are not supported."
                        )
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

        # Clip parameters before validation
        try:
            self._clip_parameters(parameters)
        except ValueError as e:
            # Clipping rejected a parameter (e.g., negative value where not allowed)
            return False, str(e)
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
        # window_sill_height is now auto-calculated from window geometry
        # if not has_param("window_sill_height", "window_sill_height"):
        #     missing.append("window_sill_height")
        if not has_param("window_frame_ratio", "window_frame_ratio"):
            missing.append("window_frame_ratio")
        # window_height is now auto-calculated from window geometry (z1, z2)
        # if not has_param("window_height"):
        #     missing.append("window_height")
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
        room_polygon = parameters.get("room_polygon")

        # Check if we have window coordinates (either as nested object or flat)
        has_window_coords = (
            "window_geometry" in parameters or
            all(k in parameters for k in ["x1", "y1", "z1", "x2", "y2", "z2"])
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
            floor_height = parameters.get("floor_height_above_terrain")
            height_roof_over_floor = parameters.get("height_roof_over_floor")

            if floor_height is not None and height_roof_over_floor is not None:
                roof_height = floor_height + height_roof_over_floor

                is_valid, error_msg = WindowHeightValidator.validate_from_parameters(
                    window_geometry_data=parameters.get("window_geometry") or parameters,
                    floor_height=floor_height,
                    roof_height=roof_height
                )
                if not is_valid:
                    return False, f"Window height validation failed: {error_msg}"

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
