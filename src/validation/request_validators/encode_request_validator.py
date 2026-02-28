"""Validator for encode requests (SRP: validates only encode request structure)"""
from typing import Any, Dict, Optional
from src.validation.base import BaseValidator, ValidationResult, ValidationError
from src.validation.enums import ValidationErrorType
from src.validation.parameter_validators import (
    ModelTypeValidator,
    EncodingSchemeValidator,
    PolygonValidator,
    HeightValidator,
    ReflectanceValidator,
    AngleValidator,
    RatioValidator,
    WindowCoordinatesValidator,
)
from src.core.enums import ParameterName, ResponseKey


class EncodeRequestValidator(BaseValidator):
    """Validates encode request structure and parameters"""

    def __init__(self):
        """Initialize encode request validator with parameter validators"""
        # Top-level validators
        self._model_type_validator = ModelTypeValidator()
        self._encoding_scheme_validator = EncodingSchemeValidator()

        # Parameter validators (Strategy Pattern)
        self._validators = {
            ParameterName.ROOM_POLYGON: PolygonValidator(ParameterName.ROOM_POLYGON.value),
            ParameterName.HEIGHT_ROOF_OVER_FLOOR: HeightValidator(
                ParameterName.HEIGHT_ROOF_OVER_FLOOR.value, min_value=0.1, max_value=20.0
            ),
            ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN: HeightValidator(
                ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value, min_value=0.0, max_value=100.0
            ),
            ParameterName.WINDOW_SILL_HEIGHT: HeightValidator(
                ParameterName.WINDOW_SILL_HEIGHT.value, min_value=0.0, max_value=10.0
            ),
            ParameterName.WINDOW_HEIGHT: HeightValidator(
                ParameterName.WINDOW_HEIGHT.value, min_value=0.1, max_value=10.0
            ),
            ParameterName.HORIZON: AngleValidator(ParameterName.HORIZON.value, allow_negative=False),
            ParameterName.ZENITH: AngleValidator(ParameterName.ZENITH.value, allow_negative=False),
            ParameterName.WINDOW_ORIENTATION: AngleValidator(ParameterName.WINDOW_ORIENTATION.value),
            ParameterName.DIRECTION_ANGLE: AngleValidator(ParameterName.DIRECTION_ANGLE.value),
            ParameterName.FACADE_REFLECTANCE: ReflectanceValidator(ParameterName.FACADE_REFLECTANCE.value),
            ParameterName.TERRAIN_REFLECTANCE: ReflectanceValidator(ParameterName.TERRAIN_REFLECTANCE.value),
            ParameterName.CEILING_REFLECTANCE: ReflectanceValidator(ParameterName.CEILING_REFLECTANCE.value),
            ParameterName.FLOOR_REFLECTANCE: ReflectanceValidator(ParameterName.FLOOR_REFLECTANCE.value),
            ParameterName.WALL_REFLECTANCE: ReflectanceValidator(ParameterName.WALL_REFLECTANCE.value),
            ParameterName.WINDOW_FRAME_REFLECTANCE: ReflectanceValidator(ParameterName.WINDOW_FRAME_REFLECTANCE.value),
            ParameterName.BALCONY_REFLECTANCE: ReflectanceValidator(ParameterName.BALCONY_REFLECTANCE.value),
            ParameterName.CONTEXT_REFLECTANCE: ReflectanceValidator(ParameterName.CONTEXT_REFLECTANCE.value),
            ParameterName.WINDOW_FRAME_RATIO: RatioValidator(ParameterName.WINDOW_FRAME_RATIO.value),
        }

        self._window_coordinates_validator = WindowCoordinatesValidator(require_3d=True)

        # Required top-level fields
        self._required_fields = {
            ResponseKey.MODEL_TYPE.value,
            ResponseKey.PARAMETERS.value,
        }

        # Required parameters
        self._required_parameters = {
            ParameterName.ROOM_POLYGON.value,
            ParameterName.HEIGHT_ROOF_OVER_FLOOR.value,
            ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value,
        }

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate encode request.

        Args:
            value: Request data to validate
            context: Optional context (unused)

        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult()

        # Type check - must be a dict
        if not isinstance(value, dict):
            error = ValidationError(
                error_type=ValidationErrorType.INVALID_TYPE,
                message=f"Request must be a dictionary, got {type(value).__name__}",
                parameter_name="request"
            )
            result.add_error(error)
            return result

        # Check required top-level fields
        missing_fields = self._required_fields - set(value.keys())
        if missing_fields:
            for field in missing_fields:
                error = ValidationError(
                    error_type=ValidationErrorType.MISSING_PARAMETER,
                    message=f"Missing required field: {field}",
                    parameter_name=field
                )
                result.add_error(error)

        # Validate model_type if present
        if ResponseKey.MODEL_TYPE.value in value:
            model_type_result = self._model_type_validator.validate(value[ResponseKey.MODEL_TYPE.value])
            for error in model_type_result.errors:
                result.add_error(error)

        # Validate encoding_scheme if present
        if ParameterName.ENCODING_SCHEME.value in value:
            encoding_scheme_result = self._encoding_scheme_validator.validate(
                value[ParameterName.ENCODING_SCHEME.value]
            )
            for error in encoding_scheme_result.errors:
                result.add_error(error)

        # Validate parameters
        if ResponseKey.PARAMETERS.value in value:
            parameters = value[ResponseKey.PARAMETERS.value]

            if not isinstance(parameters, dict):
                error = ValidationError(
                    error_type=ValidationErrorType.INVALID_TYPE,
                    message=f"Parameters must be a dictionary, got {type(parameters).__name__}",
                    parameter_name=ResponseKey.PARAMETERS.value
                )
                result.add_error(error)
                return result

            # Check required parameters
            missing_params = self._required_parameters - set(parameters.keys())
            if missing_params:
                for param in missing_params:
                    error = ValidationError(
                        error_type=ValidationErrorType.MISSING_PARAMETER,
                        message=f"Missing required parameter: {param}",
                        parameter_name=param
                    )
                    result.add_error(error)

            # Validate each parameter using appropriate validator
            for param_name, param_value in parameters.items():
                # Skip windows - validate separately
                if param_name == ParameterName.WINDOWS.value:
                    continue

                # Get validator for this parameter
                param_enum = self._get_parameter_enum(param_name)
                if param_enum and param_enum in self._validators:
                    validator = self._validators[param_enum]
                    param_result = validator.validate(param_value)
                    for error in param_result.errors:
                        result.add_error(error)

            # Validate windows if present
            if ParameterName.WINDOWS.value in parameters:
                windows_result = self._window_coordinates_validator.validate(
                    parameters[ParameterName.WINDOWS.value]
                )
                for error in windows_result.errors:
                    result.add_error(error)

        return result

    def _get_parameter_enum(self, param_name: str) -> Optional[ParameterName]:
        """
        Get ParameterName enum from string parameter name.

        Args:
            param_name: String parameter name

        Returns:
            ParameterName enum or None if not found
        """
        try:
            return ParameterName(param_name)
        except ValueError:
            return None
