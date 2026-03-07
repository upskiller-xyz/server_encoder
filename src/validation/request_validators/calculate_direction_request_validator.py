"""Validator for calculate direction requests (SRP: validates only direction calculation requests)"""
from typing import Any, Dict, Optional
from src.validation.base import BaseValidator, ValidationResult, ValidationError
from src.validation.enums import ValidationErrorType
from src.validation.parameter_validators import PolygonValidator, WindowCoordinatesValidator
from src.core.enums import ParameterName


class CalculateDirectionRequestValidator(BaseValidator):
    """Validates calculate direction request structure"""

    def __init__(self):
        """Initialize calculate direction request validator"""
        self._polygon_validator = PolygonValidator(ParameterName.ROOM_POLYGON.value)
        self._window_coordinates_validator = WindowCoordinatesValidator(require_3d=False)

        self._required_fields = {
            ParameterName.ROOM_POLYGON.value,
            ParameterName.WINDOWS.value,
        }

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate calculate direction request.

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

        # Check required fields
        missing_fields = self._required_fields - set(value.keys())
        if missing_fields:
            for field in missing_fields:
                error = ValidationError(
                    error_type=ValidationErrorType.MISSING_PARAMETER,
                    message=f"Missing required field: {field}",
                    parameter_name=field
                )
                result.add_error(error)

        # Validate room_polygon if present
        if ParameterName.ROOM_POLYGON.value in value:
            polygon_result = self._polygon_validator.validate(value[ParameterName.ROOM_POLYGON.value])
            for error in polygon_result.errors:
                result.add_error(error)

        # Validate windows if present
        if ParameterName.WINDOWS.value in value:
            windows_result = self._window_coordinates_validator.validate(value[ParameterName.WINDOWS.value])
            for error in windows_result.errors:
                result.add_error(error)

        return result
