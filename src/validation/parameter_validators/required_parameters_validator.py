"""Validator for required parameters (SRP: validates only presence of required parameters)"""
from typing import Any, Dict, Optional, List
from src.validation.base import BaseValidator, ValidationResult, ValidationError
from src.validation.enums import ValidationErrorType
from src.core.enums import ParameterName


class RequiredParametersValidator(BaseValidator):
    """Validates that all required parameters are present"""

    def __init__(self):
        """Initialize required parameters validator"""
        # Required parameters for all models
        self._required_parameters = [
            ParameterName.HEIGHT_ROOF_OVER_FLOOR,
            ParameterName.WINDOW_FRAME_RATIO,
            ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN,
            ParameterName.HORIZON,
            ParameterName.ZENITH,
            ParameterName.ROOM_POLYGON,
        ]

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate required parameters are present.

        Args:
            value: Parameters dict to validate
            context: Optional context (unused)

        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult()

        # Type check - must be a dict
        if not isinstance(value, dict):
            error = ValidationError(
                error_type=ValidationErrorType.INVALID_TYPE,
                message=f"Parameters must be a dictionary, got {type(value).__name__}",
                parameter_name="parameters"
            )
            result.add_error(error)
            return result

        # Check for missing required parameters
        missing = []
        for param_enum in self._required_parameters:
            param_name = param_enum.value
            if param_name not in value:
                missing.append(param_name)

        if missing:
            error = ValidationError(
                error_type=ValidationErrorType.MISSING_PARAMETER,
                message=f"Missing required parameters: {', '.join(missing)}",
                parameter_name="parameters"
            )
            result.add_error(error)

        return result
