"""Validator for ratio parameters (SRP: validates only ratio values)"""
from typing import Any, Dict, Optional
from src.validation.base import BaseValidator, ValidationResult, ValidationError
from src.validation.enums import ValidationErrorType


class RatioValidator(BaseValidator):
    """Validates ratio values (must be between 0.0 and 1.0)"""

    def __init__(self, parameter_name: str):
        """
        Initialize ratio validator.

        Args:
            parameter_name: Name of the parameter being validated
        """
        self._parameter_name = parameter_name
        self._min_value = 0.0
        self._max_value = 1.0

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate ratio value.

        Args:
            value: Value to validate
            context: Optional context (unused for ratio)

        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult()

        # Type check
        if not isinstance(value, (int, float)):
            error = ValidationError(
                error_type=ValidationErrorType.INVALID_TYPE,
                message=f"{self._parameter_name} must be a number, got {type(value).__name__}",
                parameter_name=self._parameter_name
            )
            result.add_error(error)
            return result

        # Range check
        if not (self._min_value <= value <= self._max_value):
            error = ValidationError(
                error_type=ValidationErrorType.INVALID_RANGE,
                message=f"{self._parameter_name} must be between {self._min_value} and {self._max_value}, got {value}",
                parameter_name=self._parameter_name
            )
            result.add_error(error)

        return result
