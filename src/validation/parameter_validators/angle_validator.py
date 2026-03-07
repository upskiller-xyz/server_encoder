"""Validator for angle parameters (SRP: validates only angle values)"""
from typing import Any, Dict, Optional
import math
from src.validation.base import BaseValidator, ValidationResult, ValidationError
from src.validation.enums import ValidationErrorType


class AngleValidator(BaseValidator):
    """Validates angle values in radians"""

    def __init__(self, parameter_name: str, allow_negative: bool = True):
        """
        Initialize angle validator.

        Args:
            parameter_name: Name of the parameter being validated
            allow_negative: Whether to allow negative angles (default: True)
        """
        self._parameter_name = parameter_name
        self._allow_negative = allow_negative
        self._min_value = -2 * math.pi if allow_negative else 0.0
        self._max_value = 2 * math.pi

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate angle value.

        Args:
            value: Value to validate
            context: Optional context (unused for angle)

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
                message=f"{self._parameter_name} must be between {self._min_value} and {self._max_value} radians, got {value}",
                parameter_name=self._parameter_name
            )
            result.add_error(error)

        return result
