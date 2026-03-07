"""Validator for height parameters (SRP: validates only height values)"""
from typing import Any, Dict, Optional
from src.validation.base import BaseValidator, ValidationResult, ValidationError
from src.validation.enums import ValidationErrorType


class HeightValidator(BaseValidator):
    """Validates height values (must be positive numbers)"""

    def __init__(self, parameter_name: str, min_value: float = 0.0, max_value: Optional[float] = None):
        """
        Initialize height validator.

        Args:
            parameter_name: Name of the parameter being validated
            min_value: Minimum allowed height (default: 0.0)
            max_value: Maximum allowed height (default: None, no upper limit)
        """
        self._parameter_name = parameter_name
        self._min_value = min_value
        self._max_value = max_value

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate height value.

        Args:
            value: Value to validate
            context: Optional context (unused for height)

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

        # Range check (minimum)
        if value < self._min_value:
            error = ValidationError(
                error_type=ValidationErrorType.INVALID_RANGE,
                message=f"{self._parameter_name} must be >= {self._min_value}, got {value}",
                parameter_name=self._parameter_name
            )
            result.add_error(error)

        # Range check (maximum)
        if self._max_value is not None and value > self._max_value:
            error = ValidationError(
                error_type=ValidationErrorType.INVALID_RANGE,
                message=f"{self._parameter_name} must be <= {self._max_value}, got {value}",
                parameter_name=self._parameter_name
            )
            result.add_error(error)

        return result
