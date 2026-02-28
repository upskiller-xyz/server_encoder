"""Validator for model type parameter (SRP: validates only model type values)"""
from typing import Any, Dict, Optional
from src.validation.base import BaseValidator, ValidationResult, ValidationError
from src.validation.enums import ValidationErrorType
from src.core.enums import ModelType


class ModelTypeValidator(BaseValidator):
    """Validates model type values against allowed model types"""

    def __init__(self):
        """Initialize model type validator"""
        self._parameter_name = "model_type"
        self._valid_types = {model_type.value for model_type in ModelType}

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate model type value.

        Args:
            value: Value to validate
            context: Optional context (unused for model type)

        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult()

        # Type check
        if not isinstance(value, str):
            error = ValidationError(
                error_type=ValidationErrorType.INVALID_TYPE,
                message=f"{self._parameter_name} must be a string, got {type(value).__name__}",
                parameter_name=self._parameter_name
            )
            result.add_error(error)
            return result

        # Value check
        if value not in self._valid_types:
            error = ValidationError(
                error_type=ValidationErrorType.INVALID_VALUE,
                message=f"{self._parameter_name} must be one of {sorted(self._valid_types)}, got '{value}'",
                parameter_name=self._parameter_name
            )
            result.add_error(error)

        return result
