"""Validator for encoding scheme parameter (SRP: validates only encoding scheme values)"""
from typing import Any, Dict, Optional
from src.validation.base import BaseValidator, ValidationResult, ValidationError
from src.validation.enums import ValidationErrorType
from src.core.enums import EncodingScheme


class EncodingSchemeValidator(BaseValidator):
    """Validates encoding scheme values against allowed schemes"""

    def __init__(self):
        """Initialize encoding scheme validator"""
        self._parameter_name = "encoding_scheme"
        self._valid_schemes = {scheme.value for scheme in EncodingScheme}

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate encoding scheme value.

        Args:
            value: Value to validate
            context: Optional context (unused for encoding scheme)

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
        if value not in self._valid_schemes:
            error = ValidationError(
                error_type=ValidationErrorType.INVALID_VALUE,
                message=f"{self._parameter_name} must be one of {sorted(self._valid_schemes)}, got '{value}'",
                parameter_name=self._parameter_name
            )
            result.add_error(error)

        return result
