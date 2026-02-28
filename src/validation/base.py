"""Base classes for validation system following OOP and SRP principles"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from src.validation.enums import ValidationErrorType


class ValidationError(Exception):
    """Custom exception for validation errors"""

    def __init__(self, error_type: ValidationErrorType, message: str, parameter_name: Optional[str] = None):
        """
        Initialize validation error.

        Args:
            error_type: Type of validation error (enum)
            message: Human-readable error message
            parameter_name: Name of the parameter that failed validation
        """
        self.error_type = error_type
        self.parameter_name = parameter_name
        super().__init__(message)


class ValidationResult:
    """Result of a validation operation"""

    def __init__(self, is_valid: bool = True, errors: Optional[List[ValidationError]] = None):
        """
        Initialize validation result.

        Args:
            is_valid: Whether validation passed
            errors: List of validation errors if validation failed
        """
        self._is_valid = is_valid
        self._errors = errors or []

    @property
    def is_valid(self) -> bool:
        """Check if validation passed"""
        return self._is_valid

    @property
    def errors(self) -> List[ValidationError]:
        """Get list of validation errors"""
        return self._errors

    def add_error(self, error: ValidationError) -> None:
        """
        Add a validation error.

        Args:
            error: Validation error to add
        """
        self._errors.append(error)
        self._is_valid = False


class BaseValidator(ABC):
    """Abstract base class for all validators (Interface)"""

    @abstractmethod
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate a value.

        Args:
            value: Value to validate
            context: Optional context dictionary with additional information

        Returns:
            ValidationResult with validation status and errors
        """
        pass
