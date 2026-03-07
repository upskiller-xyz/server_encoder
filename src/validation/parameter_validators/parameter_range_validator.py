"""Validator for parameter ranges (SRP: validates only parameter value ranges)"""
from typing import Any, Dict, Optional
import numpy as np
from src.validation.base import BaseValidator, ValidationResult, ValidationError
from src.validation.enums import ValidationErrorType


class ParameterRangeValidator(BaseValidator):
    """Validates that parameter values are within valid ranges"""

    def __init__(self, encoder_factory, clipping_config: Dict[str, Any]):
        """
        Initialize parameter range validator.

        Args:
            encoder_factory: Factory to get parameter ranges
            clipping_config: Configuration for parameters with clipping enabled
        """
        self._encoder_factory = encoder_factory
        self._clipping_config = clipping_config

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate parameter ranges.

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

        # Validate each parameter's range
        for param_name, param_value in value.items():
            # Skip array parameters (obstruction angles)
            if isinstance(param_value, (list, np.ndarray)):
                continue

            try:
                # Get valid range
                min_val, max_val = self._encoder_factory.get_parameter_range(param_name)

                # Validate range (handle reversed ranges)
                actual_min = min(min_val, max_val)
                actual_max = max(min_val, max_val)

                # Skip validation for parameters with clipping enabled
                # They are already validated and clipped elsewhere
                if param_name in self._clipping_config:
                    continue

                # For other parameters, validate normally
                try:
                    float_value = float(param_value)
                except (TypeError, ValueError) as e:
                    error = ValidationError(
                        error_type=ValidationErrorType.INVALID_TYPE,
                        message=f"Parameter '{param_name}' has invalid value type: {type(param_value).__name__}. Expected numeric value, got: {param_value}. Error: {str(e)}",
                        parameter_name=param_name
                    )
                    result.add_error(error)
                    continue

                if not (actual_min <= float_value <= actual_max):
                    error = ValidationError(
                        error_type=ValidationErrorType.INVALID_RANGE,
                        message=f"Parameter '{param_name}' value {param_value} outside valid range [{min_val}, {max_val}]",
                        parameter_name=param_name
                    )
                    result.add_error(error)

            except ValueError as e:
                # Unknown parameter - skip (might be for future use)
                if "Unknown parameter" in str(e):
                    continue
                # Re-raise if it's a different ValueError
                error = ValidationError(
                    error_type=ValidationErrorType.INVALID_VALUE,
                    message=f"Error validating parameter '{param_name}': {str(e)}",
                    parameter_name=param_name
                )
                result.add_error(error)

            except Exception as e:
                # Catch any unexpected errors and report them with context
                error = ValidationError(
                    error_type=ValidationErrorType.INVALID_VALUE,
                    message=f"Unexpected error validating parameter '{param_name}' with value {param_value}: {type(e).__name__}: {str(e)}",
                    parameter_name=param_name
                )
                result.add_error(error)

        return result
