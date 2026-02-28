"""Validator for window coordinates (SRP: validates only window coordinate structures)"""
from typing import Any, Dict, Optional, Set
from src.validation.base import BaseValidator, ValidationResult, ValidationError
from src.validation.enums import ValidationErrorType
from src.core.enums import ParameterName, REQUIRED_WINDOW_COORDINATES, REQUIRED_WINDOW_2D_COORDINATES


class WindowCoordinatesValidator(BaseValidator):
    """Validates window coordinate structures"""

    def __init__(self, require_3d: bool = True):
        """
        Initialize window coordinates validator.

        Args:
            require_3d: Whether to require 3D coordinates (x1, y1, z1, x2, y2, z2)
                       If False, only 2D coordinates are required (x1, y1, x2, y2)
        """
        self._require_3d = require_3d
        self._required_coords = REQUIRED_WINDOW_COORDINATES if require_3d else REQUIRED_WINDOW_2D_COORDINATES

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate window coordinates.

        Args:
            value: Value to validate (should be Dict[str, Dict[str, float]])
            context: Optional context (unused for window coordinates)

        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult()
        parameter_name = ParameterName.WINDOWS.value

        # Type check - must be a dict
        if not isinstance(value, dict):
            error = ValidationError(
                error_type=ValidationErrorType.INVALID_TYPE,
                message=f"{parameter_name} must be a dictionary, got {type(value).__name__}",
                parameter_name=parameter_name
            )
            result.add_error(error)
            return result

        # Must have at least one window
        if len(value) == 0:
            error = ValidationError(
                error_type=ValidationErrorType.INVALID_LENGTH,
                message=f"{parameter_name} must contain at least one window",
                parameter_name=parameter_name
            )
            result.add_error(error)
            return result

        # Validate each window
        for window_id, window_data in value.items():
            if not isinstance(window_data, dict):
                error = ValidationError(
                    error_type=ValidationErrorType.INVALID_TYPE,
                    message=f"{parameter_name}[{window_id}] must be a dictionary, got {type(window_data).__name__}",
                    parameter_name=f"{parameter_name}.{window_id}"
                )
                result.add_error(error)
                continue

            # Check required coordinates
            missing_coords = self._required_coords - set(window_data.keys())
            if missing_coords:
                error = ValidationError(
                    error_type=ValidationErrorType.MISSING_PARAMETER,
                    message=f"{parameter_name}[{window_id}] missing required coordinates: {', '.join(sorted(missing_coords))}",
                    parameter_name=f"{parameter_name}.{window_id}"
                )
                result.add_error(error)
                continue

            # Validate coordinate values are numbers
            for coord_name in self._required_coords:
                if coord_name in window_data:
                    coord_value = window_data[coord_name]
                    if not isinstance(coord_value, (int, float)):
                        error = ValidationError(
                            error_type=ValidationErrorType.INVALID_TYPE,
                            message=f"{parameter_name}[{window_id}].{coord_name} must be a number, got {type(coord_value).__name__}",
                            parameter_name=f"{parameter_name}.{window_id}.{coord_name}"
                        )
                        result.add_error(error)

        return result
