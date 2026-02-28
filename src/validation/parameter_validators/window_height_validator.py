"""Validator for window height placement (SRP: validates only window height bounds)"""
from typing import Any, Dict, Optional
from src.validation.base import BaseValidator, ValidationResult, ValidationError
from src.validation.enums import ValidationErrorType
from src.core.enums import ParameterName
from src.components.geometry import WindowHeightValidator as GeometryWindowHeightValidator


class WindowHeightValidator(BaseValidator):
    """Validates that window height is between floor and roof"""

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate window height bounds.

        Args:
            value: Window parameters dict (with z1, z2)
            context: Required context with 'floor_height' and 'height_roof_over_floor' keys

        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult()

        # Type check - value must be a dict
        if not isinstance(value, dict):
            error = ValidationError(
                error_type=ValidationErrorType.INVALID_TYPE,
                message=f"Window data must be a dictionary, got {type(value).__name__}",
                parameter_name="window"
            )
            result.add_error(error)
            return result

        # Context check - must have floor and roof height
        if not context:
            error = ValidationError(
                error_type=ValidationErrorType.MISSING_PARAMETER,
                message="Context with floor_height and height_roof_over_floor is required",
                parameter_name="context"
            )
            result.add_error(error)
            return result

        floor_height = context.get("floor_height")
        height_roof_over_floor = context.get("height_roof_over_floor")

        # Skip validation if floor/roof heights not provided
        if floor_height is None or height_roof_over_floor is None:
            return result

        # Calculate roof height
        roof_height = floor_height + height_roof_over_floor

        # Get window geometry data (either nested or flat)
        window_geometry_data = value.get(ParameterName.WINDOW_GEOMETRY.value) or value

        # Use existing WindowHeightValidator from geometry components
        is_valid, error_msg = GeometryWindowHeightValidator.validate_from_parameters(
            window_geometry_data=window_geometry_data,
            floor_height=floor_height,
            roof_height=roof_height
        )

        if not is_valid:
            error = ValidationError(
                error_type=ValidationErrorType.INVALID_RANGE,
                message=f"Window height validation failed: {error_msg}",
                parameter_name="window_height"
            )
            result.add_error(error)

        return result
