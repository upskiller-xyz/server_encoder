"""Validator for window border placement (SRP: validates only window border positioning)"""
from typing import Any, Dict, Optional
from src.validation.base import BaseValidator, ValidationResult, ValidationError
from src.validation.enums import ValidationErrorType
from src.validation.utils import ValidationUtils
from src.core.enums import ParameterName
from src.components.geometry import WindowBorderValidator as GeometryWindowBorderValidator


class WindowBorderValidator(BaseValidator):
    """Validates that window is positioned on the room polygon border"""

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate window border placement.

        Args:
            value: Window parameters dict (with x1, y1, z1, x2, y2, z2)
            context: Required context with 'room_polygon' key

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

        # Context check - must have room_polygon
        if not context or "room_polygon" not in context:
            error = ValidationError(
                error_type=ValidationErrorType.MISSING_PARAMETER,
                message="room_polygon is required in context for window border validation",
                parameter_name="room_polygon"
            )
            result.add_error(error)
            return result

        room_polygon = context["room_polygon"]

        # Check if we have window coordinates (either as nested object or flat)
        has_window_coords = ValidationUtils.has_window_coordinates(value, require_3d=True)

        if not has_window_coords:
            # Skip validation if no window coordinates present
            return result

        # Use existing WindowBorderValidator from geometry components
        is_valid, error_msg = GeometryWindowBorderValidator.validate_from_dict(
            window_data=value,
            polygon_data=room_polygon
        )

        if not is_valid:
            error = ValidationError(
                error_type=ValidationErrorType.INVALID_VALUE,
                message=f"Window geometry validation failed: {error_msg}",
                parameter_name="window_geometry"
            )
            result.add_error(error)

        return result
