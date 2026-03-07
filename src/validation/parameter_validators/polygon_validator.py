"""Validator for polygon parameters (SRP: validates only polygon structures)"""
from typing import Any, Dict, Optional, List
from src.validation.base import BaseValidator, ValidationResult, ValidationError
from src.validation.enums import ValidationErrorType


class PolygonValidator(BaseValidator):
    """Validates polygon structures (list of [x, y] coordinate pairs)"""

    def __init__(self, parameter_name: str, min_vertices: int = 3):
        """
        Initialize polygon validator.

        Args:
            parameter_name: Name of the parameter being validated
            min_vertices: Minimum number of vertices required (default: 3)
        """
        self._parameter_name = parameter_name
        self._min_vertices = min_vertices

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate polygon structure.

        Args:
            value: Value to validate (should be List[List[float]])
            context: Optional context (unused for polygon)

        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult()

        # Type check - must be a list
        if not isinstance(value, list):
            error = ValidationError(
                error_type=ValidationErrorType.INVALID_TYPE,
                message=f"{self._parameter_name} must be a list, got {type(value).__name__}",
                parameter_name=self._parameter_name
            )
            result.add_error(error)
            return result

        # Length check - must have minimum vertices
        if len(value) < self._min_vertices:
            error = ValidationError(
                error_type=ValidationErrorType.INVALID_LENGTH,
                message=f"{self._parameter_name} must have at least {self._min_vertices} vertices, got {len(value)}",
                parameter_name=self._parameter_name
            )
            result.add_error(error)
            return result

        # Validate each vertex
        for i, vertex in enumerate(value):
            if not isinstance(vertex, (list, tuple)):
                error = ValidationError(
                    error_type=ValidationErrorType.INVALID_FORMAT,
                    message=f"{self._parameter_name}[{i}] must be a list or tuple, got {type(vertex).__name__}",
                    parameter_name=self._parameter_name
                )
                result.add_error(error)
                continue

            if len(vertex) != 2:
                error = ValidationError(
                    error_type=ValidationErrorType.INVALID_FORMAT,
                    message=f"{self._parameter_name}[{i}] must have exactly 2 coordinates [x, y], got {len(vertex)}",
                    parameter_name=self._parameter_name
                )
                result.add_error(error)
                continue

            # Validate coordinates are numbers
            for j, coord in enumerate(vertex):
                if not isinstance(coord, (int, float)):
                    error = ValidationError(
                        error_type=ValidationErrorType.INVALID_TYPE,
                        message=f"{self._parameter_name}[{i}][{j}] must be a number, got {type(coord).__name__}",
                        parameter_name=self._parameter_name
                    )
                    result.add_error(error)

        return result
