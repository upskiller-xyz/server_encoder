"""Validation enums for type-safe validation"""
from enum import Enum


class ValidationErrorType(Enum):
    """Types of validation errors"""
    MISSING_PARAMETER = "missing_parameter"
    INVALID_TYPE = "invalid_type"
    INVALID_VALUE = "invalid_value"
    INVALID_RANGE = "invalid_range"
    INVALID_LENGTH = "invalid_length"
    INVALID_FORMAT = "invalid_format"
    INVALID_POLYGON = "invalid_polygon"
    INVALID_COORDINATES = "invalid_coordinates"


class ValidationType(Enum):
    """Types of validation to perform"""
    REQUIRED = "required"
    TYPE_CHECK = "type_check"
    RANGE_CHECK = "range_check"
    FORMAT_CHECK = "format_check"
    POLYGON_CHECK = "polygon_check"
    COORDINATES_CHECK = "coordinates_check"


class RequestType(Enum):
    """Types of API requests"""
    ENCODE = "encode"
    CALCULATE_DIRECTION = "calculate_direction"
    GET_REFERENCE_POINT = "get_reference_point"
    GET_EXTERNAL_REFERENCE_POINT = "get_external_reference_point"
