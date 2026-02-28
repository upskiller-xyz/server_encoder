"""Validator manager to orchestrate validation based on request type (Strategy Pattern)"""
from typing import Any, Dict, Type
from src.validation.base import BaseValidator, ValidationResult
from src.validation.enums import RequestType
from src.validation.request_validators import (
    EncodeRequestValidator,
    CalculateDirectionRequestValidator,
    ReferencePointRequestValidator,
    ExternalReferencePointRequestValidator,
)


class ValidatorManager:
    """
    Manages validation for different request types using Strategy Pattern.

    Single Responsibility: Route validation requests to appropriate validators.
    Stateless implementation using class methods.
    """

    # Strategy Pattern: Map request types to validator classes
    _VALIDATORS: Dict[RequestType, Type[BaseValidator]] = {
        RequestType.ENCODE: EncodeRequestValidator,
        RequestType.CALCULATE_DIRECTION: CalculateDirectionRequestValidator,
        RequestType.GET_REFERENCE_POINT: ReferencePointRequestValidator,
        RequestType.GET_EXTERNAL_REFERENCE_POINT: ExternalReferencePointRequestValidator,
    }

    @classmethod
    def validate(cls, request_type: RequestType, data: Any) -> ValidationResult:
        """
        Validate request data based on request type.

        Args:
            request_type: Type of request (enum)
            data: Request data to validate

        Returns:
            ValidationResult with validation status and errors

        Raises:
            ValueError: If request_type is not supported
        """
        validator_class = cls._VALIDATORS.get(request_type)

        if validator_class is None:
            raise ValueError(
                f"No validator found for request type: {request_type.value}. "
                f"Supported types: {', '.join(rt.value for rt in RequestType)}"
            )

        # Instantiate validator and call validate
        validator = validator_class()
        return validator.validate(data)

    @classmethod
    def get_validator(cls, request_type: RequestType) -> BaseValidator:
        """
        Get validator instance for a specific request type.

        Args:
            request_type: Type of request (enum)

        Returns:
            Validator instance for the request type

        Raises:
            ValueError: If request_type is not supported
        """
        validator_class = cls._VALIDATORS.get(request_type)

        if validator_class is None:
            raise ValueError(
                f"No validator found for request type: {request_type.value}. "
                f"Supported types: {', '.join(rt.value for rt in RequestType)}"
            )

        return validator_class()
