"""Validator for external reference point requests (SRP: validates only external reference point requests)"""
from src.validation.request_validators.reference_point_request_validator import ReferencePointRequestValidator


class ExternalReferencePointRequestValidator(ReferencePointRequestValidator):
    """
    Validates external reference point request structure.

    Inherits from ReferencePointRequestValidator since validation logic is identical.
    """
    pass
