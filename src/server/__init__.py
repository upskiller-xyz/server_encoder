"""Server module initialization"""
from src.server.application import ServerApplication
from src.server.launcher import ServerLauncher
from src.server.decorators import endpoint_error_handler
from src.server.openapi import OpenAPISpecGenerator
from src.server.key_manager import KeyManager, KeyType
from src.server.schemas import (
    EncodeRequest,
    CalculateDirectionRequest,
    ReferencePointRequest,
    DirectionAngleResponse,
    ReferencePointResponse,
    ErrorResponse,
)

__all__ = [
    "ServerApplication",
    "ServerLauncher",
    "endpoint_error_handler",
    "OpenAPISpecGenerator",
    "KeyManager",
    "KeyType",
    "EncodeRequest",
    "CalculateDirectionRequest",
    "ReferencePointRequest",
    "DirectionAngleResponse",
    "ReferencePointResponse",
    "ErrorResponse",
]
