from typing import Dict
from src.core import EncodingScheme
from src.server.services.encoding_service import EncodingService
from src.server.services.encoding_service_v5 import V5EncodingService


class EncodingServiceFactory:
    """Factory for creating encoding service instances (Singleton Pattern per encoding scheme)"""

    _instances: Dict[EncodingScheme, EncodingService] = {}

    @classmethod
    def get_instance(cls, encoding_scheme: EncodingScheme = EncodingScheme.V2) -> EncodingService:
        """
        Get singleton instance of encoding service for the specified encoding scheme.

        V5 and V6 use V5EncodingService (geometric mask family, float32 single-channel output).
        All other schemes (V1–V4, V7, V8) use the base EncodingService (RGBA uint8 output).

        Args:
            encoding_scheme: Encoding scheme to use (default: V2)

        Returns:
            EncodingService instance for the requested scheme
        """
        if encoding_scheme not in cls._instances:
            if encoding_scheme in (EncodingScheme.V5, EncodingScheme.V6):
                cls._instances[encoding_scheme] = V5EncodingService(encoding_scheme)
            else:
                cls._instances[encoding_scheme] = EncodingService(encoding_scheme)
        return cls._instances[encoding_scheme]

    @classmethod
    def reset_instance(cls, encoding_scheme: EncodingScheme | None = None) -> None:
        """
        Reset singleton instance(s) (useful for testing)

        Args:
            encoding_scheme: If specified, reset only that scheme's instance. Otherwise, reset all.
        """
        if encoding_scheme is None:
            cls._instances = {}
        else:
            cls._instances.pop(encoding_scheme, None)
