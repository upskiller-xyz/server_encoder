from typing import Dict
from src.core import EncodingScheme
from src.server.services.encoding_service import EncodingService


class EncodingServiceFactory:
    """Factory for creating encoding service instances (Singleton Pattern per encoding scheme)"""

    _instances: Dict[EncodingScheme, EncodingService] = {}

    @classmethod
    def get_instance(cls,  encoding_scheme: EncodingScheme = EncodingScheme.RGB) -> EncodingService:
        """
        Get singleton instance of encoding service for specified encoding scheme

        Args:
            logger: Logger instance
            encoding_scheme: Encoding scheme to use (default: HSV)

        Returns:
            EncodingService instance
        """
        if encoding_scheme not in cls._instances:
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
