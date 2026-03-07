from typing import Dict, Any, Optional
import logging

from src.core.enums import ResponseKey
from src.server.enums import ModelStatus, ServerStatus
logger = logging.getLogger(__name__)

class ServerController:
    """Generic server controller implementing IServerController interface"""

    def __init__(
        self,
        services: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the controller with dependencies

        Args:
            services: Optional dictionary of service name -> service instance
        """
        self._services = services or {}
        self._status = ServerStatus.STARTING

    def initialize(self) -> None:
        """Initialize the server and its components"""
        logger.info("Initializing server controller")
        try:
            # Initialize all registered services
            for service_name, service in self._services.items():
                if hasattr(service, 'initialize'):
                    logger.debug(f"Initializing service: {service_name}")
                    service.initialize()

            self._status = ServerStatus.RUNNING
            logger.info("Server controller initialized successfully")
        except Exception as e:
            self._status = ServerStatus.ERROR
            logger.error(f"Failed to initialize server controller: {str(e)}")
            raise

    def get_status(self) -> Dict[str, Any]:
        """
        Get current server status

        Returns:
            Dictionary containing status information
        """
        components = {}
        for service_name, service in self._services.items():
            if hasattr(service, 'get_status'):
                components[service_name] = service.get_status()
            else:
                components[service_name] = ModelStatus.READY.value

        return {
            ResponseKey.STATUS.value: self._status.value,
            ResponseKey.SERVICES.value: components
        }
