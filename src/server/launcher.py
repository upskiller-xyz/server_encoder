"""Server launcher for starting the Flask application"""
import logging
from src.server.application import ServerApplication


logger = logging.getLogger("logger")


class ServerLauncher:
    """Launcher class for the server application"""

    @staticmethod
    def create_application() -> ServerApplication:
        """
        Create and configure the Flask application.
        
        Returns:
            ServerApplication: Configured server application instance
        """
        return ServerApplication()

    @staticmethod
    def run_server(
        app: ServerApplication,
        host: str = "0.0.0.0",
        port: int = 8080,
        debug: bool = True
    ) -> None:
        """
        Run the Flask development server.
        
        Args:
            app: ServerApplication instance to run
            host: Host to bind to (default: 0.0.0.0)
            port: Port to listen on (default: 8080)
            debug: Enable debug mode (default: True)
        """
        log_msg = (
            f"Flask app '{app.app.name}' starting on "
            f"host {host}, port {port}. Debug mode: {debug}"
        )
        app.app.logger.info(log_msg)
        # Disable reloader to prevent bus errors/hangs on WSL2
        app.app.run(host=host, port=port, debug=debug, use_reloader=False)
