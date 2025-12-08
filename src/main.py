import os
from typing import Dict, Any

# Disable GPU/CUDA to prevent bus errors on WSL2
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
import io
import zipfile
import traceback
import re

from src.server.enums import ContentType, HTTPStatus, LogLevel
from src.components.enums import ModelType, ParameterName
from src.components.encoding_service import EncodingServiceFactory
from src.server.services.logging import StructuredLogger
from src.server.controllers.base_controller import ServerController




class ServerApplication:
    """Main application class implementing dependency injection and OOP principles"""

    def __init__(self, app_name: str = "Server Application"):
        self._app = Flask(app_name)
        CORS(self._app)
        self._controller = None
        self._logger = None
        self._encoding_service = None
        self._setup_dependencies()
        self._setup_routes()

    def _setup_dependencies(self) -> None:
        """Setup all dependencies using dependency injection"""
        # Logger
        self._logger = StructuredLogger("Server", LogLevel.INFO)

        # Encoding service
        self._encoding_service = EncodingServiceFactory.get_instance(self._logger)

        # Services dict
        services = {
            "encoding_service": self._encoding_service
        }

        # Controller
        self._controller = ServerController(
            logger=self._logger,
            services=services
        )

        # Initialize controller
        self._controller.initialize()

    def _setup_routes(self) -> None:
        """Setup Flask routes"""
        self._app.add_url_rule("/", "get_status", self._get_status, methods=["GET"])
        self._app.add_url_rule("/encode", "encode", self._encode_room, methods=["POST"])

    def _extract_model_type_prefix(self, model_type_str: str) -> str:
        """
        Extract model type prefix by removing version suffix.

        Supports model types with version suffixes like:
        - "df_default_2.0.1" -> "df_default"
        - "da_custom_1.5" -> "da_custom"
        - "df_default" -> "df_default" (unchanged)

        Args:
            model_type_str: Model type string possibly with version suffix

        Returns:
            Model type prefix without version suffix
        """
        # Match pattern: anything followed by underscore and version number (e.g., "_2.0.1", "_1.5")
        # Version pattern: _<digit(s)>.<digit(s)> optionally followed by .<digit(s)>
        version_pattern = r'_\d+\.\d+(?:\.\d+)?$'

        # Remove version suffix if present
        model_type_prefix = re.sub(version_pattern, '', model_type_str)

        return model_type_prefix

    def _get_status(self) -> Dict[str, Any]:
        """Get server status endpoint"""
        return jsonify(self._controller.get_status())

    def _encode_room(self):
        """
        Encode room parameters into an image or multiple images (one per window)

        Expected JSON payload (unified structure):
        {
            "model_type": "df_default" | "da_default" | "df_custom" | "da_custom",
            "parameters": {
                "height_roof_over_floor": 3.0,
                "floor_height_above_terrain": 2.0,
                ... (shared room parameters)
                "windows": {
                    "window1": {
                        "window_sill_height": 0.9,
                        "window_frame_ratio": 0.8,
                        "window_height": 1.5,
                        "x1": -0.6, "y1": 0.0, "z1": 0.9,
                        "x2": 0.6, "y2": 0.0, "z2": 2.4,
                        "obstruction_angle_horizon": 45.0,
                        "obstruction_angle_zenith": 30.0,
                        ... (window-specific parameters)
                    },
                    "window2": {
                        ... (optional additional windows)
                    }
                }
            }
        }

        Returns:
            PNG image file (single window) or ZIP file (multiple windows)
        """
        try:
            # Get JSON data
            data = request.get_json()
            if not data:
                raise BadRequest("No JSON data provided")

            # Validate required fields
            if "model_type" not in data:
                raise BadRequest("Missing 'model_type' field")
            if "parameters" not in data:
                raise BadRequest("Missing 'parameters' field")

            # Parse model type
            # Extract model type prefix (strip version suffix like "_2.0.1")
            # e.g., "df_default_2.0.1" -> "df_default"
            model_type_str = data["model_type"]
            model_type_prefix = self._extract_model_type_prefix(model_type_str)

            try:
                model_type = ModelType(model_type_prefix)
            except ValueError:
                valid_types = [mt.value for mt in ModelType]
                raise BadRequest(
                    f"Invalid model_type '{model_type_str}' (parsed as '{model_type_prefix}'). "
                    f"Valid types: {', '.join(valid_types)}"
                )

            # Get parameters
            parameters = data["parameters"]

            # Validate windows structure
            if ParameterName.WINDOWS.value not in parameters:
                raise BadRequest("Missing 'windows' field in parameters. At least one window must be provided.")

            if not isinstance(parameters[ParameterName.WINDOWS.value], dict) or len(parameters[ParameterName.WINDOWS.value]) == 0:
                raise BadRequest("'windows' must be a non-empty dictionary with at least one window.")

            # Determine if single or multi-window
            window_count = len(parameters[ParameterName.WINDOWS.value])
            is_single_window = window_count == 1

            # Log request
            self._logger.info(
                f"{'Single' if is_single_window else 'Multi'}-window encoding request - "
                f"model_type: {model_type_str}, window_count: {window_count}"
            )

            # Encode image(s)
            if is_single_window:
                # Single window - return PNG file
                image_bytes = self._encoding_service.encode_room_image(
                    parameters=parameters,
                    model_type=model_type
                )

                return send_file(
                    io.BytesIO(image_bytes),
                    mimetype='image/png',
                    as_attachment=True,
                    download_name='encoded_room.png'
                )
            else:
                # Multiple windows - return ZIP file
                image_dict = self._encoding_service.encode_multi_window_images(
                    parameters=parameters,
                    model_type=model_type
                )

                # Create ZIP file in memory
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for window_id, image_bytes in image_dict.items():
                        zip_file.writestr(f'{window_id}.png', image_bytes)

                zip_buffer.seek(0)

                self._logger.info(
                    f"Multi-window encoding complete - {len(image_dict)} images in ZIP"
                )

                return send_file(
                    zip_buffer,
                    mimetype='application/zip',
                    as_attachment=True,
                    download_name='encoded_room_windows.zip'
                )

        except BadRequest:
            raise
        except ValueError as e:
            # Log validation error with traceback
            self._logger.error(
                f"Validation error: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            return jsonify({"error": str(e)}), HTTPStatus.BAD_REQUEST.value
        except Exception as e:
            # Log encoding error with full traceback
            error_trace = traceback.format_exc()
            self._logger.error(
                f"Encoding failed: {str(e)}\n"
                f"Error type: {type(e).__name__}\n"
                f"Traceback:\n{error_trace}"
            )
            return jsonify({
                "error": f"Encoding failed: {str(e)}",
                "error_type": type(e).__name__
            }), HTTPStatus.INTERNAL_SERVER_ERROR.value

    @property
    def app(self) -> Flask:
        """Get Flask application instance"""
        return self._app


class ServerLauncher:
    """Launcher class for the server application"""

    @staticmethod
    def create_application() -> ServerApplication:
        """Create and configure the application"""
        return ServerApplication()

    @staticmethod
    def run_server(
        app: ServerApplication,
        host: str = "0.0.0.0",
        port: int = 8080,
        debug: bool = True
    ) -> None:
        """Run the server"""
        """Run the server"""
        log_msg = (
            f"Flask app '{app.app.name}' starting on "
            f"host {host}, port {port}. Debug mode: {debug}"
        )
        app.app.logger.info(log_msg)
        # Disable reloader to prevent bus errors/hangs on WSL2
        app.app.run(host=host, port=port, debug=debug, use_reloader=False)


def main() -> None:
    """Main entry point"""
    launcher = ServerLauncher()
    application = launcher.create_application()
    port = int(os.getenv("PORT", 8082))
    launcher.run_server(application, port=port, debug=True)


# Create app instance for gunicorn only when needed
# Don't create at module import time to avoid bus errors
def create_app():
    """Factory function for creating the Flask app (for gunicorn)"""
    _application = ServerApplication()
    return _application.app


# Only create app instance if not running as main (i.e., when imported by gunicorn)
if __name__ != "__main__":
    app = create_app()
else:
    # Running as main script
    main()