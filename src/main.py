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
from src.components.enums import ModelType, ParameterName, EncodingScheme
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

        # Encoding services (both RGB and HSV)
        self._encoding_service_hsv = EncodingServiceFactory.get_instance(self._logger, EncodingScheme.HSV)
        self._encoding_service_rgb = EncodingServiceFactory.get_instance(self._logger, EncodingScheme.RGB)

        # Services dict (default to HSV)
        services = {
            "encoding_service": self._encoding_service_hsv,
            "encoding_service_hsv": self._encoding_service_hsv,
            "encoding_service_rgb": self._encoding_service_rgb
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
        self._app.add_url_rule("/encode", "encode", self._encode_room_arrays, methods=["POST"])
        self._app.add_url_rule("/calculate-direction", "calculate_direction", self._calculate_direction, methods=["POST"])

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

    def _encode_room_arrays(self):
        """
        Encode room parameters into numpy arrays (returns .npz file)

        Expected JSON payload: Same as /encode endpoint

        Returns:
            .npz file containing:
            - For single window: 'image' and 'mask' arrays
            - For multi-window: 'window1_image', 'window1_mask', 'window2_image', etc.
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

            # Parse encoding scheme (default to HSV)
            encoding_scheme_str = data.get("encoding_scheme", "hsv")
            try:
                encoding_scheme = EncodingScheme(encoding_scheme_str)
            except ValueError:
                valid_schemes = [es.value for es in EncodingScheme]
                raise BadRequest(
                    f"Invalid encoding_scheme '{encoding_scheme_str}'. "
                    f"Valid schemes: {', '.join(valid_schemes)}"
                )

            # Parse model type
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

            # Select encoding service based on encoding scheme
            encoding_service = (
                self._encoding_service_hsv if encoding_scheme == EncodingScheme.HSV
                else self._encoding_service_rgb
            )

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
                f"{'Single' if is_single_window else 'Multi'}-window array encoding request - "
                f"model_type: {model_type_str}, encoding_scheme: {encoding_scheme_str}, window_count: {window_count}"
            )

            # Encode arrays
            import numpy as np

            if is_single_window:
                # Single window
                image_array, mask_array = encoding_service.encode_room_image_arrays(
                    parameters=parameters,
                    model_type=model_type
                )

                # Create NPZ file
                npz_buffer = io.BytesIO()
                arrays_dict = {'image': image_array}
                if mask_array is not None:
                    arrays_dict['mask'] = mask_array
                np.savez_compressed(npz_buffer, **arrays_dict)
                npz_buffer.seek(0)

                return send_file(
                    npz_buffer,
                    mimetype='application/octet-stream',
                    as_attachment=True,
                    download_name='encoded_room.npz'
                )
            else:
                # Multiple windows
                images_dict, masks_dict = encoding_service.encode_multi_window_images_arrays(
                    parameters=parameters,
                    model_type=model_type
                )

                # Create NPZ file with all arrays
                npz_buffer = io.BytesIO()
                arrays_dict = {}
                for window_id in images_dict.keys():
                    arrays_dict[f'{window_id}_image'] = images_dict[window_id]
                    if masks_dict[window_id] is not None:
                        arrays_dict[f'{window_id}_mask'] = masks_dict[window_id]

                np.savez_compressed(npz_buffer, **arrays_dict)
                npz_buffer.seek(0)

                self._logger.info(
                    f"Multi-window array encoding complete - {len(images_dict)} images in NPZ"
                )

                return send_file(
                    npz_buffer,
                    mimetype='application/octet-stream',
                    as_attachment=True,
                    download_name='encoded_room_windows.npz'
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
                f"Array encoding failed: {str(e)}\n"
                f"Error type: {type(e).__name__}\n"
                f"Traceback:\n{error_trace}"
            )
            return jsonify({
                "error": f"Array encoding failed: {str(e)}",
                "error_type": type(e).__name__
            }), HTTPStatus.INTERNAL_SERVER_ERROR.value

    def _calculate_direction(self):
        """
        Calculate direction_angle for window(s) from room polygon and window coordinates

        Expected JSON payload:
        {
            "parameters": {
                "room_polygon": [[x1, y1], [x2, y2], [x3, y3], ...],
                "windows": {
                    "window1": {
                        "x1": -0.6, "y1": 0.0,
                        "x2": 0.6, "y2": 0.0
                    },
                    "window2": {
                        ... (optional additional windows)
                    }
                }
            }
        }

        Returns:
            JSON with direction_angle for each window:
            {
                "direction_angles": {
                    "window1": 3.14159,  # radians
                    "window2": 1.5708
                },
                "direction_angles_degrees": {
                    "window1": 180.0,
                    "window2": 90.0
                }
            }
        """
        try:
            # Get JSON data
            data = request.get_json()
            if not data:
                raise BadRequest("No JSON data provided")

            # Validate required fields
            if "parameters" not in data:
                raise BadRequest("Missing 'parameters' field")

            parameters = data["parameters"]

            # Calculate direction angles (use HSV service, but doesn't matter which)
            direction_angles_rad = self._encoding_service_hsv.calculate_direction_angle(parameters)

            # Convert to degrees for convenience
            import math
            direction_angles_deg = {
                window_id: angle * 180 / math.pi
                for window_id, angle in direction_angles_rad.items()
            }

            # Log success
            self._logger.info(
                f"Direction angle calculation successful - "
                f"window_count: {len(direction_angles_rad)}"
            )

            return jsonify({
                "direction_angles": direction_angles_rad,
                "direction_angles_degrees": direction_angles_deg
            }), HTTPStatus.OK.value

        except BadRequest:
            raise
        except ValueError as e:
            # Log validation error
            self._logger.error(f"Direction angle calculation error: {str(e)}")
            return jsonify({"error": str(e)}), HTTPStatus.BAD_REQUEST.value
        except Exception as e:
            # Log unexpected error with traceback
            error_trace = traceback.format_exc()
            self._logger.error(
                f"Direction angle calculation failed: {str(e)}\n"
                f"Error type: {type(e).__name__}\n"
                f"Traceback:\n{error_trace}"
            )
            return jsonify({
                "error": f"Direction angle calculation failed: {str(e)}",
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