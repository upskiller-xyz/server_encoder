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
import logging
import io
import traceback
import re
import numpy as np

from src.server.enums import  HTTPStatus, LogLevel
from src.core import ParameterName, EncodingScheme
from src.server.services import EncodingServiceFactory
from src.server.controllers.base_controller import ServerController

logger = logging.getLogger("logger")


class ServerApplication:
    """Main application class implementing dependency injection and OOP principles"""

    def __init__(self, app_name: str = "Server Application"):
        self._app = Flask(app_name)
        CORS(self._app)
        self._controller = None
        self._encoding_service = None
        self._setup_dependencies()
        self._setup_routes()

    def _setup_dependencies(self) -> None:
        """Setup all dependencies using dependency injection"""
        # Logger

        # Encoding services (both RGB and HSV)
        self._encoding_service_hsv = EncodingServiceFactory.get_instance(EncodingScheme.HSV)
        self._encoding_service_rgb = EncodingServiceFactory.get_instance(EncodingScheme.RGB)

        # Services dict (default to HSV)
        services = {
            "encoding_service": self._encoding_service_hsv,
            "encoding_service_hsv": self._encoding_service_hsv,
            "encoding_service_rgb": self._encoding_service_rgb
        }

        # Controller
        self._controller = ServerController(
            services=services
        )

        # Initialize controller
        self._controller.initialize()

    def _setup_routes(self) -> None:
        """Setup Flask routes"""
        self._app.add_url_rule("/", "get_status", self._get_status, methods=["GET"])
        self._app.add_url_rule("/encode", "encode", self._encode_room_arrays, methods=["POST"])
        self._app.add_url_rule("/calculate-direction", "calculate_direction", self._calculate_direction, methods=["POST"])
        self._app.add_url_rule("/get-reference-point", "get_reference_point", self._get_reference_point, methods=["POST"])

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

            # Select encoding service based on encoding scheme
            encoding_service = (
                self._encoding_service_hsv if encoding_scheme == EncodingScheme.HSV
                else self._encoding_service_rgb
            )

            # Parse request using typed model
            try:
                room_request = encoding_service.parse_request(data) # type: ignore
            except ValueError as e:
                raise BadRequest(str(e))

            # Log request
            logger.info(
                f"{'Single' if len(room_request.windows) == 1 else 'Multi'}-window array encoding request - "
                f"model_type: {room_request.model_type.value}, encoding_scheme: {encoding_scheme_str}, "
                f"window_count: {len(room_request.windows)}"
            )

            # For backward compatibility, convert back to flat dict
            parameters = room_request.to_flat_dict()
            model_type = room_request.model_type

            # Determine if single or multi-window
            is_single_window = len(room_request.windows) == 1

            # Encode arrays
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
                    download_name='result.npz'
                )
            else:
                # Multiple windows
                result = encoding_service.encode_multi_window_images_arrays(
                    parameters=parameters,
                    model_type=model_type
                )

                # Create NPZ file with all arrays
                npz_buffer = io.BytesIO()
                arrays_dict = {}
                for window_id in result.window_ids():
                    arrays_dict[f'{window_id}_image'] = result.get_image(window_id)
                    mask = result.get_mask(window_id)
                    if mask is not None:
                        arrays_dict[f'{window_id}_mask'] = mask

                np.savez_compressed(npz_buffer, **arrays_dict)
                npz_buffer.seek(0)

                logger.info(
                    f"Multi-window array encoding complete - {len(result.images)} images in NPZ"
                )

                return send_file(
                    npz_buffer,
                    mimetype='application/octet-stream',
                    as_attachment=True,
                    download_name='result.npz'
                )

        except BadRequest:
            raise
        except ValueError as e:
            # Log validation error with traceback
            logger.error(
                f"Validation error: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            return jsonify({"error": str(e)}), HTTPStatus.BAD_REQUEST.value
        except Exception as e:
            # Log encoding error with full traceback
            error_trace = traceback.format_exc()
            logger.error(
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

        Returns:
            JSON with direction_angle for each window:
            {
                "direction_angle": {
                    "window1": 3.14159,  # radians
                    "window2": 1.5708
                }
            }
        """
        try:
            # Get JSON data
            data = request.get_json()
            if not data:
                raise BadRequest("No JSON data provided")

            # Calculate direction angles (use HSV service, but doesn't matter which)
            direction_angles_rad = self._encoding_service_hsv.calculate_direction_angle(data)

            # Log success
            logger.info(
                f"Direction angle calculation successful - "
                f"window_count: {len(direction_angles_rad)}"
            )

            return jsonify({
                ParameterName.DIRECTION_ANGLE.value: direction_angles_rad
            }), HTTPStatus.OK.value

        except BadRequest:
            raise
        except ValueError as e:
            # Log validation error
            logger.error(f"Direction angle calculation error: {str(e)}")
            return jsonify({"error": str(e)}), HTTPStatus.BAD_REQUEST.value
        except Exception as e:
            # Log unexpected error with traceback
            error_trace = traceback.format_exc()
            logger.error(
                f"Direction angle calculation failed: {str(e)}\n"
                f"Error type: {type(e).__name__}\n"
                f"Traceback:\n{error_trace}"
            )
            return jsonify({
                "error": f"Direction angle calculation failed: {str(e)}",
                "error_type": type(e).__name__
            }), HTTPStatus.INTERNAL_SERVER_ERROR.value

    def _get_reference_point(self):
        """
        Calculate reference point for window(s) from room polygon and window coordinates

        Expected JSON payload:
        {
            "room_polygon": [[x1, y1], [x2, y2], [x3, y3], ...],
            "windows": {
                "window1": {
                    "x1": -0.6, "y1": 0.0, "z1": 1.0,
                    "x2": 0.6, "y2": 0.0, "z2": 2.5
                },
                "window2": {
                    ... (optional additional windows)
                }
            }
        }

        Returns:
            JSON with reference_point for each window:
            {
                "reference_point": {
                    "window1": {"x": 0.0, "y": 0.0, "z": 1.75},
                    "window2": {"x": 2.0, "y": 1.0, "z": 1.5}
                }
            }
        """
        try:
            # Get JSON data
            data = request.get_json()
            if not data:
                raise BadRequest("No JSON data provided")

            # Calculate reference points (use HSV service, but doesn't matter which)
            reference_points = self._encoding_service_hsv.calculate_reference_point(data)

            # Log success
            logger.info(
                f"Reference point calculation successful - "
                f"window_count: {len(reference_points)}"
            )

            return jsonify({
                "reference_point": reference_points
            }), HTTPStatus.OK.value

        except BadRequest:
            raise
        except ValueError as e:
            # Log validation error
            logger.error(f"Reference point calculation error: {str(e)}")
            return jsonify({"error": str(e)}), HTTPStatus.BAD_REQUEST.value
        except Exception as e:
            # Log unexpected error with traceback
            error_trace = traceback.format_exc()
            logger.error(
                f"Reference point calculation failed: {str(e)}\n"
                f"Error type: {type(e).__name__}\n"
                f"Traceback:\n{error_trace}"
            )
            return jsonify({
                "error": f"Reference point calculation failed: {str(e)}",
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