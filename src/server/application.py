"""Server application implementation"""
from typing import Dict, Any
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
import logging
import io
import numpy as np

from src.core import ParameterName, EncodingScheme, ResponseKey
from src.core.model_type_manager import ModelTypeManager
from src.server.enums import HTTPStatus, Endpoint
from src.server.services import EncodingServiceFactory
from src.server.controllers.base_controller import ServerController
from src.server.decorators import endpoint_error_handler
from src.server.schemas import (
    EncodeRequest,
    CalculateDirectionRequest,
    ReferencePointRequest,
)
from src.server.openapi import OpenAPISpecGenerator


logger = logging.getLogger("logger")


class ServerApplication:
    """Main application class implementing dependency injection and OOP principles"""

    def __init__(self, app_name: str = "Server Application") -> None:
        """
        Initialize the Flask application with dependencies.
        
        Args:
            app_name: Name of the Flask application
        """
        self._app: Flask = Flask(app_name)
        CORS(self._app)
        self._controller: ServerController | None = None
        self._encoding_service_hsv = None
        self._encoding_service_rgb = None
        self._setup_dependencies()
        self._setup_routes()

    def _setup_dependencies(self) -> None:
        """Setup all dependencies using dependency injection"""
        # Encoding services (both RGB and HSV)
        self._encoding_service_hsv = EncodingServiceFactory.get_instance(EncodingScheme.HSV)
        self._encoding_service_rgb = EncodingServiceFactory.get_instance(EncodingScheme.RGB)

        # Services dict (default to HSV)
        from src.server.enums import ServiceName
        services = {
            ServiceName.ENCODING_SERVICE.value: self._encoding_service_hsv,
            ServiceName.ENCODING_SERVICE_HSV.value: self._encoding_service_hsv,
            ServiceName.ENCODING_SERVICE_RGB.value: self._encoding_service_rgb
        }

        # Controller
        self._controller = ServerController(services=services)

        # Initialize controller
        self._controller.initialize()

    def _setup_routes(self) -> None:
        """Setup Flask routes"""
        self._app.add_url_rule("/", "get_status", self._get_status, methods=["GET"])
        self._app.add_url_rule("/encode", "encode", self._encode_room_arrays, methods=["POST"])
        self._app.add_url_rule(
            "/calculate-direction",
            "calculate_direction",
            self._calculate_direction,
            methods=["POST"]
        )
        self._app.add_url_rule(
            "/get-reference-point",
            "get_reference_point",
            self._get_reference_point,
            methods=["POST"]
        )
        
        # Documentation endpoints
        self._app.add_url_rule("/openapi.json", "openapi_spec", self._openapi_spec, methods=["GET"])
        self._app.add_url_rule("/docs", "swagger_ui", self._swagger_ui, methods=["GET"])
        self._app.add_url_rule("/redoc", "redoc", self._redoc, methods=["GET"])

    def _get_status(self) -> Dict[str, Any]:
        """
        Get server status endpoint.
        
        Returns:
            JSON response with server status information
        """
        return jsonify(self._controller.get_status())

    @endpoint_error_handler(Endpoint.ENCODE)
    # Optional: Add Pydantic validation for type safety: @endpoint_error_handler(Endpoint.ENCODE, EncodeRequest)
    def _encode_room_arrays(self, data: Dict[str, Any]) -> tuple:
        """
        Encode room parameters into NPZ arrays.

        Expected JSON payload:
        {
            "model_type": "df_default",
            "encoding_scheme": "hsv" (optional, defaults to hsv),
            "parameters": {
                ... encoding parameters ...
            }
        }

        Returns:
            tuple: (response, status_code) with NPZ file containing image and mask arrays
        """

        # Parse encoding scheme (default to HSV)
        encoding_scheme_str = data.get(ParameterName.ENCODING_SCHEME.value, EncodingScheme.HSV.value)
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

        # Handle versioned model types (e.g., "df_default_2.0.1" -> "df_default")
        if ResponseKey.MODEL_TYPE.value in data:
            data[ResponseKey.MODEL_TYPE.value] = ModelTypeManager.extract_prefix(
                data[ResponseKey.MODEL_TYPE.value]
            )

        # Parse request using typed model
        try:
            room_request = encoding_service.parse_request(data)  # type: ignore
        except ValueError as e:
            raise BadRequest(str(e))

        # Log request
        logger.info(
            f"{'Single' if len(room_request.windows) == 1 else 'Multi'}-window encoding request - "
            f"model_type: {room_request.model_type.value}, encoding_scheme: {encoding_scheme_str}, "
            f"window_count: {len(room_request.windows)}"
        )

        # For backward compatibility, convert back to flat dict
        parameters = room_request.to_flat_dict()
        model_type = room_request.model_type

        # Determine if single or multi-window
        is_single_window = len(room_request.windows) == 1

        # Encode images
        if is_single_window:
            # Single window - encode and return arrays in NPZ format
            image_array, mask_array = encoding_service.encode_room_image_arrays(
                parameters=parameters,
                model_type=model_type
            )

            logger.info(f"Single-window array encoding complete - image shape: {image_array.shape}")

            # Create NPZ file with image and mask arrays
            npz_buffer = io.BytesIO()
            arrays_dict = {
                'window1_image': image_array,
            }
            if mask_array is not None:
                arrays_dict['window1_mask'] = mask_array

            np.savez_compressed(npz_buffer, **arrays_dict)
            npz_buffer.seek(0)

            return send_file(
                npz_buffer,
                mimetype='application/octet-stream',
                as_attachment=True,
                download_name='result.npz'
            )
        else:
            # Multiple windows - return NPZ file with arrays
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

    @endpoint_error_handler(Endpoint.CALCULATE_DIRECTION)
    def _calculate_direction(self, data: Dict[str, Any]) -> tuple:
        """
        Calculate direction_angle for window(s) from room polygon and window coordinates.

        Expected JSON payload:
        {
            "parameters": {
                "room_polygon": [[x1, y1], [x2, y2], [x3, y3], ...],
                "windows": {
                    "window1": {
                        "x1": -0.6, "y1": 0.0,
                        "x2": 0.6, "y2": 0.0
                    },
                    "window2": {...}
                }
            }
        }

        Returns:
            tuple: (response_dict, status_code) with direction_angle for each window in radians
        """

        # Extract parameters from request (handle wrapper structure)
        parameters = data.get(ResponseKey.PARAMETERS.value, data)

        # Calculate direction angles (use HSV service, but doesn't matter which)
        direction_angles_rad = self._encoding_service_hsv.calculate_direction_angle(parameters)

        # Log success
        logger.info(
            f"Direction angle calculation successful - "
            f"window_count: {len(direction_angles_rad)}"
        )

        return jsonify({
            ParameterName.DIRECTION_ANGLE.value: direction_angles_rad
        }), HTTPStatus.OK.value

    @endpoint_error_handler(Endpoint.GET_REFERENCE_POINT)
    def _get_reference_point(self, data: Dict[str, Any]) -> tuple:
        """
        Calculate reference point for window(s) from room polygon and window coordinates.

        Expected JSON payload:
        {
            "room_polygon": [[x1, y1], [x2, y2], [x3, y3], ...],
            "windows": {
                "window1": {
                    "x1": -0.6, "y1": 0.0, "z1": 1.0,
                    "x2": 0.6, "y2": 0.0, "z2": 2.5
                },
                "window2": {...}
            }
        }

        Returns:
            tuple: (response_dict, status_code) with reference_point for each window
        """
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

    def _openapi_spec(self) -> Dict[str, Any]:
        """
        Return OpenAPI 3.0 specification.
        
        Returns:
            JSON with complete API specification
        """
        spec = OpenAPISpecGenerator.generate_spec(
            title="Server Encoder API",
            description="Room encoding service with support for multiple encoding schemes",
            version="1.0.0",
            base_url="/"
        )
        return jsonify(spec)

    def _swagger_ui(self) -> str:
        """
        Return Swagger UI HTML.
        
        Interactive API documentation at /docs
        """
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Server Encoder API - Swagger UI</title>
            <meta charset="utf-8"/>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3/swagger-ui.css">
        </head>
        <body>
            <div id="swagger-ui"></div>
            <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3/swagger-ui-bundle.js"></script>
            <script>
            window.onload = function() {
                window.ui = SwaggerUIBundle({
                    url: "/openapi.json",
                    dom_id: '#swagger-ui',
                    presets: [
                        SwaggerUIBundle.presets.apis,
                        SwaggerUIBundle.SwaggerUIStandalonePreset
                    ],
                    layout: "BaseLayout",
                    requestInterceptor: (request) => {
                        request.headers['X-API-Version'] = '1.0.0';
                        return request;
                    }
                })
            }
            </script>
        </body>
        </html>
        """

    def _redoc(self) -> str:
        """
        Return ReDoc HTML.
        
        Alternative interactive API documentation at /redoc
        """
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Server Encoder API - ReDoc</title>
            <meta charset="utf-8"/>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
            <style>
              body {
                margin: 0;
                padding: 0;
              }
            </style>
        </head>
        <body>
            <redoc spec-url='/openapi.json'></redoc>
            <script src="https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js"></script>
        </body>
        </html>
        """

    @property
    def app(self) -> Flask:
        """
        Get Flask application instance.
        
        Returns:
            Flask: The Flask application object
        """
        return self._app
