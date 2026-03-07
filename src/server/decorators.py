"""Server-side decorators for endpoint handlers"""
from functools import wraps
from typing import Callable, Any, Dict, Type, Optional
from flask import request, jsonify
from werkzeug.exceptions import BadRequest, UnsupportedMediaType
from pydantic import BaseModel, ValidationError
import traceback
import logging

from src.core import ResponseKey
from src.server.enums import Endpoint, HTTPStatus

logger = logging.getLogger(__name__)


def endpoint_error_handler(
    endpoint: Endpoint,
    request_model: Optional[Type[BaseModel]] = None
) -> Callable:
    """
    Decorator for endpoint handlers to provide unified error handling and JSON extraction.
    
    Handles:
    - JSON data extraction and validation
    - Pydantic model validation (optional, for type safety)
    - BadRequest exceptions (logged and returned as 400)
    - ValueError exceptions (logged and returned as 400)
    - Pydantic ValidationError (logged and returned as 400)
    - Generic exceptions (logged and returned as 500)
    
    The decorated function should accept data as first parameter after self:
        @endpoint_error_handler(Endpoint.ENCODE)
        def _encode_room_arrays(self, data: Dict[str, Any]):
            # endpoint logic here
    
    Or with Pydantic model for type safety and validation:
        @endpoint_error_handler(Endpoint.ENCODE, EncodeRequest)
        def _encode_room_arrays(self, data: EncodeRequest):
            # data is now type-safe EncodeRequest with validated fields
    
    Args:
        endpoint: The Endpoint enum member for this handler
        request_model: Optional Pydantic BaseModel class for request validation
    
    Returns:
        Decorated function with JSON extraction, validation, and error handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                # Get and validate JSON data
                try:
                    raw_data = request.get_json(force=True)
                except (BadRequest, UnsupportedMediaType) as e:
                    # If no JSON data provided, use empty dict as default
                    raw_data = None
                
                if raw_data is None:
                    raw_data = {}
                
                # Validate with Pydantic model if provided
                if request_model:
                    try:
                        data = request_model(**raw_data)
                    except ValidationError as e:
                        # Format validation errors
                        error_msgs = "; ".join([
                            f"{error['loc'][0]}: {error['msg']}" 
                            for error in e.errors()
                        ])
                        return jsonify({ResponseKey.ERROR.value: f"Validation error: {error_msgs}"}), HTTPStatus.BAD_REQUEST.value
                else:
                    data = raw_data
                
                # Call the endpoint function with data
                return func(*args, data, **kwargs)
            except BadRequest as e:
                # Log bad request error
                logger.error(f"{endpoint.value} bad request: {str(e)}")
                return jsonify({ResponseKey.ERROR.value: str(e)}), HTTPStatus.BAD_REQUEST.value
            except ValidationError as e:
                # Log validation error
                error_msgs = "; ".join([
                    f"{error['loc'][0]}: {error['msg']}" 
                    for error in e.errors()
                ])
                logger.error(f"{endpoint.value} validation error: {error_msgs}")
                return jsonify({ResponseKey.ERROR.value: f"Validation error: {error_msgs}"}), HTTPStatus.BAD_REQUEST.value
            except ValueError as e:
                # Log validation error
                logger.error(f"{endpoint.value} error: {str(e)}")
                return jsonify({ResponseKey.ERROR.value: str(e)}), HTTPStatus.BAD_REQUEST.value
            except Exception as e:
                # Log unexpected error with traceback
                error_trace = traceback.format_exc()
                logger.error(
                    f"{endpoint.value} failed: {str(e)}\n"
                    f"Error type: {type(e).__name__}\n"
                    f"Traceback:\n{error_trace}"
                )
                return jsonify({
                    ResponseKey.ERROR.value: f"{endpoint.value} failed: {str(e)}",
                    ResponseKey.ERROR_TYPE.value: type(e).__name__
                }), HTTPStatus.INTERNAL_SERVER_ERROR.value
        
        return wrapper
    
    return decorator



