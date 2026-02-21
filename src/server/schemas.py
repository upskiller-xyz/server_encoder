"""API request/response models using Pydantic for type safety and validation in Flask"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import io
import numpy as np
from flask import send_file

from src.server.key_manager import KeyManager


class EncodeRequest(BaseModel):
    """
    Room encoding request model for type safety and validation.
    
    Can be used with endpoint_error_handler decorator for automatic validation:
        @endpoint_error_handler(Endpoint.ENCODE, EncodeRequest)
    """
    model_type: str = Field(..., description="Model type (e.g., 'df_default', 'da_custom')")
    parameters: Dict[str, Any] = Field(..., description="Encoding parameters")
    encoding_scheme: str = Field(default="hsv", description="Encoding scheme: 'hsv' or 'rgb'")

    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "df_default",
                "encoding_scheme": "hsv",
                "parameters": {
                    "window_orientation": 3.14159,
                    "facade_reflectance": 1.0,
                    "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]
                }
            }
        }


class CalculateDirectionRequest(BaseModel):
    """
    Direction angle calculation request model for type safety and validation.
    
    Can be used with endpoint_error_handler decorator for automatic validation.
    """
    room_polygon: List[List[float]] = Field(
        ..., 
        description="Room polygon coordinates [[x1, y1], [x2, y2], ...]"
    )
    windows: Dict[str, Dict[str, Any]] = Field(
        ..., 
        description="Windows with coordinates"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "windows": {
                    "window1": {
                        "x1": -0.6,
                        "y1": 0.0,
                        "x2": 0.6,
                        "y2": 0.0
                    }
                }
            }
        }


class ReferencePointRequest(BaseModel):
    """
    Reference point calculation request model for type safety and validation.
    
    Can be used with endpoint_error_handler decorator for automatic validation.
    """
    room_polygon: List[List[float]] = Field(
        ..., 
        description="Room polygon coordinates [[x1, y1], [x2, y2], ...]"
    )
    windows: Dict[str, Dict[str, Any]] = Field(
        ..., 
        description="Windows with 3D coordinates"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "windows": {
                    "window1": {
                        "x1": -0.6,
                        "y1": 0.0,
                        "z1": 1.0,
                        "x2": 0.6,
                        "y2": 0.0,
                        "z2": 2.5
                    }
                }
            }
        }


class DirectionAngleResponse(BaseModel):
    """
    Direction angle calculation response model for type safety.
    
    Used to structure and validate response data from direction angle calculations.
    """
    direction_angle: Dict[str, float] = Field(
        ..., 
        description="Direction angle in radians for each window"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "direction_angle": {
                    "window1": 3.14159,
                    "window2": 1.5708
                }
            }
        }


class ReferencePoint(BaseModel):
    """Reference point coordinates"""
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    z: float = Field(..., description="Z coordinate (height)")


class ReferencePointResponse(BaseModel):
    """
    Reference point calculation response model for type safety.
    
    Used to structure and validate response data from reference point calculations.
    """
    reference_point: Dict[str, ReferencePoint] = Field(
        ..., 
        description="Reference point coordinates for each window"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "reference_point": {
                    "window1": {"x": 0.0, "y": 0.0, "z": 1.75},
                    "window2": {"x": 2.0, "y": 1.0, "z": 1.5}
                }
            }
        }


class ErrorResponse(BaseModel):
    """
    Standard error response model for type safety.
    
    Returned by endpoint error handler on validation or processing errors.
    """
    error: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(default=None, description="Error type/class name")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid encoding_scheme 'invalid'. Valid schemes: hsv, rgb",
                "error_type": "BadRequest"
            }
        }


class EncoderResponse:
    """
    Builder for encoder response with NPZ file generation.
    
    Handles construction of image and mask arrays into NPZ format for both
    single-window and multi-window encoding requests.
    """
    
    def __init__(self) -> None:
        """Initialize empty arrays dictionary"""
        self._arrays_dict: Dict[str, Any] = {}
    
    def add_window(self, 
                   image_array: Any, 
                   mask_array: Optional[Any] = None, 
                   window_id: Optional[str] = None) -> None:
        """
        Add a window's image and mask arrays to the response.
        
        Args:
            image_array: Image array for the window
            mask_array: Optional mask array for the window
            window_id: Optional window ID for multi-window encoding (None for single-window)
        
        Example:
            # Single window
            response = EncoderResponse()
            response.add_window(image_array, mask_array)
            
            # Multi-window
            response = EncoderResponse()
            response.add_window(window1_image, window1_mask, "window1")
            response.add_window(window2_image, window2_mask, "window2")
        """
        self._arrays_dict[KeyManager.get_image_key(window_id)] = image_array
        if mask_array is not None:
            self._arrays_dict[KeyManager.get_mask_key(window_id)] = mask_array
    
    def to_npz_response(self) -> tuple:
        """
        Create NPZ file response from all added windows.
        
        Returns:
            tuple: (response, status_code) with NPZ file for downloading
        
        Example:
            response = EncoderResponse()
            response.add_window(image_array, mask_array)
            return response.to_npz_response()
        """
        npz_buffer = io.BytesIO()
        np.savez_compressed(npz_buffer, **self._arrays_dict)
        npz_buffer.seek(0)
        
        return send_file(
            npz_buffer,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='result.npz'
        )
