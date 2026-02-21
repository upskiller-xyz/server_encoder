"""API request/response models using Pydantic for type safety and validation in Flask"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


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
