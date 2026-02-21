"""OpenAPI specification generator from Pydantic models for auto-documentation"""
from typing import Dict, Any, List
from src.server.schemas import (
    EncodeRequest,
    CalculateDirectionRequest,
    ReferencePointRequest,
    DirectionAngleResponse,
    ReferencePointResponse,
    ErrorResponse,
)


class OpenAPISpecGenerator:
    """Generates OpenAPI 3.0 specification for Flask API using Pydantic models"""

    @staticmethod
    def generate_spec(
        title: str = "Server Encoder API",
        description: str = "Room encoding service with support for multiple encoding schemes",
        version: str = "1.0.0",
        base_url: str = "/"
    ) -> Dict[str, Any]:
        """
        Generate OpenAPI 3.0 specification from Pydantic models.
        
        Args:
            title: API title
            description: API description
            version: API version
            base_url: Base URL for API endpoints
            
        Returns:
            OpenAPI 3.0 specification dict
        """
        return {
            "openapi": "3.0.0",
            "info": {
                "title": title,
                "description": description,
                "version": version,
                "contact": {
                    "name": "API Support"
                }
            },
            "servers": [
                {
                    "url": base_url,
                    "description": "Server API"
                }
            ],
            "paths": OpenAPISpecGenerator._generate_paths(),
            "components": {
                "schemas": OpenAPISpecGenerator._generate_schemas()
            }
        }

    @staticmethod
    def _generate_paths() -> Dict[str, Any]:
        """Generate API paths from endpoint definitions"""
        return {
            "/encode": {
                "post": {
                    "summary": "Encode room parameters",
                    "description": "Encode room parameters into NPZ arrays with image and mask data",
                    "tags": ["Encoding"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/EncodeRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Success - NPZ file with encoded arrays",
                            "content": {
                                "application/octet-stream": {
                                    "schema": {
                                        "type": "string",
                                        "format": "binary"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Validation error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        },
                        "500": {
                            "description": "Server error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/calculate-direction": {
                "post": {
                    "summary": "Calculate direction angles",
                    "description": "Calculate direction_angle for window(s) from room polygon and window coordinates",
                    "tags": ["Calculations"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CalculateDirectionRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Success - Direction angles for windows",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/DirectionAngleResponse"}
                                }
                            }
                        },
                        "400": {
                            "description": "Validation error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        },
                        "500": {
                            "description": "Server error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/get-reference-point": {
                "post": {
                    "summary": "Calculate reference points",
                    "description": "Calculate reference point for window(s) from room polygon and window coordinates",
                    "tags": ["Calculations"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ReferencePointRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Success - Reference points for windows",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ReferencePointResponse"}
                                }
                            }
                        },
                        "400": {
                            "description": "Validation error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        },
                        "500": {
                            "description": "Server error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            }
        }

    @staticmethod
    def _generate_schemas() -> Dict[str, Any]:
        """Generate component schemas from Pydantic models"""
        return {
            "EncodeRequest": EncodeRequest.model_json_schema(),
            "CalculateDirectionRequest": CalculateDirectionRequest.model_json_schema(),
            "ReferencePointRequest": ReferencePointRequest.model_json_schema(),
            "DirectionAngleResponse": DirectionAngleResponse.model_json_schema(),
            "ReferencePointResponse": ReferencePointResponse.model_json_schema(),
            "ErrorResponse": ErrorResponse.model_json_schema(),
        }
