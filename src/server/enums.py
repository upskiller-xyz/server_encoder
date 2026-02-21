from enum import Enum


class ModelStatus(Enum):
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class ServerStatus(Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ContentType(Enum):
    IMAGE_JPEG = "image/jpeg"
    IMAGE_PNG = "image/png"
    IMAGE_WEBP = "image/webp"
    IMAGE_BMP = "image/bmp"

    @classmethod
    def is_image(cls, content_type: str) -> bool:
        return content_type.startswith('image/')


class HTTPStatus(Enum):
    OK = 200
    BAD_REQUEST = 400
    INTERNAL_SERVER_ERROR = 500


class ServiceName(Enum):
    """Service names for dependency injection"""
    ENCODING_SERVICE = "encoding_service"
    ENCODING_SERVICE_HSV = "encoding_service_hsv"
    ENCODING_SERVICE_RGB = "encoding_service_rgb"


class Endpoint(Enum):
    """API endpoint names"""
    STATUS = "status"
    ENCODE = "encode"
    CALCULATE_DIRECTION = "calculate_direction"
    GET_REFERENCE_POINT = "get_reference_point"