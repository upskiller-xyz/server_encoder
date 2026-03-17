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
    ENCODING_SERVICE_V1 = "encoding_service_v1"
    ENCODING_SERVICE_V2 = "encoding_service_v2"
    ENCODING_SERVICE_V3 = "encoding_service_v3"
    ENCODING_SERVICE_V4 = "encoding_service_v4"


class Endpoint(Enum):
    """API endpoint names"""
    STATUS = "status"
    ENCODE = "encode"
    CALCULATE_DIRECTION = "calculate_direction"
    GET_REFERENCE_POINT = "get_reference_point"
    GET_EXTERNAL_REFERENCE_POINT = "get_external_reference_point"

