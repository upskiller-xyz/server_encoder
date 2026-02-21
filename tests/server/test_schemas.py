"""Tests for Pydantic request/response schemas"""
import pytest
from pydantic import ValidationError
from src.server.schemas import (
    EncodeRequest,
    CalculateDirectionRequest,
    ReferencePointRequest,
    DirectionAngleResponse,
    ReferencePointResponse,
    ReferencePoint,
    ErrorResponse,
)


class TestEncodeRequest:
    """Test suite for EncodeRequest model"""

    def test_valid_encode_request_with_defaults(self):
        """Test creating valid EncodeRequest with default encoding_scheme"""
        request = EncodeRequest(
            model_type="df_default",
            parameters={"window_orientation": 3.14}
        )
        assert request.model_type == "df_default"
        assert request.parameters == {"window_orientation": 3.14}
        assert request.encoding_scheme == "hsv"

    def test_valid_encode_request_with_rgb(self):
        """Test creating valid EncodeRequest with RGB encoding scheme"""
        request = EncodeRequest(
            model_type="da_custom",
            parameters={"facade_reflectance": 0.5},
            encoding_scheme="rgb"
        )
        assert request.encoding_scheme == "rgb"

    def test_encode_request_missing_model_type(self):
        """Test EncodeRequest validation fails without model_type"""
        with pytest.raises(ValidationError):
            EncodeRequest(parameters={"test": "value"})

    def test_encode_request_missing_parameters(self):
        """Test EncodeRequest validation fails without parameters"""
        with pytest.raises(ValidationError):
            EncodeRequest(model_type="df_default")

    def test_encode_request_model_dump(self):
        """Test converting EncodeRequest to dict"""
        request = EncodeRequest(
            model_type="df_default",
            parameters={"test": "value"},
            encoding_scheme="hsv"
        )
        data = request.model_dump()
        assert data["model_type"] == "df_default"
        assert data["parameters"] == {"test": "value"}
        assert data["encoding_scheme"] == "hsv"


class TestCalculateDirectionRequest:
    """Test suite for CalculateDirectionRequest model"""

    def test_valid_calculate_direction_request(self):
        """Test creating valid CalculateDirectionRequest"""
        request = CalculateDirectionRequest(
            room_polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            windows={"window1": {"x1": -0.6, "y1": 0.0, "x2": 0.6, "y2": 0.0}}
        )
        assert len(request.room_polygon) == 4
        assert "window1" in request.windows

    def test_calculate_direction_request_missing_room_polygon(self):
        """Test validation fails without room_polygon"""
        with pytest.raises(ValidationError):
            CalculateDirectionRequest(
                windows={"window1": {"x1": -0.6, "y1": 0.0}}
            )

    def test_calculate_direction_request_missing_windows(self):
        """Test validation fails without windows"""
        with pytest.raises(ValidationError):
            CalculateDirectionRequest(
                room_polygon=[[0, 0], [10, 0], [10, 10]]
            )

    def test_calculate_direction_request_multiple_windows(self):
        """Test request with multiple windows"""
        request = CalculateDirectionRequest(
            room_polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            windows={
                "window1": {"x1": -0.6, "y1": 0.0, "x2": 0.6, "y2": 0.0},
                "window2": {"x1": 2.0, "y1": 0.0, "x2": 3.0, "y2": 0.0}
            }
        )
        assert len(request.windows) == 2


class TestReferencePointRequest:
    """Test suite for ReferencePointRequest model"""

    def test_valid_reference_point_request(self):
        """Test creating valid ReferencePointRequest"""
        request = ReferencePointRequest(
            room_polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            windows={
                "window1": {
                    "x1": -0.6, "y1": 0.0, "z1": 1.0,
                    "x2": 0.6, "y2": 0.0, "z2": 2.5
                }
            }
        )
        assert request.room_polygon == [[0, 0], [10, 0], [10, 10], [0, 10]]
        assert "window1" in request.windows
        assert request.windows["window1"]["z1"] == 1.0

    def test_reference_point_request_with_3d_coordinates(self):
        """Test request with full 3D window coordinates"""
        request = ReferencePointRequest(
            room_polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            windows={
                "window1": {
                    "x1": -0.6, "y1": 0.0, "z1": 1.0,
                    "x2": 0.6, "y2": 0.0, "z2": 2.5
                }
            }
        )
        window = request.windows["window1"]
        assert window["z1"] == 1.0
        assert window["z2"] == 2.5


class TestReferencePoint:
    """Test suite for ReferencePoint model"""

    def test_valid_reference_point(self):
        """Test creating valid ReferencePoint"""
        point = ReferencePoint(x=0.0, y=0.0, z=1.75)
        assert point.x == 0.0
        assert point.y == 0.0
        assert point.z == 1.75

    def test_reference_point_with_floats(self):
        """Test ReferencePoint with various float values"""
        point = ReferencePoint(x=-2.5, y=3.7, z=0.5)
        assert point.x == -2.5
        assert point.y == 3.7
        assert point.z == 0.5

    def test_reference_point_missing_coordinate(self):
        """Test validation fails without required coordinate"""
        with pytest.raises(ValidationError):
            ReferencePoint(x=0.0, y=0.0)


class TestDirectionAngleResponse:
    """Test suite for DirectionAngleResponse model"""

    def test_valid_direction_angle_response(self):
        """Test creating valid DirectionAngleResponse"""
        response = DirectionAngleResponse(
            direction_angle={"window1": 3.14159, "window2": 1.5708}
        )
        assert response.direction_angle["window1"] == 3.14159
        assert response.direction_angle["window2"] == 1.5708

    def test_direction_angle_response_single_window(self):
        """Test response with single window"""
        response = DirectionAngleResponse(
            direction_angle={"window1": 0.0}
        )
        assert len(response.direction_angle) == 1
        assert response.direction_angle["window1"] == 0.0


class TestReferencePointResponse:
    """Test suite for ReferencePointResponse model"""

    def test_valid_reference_point_response(self):
        """Test creating valid ReferencePointResponse"""
        response = ReferencePointResponse(
            reference_point={
                "window1": ReferencePoint(x=0.0, y=0.0, z=1.75),
                "window2": ReferencePoint(x=2.0, y=1.0, z=1.5)
            }
        )
        assert response.reference_point["window1"].z == 1.75
        assert response.reference_point["window2"].x == 2.0

    def test_reference_point_response_from_dict(self):
        """Test creating response from dict data"""
        response = ReferencePointResponse(
            reference_point={
                "window1": {"x": 0.0, "y": 0.0, "z": 1.75}
            }
        )
        assert response.reference_point["window1"].x == 0.0


class TestErrorResponse:
    """Test suite for ErrorResponse model"""

    def test_valid_error_response_with_type(self):
        """Test creating valid ErrorResponse with error_type"""
        response = ErrorResponse(
            error="Invalid encoding_scheme",
            error_type="BadRequest"
        )
        assert response.error == "Invalid encoding_scheme"
        assert response.error_type == "BadRequest"

    def test_valid_error_response_without_type(self):
        """Test creating valid ErrorResponse without error_type"""
        response = ErrorResponse(error="Something went wrong")
        assert response.error == "Something went wrong"
        assert response.error_type is None

    def test_error_response_missing_error_message(self):
        """Test validation fails without error message"""
        with pytest.raises(ValidationError):
            ErrorResponse(error_type="CustomError")
