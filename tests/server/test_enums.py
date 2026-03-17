"""Tests for server enums (ServiceName, Endpoint)"""
import pytest
from src.server.enums import ServiceName, Endpoint, HTTPStatus


class TestServiceNameEnum:
    """Test suite for ServiceName enum"""

    def test_encoding_service_value(self):
        """Test ENCODING_SERVICE enum value"""
        assert ServiceName.ENCODING_SERVICE.value == "encoding_service"

    def test_encoding_service_v1_value(self):
        """Test ENCODING_SERVICE_V1 enum value"""
        assert ServiceName.ENCODING_SERVICE_V1.value == "encoding_service_v1"

    def test_encoding_service_v2_value(self):
        """Test ENCODING_SERVICE_V2 enum value"""
        assert ServiceName.ENCODING_SERVICE_V2.value == "encoding_service_v2"

    def test_encoding_service_v3_value(self):
        """Test ENCODING_SERVICE_V3 enum value"""
        assert ServiceName.ENCODING_SERVICE_V3.value == "encoding_service_v3"

    def test_encoding_service_v4_value(self):
        """Test ENCODING_SERVICE_V4 enum value"""
        assert ServiceName.ENCODING_SERVICE_V4.value == "encoding_service_v4"

    def test_service_name_all_members(self):
        """Test all ServiceName members are defined"""
        members = [m for m in ServiceName]
        assert len(members) == 5
        assert ServiceName.ENCODING_SERVICE in members
        assert ServiceName.ENCODING_SERVICE_V1 in members
        assert ServiceName.ENCODING_SERVICE_V2 in members
        assert ServiceName.ENCODING_SERVICE_V3 in members
        assert ServiceName.ENCODING_SERVICE_V4 in members


class TestEndpointEnum:
    """Test suite for Endpoint enum"""

    def test_status_value(self):
        """Test STATUS endpoint value"""
        assert Endpoint.STATUS.value == "status"

    def test_encode_value(self):
        """Test ENCODE endpoint value"""
        assert Endpoint.ENCODE.value == "encode"

    def test_calculate_direction_value(self):
        """Test CALCULATE_DIRECTION endpoint value"""
        assert Endpoint.CALCULATE_DIRECTION.value == "calculate_direction"

    def test_get_reference_point_value(self):
        """Test GET_REFERENCE_POINT endpoint value"""
        assert Endpoint.GET_REFERENCE_POINT.value == "get_reference_point"

    def test_get_external_reference_point_value(self):
        """Test GET_EXTERNAL_REFERENCE_POINT endpoint value"""
        assert Endpoint.GET_EXTERNAL_REFERENCE_POINT.value == "get_external_reference_point"

    def test_endpoint_all_members(self):
        """Test all Endpoint members are defined"""
        members = [m for m in Endpoint]
        assert len(members) == 5
        assert Endpoint.STATUS in members
        assert Endpoint.ENCODE in members
        assert Endpoint.CALCULATE_DIRECTION in members
        assert Endpoint.GET_REFERENCE_POINT in members
        assert Endpoint.GET_EXTERNAL_REFERENCE_POINT in members

    def test_endpoint_enum_names(self):
        """Test endpoint enum member names"""
        assert Endpoint.STATUS.name == "STATUS"
        assert Endpoint.ENCODE.name == "ENCODE"
        assert Endpoint.CALCULATE_DIRECTION.name == "CALCULATE_DIRECTION"
        assert Endpoint.GET_REFERENCE_POINT.name == "GET_REFERENCE_POINT"


class TestHTTPStatusEnum:
    """Test suite for HTTPStatus enum"""

    def test_http_status_ok(self):
        """Test OK status code"""
        assert HTTPStatus.OK.value == 200

    def test_http_status_bad_request(self):
        """Test BAD_REQUEST status code"""
        assert HTTPStatus.BAD_REQUEST.value == 400

    def test_http_status_internal_server_error(self):
        """Test INTERNAL_SERVER_ERROR status code"""
        assert HTTPStatus.INTERNAL_SERVER_ERROR.value == 500
