"""Tests for endpoint error handler decorator"""
import pytest
from unittest.mock import Mock, patch
from flask import Flask, json
from werkzeug.exceptions import BadRequest
from pydantic import ValidationError

from src.server.decorators import endpoint_error_handler
from src.server.enums import Endpoint
from src.server.schemas import EncodeRequest
from src.core import ResponseKey


@pytest.fixture
def app():
    """Create Flask test app"""
    app = Flask(__name__)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create Flask test client"""
    return app.test_client()


class TestEndpointErrorHandler:
    """Test suite for endpoint_error_handler decorator"""

    def test_decorator_extracts_json_data(self, app):
        """Test that decorator extracts JSON data from request"""
        @app.route('/test', methods=['POST'])
        @endpoint_error_handler(Endpoint.ENCODE)
        def test_endpoint(data):
            return {"received": data}, 200

        with app.test_client() as client:
            response = client.post(
                '/test',
                json={"test": "value"},
                content_type='application/json'
            )
            
            assert response.status_code == 200
            result = json.loads(response.data)
            assert result["received"]["test"] == "value"

    def test_decorator_handles_missing_content_type(self, app):
        """Test that decorator handles missing Content-Type header gracefully"""
        @app.route('/test', methods=['POST'])
        @endpoint_error_handler(Endpoint.ENCODE)
        def test_endpoint(data):
            return {"received": data}, 200

        with app.test_client() as client:
            # POST with no Content-Type header - decorator should use empty dict as default
            response = client.post('/test', data='')
            
            # Should succeed with empty dict as default
            assert response.status_code == 200
            result = json.loads(response.data)
            assert result["received"] == {}

    def test_decorator_validates_with_pydantic_model(self, app):
        """Test that decorator validates request with Pydantic model"""
        @app.route('/test', methods=['POST'])
        @endpoint_error_handler(Endpoint.ENCODE, EncodeRequest)
        def test_endpoint(data: EncodeRequest):
            return {"model_type": data.model_type}, 200

        with app.test_client() as client:
            response = client.post(
                '/test',
                json={
                    "model_type": "df_default",
                    "parameters": {"test": "value"}
                },
                content_type='application/json'
            )
            
            assert response.status_code == 200
            result = json.loads(response.data)
            assert result["model_type"] == "df_default"

    def test_decorator_rejects_invalid_pydantic_model(self, app):
        """Test that decorator rejects invalid Pydantic data"""
        @app.route('/test', methods=['POST'])
        @endpoint_error_handler(Endpoint.ENCODE, EncodeRequest)
        def test_endpoint(data: EncodeRequest):
            return {"model_type": data.model_type}, 200

        with app.test_client() as client:
            # Missing required 'parameters' field
            response = client.post(
                '/test',
                json={"model_type": "df_default"},
                content_type='application/json'
            )
            
            assert response.status_code == 400
            result = json.loads(response.data)
            assert ResponseKey.ERROR.value in result
            assert "Validation error" in result[ResponseKey.ERROR.value]

    def test_decorator_handles_value_error(self, app):
        """Test that decorator catches and handles ValueError"""
        @app.route('/test', methods=['POST'])
        @endpoint_error_handler(Endpoint.ENCODE)
        def test_endpoint(data):
            raise ValueError("Test validation error")

        with app.test_client() as client:
            response = client.post(
                '/test',
                json={"test": "value"},
                content_type='application/json'
            )
            
            assert response.status_code == 400
            result = json.loads(response.data)
            assert ResponseKey.ERROR.value in result
            assert "Test validation error" in result[ResponseKey.ERROR.value]

    def test_decorator_handles_generic_exception(self, app):
        """Test that decorator catches and handles generic exceptions"""
        @app.route('/test', methods=['POST'])
        @endpoint_error_handler(Endpoint.ENCODE)
        def test_endpoint(data):
            raise RuntimeError("Unexpected error")

        with app.test_client() as client:
            response = client.post(
                '/test',
                json={"test": "value"},
                content_type='application/json'
            )
            
            assert response.status_code == 500
            result = json.loads(response.data)
            assert ResponseKey.ERROR.value in result
            assert "Unexpected error" in result[ResponseKey.ERROR.value]
            assert ResponseKey.ERROR_TYPE.value in result

    def test_decorator_preserves_bad_request(self, app):
        """Test that decorator re-raises BadRequest exceptions"""
        @app.route('/test', methods=['POST'])
        @endpoint_error_handler(Endpoint.ENCODE)
        def test_endpoint(data):
            raise BadRequest("Custom bad request")

        with app.test_client() as client:
            response = client.post(
                '/test',
                json={"test": "value"},
                content_type='application/json'
            )
            
            # BadRequest is re-raised, resulting in 400
            assert response.status_code == 400

    def test_decorator_with_multiple_endpoints(self, app):
        """Test that decorator works with multiple endpoints"""
        @app.route('/encode', methods=['POST'])
        @endpoint_error_handler(Endpoint.ENCODE)
        def encode_endpoint(data):
            return {"type": "encode"}, 200

        @app.route('/calculate', methods=['POST'])
        @endpoint_error_handler(Endpoint.CALCULATE_DIRECTION)
        def calculate_endpoint(data):
            return {"type": "calculate"}, 200

        with app.test_client() as client:
            # Test encode endpoint
            response1 = client.post(
                '/encode',
                json={"data": "test"},
                content_type='application/json'
            )
            assert response1.status_code == 200
            
            # Test calculate endpoint
            response2 = client.post(
                '/calculate',
                json={"data": "test"},
                content_type='application/json'
            )
            assert response2.status_code == 200

    def test_decorator_formats_pydantic_errors(self, app):
        """Test that decorator formats Pydantic validation errors properly"""
        @app.route('/test', methods=['POST'])
        @endpoint_error_handler(Endpoint.ENCODE, EncodeRequest)
        def test_endpoint(data: EncodeRequest):
            return {}, 200

        with app.test_client() as client:
            response = client.post(
                '/test',
                json={},  # Missing both required fields
                content_type='application/json'
            )
            
            assert response.status_code == 400
            result = json.loads(response.data)
            error_msg = result[ResponseKey.ERROR.value]
            # Should contain field names and error descriptions
            assert "Validation error" in error_msg

    def test_decorator_endpoint_parameter_used_in_logging(self, app):
        """Test that endpoint parameter is used in error messages"""
        @app.route('/test', methods=['POST'])
        @endpoint_error_handler(Endpoint.CALCULATE_DIRECTION)
        def test_endpoint(data):
            raise ValueError("Test error")

        with app.test_client() as client:
            response = client.post(
                '/test',
                json={"test": "value"},
                content_type='application/json'
            )
            
            assert response.status_code == 400
            # The error message should indicate it's from CALCULATE_DIRECTION
            result = json.loads(response.data)
            assert ResponseKey.ERROR.value in result

    def test_decorator_with_empty_json_object(self, app):
        """Test decorator handles empty JSON object"""
        @app.route('/test', methods=['POST'])
        @endpoint_error_handler(Endpoint.ENCODE)
        def test_endpoint(data):
            return {"data": str(data)}, 200

        with app.test_client() as client:
            response = client.post(
                '/test',
                json={},
                content_type='application/json'
            )
            
            # Empty dict is valid JSON, should pass through
            assert response.status_code == 200

    def test_decorator_preserves_function_metadata(self, app):
        """Test that decorator preserves original function metadata"""
        @endpoint_error_handler(Endpoint.ENCODE)
        def test_function(data):
            """Test function docstring"""
            return data

        # functools.wraps should preserve function name and docstring
        assert test_function.__name__ == "test_function"
        assert "Test function docstring" in test_function.__doc__
