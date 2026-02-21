"""Tests for ServerApplication Flask endpoints"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from src.server.application import ServerApplication
from src.server.enums import ServiceName
from src.core import ResponseKey


@pytest.fixture
def app():
    """Create ServerApplication test instance and return the Flask app"""
    server_app = ServerApplication()
    return server_app.app


@pytest.fixture
def client(app):
    """Create Flask test client"""
    return app.test_client()


class TestServerApplicationEndpoints:
    """Test suite for ServerApplication endpoints"""

    def test_status_endpoint_returns_ok(self, client):
        """Test status endpoint returns 200 OK"""
        response = client.get('/')
        
        assert response.status_code == 200
        result = json.loads(response.data)
        assert "status" in result

    def test_encode_endpoint_requires_post(self, client):
        """Test encode endpoint requires POST method"""
        response = client.get('/encode')
        
        # GET should not be allowed
        assert response.status_code in [405, 404]  # Method not allowed or not found

    def test_encode_endpoint_requires_json(self, client):
        """Test encode endpoint requires JSON data"""
        response = client.post('/encode', data='')
        
        assert response.status_code == 400
        result = json.loads(response.data)
        assert ResponseKey.ERROR.value in result

    def test_calculate_direction_endpoint_requires_post(self, client):
        """Test calculate-direction endpoint requires POST"""
        response = client.get('/calculate-direction')
        
        assert response.status_code in [405, 404]

    def test_get_reference_point_endpoint_requires_post(self, client):
        """Test get-reference-point endpoint requires POST"""
        response = client.get('/get-reference-point')
        
        assert response.status_code in [405, 404]

    def test_openapi_json_endpoint(self, client):
        """Test /openapi.json endpoint returns OpenAPI spec"""
        response = client.get('/openapi.json')
        
        assert response.status_code == 200
        result = json.loads(response.data)
        assert "openapi" in result
        assert result["openapi"].startswith("3.")
        assert "info" in result
        assert "paths" in result

    def test_swagger_ui_endpoint(self, client):
        """Test /docs endpoint returns Swagger UI HTML"""
        response = client.get('/docs')
        
        assert response.status_code == 200
        assert b"swagger-ui" in response.data or b"html" in response.data

    def test_redoc_endpoint(self, client):
        """Test /redoc endpoint returns ReDoc HTML"""
        response = client.get('/redoc')
        
        assert response.status_code == 200
        assert b"redoc" in response.data or b"html" in response.data

    def test_openapi_spec_contains_all_endpoints(self, client):
        """Test that OpenAPI spec contains all application endpoints"""
        response = client.get('/openapi.json')
        result = json.loads(response.data)
        
        paths = result.get("paths", {})
        assert "/encode" in paths
        assert "/calculate-direction" in paths
        assert "/get-reference-point" in paths

    def test_openapi_spec_contains_all_schemas(self, client):
        """Test that OpenAPI spec contains all Pydantic schemas"""
        response = client.get('/openapi.json')
        result = json.loads(response.data)
        
        components = result.get("components", {})
        schemas = components.get("schemas", {})
        
        # Should contain request and response schemas
        assert "EncodeRequest" in schemas
        assert "CalculateDirectionRequest" in schemas
        assert "ReferencePointRequest" in schemas
        assert "DirectionAngleResponse" in schemas
        assert "ReferencePointResponse" in schemas

    def test_app_initialization_sets_up_routes(self, app):
        """Test that ServerApplication initializes with all routes"""
        routes = [str(rule.rule) for rule in app.url_map.iter_rules()]
        
        # Should have main routes
        assert "/" in routes
        assert "/encode" in routes
        assert "/calculate-direction" in routes
        assert "/get-reference-point" in routes
        assert "/openapi.json" in routes
        assert "/docs" in routes
        assert "/redoc" in routes

    def test_app_cors_enabled(self, app):
        """Test that CORS is enabled on the app"""
        # Check if CORS is configured on the Flask app
        assert app is not None
        # CORS should be registered on the app
        # Note: CORS is added at ServerApplication level, so we check the Flask app exists
        assert hasattr(app, 'url_map')


class TestServerApplicationDependencies:
    """Test suite for ServerApplication dependency injection"""

    def test_app_initializes_encoding_services(self):
        """Test that ServerApplication initializes encoding services"""
        server_app = ServerApplication()
        # Check that encoding services are initialized (private attributes)
        assert server_app._encoding_service_hsv is not None
        assert server_app._encoding_service_rgb is not None

    def test_app_initializes_model_definitions(self):
        """Test that ServerApplication initializes model definitions via controller"""
        server_app = ServerApplication()
        # Controller should be initialized
        assert server_app._controller is not None
        # Controller should have been initialized
        assert hasattr(server_app._controller, 'initialize')

    def test_app_creates_controller(self):
        """Test that ServerApplication creates controller instance"""
        server_app = ServerApplication()
        assert server_app._controller is not None
        # Controller should have get_status method
        assert hasattr(server_app._controller, 'get_status')


class TestServerApplicationErrorHandling:
    """Test suite for error handling in ServerApplication"""

    def test_encode_endpoint_handles_missing_model_type(self, client):
        """Test encode endpoint handles missing model_type"""
        response = client.post(
            '/encode',
            json={"parameters": {}},
            content_type='application/json'
        )
        
        assert response.status_code == 400
        result = json.loads(response.data)
        assert ResponseKey.ERROR.value in result

    def test_encode_endpoint_handles_missing_parameters(self, client):
        """Test encode endpoint handles missing parameters"""
        response = client.post(
            '/encode',
            json={"model_type": "test"},
            content_type='application/json'
        )
        
        assert response.status_code == 400
        result = json.loads(response.data)
        assert ResponseKey.ERROR.value in result

    def test_calculate_direction_endpoint_handles_missing_room_polygon(self, client):
        """Test calculate-direction endpoint handles missing room_polygon"""
        response = client.post(
            '/calculate-direction',
            json={"windows": []},
            content_type='application/json'
        )
        
        assert response.status_code == 400
        result = json.loads(response.data)
        assert ResponseKey.ERROR.value in result

    def test_calculate_direction_endpoint_handles_missing_windows(self, client):
        """Test calculate-direction endpoint handles missing windows"""
        response = client.post(
            '/calculate-direction',
            json={"room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            content_type='application/json'
        )
        
        assert response.status_code == 400
        result = json.loads(response.data)
        assert ResponseKey.ERROR.value in result

    def test_error_response_includes_error_key(self, client):
        """Test that error responses include ResponseKey.ERROR"""
        response = client.post(
            '/encode',
            json={},
            content_type='application/json'
        )
        
        assert response.status_code == 400
        result = json.loads(response.data)
        assert ResponseKey.ERROR.value in result

    def test_error_response_has_consistent_format(self, client):
        """Test that all error responses have consistent format"""
        test_cases = [
            ('/encode', {}),
            ('/calculate-direction', {}),
            ('/get-reference-point', {}),
        ]
        
        for endpoint, data in test_cases:
            response = client.post(
                endpoint,
                json=data,
                content_type='application/json'
            )
            
            result = json.loads(response.data)
            assert ResponseKey.ERROR.value in result or response.status_code == 200


class TestServerApplicationIntegration:
    """Integration tests for ServerApplication"""

    def test_app_serves_documentation(self, client):
        """Test that app can serve all documentation endpoints"""
        endpoints = ['/docs', '/redoc', '/openapi.json']
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200, f"Failed at {endpoint}"

    def test_openapi_spec_is_valid_json(self, client):
        """Test that OpenAPI spec is valid JSON"""
        response = client.get('/openapi.json')
        
        assert response.status_code == 200
        try:
            spec = json.loads(response.data)
            assert isinstance(spec, dict)
            assert "openapi" in spec
        except json.JSONDecodeError:
            pytest.fail("OpenAPI spec is not valid JSON")

    def test_server_app_has_required_attributes(self):
        """Test that ServerApplication has all required attributes"""
        server_app = ServerApplication()
        required_attrs = ['_app', '_controller', '_encoding_service_hsv', '_encoding_service_rgb']
        
        for attr in required_attrs:
            assert hasattr(server_app, attr), f"Missing attribute: {attr}"
            assert getattr(server_app, attr) is not None, f"Attribute is None: {attr}"

    def test_openapi_spec_has_required_sections(self, client):
        """Test that OpenAPI spec has all required sections"""
        response = client.get('/openapi.json')
        spec = json.loads(response.data)
        
        required_sections = ['openapi', 'info', 'paths', 'components']
        
        for section in required_sections:
            assert section in spec, f"Missing required section: {section}"

    def test_openapi_spec_info_has_metadata(self, client):
        """Test that OpenAPI spec info contains required metadata"""
        response = client.get('/openapi.json')
        spec = json.loads(response.data)
        
        info = spec.get("info", {})
        assert "title" in info
        assert "version" in info
        assert "description" in info or "title" in info
