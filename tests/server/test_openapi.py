"""Tests for OpenAPI specification generator"""
import pytest
from src.server.openapi import OpenAPISpecGenerator


class TestOpenAPISpecGenerator:
    """Test suite for OpenAPISpecGenerator"""

    def test_generate_spec_basic(self):
        """Test basic OpenAPI spec generation"""
        spec = OpenAPISpecGenerator.generate_spec()
        
        assert spec["openapi"] == "3.0.0"
        assert "info" in spec
        assert "paths" in spec
        assert "components" in spec

    def test_generate_spec_info_section(self):
        """Test info section of generated spec"""
        spec = OpenAPISpecGenerator.generate_spec(
            title="Test API",
            description="Test Description",
            version="2.0.0"
        )
        
        assert spec["info"]["title"] == "Test API"
        assert spec["info"]["description"] == "Test Description"
        assert spec["info"]["version"] == "2.0.0"

    def test_generate_spec_servers(self):
        """Test servers section of generated spec"""
        spec = OpenAPISpecGenerator.generate_spec(base_url="/api/v1")
        
        assert "servers" in spec
        assert len(spec["servers"]) > 0
        assert spec["servers"][0]["url"] == "/api/v1"

    def test_generate_spec_all_endpoints(self):
        """Test that all endpoints are included"""
        spec = OpenAPISpecGenerator.generate_spec()
        paths = spec["paths"]
        
        assert "/encode" in paths
        assert "/calculate-direction" in paths
        assert "/get-reference-point" in paths

    def test_generate_spec_encode_endpoint(self):
        """Test /encode endpoint specification"""
        spec = OpenAPISpecGenerator.generate_spec()
        encode = spec["paths"]["/encode"]["post"]
        
        assert encode["summary"] == "Encode room parameters"
        assert "requestBody" in encode
        assert "responses" in encode
        assert "200" in encode["responses"]
        assert "400" in encode["responses"]
        assert "500" in encode["responses"]

    def test_generate_spec_calculate_direction_endpoint(self):
        """Test /calculate-direction endpoint specification"""
        spec = OpenAPISpecGenerator.generate_spec()
        calc = spec["paths"]["/calculate-direction"]["post"]
        
        assert calc["summary"] == "Calculate direction angles"
        assert "POST" in spec["paths"]["/calculate-direction"] or "post" in spec["paths"]["/calculate-direction"]
        assert "requestBody" in calc

    def test_generate_spec_get_reference_point_endpoint(self):
        """Test /get-reference-point endpoint specification"""
        spec = OpenAPISpecGenerator.generate_spec()
        ref = spec["paths"]["/get-reference-point"]["post"]
        
        assert ref["summary"] == "Calculate reference points"
        assert "requestBody" in ref

    def test_generate_spec_all_schemas(self):
        """Test that all schemas are included"""
        spec = OpenAPISpecGenerator.generate_spec()
        schemas = spec["components"]["schemas"]
        
        assert "EncodeRequest" in schemas
        assert "CalculateDirectionRequest" in schemas
        assert "ReferencePointRequest" in schemas
        assert "DirectionAngleResponse" in schemas
        assert "ReferencePointResponse" in schemas
        assert "ErrorResponse" in schemas

    def test_generate_spec_encode_request_schema(self):
        """Test EncodeRequest schema"""
        spec = OpenAPISpecGenerator.generate_spec()
        schema = spec["components"]["schemas"]["EncodeRequest"]
        
        assert "properties" in schema
        assert "model_type" in schema["properties"]
        assert "parameters" in schema["properties"]
        assert "encoding_scheme" in schema["properties"]

    def test_generate_spec_error_responses(self):
        """Test that error responses are documented"""
        spec = OpenAPISpecGenerator.generate_spec()
        
        for endpoint_path in spec["paths"].values():
            for method in endpoint_path.values():
                assert "400" in method["responses"]
                assert "500" in method["responses"]

    def test_generate_spec_request_bodies_have_schemas(self):
        """Test that request bodies reference schemas"""
        spec = OpenAPISpecGenerator.generate_spec()
        
        encode = spec["paths"]["/encode"]["post"]
        assert "$ref" in str(encode["requestBody"])
        
        calc = spec["paths"]["/calculate-direction"]["post"]
        assert "$ref" in str(calc["requestBody"])

    def test_generate_spec_response_schemas(self):
        """Test that responses reference schemas"""
        spec = OpenAPISpecGenerator.generate_spec()
        
        calc = spec["paths"]["/calculate-direction"]["post"]
        response_200 = calc["responses"]["200"]
        assert "$ref" in str(response_200)

    def test_generate_spec_default_values(self):
        """Test spec generation with default values"""
        spec = OpenAPISpecGenerator.generate_spec()
        
        assert spec["info"]["title"] == "Server Encoder API"
        assert spec["info"]["version"] == "1.0.0"
        assert "Room encoding service" in spec["info"]["description"]

    def test_generate_spec_is_valid_json_serializable(self):
        """Test that spec can be serialized to JSON"""
        import json
        spec = OpenAPISpecGenerator.generate_spec()
        
        # Should not raise
        json_str = json.dumps(spec)
        assert json_str is not None
        assert len(json_str) > 0

    def test_generate_spec_tags_organized(self):
        """Test that endpoints are organized by tags"""
        spec = OpenAPISpecGenerator.generate_spec()
        
        # Encoding endpoint should have Encoding tag
        encode = spec["paths"]["/encode"]["post"]
        assert "tags" in encode
        assert "Encoding" in encode["tags"]
        
        # Calculation endpoints should have Calculations tag
        calc = spec["paths"]["/calculate-direction"]["post"]
        assert "tags" in calc
        assert "Calculations" in calc["tags"]

    def test_generate_spec_openapi_version(self):
        """Test that generated spec is OpenAPI 3.0.0"""
        spec = OpenAPISpecGenerator.generate_spec()
        assert spec["openapi"] == "3.0.0"
