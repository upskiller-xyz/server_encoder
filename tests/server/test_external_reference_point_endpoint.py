"""
Tests for external reference point calculation endpoint

Tests the /get-external-reference-point endpoint that calculates
the external face reference point for windows.
"""
import pytest
from src.server.application import ServerApplication


class TestExternalReferencePointEndpoint:
    """Test external reference point calculation endpoint"""

    @pytest.fixture
    def app(self):
        """Create test Flask application"""
        server_app = ServerApplication()
        app = server_app.app
        app.config['TESTING'] = True
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()

    def test_calculate_external_reference_point_single_window(self, client):
        """Test external reference point calculation for single window"""
        # Simple rectangular room: 4m x 3m
        # Window on south wall (y=0, internal edge) extending to y=0.4
        # Window faces south (outward), so external point is at y=-0.4
        payload = {
            "room_polygon": [
                [-2.0, 0.0],
                [2.0, 0.0],
                [2.0, 3.0],
                [-2.0, 3.0]
            ],
            "windows": {
                "window1": {
                    "x1": -0.6, "y1": 0.0, "z1": 1.0,
                    "x2": 0.6, "y2": 0.4, "z2": 2.5
                }
            }
        }

        response = client.post(
            '/get-external-reference-point',
            json=payload,
            content_type='application/json'
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "external_reference_point" in data
        assert "window1" in data["external_reference_point"]

        # External reference point should be on the outer edge
        # Window faces south (outward from room), so external point is at negative y
        ext_ref = data["external_reference_point"]["window1"]
        assert "x" in ext_ref
        assert "y" in ext_ref
        assert "z" in ext_ref

        # X should be center of window (0.0)
        assert abs(ext_ref["x"] - 0.0) < 0.01

        # Y should be at the external edge (-0.4), outside the room
        # The window faces south (negative y direction)
        assert abs(ext_ref["y"] - (-0.4)) < 0.01

        # Z should be center of window height (1.75)
        assert abs(ext_ref["z"] - 1.75) < 0.01

    def test_calculate_external_reference_point_multiple_windows(self, client):
        """Test external reference point calculation for multiple windows"""
        payload = {
            "room_polygon": [
                [-2.0, 0.0],
                [2.0, 0.0],
                [2.0, 3.0],
                [-2.0, 3.0]
            ],
            "windows": {
                "window1": {
                    "x1": -0.6, "y1": 0.0, "z1": 1.0,
                    "x2": 0.6, "y2": 0.3, "z2": 2.5
                },
                "window2": {
                    "x1": 1.0, "y1": 0.0, "z1": 0.5,
                    "x2": 1.8, "y2": 0.3, "z2": 2.0
                }
            }
        }

        response = client.post(
            '/get-external-reference-point',
            json=payload,
            content_type='application/json'
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "external_reference_point" in data
        assert "window1" in data["external_reference_point"]
        assert "window2" in data["external_reference_point"]

        # Verify both windows have external reference points
        ext_ref1 = data["external_reference_point"]["window1"]
        ext_ref2 = data["external_reference_point"]["window2"]

        # Both should have x, y, z coordinates
        assert all(k in ext_ref1 for k in ["x", "y", "z"])
        assert all(k in ext_ref2 for k in ["x", "y", "z"])

    def test_missing_parameters(self, client):
        """Test error handling for missing parameters"""
        # Missing windows parameter
        payload = {
            "room_polygon": [
                [-2.0, 0.0],
                [2.0, 0.0],
                [2.0, 3.0],
                [-2.0, 3.0]
            ]
        }

        response = client.post(
            '/get-external-reference-point',
            json=payload,
            content_type='application/json'
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "windows" in data["error"].lower()

    def test_missing_room_polygon(self, client):
        """Test error handling for missing room polygon"""
        payload = {
            "windows": {
                "window1": {
                    "x1": -0.6, "y1": 0.0, "z1": 1.0,
                    "x2": 0.6, "y2": 0.4, "z2": 2.5
                }
            }
        }

        response = client.post(
            '/get-external-reference-point',
            json=payload,
            content_type='application/json'
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "room_polygon" in data["error"].lower()

    def test_missing_window_coordinates(self, client):
        """Test error handling for missing window coordinates"""
        payload = {
            "room_polygon": [
                [-2.0, 0.0],
                [2.0, 0.0],
                [2.0, 3.0],
                [-2.0, 3.0]
            ],
            "windows": {
                "window1": {
                    "x1": -0.6, "y1": 0.0, "z1": 1.0
                    # Missing x2, y2, z2
                }
            }
        }

        response = client.post(
            '/get-external-reference-point',
            json=payload,
            content_type='application/json'
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "coordinates" in data["error"].lower()

    def test_endpoint_requires_post(self, client):
        """Test that GET requests are not allowed"""
        response = client.get('/get-external-reference-point')
        assert response.status_code == 405  # Method Not Allowed
