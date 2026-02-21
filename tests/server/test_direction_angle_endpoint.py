"""Test direction_angle calculation endpoint."""

import pytest
import json
import math
from src.main import ServerApplication


class TestDirectionAngleEndpoint:
    """Test the /calculate-direction endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = ServerApplication("TestApp")
        app.app.config['TESTING'] = True
        with app.app.test_client() as client:
            yield client

    def test_calculate_direction_single_window(self, client):
        """Test calculating direction_angle for a single window."""
        # Square room with window on south wall (y=0)
        # Window facing south (negative y direction)
        payload = {
            "parameters": {
                "room_polygon": [[0, 2], [0, -7], [-3, -7], [-3, 2]],
                "windows": {
                    "test_window": {
                        "x1": 0.0,
                        "y1": 0.2,
                        "x2": 0.0,
                        "y2": 1.8
                    }
                }
            }
        }

        response = client.post(
            '/calculate-direction',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        assert "direction_angle" in data
        assert "test_window" in data["direction_angle"]

        # Verify result is a reasonable angle (0 to 2*pi radians)
        angle_rad = data["direction_angle"]["test_window"]
        assert 0 <= angle_rad <= 2 * math.pi

    def test_calculate_direction_multiple_windows(self, client):
        """Test calculating direction_angle for multiple windows."""
        payload = {
            "parameters": {
                "room_polygon": [[0, 2], [0, -7], [-3, -7], [-3, 2]],
                "windows": {
                    "window1": {
                        "x1": 0.0,
                        "y1": 0.2,
                        "x2": 0.0,
                        "y2": 1.8
                    },
                    "window2": {
                        "x1": -0.5,
                        "y1": -7.0,
                        "x2": -2.5,
                        "y2": -7.0
                    }
                }
            }
        }

        response = client.post(
            '/calculate-direction',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        assert "window1" in data["direction_angle"]
        assert "window2" in data["direction_angle"]

    def test_missing_parameters(self, client):
        """Test error handling for missing parameters."""
        payload = {}

        response = client.post(
            '/calculate-direction',
            data=json.dumps(payload),
            content_type='application/json'
        )

        # Flask BadRequest returns HTML error page, not JSON
        assert response.status_code == 400

    def test_missing_room_polygon(self, client):
        """Test error handling for missing room_polygon."""
        payload = {
            "parameters": {
                "windows": {
                    "window1": {
                        "x1": 0.0,
                        "y1": 0.2,
                        "x2": 0.0,
                        "y2": 1.8
                    }
                }
            }
        }

        response = client.post(
            '/calculate-direction',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
        assert "room_polygon" in data["error"].lower()

    def test_missing_windows(self, client):
        """Test error handling for missing windows."""
        payload = {
            "parameters": {
                "room_polygon": [[0, 2], [0, -7], [-3, -7], [-3, 2]]
            }
        }

        response = client.post(
            '/calculate-direction',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
        assert "windows" in data["error"].lower()

    def test_missing_window_coordinates(self, client):
        """Test error handling for missing window coordinates."""
        payload = {
            "parameters": {
                "room_polygon": [[0, 2], [0, -7], [-3, -7], [-3, 2]],
                "windows": {
                    "window1": {
                        "x1": 0.0,
                        "y1": 0.2
                        # Missing x2, y2
                    }
                }
            }
        }

        response = client.post(
            '/calculate-direction',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_window_not_on_polygon_edge(self, client):
        """Test direction calculation for window not exactly on polygon edge.

        The endpoint projects the window to the nearest edge and calculates direction,
        rather than rejecting it. Validation happens during encoding, not here.
        """
        payload = {
            "parameters": {
                "room_polygon": [[0, 2], [0, -7], [-3, -7], [-3, 2]],
                "windows": {
                    "window1": {
                        # Window in the middle of the room, not on edge
                        # Will be projected to nearest edge for direction calculation
                        "x1": -1.5,
                        "y1": -2.0,
                        "x2": -1.5,
                        "y2": -4.0
                    }
                }
            }
        }

        response = client.post(
            '/calculate-direction',
            data=json.dumps(payload),
            content_type='application/json'
        )

        # Should succeed - endpoint calculates direction by projecting to nearest edge
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "direction_angle" in data
        assert "window1" in data["direction_angle"]
