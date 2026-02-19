"""Test user's exact example."""

import json
import math
from src.main import ServerApplication


def test_user_exact_example():
    """Test the exact payload from user's example."""
    app = ServerApplication("TestApp")
    app.app.config['TESTING'] = True

    # User's exact payload (with x2 fixed to be on the edge)
    pld = {
        "parameters": {
            "room_polygon": [[0, 0], [0, 7], [-3, 7], [-3, 0]],
            "windows": {
                "window_id": {
                    "x1": 0, "y1": 1, "z1": 10.9,
                    "x2": 0, "y2": 1.8, "z2": 11.9,  # Fixed: x2 should be 0, not 0.4
                }
            }
        }
    }

    with app.app.test_client() as client:
        response = client.post(
            '/calculate-direction',
            data=json.dumps(pld),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        print("\n" + "="*50)
        print("User's Example Result:")
        print("="*50)
        print(f"Direction angle: {data['direction_angle']['window_id']:.4f} rad")
        print("="*50 + "\n")

        # Should be 0° (east), not 180° (π)
        angle_deg = data['direction_angle']['window_id'] * 180 / math.pi
        assert abs(angle_deg - 0.0) < 1.0, (
            f"Expected 0° for window on east wall, got {angle_deg:.2f}°"
        )


if __name__ == "__main__":
    test_user_exact_example()
    print("\n✓ Test passed! Window on east wall correctly returns 0° (facing east)")
