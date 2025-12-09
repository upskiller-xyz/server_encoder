"""
Example: Calculate direction_angle for windows

This example demonstrates how to use the /calculate-direction endpoint
to calculate the direction_angle (orientation) for windows based on
their position in a room polygon.
"""

import requests
import json
import math


def calculate_direction_angles():
    """Calculate direction angles for windows in a room."""

    # API endpoint
    url = "http://localhost:3000/calculate-direction"

    # Define the room and windows
    # Room: rectangular shape from (0,2) to (-3,-7)
    # Window 1: on the east wall (x=0)
    # Window 2: on the south wall (y=-7)
    payload = {
        "parameters": {
            "room_polygon": [
                [0, 2],      # Northeast corner
                [0, -7],     # Southeast corner
                [-3, -7],    # Southwest corner
                [-3, 2]      # Northwest corner
            ],
            "windows": {
                "east_window": {
                    "x1": 0.0,
                    "y1": 0.2,
                    "x2": 0.0,
                    "y2": 1.8
                },
                "south_window": {
                    "x1": -0.5,
                    "y1": -7.0,
                    "x2": -2.5,
                    "y2": -7.0
                }
            }
        }
    }

    # Make the request
    print("Calculating direction angles...")
    print(f"Request payload:\n{json.dumps(payload, indent=2)}\n")

    try:
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()

            print("✓ Success!")
            print("\nResults:")
            print("-" * 50)

            for window_id in result["direction_angles"]:
                angle_rad = result["direction_angles"][window_id]
                angle_deg = result["direction_angles_degrees"][window_id]

                print(f"\n{window_id}:")
                print(f"  Direction angle: {angle_rad:.4f} radians")
                print(f"  Direction angle: {angle_deg:.2f}°")

                # Interpret the angle

            return result
        else:
            print(f"✗ Error {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print("✗ Connection failed. Is the server running on port 3000?")
        print("  Start the server with: python src/main.py")
        return None
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return None

def example_with_single_window():
    """Example with a single window."""
    url = "http://localhost:3000/calculate-direction"

    payload = {
        "parameters": {
            "room_polygon": [[0, 2], [0, -7], [-3, -7], [-3, 2]],
            "windows": {
                "my_window": {
                    "x1": 0.0,
                    "y1": 0.2,
                    "x2": 0.0,
                    "y2": 1.8
                }
            }
        }
    }

    print("\n" + "="*50)
    print("Example: Single window")
    print("="*50 + "\n")

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            angle_deg = result["direction_angles_degrees"]["my_window"]
           
            return result
        else:
            print(f"Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Direction Angle Calculation Example")
    print("="*50 + "\n")

    # Example 1: Multiple windows
    calculate_direction_angles()

    # Example 2: Single window
    example_with_single_window()

    print("\n" + "="*50)
    print("Done!")
    print("="*50 + "\n")
