# Direction Angle Calculation Endpoint

## Overview

The `/calculate-direction` endpoint calculates the `direction_angle` (window orientation) for one or more windows based on their position within a room polygon.

## Endpoint Details

- **URL**: `/calculate-direction`
- **Method**: `POST`
- **Content-Type**: `application/json`

## Request Format

```json
{
  "parameters": {
    "room_polygon": [[x1, y1], [x2, y2], [x3, y3], ...],
    "windows": {
      "window_id_1": {
        "x1": <number>,
        "y1": <number>,
        "x2": <number>,
        "y2": <number>
      },
      "window_id_2": {
        ...
      }
    }
  }
}
```

### Parameters

- **room_polygon**: Array of [x, y] coordinate pairs defining the room boundary
- **windows**: Dictionary mapping window IDs to window parameters
  - **x1, y1**: First corner coordinates of the window
  - **x2, y2**: Second corner coordinates of the window
  - **z1, z2**: (Optional) Height coordinates (not required for direction calculation)

**Important**: The window must lie **on** a polygon edge. Both corners (x1,y1) and (x2,y2) must be on the same edge. For example:
- ✓ Correct: Window on east wall at x=0: `x1=0, y1=1, x2=0, y2=2`
- ✗ Wrong: `x1=0, y1=1, x2=0.4, y2=2` (window crosses from edge into room)

## Response Format

### Success (200 OK)

```json
{
  "direction_angles": {
    "window_id_1": <float>,  // radians
    "window_id_2": <float>
  },
  "direction_angles_degrees": {
    "window_id_1": <float>,  // degrees
    "window_id_2": <float>
  }
}
```

### Error (400 Bad Request)

```json
{
  "error": "<error message>"
}
```

## Example Usage

### Python

```python
import requests
import json

url = "http://localhost:3000/calculate-direction"

payload = {
    "parameters": {
        "room_polygon": [
            [0, 2],
            [0, -7],
            [-3, -7],
            [-3, 2]
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

response = requests.post(url, json=payload)
result = response.json()

print(f"East window: {result['direction_angles_degrees']['east_window']:.2f}°")
print(f"South window: {result['direction_angles_degrees']['south_window']:.2f}°")
```

### cURL

```bash
curl -X POST http://localhost:3000/calculate-direction \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Understanding Direction Angles

The direction angle represents the orientation the window is facing:
- **0° / 0 rad**: Facing East (positive X direction)
- **90° / π/2 rad**: Facing North (positive Y direction)
- **180° / π rad**: Facing West (negative X direction)
- **270° / 3π/2 rad**: Facing South (negative Y direction)

The direction angle is calculated as the perpendicular to the polygon edge containing the window, pointing away from the room interior.

## Error Handling

The endpoint validates:
1. Required parameters are present (`room_polygon`, `windows`)
2. Each window has required coordinates (`x1`, `y1`, `x2`, `y2`)
3. Windows are positioned on the polygon edges
4. Room polygon is valid

Common errors:
- **"Missing required parameter: 'room_polygon'"**: Include room_polygon in parameters
- **"Missing required parameter: 'windows'"**: Include at least one window
- **"Window 'X' missing required coordinates"**: Ensure all windows have x1, y1, x2, y2
- **"Window at (...) does not lie on any polygon edge"**: Window must be positioned on room boundary

## Integration with Encoding Endpoint

The calculated `direction_angle` can be used directly in the `/encode` endpoint for DA (Daylight Autonomy) models:

```python
# Step 1: Calculate direction angle
direction_result = requests.post(
    "http://localhost:3000/calculate-direction",
    json={"parameters": {"room_polygon": [...], "windows": {...}}}
)
direction_angle = direction_result.json()["direction_angles"]["window1"]

# Step 2: Use in encoding
encoding_result = requests.post(
    "http://localhost:3000/encode",
    json={
        "model_type": "da_default",
        "parameters": {
            "room_polygon": [...],
            "height_roof_over_floor": 2.7,
            "floor_height_above_terrain": 1.0,
            "windows": {
                "window1": {
                    "x1": 0, "y1": 0.2, "z1": 1.9,
                    "x2": 0, "y2": 1.8, "z2": 3.4,
                    "window_frame_ratio": 0.2,
                    "window_orientation": direction_angle,  # Use calculated angle
                    "horizon": 0,
                    "zenith": 0
                }
            }
        }
    }
)
```

## Examples

See [example/calculate_direction_angle_example.py](example/calculate_direction_angle_example.py) for a complete working example.

## Testing

Run tests with:
```bash
pytest tests/test_direction_angle_endpoint.py -v
```
