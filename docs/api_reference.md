# API Reference

## Base URL
```
http://localhost:8081
```

## Endpoints

### GET `/`
Health check endpoint.

**Response:**
```json
{
  "status": "running",
  "service": "encoding_service",
  "timestamp": "2025-10-26T12:34:56"
}
```

### POST `/encode`
Encode room geometry and parameters into daylight prediction images.

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "model_type": "df_default" | "da_default" | "df_custom" | "da_custom",
  "encoding_scheme": "v1" | "v2" | "v3" | "v4" | "v5" (optional, default: "v2"),
  "parameters": {
    // See request_schema.md for complete structure
  }
}
```

**Response:**

**Single Window (1 window in request):**
- **Content-Type:** `image/png`
- **Body:** PNG image bytes
- **Filename:** `encoded_room.png`

**Multiple Windows (2+ windows in request):**
- **Content-Type:** `application/zip`
- **Body:** ZIP archive containing one PNG per window
- **Filename:** `encoded_room_windows.zip`
- **Archive contents:** `{window_id}.png` for each window

**Status Codes:**
- `200 OK` - Success
- `400 Bad Request` - Invalid parameters or missing required fields
- `500 Internal Server Error` - Encoding failure

**Error Response:**
```json
{
  "error": "Error message description"
}
```

## Model Types

| Value | Description |
|-------|-------------|
| `df_default` | Daylight Factor with fixed default materials (reflectance = 0.8) |
| `da_default` | Daylight Autonomy with fixed default materials |
| `df_custom` | Daylight Factor with customizable material reflectances |
| `da_custom` | Daylight Autonomy with customizable material reflectances |

## Encoding Schemes

| Value | Output | Description |
|-------|--------|-------------|
| `v1` | `uint8` RGBA | Legacy RGB-style channel mapping; obstruction bar at right edge |
| `v2` | `uint8` RGBA | **Default.** HSV-style channel mapping; obstruction bar at right edge |
| `v3` | `uint8` RGBA | HSV-style, no obstruction bar |
| `v4` | `uint8` RGBA | HSV-style, compact bounding-box obstruction |
| `v5` | `float32` single-channel | Geometry-only mask (background=0, room=1, window=0.6); no parameter encoding |

**Note:** V1–V4 use RGBA channels (128×128×4 uint8). V5 returns a single-channel float32 array (128×128×1). See [encoding_schemes.md](encoding_schemes.md) for full details.

### Python Example (V5 geometry-only)
```python
import requests, numpy as np, io

payload = {
    "model_type": "df_default",
    "encoding_scheme": "v5",
    "parameters": {
        "room_polygon": [[0, 0], [5, 0], [5, 4], [0, 4]],
        "windows": {
            "main_window": {
                "x1": -0.6, "y1": 0.0, "z1": 0.9,
                "x2":  0.6, "y2": 0.0, "z2": 2.4
            }
        }
    }
}

response = requests.post("http://localhost:8081/encode", json=payload)
npz = np.load(io.BytesIO(response.content))
image = npz["main_window_image"]  # (128, 128, 1) float32
print(image.shape, image.dtype)   # (128, 128, 1) float32
```

## Examples

### Single Window Request
```bash
curl -X POST http://localhost:8081/encode \
  -H "Content-Type: application/json" \
  -d @single_window.json \
  -o encoded_room.png
```

### Multiple Windows Request
```bash
curl -X POST http://localhost:8081/encode \
  -H "Content-Type: application/json" \
  -d @multi_window.json \
  -o encoded_room_windows.zip
```

### Python Example (V2 encoding - default)
```python
import requests
import json

url = "http://localhost:8081/encode"
payload = {
    "model_type": "df_default",
    "encoding_scheme": "v2",  # Optional - v2 is default
    "parameters": {
        "height_roof_over_floor": 2.7,
        "floor_height_above_terrain": 3.0,
        "room_polygon": [[0, 0], [5, 0], [5, 4], [0, 4]],
        "windows": {
            "main_window": {
                "window_sill_height": 0.9,
                "window_frame_ratio": 0.15,
                "window_height": 1.5,
                "x1": -0.6, "y1": 0.0, "z1": 0.9,
                "x2": 0.6, "y2": 0.0, "z2": 2.4,
                "horizon": 15.0,
                "zenith": 10.0
            }
        }
    }
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    with open("encoded_room.png", "wb") as f:
        f.write(response.content)
else:
    print(f"Error: {response.json()}")
```

### Python Example (V1 encoding - legacy RGB)
```python
# Same as above, but specify "encoding_scheme": "v1" for legacy RGB encoding
payload = {
    "model_type": "df_default",
    "encoding_scheme": "v1",  # Legacy RGB channel mapping
    "parameters": {
        # ... same parameters as above
    }
}
```
