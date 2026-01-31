# Room Mask Feature

## Overview

The `/encode` endpoint returns **NumPy arrays** in `.npz` format:
- **Image**: (128, 128, 4) RGBA encoded room image
- **Mask**: (128, 128) binary mask (1=room, 0=other areas)

## API Endpoint

**Endpoint:** `POST /encode`

Returns `.npz` file with NumPy arrays (no PNG decoding needed).

### Response Format

**Single or Multiple Windows:** NPZ file containing:
- `{window_id}_image`: (128, 128, 4) uint8 RGBA array
- `{window_id}_mask`: (128, 128) uint8 binary array (1=room, 0=other)

**Note:** Keys use window IDs from your `windows` dict (e.g., `"window1_image"`, `"window1_mask"`).

## Usage Example

```python
import requests
import numpy as np
import io
import matplotlib.pyplot as plt

# Send encoding request
response = requests.post('http://localhost:8082/encode', json={
    "model_type": "df_default",
    "encoding_scheme": "hsv",
    "parameters": {
        "height_roof_over_floor": 3.0,
        "floor_height_above_terrain": 2.0,
        "room_polygon": [
            [-2.0, 0.0],
            [2.0, 0.0],
            [2.0, 5.0],
            [-2.0, 5.0]
        ],
        "windows": {
            "window1": {
                "window_sill_height": 0.9,
                "window_frame_ratio": 0.8,
                "x1": -0.6, "y1": 0.0, "z1": 2.1,
                "x2": 0.6, "y2": 0.0, "z2": 3.4,
                "horizon": 45.0,
                "zenith": 30.0
            }
        }
    }
})

# Load NPZ
npz_data = np.load(io.BytesIO(response.content))

# Access arrays (use your window ID)
image = npz_data['window1_image']  # (128, 128, 4) RGBA
mask = npz_data['window1_mask']    # (128, 128) binary

# Display
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Encoded Image")
plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title("Room Mask")
plt.show()
```

## Key Changes

- **Simplified API**: Single `/encode` endpoint returns NPZ format
- **Room mask**: Binary mask (1=room, 0=other) included automatically
- **Direction angle**: Now correctly points outward from building
- **Rotation**: Automatic alignment to east-facing (0°) using negative rotation
- **No decoding needed**: Direct NumPy array access
