# Room Encoding Server

Encode room geometry and parameters into images for daylight prediction models.

## Overview

The **Room Encoding Server** transforms 3D room geometry and physical parameters into 2D encoded images for use with Daylight Factor (DF) and Daylight Autonomy (DA) prediction models.

### Key Features

- Encode room geometry, materials, and environmental context into RGBA images
- Support for single and multi-window rooms
- Automatic facade rotation for windows on different orientations
- Multiple model types: DF/DA with default or custom materials
- RESTful API with comprehensive validation
- Object-oriented design following SOLID principles

## Quick Start

```python
import requests

url = "http://localhost:8081/encode"

payload = {
    "model_type": "df_default",
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

# Save the encoded image
with open("encoded_room.png", "wb") as f:
    f.write(response.content)
```

## Installation

### Using Poetry (recommended)

```sh
git clone https://github.com/upskiller-xyz/server_encoder.git
cd server_encoder
poetry install
poetry shell
python -m src.main
```

### Using pip

```sh
git clone https://github.com/upskiller-xyz/server_encoder.git
cd server_encoder
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install flask flask-cors numpy opencv-python-headless shapely
python -m src.main
```

## Documentation

- **[API Reference](api_reference.md)** - Endpoint documentation and examples
- **[Request Schema](request_schema.md)** - Complete parameter reference and validation

## Model Types

| Model Type | Description |
|------------|-------------|
| `df_default` | Daylight Factor with default materials (reflectance = 0.8) |
| `da_default` | Daylight Autonomy with default materials |
| `df_custom` | Daylight Factor with custom material reflectances |
| `da_custom` | Daylight Autonomy with custom material reflectances |

## License

GPL-3.0 - See [LICENSE](https://github.com/upskiller-xyz/server_encoder/blob/master/docs/LICENSE) for details.

## Contact

Stanislava Fedorova - [stasya.fedorova@gmail.com](mailto:stasya.fedorova@gmail.com)

Project Link: [https://github.com/upskiller-xyz/server_encoder](https://github.com/upskiller-xyz/server_encoder)
