<a name="readme-top"></a>

<!-- Badges commented out - uncomment if needed
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/upskiller-xyz/server_encoder">
    <img src="https://github.com/upskiller-xyz/DaylightFactor/blob/main/docs/images/logo_upskiller.png" alt="Logo" height="100" >
  </a>

  <h3 align="center">Room Encoding Server</h3>

  <p align="center">
    Encode room geometry and parameters into images for daylight prediction models
    <br />
    <a href="docs/api_reference.md">API Reference</a>
    ·
    <a href="docs/request_schema.md">Request Schema</a>
    <br />
    <a href="https://github.com/upskiller-xyz/server_encoder/issues">Report Bug</a>
    ·
    <a href="https://github.com/upskiller-xyz/server_encoder/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a>
        <li><a href="#api-endpoints">API Endpoints</a></li>
        <li><a href="#deployment">Deployment</a></li>
    </li>
    <li><a href="#design">Design</a></li>
    <li><a href="#testing">Testing</a></li>
    <li><a href="#documentation">Documentation</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contribution">Contribution</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

The **Room Encoding Server** transforms 3D room geometry and physical parameters into 2D encoded images for use with Daylight Factor (DF) and Daylight Autonomy (DA) prediction models.

**Key Features:**
- Encode room geometry, materials, and environmental context into RGBA images
- Support for single and multi-window rooms with complex polygons
- Automatic facade rotation and direction angle calculation from room geometry
- Multiple model types: DF/DA with default or custom materials
- RESTful API with comprehensive validation and error handling
- Object-oriented design following SOLID principles and design patterns
- Structured logging with proper error reporting

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [Python 3.10+](https://www.python.org/)
* [Flask](https://flask.palletsprojects.com/) - Web framework
* [OpenCV](https://opencv.org/) - Image processing
* [NumPy](https://numpy.org/) - Numerical computing
* [Shapely](https://shapely.readthedocs.io/) - Geometric operations

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Python 3.10 or higher
* Poetry (recommended) or pip

### Installation

#### Using Poetry (recommended)

1. Clone the repository
   ```sh
   git clone https://github.com/upskiller-xyz/server_encoder.git
   cd server_encoder
   ```

2. Install dependencies
   ```sh
   poetry install
   ```

3. Activate virtual environment
   ```sh
   poetry shell
   ```

4. Run the server
   ```sh
   python -m src.main
   ```

#### Using pip

1. Clone the repository
   ```sh
   git clone https://github.com/upskiller-xyz/server_encoder.git
   cd server_encoder
   ```

2. Create virtual environment
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```sh
   pip install flask flask-cors numpy opencv-python-headless shapely
   ```

4. Run the server
   ```sh
   python -m src.main
   ```

#### Environment Variables

```sh
export PORT=8081  # Default: 8081
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### Quick Start

```python
import requests
import json

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
                "obstruction_angle_horizon": 15.0,
                "obstruction_angle_zenith": 10.0
            }
        }
    }
}

response = requests.post(url, json=payload)

# Save the encoded image
with open("encoded_room.png", "wb") as f:
    f.write(response.content)
```

### API Endpoints

#### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "running",
  "service": "encoding_service",
  "timestamp": "2025-10-26T12:34:56"
}
```

#### `POST /encode`
Encode room geometry and parameters into image(s).

**Request:** JSON with `model_type` and `parameters`

**Response:**
- Single window: PNG image
- Multiple windows: ZIP archive with one PNG per window

See [API Reference](docs/api_reference.md) for complete documentation.

### Model Types

| Model Type | Description |
|------------|-------------|
| `df_default` | Daylight Factor with default materials (reflectance = 0.8) |
| `da_default` | Daylight Autonomy with default materials |
| `df_custom` | Daylight Factor with custom material reflectances |
| `da_custom` | Daylight Autonomy with custom material reflectances |

### Multi-Window Support

For rooms with windows on different facades:

```python
payload = {
    "model_type": "df_custom",
    "parameters": {
        "height_roof_over_floor": 2.7,
        "room_polygon": [[0, 0], [5, 0], [5, 4], [0, 4]],
        "windows": {
            "south_window": {
                "x1": -0.6, "y1": 0.0, "z1": 0.9,
                "x2": 0.6, "y2": 0.0, "z2": 2.4,
                # ... other parameters
            },
            "west_window": {
                "x1": 0.0, "y1": -0.5, "z1": 1.0,
                "x2": 0.0, "y2": 0.5, "z2": 2.0,
                # ... other parameters
            }
        }
    }
}

response = requests.post(url, json=payload)
# Returns ZIP file with south_window.png and west_window.png
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Deployment

#### Local Development

```sh
export PORT=8081
python -m src.main
```

Server runs on `http://localhost:8081` with debug mode enabled.

#### Docker

```sh
docker build -t room-encoder .
docker run -p 8081:8081 room-encoder
```

#### Production

Use a production WSGI server:

```sh
gunicorn -w 4 -b 0.0.0.0:8081 src.main:app
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DESIGN -->
## Design

### Architecture

The server follows strict **Object-Oriented Programming** principles and design patterns:

**Core Components:**
```
ServerApplication
├── EncodingService (handles encoding logic)
│   ├── RoomImageBuilder (constructs images)
│   ├── RoomImageDirector (orchestrates building)
│   └── RegionEncoders (encode specific regions)
│       ├── BackgroundRegionEncoder
│       ├── RoomRegionEncoder
│       ├── WindowRegionEncoder
│       └── ObstructionBarRegionEncoder
├── StructuredLogger (logging system)
└── ServerController (request handling)
```

**Design Patterns Used:**
- **Builder Pattern**: Image construction
- **Director Pattern**: Orchestration of building process
- **Factory Pattern**: Encoder and service creation
- **Strategy Pattern**: Channel mappings, validation rules
- **Singleton Pattern**: Service instances
- **Adapter Pattern**: Parameter transformation
- **Enumerator Pattern**: Constants and magic strings

**Principles:**
- Single Responsibility Principle (SRP)
- Dependency Injection
- Separation of Concerns
- Type Safety with Type Hints

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- TESTING -->
## Testing

Run the test suite:

```sh
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific test file
poetry run pytest tests/test_window.py -v
```

**Test Coverage:**
- 101 unit tests across 5 test modules
- Coverage for all region encoders
- Validation and integration tests
- Image scaling and positioning tests

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DOCUMENTATION -->
## Documentation

Comprehensive documentation available in the `docs/` directory:

- **[API Reference](docs/api_reference.md)** - Endpoint documentation and examples
- **[Request Schema](docs/request_schema.md)** - Complete parameter reference and validation

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/upskiller-xyz/server_encoder/issues) for a full list of proposed features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTION -->
## Contribution

Contributions are welcome! Please follow these guidelines:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Follow the development guidelines in [CLAUDE.md](CLAUDE.md)
4. Write tests for new functionality
5. Ensure all tests pass: `poetry run pytest`
6. Commit your Changes using [conventional commits](https://www.conventionalcommits.org/)
7. Push to the Branch (`git push origin feature/AmazingFeature`)
8. Open a Pull Request

**Development Guidelines:**
- Follow Object-Oriented Programming principles
- Use design patterns appropriately
- Add type hints to all functions
- Write self-documenting code
- Update documentation for API changes
- Maintain test coverage above 90%

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

See [License](./docs/LICENSE) for details - [GPL-3.0](https://choosealicense.com/licenses/gpl-3.0/).

**Summary:**
Strong copyleft. You can use, distribute and modify this code in academic and commercial contexts, but you must keep the code open-source under GPL-3.0 and provide appropriate attribution.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Stanislava Fedorova - [stasya.fedorova@gmail.com](mailto:stasya.fedorova@gmail.com)

Project Link: [https://github.com/upskiller-xyz/server_encoder](https://github.com/upskiller-xyz/server_encoder)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [README template](https://github.com/othneildrew/Best-README-Template)
* [Flask](https://flask.palletsprojects.com/) - Web framework
* [OpenCV](https://opencv.org/) - Image processing
* [NumPy](https://numpy.org/) - Numerical computing
* [Shapely](https://shapely.readthedocs.io/) - Geometric operations

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- Commented out - uncomment if needed
[contributors-shield]: https://img.shields.io/github/contributors/upskiller-xyz/server_encoder.svg?style=for-the-badge
[contributors-url]: https://github.com/upskiller-xyz/server_encoder/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/upskiller-xyz/server_encoder.svg?style=for-the-badge
[forks-url]: https://github.com/upskiller-xyz/server_encoder/network/members
[stars-shield]: https://img.shields.io/github/stars/upskiller-xyz/server_encoder.svg?style=for-the-badge
[stars-url]: https://github.com/upskiller-xyz/server_encoder/stargazers
[issues-shield]: https://img.shields.io/github/issues/upskiller-xyz/server_encoder.svg?style=for-the-badge
[issues-url]: https://github.com/upskiller-xyz/server_encoder/issues
[license-shield]: https://img.shields.io/github/license/upskiller-xyz/server_encoder.svg?style=for-the-badge
[license-url]: https://github.com/upskiller-xyz/server_encoder/blob/master/docs/LICENSE.txt
-->
