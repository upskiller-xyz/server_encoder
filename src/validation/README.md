# Validation System

A comprehensive validation system following OOP principles, SRP, and design patterns.

## Architecture

### Design Patterns Used

1. **Strategy Pattern**: `ValidatorManager` routes validation to appropriate validators based on `RequestType` enum
2. **Single Responsibility Principle (SRP)**: Each validator validates exactly one thing
3. **Inheritance**: All validators inherit from `BaseValidator` abstract base class
4. **Enumerator Pattern**: All types and error types use enums (no magic strings)

### Directory Structure

```
src/validation/
├── __init__.py                    # Public API
├── base.py                        # Base classes (BaseValidator, ValidationResult, ValidationError)
├── enums.py                       # Enums (RequestType, ValidationErrorType, ValidationType)
├── validator_manager.py           # ValidatorManager (Strategy Pattern orchestrator)
├── parameter_validators/          # One validator per parameter type
│   ├── __init__.py
│   ├── angle_validator.py         # Validates angles (radians)
│   ├── encoding_scheme_validator.py # Validates encoding schemes
│   ├── height_validator.py        # Validates heights (positive numbers)
│   ├── model_type_validator.py    # Validates model types
│   ├── polygon_validator.py       # Validates polygon structures
│   ├── ratio_validator.py         # Validates ratios (0-1)
│   ├── reflectance_validator.py   # Validates reflectance (0-1)
│   └── window_coordinates_validator.py # Validates window coordinates
└── request_validators/            # One validator per request type
    ├── __init__.py
    ├── encode_request_validator.py
    ├── calculate_direction_request_validator.py
    ├── reference_point_request_validator.py
    └── external_reference_point_request_validator.py
```

## Core Classes

### BaseValidator (Abstract)

All validators inherit from this interface.

```python
class BaseValidator(ABC):
    @abstractmethod
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        pass
```

### ValidationResult

Encapsulates validation outcome and errors.

```python
result = ValidationResult()
if not result.is_valid:
    for error in result.errors:
        print(error.error_type.value, error.parameter_name, str(error))
```

### ValidationError

Custom exception with error type enum and parameter name.

```python
error = ValidationError(
    error_type=ValidationErrorType.INVALID_RANGE,
    message="height must be > 0",
    parameter_name="height_roof_over_floor"
)
```

## Parameter Validators

Each parameter validator validates **exactly one type** of parameter (SRP).

| Validator | Validates | Range/Rules |
|-----------|-----------|-------------|
| `ReflectanceValidator` | Reflectance values | 0.0 - 1.0 |
| `RatioValidator` | Ratio values | 0.0 - 1.0 |
| `HeightValidator` | Height values | >= min_value, <= max_value (configurable) |
| `AngleValidator` | Angle values | In radians, configurable range |
| `PolygonValidator` | Polygon structures | List of [x, y] pairs, min 3 vertices |
| `WindowCoordinatesValidator` | Window coordinates | x1, y1, z1, x2, y2, z2 (or 2D if configured) |
| `ModelTypeValidator` | Model type values | Must be valid `ModelType` enum value |
| `EncodingSchemeValidator` | Encoding scheme values | Must be valid `EncodingScheme` enum value |

## Request Validators

Each request validator validates **exactly one type** of request (SRP).

| Validator | Request Type | Required Fields |
|-----------|-------------|-----------------|
| `EncodeRequestValidator` | Encode requests | model_type, parameters (with room_polygon, height_roof_over_floor, etc.) |
| `CalculateDirectionRequestValidator` | Direction calculation | room_polygon, windows (2D coordinates) |
| `ReferencePointRequestValidator` | Reference point calculation | room_polygon, windows (3D coordinates) |
| `ExternalReferencePointRequestValidator` | External reference point | Same as ReferencePointRequestValidator |

## ValidatorManager

Orchestrates validation using **Strategy Pattern**. Routes requests to appropriate validators based on `RequestType` enum.

```python
from src.validation import ValidatorManager, RequestType

manager = ValidatorManager()

# Validate encode request
result = manager.validate(RequestType.ENCODE, request_data)

if not result.is_valid:
    for error in result.errors:
        print(f"{error.error_type.value}: {error}")
```

## Usage Examples

### Basic Validation

```python
from src.validation import ValidatorManager, RequestType

manager = ValidatorManager()

request = {
    "model_type": "df_default",
    "parameters": {
        "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "height_roof_over_floor": 2.7,
        "floor_height_above_terrain": 0.0
    }
}

result = manager.validate(RequestType.ENCODE, request)
print(result.is_valid)  # True or False
```

### Integration with Flask Endpoint

```python
from src.validation import ValidatorManager, RequestType
from werkzeug.exceptions import BadRequest

manager = ValidatorManager()

@app.route('/encode', methods=['POST'])
def encode():
    data = request.get_json()

    # Validate request
    result = manager.validate(RequestType.ENCODE, data)

    if not result.is_valid:
        error_messages = [str(error) for error in result.errors]
        raise BadRequest("; ".join(error_messages))

    # Proceed with encoding
    # ...
```

### Custom Validator

To add a new parameter validator:

```python
from src.validation.base import BaseValidator, ValidationResult, ValidationError
from src.validation.enums import ValidationErrorType

class MyCustomValidator(BaseValidator):
    def __init__(self, parameter_name: str):
        self._parameter_name = parameter_name

    def validate(self, value, context=None):
        result = ValidationResult()

        # Type check
        if not isinstance(value, str):
            result.add_error(ValidationError(
                error_type=ValidationErrorType.INVALID_TYPE,
                message=f"{self._parameter_name} must be a string",
                parameter_name=self._parameter_name
            ))

        return result
```

## Enums

### RequestType

```python
class RequestType(Enum):
    ENCODE = "encode"
    CALCULATE_DIRECTION = "calculate_direction"
    GET_REFERENCE_POINT = "get_reference_point"
    GET_EXTERNAL_REFERENCE_POINT = "get_external_reference_point"
```

### ValidationErrorType

```python
class ValidationErrorType(Enum):
    MISSING_PARAMETER = "missing_parameter"
    INVALID_TYPE = "invalid_type"
    INVALID_VALUE = "invalid_value"
    INVALID_RANGE = "invalid_range"
    INVALID_LENGTH = "invalid_length"
    INVALID_FORMAT = "invalid_format"
    INVALID_POLYGON = "invalid_polygon"
    INVALID_COORDINATES = "invalid_coordinates"
```

## Principles Followed

1. **OOP**: Everything is a class, encapsulated with proper interfaces
2. **SRP**: Each validator has exactly one responsibility
3. **Strategy Pattern**: ValidatorManager routes to validators based on request type
4. **Enumerator Pattern**: All types use enums, no magic strings
5. **Inheritance**: All validators inherit from BaseValidator ABC
6. **Type Safety**: Type hints throughout
7. **Composition**: Request validators compose parameter validators

## Testing

See `example/validation_example.py` for comprehensive usage examples.

## Future Enhancements

- Add validation caching for performance
- Add async validation support
- Add validation rule composition (AND/OR validators)
- Add validation warnings (non-blocking issues)
