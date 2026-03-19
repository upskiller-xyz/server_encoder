"""
Example of using the validation system.

This demonstrates how to integrate the ValidatorManager into the application.
"""
from src.validation import ValidatorManager, RequestType, ValidationResult


def example_encode_validation():
    """Example: Validate an encode request"""
    manager = ValidatorManager()

    # Valid request
    valid_request = {
        "model_type": "df_default",
        "encoding_scheme": "v2",
        "parameters": {
            "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "height_roof_over_floor": 2.7,
            "floor_height_above_terrain": 0.0,
            "windows": {
                "window1": {
                    "x1": -0.6, "y1": 0.0, "z1": 1.0,
                    "x2": 0.6, "y2": 0.0, "z2": 2.5
                }
            }
        }
    }

    result = manager.validate(RequestType.ENCODE, valid_request)
    print(f"Valid request validation: {result.is_valid}")

    # Invalid request - missing required field
    invalid_request = {
        "model_type": "df_default",
        "parameters": {
            "room_polygon": [[0, 0], [10, 0], [10, 10]]  # Invalid polygon (< 3 vertices)
        }
    }

    result = manager.validate(RequestType.ENCODE, invalid_request)
    print(f"Invalid request validation: {result.is_valid}")
    for error in result.errors:
        print(f"  - {error.error_type.value}: {error}")


def example_calculate_direction_validation():
    """Example: Validate a calculate direction request"""
    manager = ValidatorManager()

    # Valid request
    valid_request = {
        "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "windows": {
            "window1": {
                "x1": -0.6, "y1": 0.0,
                "x2": 0.6, "y2": 0.0
            }
        }
    }

    result = manager.validate(RequestType.CALCULATE_DIRECTION, valid_request)
    print(f"Valid direction request: {result.is_valid}")

    # Invalid request - missing coordinates
    invalid_request = {
        "room_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "windows": {
            "window1": {
                "x1": -0.6  # Missing y1, x2, y2
            }
        }
    }

    result = manager.validate(RequestType.CALCULATE_DIRECTION, invalid_request)
    print(f"Invalid direction request: {result.is_valid}")
    for error in result.errors:
        print(f"  - {error.error_type.value}: {error}")


def example_integration_with_endpoint():
    """
    Example: How to integrate validation into an endpoint.

    This shows how you would use the ValidatorManager in application.py endpoints.
    """
    from werkzeug.exceptions import BadRequest

    manager = ValidatorManager()

    def encode_endpoint(request_data: dict):
        """Example endpoint using validation"""
        # Validate request
        result = manager.validate(RequestType.ENCODE, request_data)

        if not result.is_valid:
            # Collect all error messages
            error_messages = [str(error) for error in result.errors]
            raise BadRequest("; ".join(error_messages))

        # If validation passes, proceed with encoding
        print("Request is valid, proceeding with encoding...")
        # ... rest of endpoint logic ...

    # Test with invalid request
    try:
        invalid_request = {
            "model_type": "invalid_model",
            "parameters": {}
        }
        encode_endpoint(invalid_request)
    except BadRequest as e:
        print(f"Caught validation error: {e.description}")


if __name__ == "__main__":
    print("=== Encode Request Validation ===")
    example_encode_validation()

    print("\n=== Calculate Direction Request Validation ===")
    example_calculate_direction_validation()

    print("\n=== Integration Example ===")
    example_integration_with_endpoint()
