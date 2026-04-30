"""
Encoding parameter validator.

SRP: owns all parameter-level validation, clipping, and scheme-specific
preprocessing for encoding requests. EncodingService delegates to this class
and performs no parameter logic itself.
"""
from typing import Any, Dict, Tuple
import logging

import numpy as np

from src.core.enums import DEFAULT_PARAMETER_VALUES, EncodingScheme, ParameterName
from src.components.parameter_encoders import EncoderFactory
from src.components.geometry import WindowGeometry, RoomPolygon
from src.components.calculators import ParameterCalculatorRegistry
from src.validation.utils import ValidationUtils
from src.validation.parameter_validators.window_border_validator import WindowBorderValidator
from src.validation.parameter_validators.window_height_validator import WindowHeightValidator
from src.core import ModelType

logger = logging.getLogger(__name__)


class EncodingParameterValidator:
    """
    Validates and preprocesses flat encoding parameter dictionaries.

    Responsibilities:
    - Auto-fill direction_angle from window geometry when absent
    - Apply scheme-specific preprocessing (V7/V8 height defaults, V8 vector unpacking)
    - Check required parameters
    - Validate parameter ranges
    - Validate window geometry placement (border + height)
    - Clip parameters to model-safe ranges
    """

    # Format: {param_name: (clip_min, clip_max, reject_below_min)}
    _CLIPPING_CONFIG = {
        ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value: (0.0, 10.0, True),
        ParameterName.HEIGHT_ROOF_OVER_FLOOR.value: (15.0, 30.0, True),
        ParameterName.HORIZON.value: (0.0, 90.0, False),
        ParameterName.ZENITH.value: (0.0, 70.0, False),
    }

    # Parameters that must be present for all encoding schemes
    _REQUIRED_PARAMETERS = (
        ParameterName.HEIGHT_ROOF_OVER_FLOOR,
        ParameterName.WINDOW_FRAME_RATIO,
        ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN,
        ParameterName.HORIZON,
        ParameterName.ZENITH,
        ParameterName.ROOM_POLYGON,
    )

    def __init__(self, encoding_scheme: EncodingScheme, encoder_factory: EncoderFactory) -> None:
        self._encoding_scheme = encoding_scheme
        self._encoder_factory = encoder_factory

    # ------------------------------------------------------------------
    # Direction angle auto-fill
    # ------------------------------------------------------------------

    def ensure_direction_angle(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Auto-calculate direction_angle from window geometry when absent.

        Handles both flat (single-window) and windows-dict structures.
        Modifies parameters in-place.
        """
        room_polygon = None
        if ParameterName.ROOM_POLYGON.value in parameters:
            room_polygon = RoomPolygon.from_dict(parameters[ParameterName.ROOM_POLYGON.value])

        if ValidationUtils.has_flat_window_coordinates(parameters) and room_polygon:
            if ParameterName.DIRECTION_ANGLE.value not in parameters:
                self._fill_direction_angle(parameters, room_polygon)

        elif ParameterName.WINDOWS.value in parameters and room_polygon:
            windows = parameters[ParameterName.WINDOWS.value]
            if isinstance(windows, dict):
                for window_id, window_params in windows.items():
                    if (
                        isinstance(window_params, dict)
                        and ParameterName.DIRECTION_ANGLE.value not in window_params
                        and ValidationUtils.has_flat_window_coordinates(window_params)
                    ):
                        self._fill_direction_angle(window_params, room_polygon, window_id)

        return parameters

    def _fill_direction_angle(
        self,
        parameters: Dict[str, Any],
        room_polygon: RoomPolygon,
        window_id: str = "",
    ) -> None:
        label = f" for '{window_id}'" if window_id else ""
        try:
            angle = WindowGeometry.from_dict(parameters).calculate_direction_from_polygon(room_polygon)
            parameters[ParameterName.DIRECTION_ANGLE.value] = angle
            logger.info("Auto-calculated direction_angle%s: %.4f rad", label, angle)
        except Exception as exc:
            logger.warning("Could not auto-calculate direction_angle%s: %s", label, exc)

    # ------------------------------------------------------------------
    # Public validation entry point
    # ------------------------------------------------------------------

    def validate(self, parameters: Dict[str, Any], model_type: ModelType) -> Tuple[bool, str]:
        """
        Validate parameters, handling both flat and windows-dict structures.

        Args:
            parameters: Flat or windows-dict parameter dictionary (mutated in place)
            model_type: Model type being encoded

        Returns:
            (is_valid, error_message)
        """
        if ParameterName.WINDOWS.value in parameters:
            if not isinstance(parameters[ParameterName.WINDOWS.value], dict):
                return False, "Parameter 'windows' must be a dictionary mapping window_id to window parameters"

            for window_id, window_params in parameters[ParameterName.WINDOWS.value].items():
                if not isinstance(window_params, dict):
                    return False, f"Window '{window_id}' parameters must be a dictionary, got {type(window_params).__name__}"

                merged = {**parameters, **window_params}
                merged.pop(ParameterName.WINDOWS.value, None)

                is_valid, error_msg = self._validate_flat(merged)
                if not is_valid:
                    return False, f"Window '{window_id}': {error_msg}"

                # Write clipped values back to window_params and shared params
                for param_name in list(window_params.keys()):
                    if param_name in merged:
                        window_params[param_name] = merged[param_name]
                for param_name in self._CLIPPING_CONFIG:
                    if param_name in merged and param_name not in window_params:
                        parameters[param_name] = merged[param_name]

            return True, ""

        return self._validate_flat(parameters)

    # ------------------------------------------------------------------
    # Flat parameter validation
    # ------------------------------------------------------------------

    def _validate_flat(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate a single flat parameter dict."""
        try:
            self._preprocess_scheme_params(parameters)
        except ValueError as exc:
            return False, str(exc)

        calculated = ParameterCalculatorRegistry.calculate_derived_parameters(parameters)
        parameters.update(calculated)

        missing = [p.value for p in self._REQUIRED_PARAMETERS if p.value not in parameters]
        if missing:
            return False, f"Missing required parameters: {', '.join(missing)}"

        range_error = self._validate_ranges(parameters)
        if range_error:
            return False, range_error

        geometry_error = self._validate_geometry(parameters)
        if geometry_error:
            return False, geometry_error

        try:
            self.clip(parameters)
        except ValueError as exc:
            return False, str(exc)

        return True, ""

    def _validate_ranges(self, parameters: Dict[str, Any]) -> str:
        """Validate numeric ranges. Returns error string or empty string."""
        for param_name, value in parameters.items():
            if isinstance(value, (list, np.ndarray)):
                continue
            if param_name in self._CLIPPING_CONFIG:
                continue  # handled by clip()
            try:
                min_val, max_val = self._encoder_factory.get_parameter_range(param_name)
                actual_min, actual_max = min(min_val, max_val), max(min_val, max_val)
                try:
                    float_value = float(value)
                except (TypeError, ValueError) as exc:
                    return (
                        f"Parameter '{param_name}' has invalid value type: {type(value).__name__}. "
                        f"Expected numeric value, got: {value}. Error: {exc}"
                    )
                if not (actual_min <= float_value <= actual_max):
                    return f"Parameter '{param_name}' value {value} outside valid range [{min_val}, {max_val}]"
            except ValueError as exc:
                if "Unknown parameter" in str(exc):
                    continue
                return f"Error validating parameter '{param_name}': {exc}"
            except Exception as exc:
                return f"Unexpected error validating '{param_name}' with value {value}: {type(exc).__name__}: {exc}"
        return ""

    def _validate_geometry(self, parameters: Dict[str, Any]) -> str:
        """Validate window geometry placement. Returns error string or empty string."""
        room_polygon = parameters.get(ParameterName.ROOM_POLYGON.value)
        if not ValidationUtils.has_window_coordinates(parameters, require_3d=True) or not room_polygon:
            return ""

        border_result = WindowBorderValidator().validate(parameters, {"room_polygon": room_polygon})
        if not border_result.is_valid:
            return "; ".join(str(e) for e in border_result.errors)

        floor_height = parameters.get(ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value)
        height_roof = parameters.get(ParameterName.HEIGHT_ROOF_OVER_FLOOR.value)
        if floor_height is not None and height_roof is not None:
            height_result = WindowHeightValidator().validate(
                parameters, {"floor_height": floor_height, "height_roof_over_floor": height_roof}
            )
            if not height_result.is_valid:
                return "; ".join(str(e) for e in height_result.errors)

        return ""

    # ------------------------------------------------------------------
    # Clipping
    # ------------------------------------------------------------------

    def clip(self, parameters: Dict[str, Any]) -> None:
        """
        Clip parameters to model-safe ranges in-place.

        Raises:
            ValueError: For values that cannot be clipped (e.g. negative heights)
        """
        for param_name, (min_val, max_val, reject_below_min) in self._CLIPPING_CONFIG.items():
            if param_name not in parameters:
                continue

            value = parameters[param_name]
            if isinstance(value, (list, np.ndarray)):
                continue

            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Parameter '{param_name}' has invalid value: {parameters[param_name]}. Error: {exc}")

            original, clipped = value, False

            if param_name == ParameterName.HEIGHT_ROOF_OVER_FLOOR.value:
                if value <= 0.0:
                    raise ValueError(f"Parameter '{param_name}' value {value} not supported. Must be greater than 0.")
                if value < min_val:
                    value, clipped = min_val, True
            elif value < min_val:
                if reject_below_min:
                    raise ValueError(
                        f"Parameter '{param_name}' value {value} not supported. "
                        f"Valid range is [{min_val}, {max_val}]."
                    )
                value, clipped = min_val, True

            if value > max_val:
                value, clipped = max_val, True

            if clipped:
                logger.warning(
                    "Parameter '%s' value %s outside range [%s, %s]. Clipped to %s.",
                    param_name, original, min_val, max_val, value,
                )
                parameters[param_name] = value

    # ------------------------------------------------------------------
    # Scheme-specific preprocessing
    # ------------------------------------------------------------------

    def _preprocess_scheme_params(self, parameters: Dict[str, Any]) -> None:
        """
        Apply scheme-specific transformations before validation.

        V8 / V10: unpack height_vector → height_roof_over_floor, floor_height_above_terrain
        V7 / V8 / V9 / V10: inject defaults for missing height parameters

        Raises:
            ValueError: If height_vector has wrong shape
        """
        if self._encoding_scheme in (EncodingScheme.V8, EncodingScheme.V10, EncodingScheme.V11):
            key = ParameterName.HEIGHT_VECTOR.value
            if key in parameters:
                vec = parameters[key]
                if not hasattr(vec, "__len__") or len(vec) != 2:
                    raise ValueError(
                        f"'{key}' must be a two-element sequence "
                        f"[height_roof_over_floor, floor_height_above_terrain], got: {vec!r}"
                    )
                parameters[ParameterName.HEIGHT_ROOF_OVER_FLOOR.value] = float(vec[0])
                parameters[ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value] = float(vec[1])

        if self._encoding_scheme in (EncodingScheme.V7, EncodingScheme.V8, EncodingScheme.V9, EncodingScheme.V10, EncodingScheme.V11):
            for param in (ParameterName.HEIGHT_ROOF_OVER_FLOOR, ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN):
                if param.value not in parameters:
                    parameters[param.value] = DEFAULT_PARAMETER_VALUES[param]
