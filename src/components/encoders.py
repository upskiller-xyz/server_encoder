from typing import Tuple
import numpy as np
from src.components.interfaces import IChannelEncoder
from src.components.enums import ParameterName, EncoderType


class LinearChannelEncoder(IChannelEncoder):
    """Encodes values using linear mapping to [0-255]"""

    def __init__(
        self,
        min_value: float,
        max_value: float,
        output_min: float = 0.0,
        output_max: float = 1.0
    ):
        """
        Initialize linear encoder

        Args:
            min_value: Minimum value of input range
            max_value: Maximum value of input range
            output_min: Minimum value of normalized output range (default 0.0)
            output_max: Maximum value of normalized output range (default 1.0)
        """
        self._min_value = min_value
        self._max_value = max_value
        self._range = max_value - min_value
        self._output_min = output_min
        self._output_max = output_max
        self._output_range = output_max - output_min

    def encode(self, value: float) -> int:
        """Encode value using linear normalization"""
        # Handle reversed ranges (min > max)
        actual_min = min(self._min_value, self._max_value)
        actual_max = max(self._min_value, self._max_value)

        # Clamp value to input range
        clamped = np.clip(value, actual_min, actual_max)

        # Normalize to [0, 1]
        normalized = 0.0
        if self._range != 0:
            normalized = (clamped - self._min_value) / self._range

        # Map to output range
        output_normalized = self._output_min + (normalized * self._output_range)

        # Scale to [0, 255]
        return int(np.clip(output_normalized * 255, 0, 255))

    def get_range(self) -> Tuple[float, float]:
        """Get valid input range"""
        return (self._min_value, self._max_value)


class AngleChannelEncoder(IChannelEncoder):
    """Encodes angular values (degrees) to [0-255]"""

    def __init__(
        self,
        min_angle: float,
        max_angle: float,
        output_min: float = 0.0,
        output_max: float = 1.0
    ):
        """
        Initialize angle encoder

        Args:
            min_angle: Minimum angle in degrees
            max_angle: Maximum angle in degrees
            output_min: Minimum value of normalized output range (default 0.0)
            output_max: Maximum value of normalized output range (default 1.0)
        """
        self._encoder = LinearChannelEncoder(
            min_angle, max_angle, output_min, output_max
        )

    def encode(self, value: float) -> int:
        """Encode angle value"""
        return self._encoder.encode(value)

    def get_range(self) -> Tuple[float, float]:
        """Get valid angle range"""
        return self._encoder.get_range()


class ReflectanceChannelEncoder(IChannelEncoder):
    """Encodes reflectance values [0-1] to [0-255]"""

    def __init__(
        self,
        min_reflectance: float = 0.0,
        max_reflectance: float = 1.0,
        output_min: float = 0.0,
        output_max: float = 1.0
    ):
        """
        Initialize reflectance encoder

        Args:
            min_reflectance: Minimum reflectance value
            max_reflectance: Maximum reflectance value
            output_min: Minimum value of normalized output range (default 0.0)
            output_max: Maximum value of normalized output range (default 1.0)
        """
        self._encoder = LinearChannelEncoder(
            min_reflectance, max_reflectance, output_min, output_max
        )

    def encode(self, value: float) -> int:
        """Encode reflectance value"""
        return self._encoder.encode(value)

    def get_range(self) -> Tuple[float, float]:
        """Get valid reflectance range"""
        return self._encoder.get_range()


class EncoderFactory:
    """Factory for creating channel encoders (Factory Pattern)"""

    _ENCODER_CONFIGS = {
        # Background encoders - using enum keys
        # Red channel: facade_reflectance (0-1 → 0-1, optional, default=1)
        ParameterName.FACADE_REFLECTANCE.value: (EncoderType.REFLECTANCE.value, 0.0, 1.0, 0.0, 1.0),
        # Green channel: floor_height_above_terrain (0-10m → 0.1-1, REQUIRED)
        ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value: (EncoderType.LINEAR.value, 0.0, 10.0, 0.1, 1.0),
        # Blue channel: terrain_reflectance (0-1 → 0-1, optional, default=1)
        ParameterName.TERRAIN_REFLECTANCE.value: (EncoderType.REFLECTANCE.value, 0.0, 1.0, 0.0, 1.0),
        # Alpha channel: window_orientation (0-360° → 0-1, optional, default=0.8)
        ParameterName.WINDOW_ORIENTATION.value: (EncoderType.ANGLE.value, 0.0, 360.0, 0.0, 1.0),

        # Legacy background encoder names (backward compatibility)
        "facade_reflectance": (EncoderType.REFLECTANCE.value, 0.0, 1.0, 0.0, 1.0),
        "floor_height_above_terrain": (EncoderType.LINEAR.value, 0.0, 10.0, 0.1, 1.0),

        # Room encoders - using enum keys
        # Red channel: height_roof_over_floor (0-30m → 0-1, REQUIRED)
        ParameterName.HEIGHT_ROOF_OVER_FLOOR.value: (EncoderType.LINEAR.value, 0.0, 30.0, 0.0, 1.0),
        # Green channel: horizontal_reflectance (0-1 → 0-1, optional, default=1)
        ParameterName.FLOOR_REFLECTANCE.value: (EncoderType.REFLECTANCE.value, 0.0, 1.0, 0.0, 1.0),
        # Blue channel: vertical_reflectance (0-1 → 0-1, optional, default=1)
        ParameterName.WALL_REFLECTANCE.value: (EncoderType.REFLECTANCE.value, 0.0, 1.0, 0.0, 1.0),
        # Alpha channel: ceiling_reflectance (0.5-1 → 0-1, optional, default=1)
        ParameterName.CEILING_REFLECTANCE.value: (EncoderType.REFLECTANCE.value, 0.5, 1.0, 0.0, 1.0),

        # Legacy room encoder names (backward compatibility)
        "height_roof_over_floor": (EncoderType.LINEAR.value, 0.0, 30.0, 0.0, 1.0),
        "horizontal_reflectance": (EncoderType.REFLECTANCE.value, 0.0, 1.0, 0.0, 1.0),
        "vertical_reflectance": (EncoderType.REFLECTANCE.value, 0.0, 1.0, 0.0, 1.0),

        # Window encoders - using enum keys
        # Red channel: sill_height (0-5m → 0-1)
        ParameterName.WINDOW_SILL_HEIGHT.value: (EncoderType.LINEAR.value, 0.0, 5.0, 0.0, 1.0),
        # Green channel: frame_ratio (1-0 → 0-1, REVERSED)
        ParameterName.WINDOW_FRAME_RATIO.value: (EncoderType.LINEAR.value, 1.0, 0.0, 0.0, 1.0),
        # Blue channel: window_height (0.2-5m → 0.99-0.01, REVERSED)
        ParameterName.WINDOW_HEIGHT.value: (EncoderType.LINEAR.value, 0.2, 5.0, 0.99, 0.01),
        # Alpha channel: window_frame_reflectance (0-1 → 0-1, optional, default=0.8)
        ParameterName.WINDOW_FRAME_REFLECTANCE.value: (EncoderType.REFLECTANCE.value, 0.0, 1.0, 0.0, 1.0),

        # Obstruction bar encoders - using enum keys
        # Alpha channel: balcony_reflectance (0-1 → 0-1, optional, default=0.8)
        ParameterName.BALCONY_REFLECTANCE.value: (EncoderType.REFLECTANCE.value, 0.0, 1.0, 0.0, 1.0),
        # Red channel: horizon (0-90° → 0-1)
        ParameterName.HORIZON.value: (EncoderType.ANGLE.value, 0.0, 90.0, 0.0, 1.0),
        # Green channel: context_reflectance (0.1-0.6 → 0-1)
        ParameterName.CONTEXT_REFLECTANCE.value: (EncoderType.REFLECTANCE.value, 0.1, 0.6, 0.0, 1.0),
        # Blue channel: zenith (0-70° → 0.2-0.8)
        ParameterName.ZENITH.value: (EncoderType.ANGLE.value, 0.0, 70.0, 0.2, 0.8),

        # Legacy obstruction bar encoder names (backward compatibility)
        "balcony_reflectance": (EncoderType.REFLECTANCE.value, 0.0, 1.0, 0.0, 1.0),
        "horizon": (EncoderType.ANGLE.value, 0.0, 90.0, 0.0, 1.0),
        "zenith": (EncoderType.ANGLE.value, 0.0, 70.0, 0.2, 0.8),
        "context_reflectance": (EncoderType.REFLECTANCE.value, 0.1, 0.6, 0.0, 1.0),
    }

    @classmethod
    def create_encoder(cls, parameter_name: str) -> IChannelEncoder:
        """
        Create appropriate encoder for a parameter

        Args:
            parameter_name: Name of the parameter

        Returns:
            Appropriate channel encoder

        Raises:
            ValueError: If parameter name is unknown
        """
        if parameter_name not in cls._ENCODER_CONFIGS:
            raise ValueError(f"Unknown parameter: {parameter_name}")

        config = cls._ENCODER_CONFIGS[parameter_name]
        encoder_type, min_val, max_val, output_min, output_max = config

        # Strategy pattern: select encoder type using enum
        encoder_strategies = {
            EncoderType.LINEAR.value: lambda: LinearChannelEncoder(
                min_val, max_val, output_min, output_max
            ),
            EncoderType.ANGLE.value: lambda: AngleChannelEncoder(
                min_val, max_val, output_min, output_max
            ),
            EncoderType.REFLECTANCE.value: lambda: ReflectanceChannelEncoder(
                min_val, max_val, output_min, output_max
            ),
        }

        creator = encoder_strategies.get(encoder_type)
        if not creator:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        return creator()

    @classmethod
    def get_parameter_range(cls, parameter_name: str) -> Tuple[float, float]:
        """Get the valid input range for a parameter"""
        if parameter_name not in cls._ENCODER_CONFIGS:
            raise ValueError(f"Unknown parameter: {parameter_name}")

        _, min_val, max_val, _, _ = cls._ENCODER_CONFIGS[parameter_name]
        return (min_val, max_val)
