import math
from typing import Tuple
from src.components.parameter_encoders.base import BaseChannelEncoder
from src.core import ParameterName, EncoderType
from src.components.parameter_encoders.linear_channel_encoder import LinearChannelEncoder
from src.components.parameter_encoders.angle_channel_encoder import AngleChannelEncoder
from src.components.parameter_encoders.reflectance_channel_encoder import ReflectanceChannelEncoder


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
        # Alpha channel: window_orientation (0-2π rad → 0-1, optional)
        ParameterName.WINDOW_ORIENTATION.value: (EncoderType.ANGLE.value, 0.0, 2 * math.pi, 0.0, 1.0),

        # Room encoders - using enum keys
        # Red channel: height_roof_over_floor (0-30m → 0-1, REQUIRED)
        ParameterName.HEIGHT_ROOF_OVER_FLOOR.value: (EncoderType.LINEAR.value, 0.0, 30.0, 0.0, 1.0),
        # Green channel: horizontal_reflectance (0-1 → 0-1, optional, default=1)
        ParameterName.FLOOR_REFLECTANCE.value: (EncoderType.REFLECTANCE.value, 0.0, 1.0, 0.0, 1.0),
        # Blue channel: vertical_reflectance (0-1 → 0-1, optional, default=1)
        ParameterName.WALL_REFLECTANCE.value: (EncoderType.REFLECTANCE.value, 0.0, 1.0, 0.0, 1.0),
        # Alpha channel: ceiling_reflectance (0.5-1 → 0-1, optional, default=1)
        ParameterName.CEILING_REFLECTANCE.value: (EncoderType.REFLECTANCE.value, 0.5, 1.0, 0.0, 1.0),

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
        ParameterName.ZENITH.value: (EncoderType.ANGLE.value, 0.0, 70.0, 0.2, 0.8)
    }

    @classmethod
    def create_encoder(cls, parameter_name: str) -> BaseChannelEncoder:
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
