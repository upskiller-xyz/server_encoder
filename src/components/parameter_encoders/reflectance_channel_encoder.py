from typing import Tuple

from src.components.parameter_encoders.base import BaseChannelEncoder
from src.components.parameter_encoders.linear_channel_encoder import LinearChannelEncoder


class ReflectanceChannelEncoder(BaseChannelEncoder):
    """Encodes reflectance values [0-1] to [0-255]"""

    def __init__(
        self,
        min_value: float,
        max_value: float,
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
            min_value, max_value, output_min, output_max
        )

    def encode(self, value: float) -> int:
        """Encode reflectance value"""
        return self._encoder.encode(value)

    def get_range(self) -> Tuple[float, float]:
        """Get valid reflectance range"""
        return self._encoder.get_range()
