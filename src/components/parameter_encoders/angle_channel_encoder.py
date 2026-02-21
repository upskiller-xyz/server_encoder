from typing import Tuple
from src.components.parameter_encoders.base import BaseChannelEncoder
from src.components.parameter_encoders.linear_channel_encoder import LinearChannelEncoder


class AngleChannelEncoder(BaseChannelEncoder):
    """Encodes angular values (degrees) to [0-255]"""

    def __init__(
        self,
        min_value: float,
        max_value: float,
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
            min_value, max_value, output_min, output_max
        )

    def encode(self, value: float) -> int:
        """Encode angle value"""
        return self._encoder.encode(value)

    def get_range(self) -> Tuple[float, float]:
        """Get valid angle range"""
        return self._encoder.get_range()
