

from typing import Tuple


class BaseChannelEncoder:
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
        self._min_value = min_value
        self._max_value = max_value
        self._range = max_value - min_value
        self._output_min = output_min
        self._output_max = output_max
        self._output_range = output_max - output_min

    def encode(self, value: float) -> int:
        """Encode angle value"""
        return int(value)

    def get_range(self) -> Tuple[float, float]:
        """Get valid angle range"""
        return self._output_min, self._output_max