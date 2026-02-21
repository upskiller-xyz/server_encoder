from typing import Tuple
import numpy as np


class LinearChannelEncoder:
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
