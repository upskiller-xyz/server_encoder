"""
Parameter validation and clipping module

This module provides validation and clipping functionality for encoding parameters,
following OOP principles and design patterns.
"""

from typing import Dict, Any, Tuple
import numpy as np
from abc import ABC, abstractmethod
from src.components.enums import ParameterName, ModelType
from src.components.encoders import EncoderFactory
from src.components.parameter_calculators import ParameterCalculatorRegistry
from src.components.geometry import WindowBorderValidator, WindowHeightValidator
from src.server.services.logging import StructuredLogger


class IParameterValidator(ABC):
    """Interface for parameter validators"""

    @abstractmethod
    def validate(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> Tuple[bool, str]:
        """
        Validate parameters

        Args:
            parameters: Parameters to validate
            model_type: Model type being used

        Returns:
            (is_valid, error_message)
        """
        pass
