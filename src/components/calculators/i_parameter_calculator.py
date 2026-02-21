from typing import Any, Dict
from abc import ABC, abstractmethod


class IParameterCalculator(ABC):
    """
    Abstract base class for parameter calculators (Strategy Pattern)

    Calculates derived parameters from input data.
    """

    @abstractmethod
    def can_calculate(self, parameters: Dict[str, Any]) -> bool:
        """
        Check if this calculator can compute from given parameters

        Args:
            parameters: Encoding parameters dictionary

        Returns:
            True if calculator has required inputs
        """
        pass

    @abstractmethod
    def calculate(self, parameters: Dict[str, Any]) -> Any:
        """
        Calculate the parameter value

        Args:
            parameters: Input parameters

        Returns:
            Calculated value

        Raises:
            ValueError: If calculation cannot be performed
        """
        pass

    @abstractmethod
    def get_parameter_name(self) -> str:
        """Get the name of the parameter this calculator produces"""
        pass
