from typing import Dict, Any
import logging
from src.components.calculators.window_sill_height_calculator import WindowSillHeightCalculator
from src.components.calculators.window_height_calculator import WindowHeightCalculator

logger = logging.getLogger(__name__)


class ParameterCalculatorRegistry:
    """
    Registry of parameter calculators (Factory + Registry Pattern)

    Manages automatic calculation of derived parameters.
    """

    # Available calculators (order matters: dependencies should be calculated first)
    _CALCULATORS = [
        WindowHeightCalculator(),         # No dependencies
        WindowSillHeightCalculator(),     # Depends on floor_height_above_terrain
    ]

    @classmethod
    def calculate_derived_parameters(
        cls,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate all derived parameters that can be computed

        Args:
            parameters: Input parameters
            logger: Optional logger for warnings

        Returns:
            Dictionary with calculated parameter values

        Raises:
            ValueError: If strict mode (logger=None) and calculation fails
        """
        result = {}

        for calculator in cls._CALCULATORS:
            param_name = calculator.get_parameter_name()

            # Skip if parameter already provided
            if param_name in parameters:
                continue

            # Try to calculate
            if calculator.can_calculate(parameters):
                try:
                    calculated_value = calculator.calculate(parameters)
                    result[param_name] = calculated_value
                    logger.debug(
                            f"Auto-calculated {param_name} = {calculated_value}"
                        )
                except ValueError as e:
                    logger.warning(
                            f"Could not auto-calculate {param_name}: {str(e)}"
                        )

        return result
