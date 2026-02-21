from typing import Dict, Any, List
from src.core import RegionType, REQUIRED_PARAMETERS


def validate_required_parameters(
    region_type: RegionType,
    parameters: Dict[str, Any]
) -> List[str]:
    """
    Validate required parameters for a region using list comprehension (Strategy Pattern)

    Args:
        region_type: The region type to validate
        parameters: Parameter dictionary to check

    Returns:
        List of missing parameter names (empty if all present)
    """
    required = REQUIRED_PARAMETERS.get(region_type, [])
    missing = [param.value for param in required if param.value not in parameters]
    return missing
