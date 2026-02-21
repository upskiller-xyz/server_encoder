from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class RegionParameters:
    """
    Parameters for a specific region (background, room, window, obstruction_bar)

    Replaces Dict[str, Any] with a proper type-safe class
    """
    parameters: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default=None) -> Any:
        """Get parameter value"""
        return self.parameters.get(key, default)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set parameter value"""
        self.parameters[key] = value

    def __getitem__(self, key: str) -> Any:
        """Get parameter value (dict-like access)"""
        return self.parameters[key]

    def __contains__(self, key: str) -> bool:
        """Check if parameter exists"""
        return key in self.parameters

    def update(self, other: Dict[str, Any]) -> None:
        """Update parameters"""
        self.parameters.update(other)

    def keys(self):
        """Get parameter keys"""
        return self.parameters.keys()

    def values(self):
        """Get parameter values"""
        return self.parameters.values()

    def items(self):
        """Get parameter items"""
        return self.parameters.items()
