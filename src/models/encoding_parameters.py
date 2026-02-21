from typing import Dict, Any
from dataclasses import dataclass, field
from src.core import RegionType
from src.models.region_parameters import RegionParameters


@dataclass
class EncodingParameters:
    """
    Complete set of parameters for encoding, organized by region

    Replaces Dict[str, Dict[str, Any]] with a proper type-safe class
    """
    background: RegionParameters = field(default_factory=RegionParameters)
    room: RegionParameters = field(default_factory=RegionParameters)
    window: RegionParameters = field(default_factory=RegionParameters)
    obstruction_bar: RegionParameters = field(default_factory=RegionParameters)
    # Top-level parameters (like direction_angle)
    global_params: Dict[str, Any] = field(default_factory=dict)

    def get_region(self, region_type: RegionType) -> RegionParameters:
        """Get parameters for a specific region"""
        region_map = {
            RegionType.BACKGROUND: self.background,
            RegionType.ROOM: self.room,
            RegionType.WINDOW: self.window,
            RegionType.OBSTRUCTION_BAR: self.obstruction_bar,
        }
        return region_map.get(region_type, RegionParameters())

    def set_global(self, key: str, value: Any) -> None:
        """Set a global parameter"""
        self.global_params[key] = value

    def get_global(self, key: str, default=None) -> Any:
        """Get a global parameter"""
        return self.global_params.get(key, default)

    # Dict-like interface for backwards compatibility
    def get(self, key: str, default=None) -> Any:
        """Get region or global parameter (backwards compatibility)"""
        if key in [r.value for r in RegionType]:
            return self.get_region(RegionType(key))
        return self.global_params.get(key, default)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set parameter (backwards compatibility)"""
        if key in [r.value for r in RegionType]:
            # Setting a region - not typical, but support it
            pass
        else:
            self.global_params[key] = value

    def __getitem__(self, key: str) -> Any:
        """Get parameter (backwards compatibility)"""
        # Check all regions for the parameter
        for region in [self.background, self.room, self.window, self.obstruction_bar]:
            if key in region.parameters:
                return region.parameters[key]
        # Check global parameters
        if key in self.global_params:
            return self.global_params[key]
        # Not found
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Check if parameter exists (backwards compatibility)"""
        # Check all regions for the parameter
        for region in [self.background, self.room, self.window, self.obstruction_bar]:
            if key in region.parameters:
                return True
        # Check global parameters
        return key in self.global_params

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncodingParameters':
        """Create from dictionary (for backwards compatibility)"""
        params = cls()

        # Separate region parameters from global parameters
        for key, value in data.items():
            if key == RegionType.BACKGROUND.value:
                params.background = RegionParameters(parameters=value if isinstance(value, dict) else {})
            elif key == RegionType.ROOM.value:
                params.room = RegionParameters(parameters=value if isinstance(value, dict) else {})
            elif key == RegionType.WINDOW.value:
                params.window = RegionParameters(parameters=value if isinstance(value, dict) else {})
            elif key == RegionType.OBSTRUCTION_BAR.value:
                params.obstruction_bar = RegionParameters(parameters=value if isinstance(value, dict) else {})
            else:
                params.global_params[key] = value

        return params

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for backwards compatibility)"""
        result = dict(self.global_params)
        if self.background.parameters:
            result[RegionType.BACKGROUND.value] = self.background.parameters
        if self.room.parameters:
            result[RegionType.ROOM.value] = self.room.parameters
        if self.window.parameters:
            result[RegionType.WINDOW.value] = self.window.parameters
        if self.obstruction_bar.parameters:
            result[RegionType.OBSTRUCTION_BAR.value] = self.obstruction_bar.parameters
        return result
