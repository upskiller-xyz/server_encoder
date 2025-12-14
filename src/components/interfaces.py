from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field
import numpy as np
from src.components.enums import ChannelType, RegionType, ModelType, ParameterName


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
        if key in [r.value for r in RegionType]:
            return self.get_region(RegionType(key)).parameters
        return self.global_params.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if parameter exists (backwards compatibility)"""
        if key in [r.value for r in RegionType]:
            region_params = self.get_region(RegionType(key)).parameters
            return bool(region_params)  # True if region has any parameters
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


class IChannelEncoder(ABC):
    """Interface for encoding values into image channels"""

    @abstractmethod
    def encode(self, value: float) -> int:
        """
        Encode a value into pixel intensity [0-255]

        Args:
            value: The value to encode

        Returns:
            Pixel intensity value [0-255]
        """
        pass

    @abstractmethod
    def get_range(self) -> Tuple[float, float]:
        """Get the valid range for this encoder"""
        pass


class IRegionEncoder(ABC):
    """Interface for encoding specific image regions"""

    @abstractmethod
    def encode_region(
        self,
        image: np.ndarray,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> np.ndarray:
        """
        Encode parameters into a specific region of the image

        Args:
            image: The RGBA image array to modify
            parameters: Dictionary of parameter name -> value
            model_type: The model type being used

        Returns:
            Modified image array
        """
        pass

    @abstractmethod
    def get_region_type(self) -> RegionType:
        """Get the region type this encoder handles"""
        pass


class IImageBuilder(ABC):
    """Interface for building encoded images using Builder pattern"""

    @abstractmethod
    def reset(self) -> 'IImageBuilder':
        """Reset the builder to initial state"""
        pass

    @abstractmethod
    def set_model_type(self, model_type: ModelType) -> 'IImageBuilder':
        """Set the model type"""
        pass

    @abstractmethod
    def encode_region(self, region_type: RegionType, parameters: Dict[str, Any]) -> 'IImageBuilder':
        """Encode a region using its encoder"""
        pass

    @abstractmethod
    def build(self) -> np.ndarray:
        """Build and return the final encoded image"""
        pass


@dataclass
class EncodingResult:
    """
    Result of encoding operation containing images and masks

    Encapsulates encoding results to avoid Dict[str, ...] returns
    """
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    masks: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)

    def get_image(self, window_id: str = "window_1") -> np.ndarray:
        """Get image for specific window"""
        return self.images.get(window_id)

    def get_mask(self, window_id: str = "window_1") -> Optional[np.ndarray]:
        """Get mask for specific window"""
        return self.masks.get(window_id)

    def add_window(self, window_id: str, image: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        """Add window encoding result"""
        self.images[window_id] = image
        self.masks[window_id] = mask

    def window_ids(self) -> list:
        """Get list of window IDs"""
        return list(self.images.keys())

    def is_single_window(self) -> bool:
        """Check if this is a single window result"""
        return len(self.images) == 1

    def get_first_image(self) -> np.ndarray:
        """Get first image (useful for single-window case)"""
        return next(iter(self.images.values()))

    def get_first_mask(self) -> Optional[np.ndarray]:
        """Get first mask (useful for single-window case)"""
        return next(iter(self.masks.values()))


@dataclass
class EncodedBytesResult:
    """
    Result of encoding operation containing PNG bytes and masks

    Encapsulates encoding results to avoid Dict[str, bytes] returns
    """
    images: Dict[str, bytes] = field(default_factory=dict)
    masks: Dict[str, Optional[bytes]] = field(default_factory=dict)

    def get_image(self, window_id: str = "window_1") -> bytes:
        """Get image bytes for specific window"""
        return self.images.get(window_id)

    def get_mask(self, window_id: str = "window_1") -> Optional[bytes]:
        """Get mask bytes for specific window"""
        return self.masks.get(window_id)

    def add_window(self, window_id: str, image: bytes, mask: Optional[bytes] = None) -> None:
        """Add window encoding result"""
        self.images[window_id] = image
        self.masks[window_id] = mask

    def window_ids(self) -> list:
        """Get list of window IDs"""
        return list(self.images.keys())


class IEncodingService(ABC):
    """Interface for the encoding service"""

    @abstractmethod
    def encode_room_image(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> bytes:
        """
        Encode room parameters into an image

        Args:
            parameters: Dictionary of all encoding parameters
            model_type: The model type to use

        Returns:
            Encoded image as PNG bytes
        """
        pass

    @abstractmethod
    def validate_parameters(
        self,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> Tuple[bool, str]:
        """
        Validate encoding parameters

        Returns:
            (is_valid, error_message)
        """
        pass
