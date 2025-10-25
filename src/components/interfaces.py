from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
from src.components.enums import ChannelType, RegionType, ModelType


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
