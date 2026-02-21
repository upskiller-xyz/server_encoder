from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from src.core import RegionType, ModelType
# from src.models.encoding_parameters import EncodingParameters


class IRegionEncoder(ABC):
    """
    Abstract base class for region encoders (Strategy Pattern)

    Each encoder handles a specific region type (background, room, window, obstruction bar)
    and encodes its parameters into the image.
    """

    @abstractmethod
    def get_region_type(self) -> RegionType:
        """Get the region type this encoder handles"""
        pass

    @abstractmethod
    def encode_region(
        self,
        image: np.ndarray,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> np.ndarray:
        """
        Encode region parameters into the image

        Args:
            image: Image array to encode into (modified in-place)
            parameters: Region-specific parameters
            model_type: Model type (DF_DEFAULT, DF_CUSTOM, DA_DEFAULT, DA_CUSTOM)

        Returns:
            Modified image array

        Raises:
            ValueError: If required parameters are missing
        """
        pass

    @abstractmethod
    def get_last_mask(self) -> np.ndarray:
        """
        Get the last generated mask for this region

        Returns:
            Binary mask array where True indicates the region area
        """
        pass
