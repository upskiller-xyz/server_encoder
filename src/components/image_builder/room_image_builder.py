from typing import Dict, Any, Optional
import numpy as np
from src.core import RegionType, ModelType, ParameterName, EncodingScheme
from src.components.region_encoders import RegionEncoderFactory
from src.core.graphics_constants import GRAPHICS_CONSTANTS
from src.models.encoding_parameters import EncodingParameters


class RoomImageBuilder:
    """
    Builder for creating encoded room images (Builder Pattern)

    Constructs 128×128 RGBA images with encoded room parameters
    """

    @classmethod
    def _empty(cls) -> np.ndarray:
        return np.zeros((128, 128, 4), dtype=np.uint8)
    
    @classmethod
    def _is_empty(cls, arr)->bool:
        return np.array_equal(arr, cls._empty())

    def __init__(self, encoding_scheme: EncodingScheme = EncodingScheme.RGB):
        self._image: np.ndarray = self._empty()
        self._model_type: ModelType = ModelType.DF_DEFAULT
        self._encoding_scheme = encoding_scheme
        self._region_encoder_factory = RegionEncoderFactory()
        self._room_mask: np.ndarray = self._empty()
        self.reset()

    def reset(self) -> 'RoomImageBuilder':
        """Reset builder to initial state"""
        # Create 128×128 RGBA image initialized to zeros
        self._image = self._empty()
        self._model_type = ModelType.DF_DEFAULT
        self._room_mask = self._empty()
        return self

    def set_model_type(self, model_type: ModelType) -> 'RoomImageBuilder':
        """Set the model type for encoding"""
        if not isinstance(model_type, ModelType):
            raise ValueError(f"Invalid model type: {model_type}")
        self._model_type = model_type
        return self

    def encode_region(self, region_type: RegionType, parameters: Dict[str, Any]) -> 'RoomImageBuilder':
        """
        Encode a region using its encoder (Single Responsibility Principle)

        Args:
            region_type: Type of region to encode
            parameters: Region-specific parameters

        Returns:
            Self for chaining
        """
        self._validate_state()

        # Snap window to room's facade edge before encoding
        # if region_type == RegionType.WINDOW and not self._is_empty(self._room_mask):
        #     self._snap_window_to_room(parameters)

        encoder = self._region_encoder_factory.get_encoder(region_type, self._encoding_scheme)
        self._image = encoder.encode_region(self._image, parameters, self._model_type)

        # Capture room mask when encoding room region
        if region_type == RegionType.ROOM:
            self._room_mask = encoder.get_last_mask()

        return self

    def _snap_window_to_room(self, window_params: EncodingParameters) -> None:
        """Inject room facade right edge into window parameters for pixel-perfect alignment."""
        center_y = self._room_mask.shape[0] // 2
        room_columns = np.where(self._room_mask[center_y])[0]
        if len(room_columns) == 0:
            return
        # window_params.right_wall = int(room_columns.max())
        window_params[ParameterName.RIGHT_WALL.value] = int(room_columns.max())

    def build(self) -> np.ndarray:
        """
        Build and return the final encoded image

        Returns:
            128×128 RGBA numpy array

        Raises:
            RuntimeError: If model type not set
        """
        self._validate_state()
        if self._image is None:
            raise RuntimeError("Image not initialized")
        return self._image.copy()

    def get_room_mask(self) -> Optional[np.ndarray]:
        """
        Get the room mask (ones in room area, zeros elsewhere)

        Returns:
            128×128 single-channel mask or None if no room was encoded
        """
        if self._room_mask is None:
            return None
        return self._room_mask.copy()

    def _validate_state(self) -> None:
        """Validate builder state before operations"""
        if self._model_type is None:
            raise RuntimeError("Model type must be set before encoding")
        if self._image is None:
            raise RuntimeError("Image not initialized. Call reset() first.")
