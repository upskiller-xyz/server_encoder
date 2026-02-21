from typing import Dict, Any, List
import numpy as np
from src.models import IRegionEncoder
from src.core import RegionType, ModelType, ChannelType, ParameterName, DEFAULT_PARAMETER_VALUES, EncodingScheme, get_channel_mapping, HSV_DEFAULT_PIXEL_OVERRIDES
from src.components.parameter_encoders import EncoderFactory as ParameterEncoderFactory
from src.components.region_encoders.validation_helpers import validate_required_parameters


class BaseRegionEncoder(IRegionEncoder):
    """Base class for region encoders with common functionality"""

    def __init__(self, region_type: RegionType, encoding_scheme: EncodingScheme = EncodingScheme.RGB):
        self._region_type = region_type
        self._encoder_factory = ParameterEncoderFactory()
        self._encoding_scheme = encoding_scheme
        self._last_mask: np.ndarray = np.array([])

    def get_region_type(self) -> RegionType:
        """Get the region type"""
        return self._region_type

    def set_encoding_scheme(self, encoding_scheme: EncodingScheme) -> None:
        """Set the encoding scheme for encoding"""
        self._encoding_scheme = encoding_scheme

    def get_last_mask(self) -> np.ndarray:
        """Get the last generated mask"""
        return self._last_mask

    def encode_region(
        self,
        image: np.ndarray,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> np.ndarray:
        """
        Encode window region

        Channels (CORRECTED):
        - Red: sill_height (z1, 0-5m → 0-1) [AUTO-CALCULATED]
        - Green: frame_ratio (1-0 → 0-1, reversed) [REQUIRED]
        - Blue: window_height (z2-z1, 0.2-5m → 0.99-0.01, reversed) [AUTO-CALCULATED]
        - Alpha: window_frame_reflectance (0-1 → 0-1, optional, default=0.8)

        Raises:
            ValueError: If required parameters are missing
        """
        parameters = self._update_parameters(parameters)
        # Validate required parameters
        self._validate_required_parameters(parameters)
        mask = self._get_area_mask(image, parameters, model_type)
        # Store the mask for later retrieval
        self._last_mask = mask.astype(np.uint8)

        # Get channel mapping for this region based on encoding scheme
        channel_map = get_channel_mapping(self._encoding_scheme)[self._region_type]

        _extra_params = self._get_extra(image, parameters)

        image[mask] = self._encode_all_channels(parameters, channel_map, model_type, *_extra_params)

        return image

    def _update_parameters(self, params):
        return params

    def _get_extra(self, image, params):
        return []

    def _validate_required_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate required background parameters using list comprehension"""
        missing = validate_required_parameters(self._region_type, parameters)
        if missing:
            raise ValueError(f"Missing required {self.__class__.__name__} parameters: {', '.join(missing)}")

    def _get_area_mask(self, image, parameters, model_type) -> np.ndarray[Any, Any]:
        height, width = image.shape[:2]
        mask = np.ones((height, width), dtype=bool)
        return mask

    def _encode_parameter(self, parameter_name: str, value: Any) -> int:
        """
        Encode a parameter value to pixel intensity

        Args:
            parameter_name: Name of the parameter
            value: Value to encode

        Returns:
            Pixel intensity [0-255]
        """
        encoder = self._encoder_factory.create_encoder(parameter_name)
        return encoder.encode(float(value))

    def _is_custom_model(self, model_type: ModelType) -> bool:
        """Check if model type uses custom materials"""
        return model_type in [ModelType.DF_CUSTOM, ModelType.DA_CUSTOM]

    def _is_da_model(self, model_type: ModelType) -> bool:
        """Check if model type is Daylight Autonomy"""
        return model_type in [ModelType.DA_DEFAULT, ModelType.DA_CUSTOM]

    def _get_parameter_with_default(
        self,
        parameters: Dict[str, Any],
        param_name: ParameterName
    ) -> Any:
        """
        Get parameter value with fallback to default value

        Args:
            parameters: Parameter dictionary
            param_name: Parameter name enum

        Returns:
            Parameter value or default value
        """
        pr = parameters #.to_dict()
        return pr.get(
            param_name.value,
            DEFAULT_PARAMETER_VALUES.get(param_name, pr.get(param_name.value))
        )

    def _encode_channel(
        self,
        parameters: Dict[str, Any],
        channel_map: Dict[ChannelType, ParameterName],
        channel_type: ChannelType,
        model_type: ModelType = ModelType.DF_DEFAULT
    ) -> int | np.ndarray:
        """
        Encode a single channel using the channel mapping (Template Method Pattern)

        Args:
            parameters: Parameter dictionary
            channel_map: Mapping from channel types to parameter names
            channel_type: Channel to encode (RED, GREEN, BLUE, or ALPHA)
            model_type: Model type (for HSV default overrides)

        Returns:
            Encoded pixel intensity [0-255]
        """
        param_name = channel_map[channel_type]
        param_value = self._get_parameter_with_default(parameters, param_name)

        if param_value is None:
            raise ValueError(
                f"{self.__class__} parameter '{param_name.value}' is missing or could not be calculated. "
                f"Available parameters: {list(parameters.keys())}. "

            )

        # Check if this is a default value and HSV encoding with an override
        is_using_default = param_name.value not in parameters
        if (is_using_default and
            self._encoding_scheme == EncodingScheme.HSV and
            model_type is not None):
            # Check for HSV pixel override
            override_key = (self._region_type, channel_type, model_type)
            if override_key in HSV_DEFAULT_PIXEL_OVERRIDES:
                return HSV_DEFAULT_PIXEL_OVERRIDES[override_key]

        return self._encode_parameter(param_name.value, param_value)

    def _encode_all_channels(
        self,
        parameters: Dict[str, Any],
        channel_map: Dict[ChannelType, ParameterName],
        model_type: ModelType = ModelType.DF_DEFAULT
    ) -> List[int | np.ndarray]:
        """
        Encode all 4 channels (RGBA) using list comprehension

        Args:
            parameters: Parameter dictionary
            channel_map: Mapping from channel types to parameter names
            model_type: Model type (for HSV default overrides)

        Returns:
            List of 4 encoded values [R, G, B, A]
        """
        return [
            self._encode_channel(parameters, channel_map, channel_type, model_type)
            for channel_type in [ChannelType.RED, ChannelType.GREEN, ChannelType.BLUE, ChannelType.ALPHA]
        ]
