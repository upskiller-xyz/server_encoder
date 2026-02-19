from typing import Any, Dict, List
import numpy as np
from src.core import RegionType, ModelType, ChannelType, ParameterName, ImageDimensions, EncodingScheme, get_channel_mapping, HSV_DEFAULT_PIXEL_OVERRIDES
from src.components.region_encoders.base_region_encoder import BaseRegionEncoder
from src.components.region_encoders.validation_helpers import validate_required_parameters
from src.core import GRAPHICS_CONSTANTS


class ObstructionBarEncoder(BaseRegionEncoder):
    """
    Encodes obstruction bar region parameters

    CORRECTED CHANNEL MAPPINGS:
    - Red: horizon (0-90° input → 0-1 normalized)
    - Green: context_reflectance (0.1-0.6 input → 0-1 normalized, default=1 if unobstructed)
    - Blue: zenith (0-70° input → 0.2-0.8 normalized)
    - Alpha: balcony_reflectance (0-1 input → 0-1 normalized, default=0.8)
    """

    def __init__(self, encoding_scheme: EncodingScheme = EncodingScheme.RGB):
        super().__init__(RegionType.OBSTRUCTION_BAR, encoding_scheme)

    def encode_region(
        self,
        image: np.ndarray,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> np.ndarray:
        """
        Override encode_region for obstruction bar to handle per-row encoding

        The obstruction bar is special because RGB channels vary per row
        while the alpha channel is constant.
        """
        parameters = self._update_parameters(parameters)
        self._validate_required_parameters(parameters)

        # Get bar position
        bar_x_start, bar_y_start, bar_x_end, bar_y_end = self._get_final_bar_position(image)

        # Get bar dimensions
        dims = ImageDimensions(image.shape[1])
        bar_height = dims.obstruction_bar_height
        actual_bar_height = bar_y_end - bar_y_start

        # Get channel mapping based on encoding scheme
        channel_map = get_channel_mapping(self._encoding_scheme)[self._region_type]
        channel_order = [ChannelType.RED, ChannelType.GREEN, ChannelType.BLUE, ChannelType.ALPHA]

        for channel_idx, channel_type in enumerate(channel_order):
            encoded = self._encode_channel(parameters, channel_map, channel_type, model_type, bar_height, actual_bar_height)
            # Squeeze and broadcast encoded array to match bar width
            encoded = np.squeeze(encoded)
            bar_width = bar_x_end - bar_x_start
            if encoded.ndim == 1:
                # Repeat 1D array across width dimension
                encoded = np.repeat(encoded[:, np.newaxis], bar_width, axis=1)
            image[bar_y_start:bar_y_end, bar_x_start:bar_x_end, channel_idx] = encoded

        return image

    def _update_parameters(self, params):
        return params

    def _get_extra(self, image, params):
        # Get obstruction bar dimensions based on image size
        dims = ImageDimensions(image.shape[1])
        _, y_start, _, y_end = self._get_final_bar_position(image)

        bar_height = dims.obstruction_bar_height
        actual_bar_height = y_end - y_start
        return [bar_height, actual_bar_height]



    def _get_area_mask(self, image, parameters, model_type) -> np.ndarray[Any, Any]:
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=bool)

        x_start, y_start, x_end, y_end = self._get_final_bar_position(image)
        mask[y_start:y_end, x_start:x_end] = True
        return mask

    def _get_final_bar_position(self, image) -> tuple[int, int, int, int]:
        height, width = image.shape[:2]

        # Get obstruction bar dimensions based on image size
        dims = ImageDimensions(width)
        x_start, y_start, x_end, y_end = dims.get_obstruction_bar_position()

        y_start = max(y_start, GRAPHICS_CONSTANTS.BORDER_PX)
        y_end = min(y_end, height - GRAPHICS_CONSTANTS.BORDER_PX)
        return (x_start, y_start, x_end, y_end)



    def _encode_all_channels(
        self,
        parameters: Dict[str, Any],
        channel_map: Dict[ChannelType, ParameterName],
        model_type: ModelType = ModelType.DF_DEFAULT,
        bar_height: int = 64,
        actual_bar_height: int = 64
    ) -> List[int | np.ndarray]:
        """
        Encode all 4 channels (RGBA) using list comprehension

        Args:
            parameters: Parameter dictionary
            channel_map: Mapping from channel types to parameter names
            model_type: Model type (for HSV default overrides)
            bar_height: Expected bar height
            actual_bar_height: Actual bar height

        Returns:
            List of 4 encoded values [R, G, B, A]
        """
        return [
            self._encode_channel(parameters, channel_map, channel_type, model_type, bar_height, actual_bar_height)
            for channel_type in [ChannelType.RED, ChannelType.GREEN, ChannelType.BLUE, ChannelType.ALPHA]
        ]

    def _encode_override(self, channel_type:ChannelType, bar_height:int,model_type:ModelType, is_using_default:bool)->np.ndarray | None:

        if (is_using_default and
            self._encoding_scheme == EncodingScheme.HSV and
            model_type is not None):
            override_key = (self._region_type, channel_type, model_type)
            if override_key in HSV_DEFAULT_PIXEL_OVERRIDES:
        
                px = HSV_DEFAULT_PIXEL_OVERRIDES[override_key]
                return np.array([px] * bar_height)[:, np.newaxis]
    

    def _encode_alpha(self,
        param_name:ParameterName, 
        param_value:float,
        channel_type: ChannelType,
        model_type: ModelType = ModelType.DF_DEFAULT,
        is_using_default:bool=True,
        bar_height: int = 64 )-> np.ndarray:

        # Check for HSV pixel override for alpha channel
        overriden = self._encode_override(channel_type, bar_height, 
                                          model_type, is_using_default)
        if overriden is not None:
            return overriden
        # Constant value across entire bar
        px = self._encode_parameter(param_name.value, param_value)
        return np.array([px] * bar_height)

    def _encode_channel(
        self,
        parameters: Dict[str, Any],
        channel_map: Dict[ChannelType, ParameterName],
        channel_type: ChannelType,
        model_type: ModelType = ModelType.DF_DEFAULT,
        bar_height: int = 64,
        actual_bar_height: int = 64
    ) -> np.ndarray:
        """
        Encode a single channel using the channel mapping (Template Method Pattern)

        Args:
            parameters: Parameter dictionary
            channel_map: Mapping from channel types to parameter names
            channel_type: Channel to encode (RED, GREEN, BLUE, or ALPHA)
            model_type: Model type (for HSV default overrides)
            bar_height: Expected bar height
            actual_bar_height: Actual bar height

        Returns:
            Encoded pixel intensity [0-255]
        """
        param_name = channel_map[channel_type]

        # Get parameter value with default
        param_value = self._get_parameter_with_default(parameters, param_name)
        if param_value is None:
            raise ValueError(f"Missing parameter '{param_name.value}' for obstruction bar")
        
        # Check if this is a default value and HSV encoding with an override
        is_using_default = param_name.value not in parameters

        # Alpha channel is constant, RGB channels vary per row
        if channel_type == ChannelType.ALPHA:
            return self._encode_alpha(param_name, param_value, channel_type, model_type, is_using_default, bar_height)
        
        # Check for HSV pixel override for alpha channel
        overriden = self._encode_override(channel_type, bar_height, 
                                        model_type, is_using_default)
        if overriden is not None:
            return overriden

        # Array values that vary per row
        normalized_array = self._normalize_parameter(param_value, bar_height)
        encoded = np.array([
            self._encode_parameter(param_name.value, val)
            for val in normalized_array[:actual_bar_height]
        ])[:, np.newaxis]

        return encoded

    def _validate_required_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Validate that required obstruction parameters are present (supports legacy names)

        Args:
            parameters: Parameter dictionary

        Raises:
            ValueError: If required parameters are missing
        """
        missing = validate_required_parameters(self._region_type, parameters)
        if missing:
            raise ValueError(f"Missing required obstruction bar parameters: {', '.join(missing)}")

    @staticmethod
    def _upsample_values(values: list, expected_length: int) -> list:
        """Upsample: Distribute values evenly when we have fewer values than needed"""
        pixels_per_value = expected_length / len(values)
        return [values[int(i / pixels_per_value)] for i in range(expected_length)]

    @staticmethod
    def _downsample_values(values: list, expected_length: int) -> list:
        """Downsample: Take evenly spaced values when we have more values than needed"""
        indices = np.linspace(0, len(values) - 1, expected_length).astype(int)
        return [values[i] for i in indices]

    @staticmethod
    def _use_exact_values(values: list, expected_length: int) -> list:
        """Exact match: Use values as-is"""
        return values

    @staticmethod
    def _replicate_scalar(value: Any, expected_length: int) -> list:
        """Replicate single scalar value across all pixels"""
        return [value] * expected_length

    def _normalize_parameter(
        self,
        value: Any,
        expected_length: int
    ) -> list:
        """
        Normalize parameter to list format, distributing values evenly across bar height

        Uses Strategy Pattern with map to select appropriate normalization strategy
        based on input type and length.

        Args:
            value: Single value or list/array
            expected_length: Expected length of the list (bar height in pixels)

        Returns:
            List of values with expected length
        """
        # Handle scalar values (non-list, non-array)
        if not isinstance(value, (list, np.ndarray)):
            return self._replicate_scalar(value, expected_length)

        # Convert to list for processing
        values = list(value)
        values_length = len(values)

        # Strategy map: length comparison -> normalization function (Strategy Pattern)
        # The comparison determines which strategy to use
        if values_length == expected_length:
            strategy = self._use_exact_values
        elif values_length < expected_length:
            strategy = self._upsample_values
        else:  # values_length > expected_length
            strategy = self._downsample_values

        return strategy(values, expected_length)
