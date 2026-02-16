from typing import Dict, Any, List
import math
import numpy as np
import cv2
from src.components.interfaces import IRegionEncoder
from src.components.enums import ParameterName, RegionType, ModelType, ChannelType, ImageDimensions, REQUIRED_PARAMETERS, DEFAULT_PARAMETER_VALUES, REGION_CHANNEL_MAPPING, EncodingScheme, get_channel_mapping, HSV_DEFAULT_PIXEL_OVERRIDES
from src.components.encoders import EncoderFactory
from src.components.geometry import RoomPolygon, WindowGeometry
from src.components.graphics_constants import GRAPHICS_CONSTANTS
from src.components.parameter_calculators import ParameterCalculatorRegistry


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


class BaseRegionEncoder(IRegionEncoder):
    """Base class for region encoders with common functionality"""

    def __init__(self, region_type: RegionType, encoding_scheme: EncodingScheme = EncodingScheme.RGB):
        self._region_type = region_type
        self._encoder_factory = EncoderFactory()
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
        
    def _get_area_mask(self, image, parameters, model_type)->np.ndarray[Any, Any]:
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
        return parameters.get(
            param_name.value,
            DEFAULT_PARAMETER_VALUES.get(param_name, parameters.get(param_name.value))
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


class BackgroundRegionEncoder(BaseRegionEncoder):
    """
    Encodes background region parameters

    Background fills entire image except obstruction bar, window, and room areas.

    CORRECTED CHANNEL MAPPINGS:
    - Red: facade_reflectance (0-1 → 0-1, default=1) [OPTIONAL]
    - Green: floor_height_above_terrain (0-10m → 0.1-1) [REQUIRED]
    - Blue: terrain_reflectance (0-1 → 0-1, default=1) [OPTIONAL]
    - Alpha: window_orientation (0-2π rad → 0-1, math convention 0=East CCW) [AUTO from direction_angle]
    """

    def __init__(self, encoding_scheme: EncodingScheme = EncodingScheme.RGB):
        super().__init__(RegionType.BACKGROUND, encoding_scheme)


class RoomRegionEncoder(BaseRegionEncoder):
    """
    Encodes room region parameters

    Room polygon construction:
    - Receives room_polygon as array of coordinates [[x,y], [x,y]..] in meters
    - 1 pixel = 0.1m (10cm) for 128x128, scales proportionally
    - Polygon positioned so rightmost side aligns with left edge of window area
    - Window is at 12 pixels from right edge (+ wall thickness)

    CORRECTED CHANNEL MAPPINGS:
    - Red: height_roof_over_floor (0-30m → 0-1) [REQUIRED]
    - Green: horizontal_reflectance (0-1 → 0-1, default=1) [OPTIONAL]
    - Blue: vertical_reflectance (0-1 → 0-1, default=1) [OPTIONAL]
    - Alpha: ceiling_reflectance (0.5-1 → 0-1, default=1) [OPTIONAL]
    """

    def __init__(self, encoding_scheme: EncodingScheme = EncodingScheme.RGB):
        super().__init__(RegionType.ROOM, encoding_scheme)

    def _get_area_mask(
        self,
        image: np.ndarray,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> np.ndarray:
        """
        Create binary mask for room polygon

        Room polygon is positioned relative to window coordinates.
        The rightmost side of the room polygon (where window is located)
        should align exactly with the left edge of the window area.

        For DA models, the mask extends to include the C-frame area
        (gap between window and obstruction bar) so the alpha channel
        uses ceiling_reflectance instead of window_orientation.

        Args:
            image: Image array
            parameters: Parameters including room_polygon and window coordinates
            model_type: Model type (DA_DEFAULT, DA_CUSTOM, DF_DEFAULT, DF_CUSTOM)

        Returns:
            Boolean mask array where True indicates room area
        """
        height, width = image.shape[:2]
        # If room_polygon provided, use it
        room_polygon_key = ParameterName.ROOM_POLYGON.value
        if room_polygon_key in parameters and parameters[room_polygon_key]:
            polygon_data = parameters[room_polygon_key]
            # Normalize to RoomPolygon if not already
            if isinstance(polygon_data, RoomPolygon):
                polygon: RoomPolygon = polygon_data
            else:
                polygon: RoomPolygon = RoomPolygon.from_dict(polygon_data)

            # Get window coordinates for positioning
            window_x1 = parameters.get(ParameterName.X1.value)
            window_y1 = parameters.get(ParameterName.Y1.value)
            window_x2 = parameters.get(ParameterName.X2.value)
            window_y2 = parameters.get(ParameterName.Y2.value)
            direction_angle = parameters.get(ParameterName.DIRECTION_ANGLE.value)

            # Also check window_geometry (already normalized to WindowGeometry class at entry point)
            if window_x1 is None and ParameterName.WINDOW_GEOMETRY.value in parameters:
                geom: WindowGeometry = parameters[ParameterName.WINDOW_GEOMETRY.value]
                window_x1 = geom.x1
                window_y1 = geom.y1
                window_x2 = geom.x2
                window_y2 = geom.y2
                if direction_angle is None:
                    direction_angle = geom.calculate_direction_from_polygon(polygon)


            # Note: Rotation is handled at a higher level (in image builder)
            # so polygon and window coordinates here are already rotated if needed

            # Create mask
            mask = np.zeros((height, width), dtype=np.uint8)
            pixel_coords = polygon.to_pixel_array(
                window_x1=window_x1,
                window_y1=window_y1,
                window_x2=window_x2,
                window_y2=window_y2,
                image_size=width,
                direction_angle=direction_angle
            )
            
            cv2.fillPoly(mask, pixel_coords, 1)

            # Enforce border
            self._enforce_border(mask, height, width)

            return mask.astype(bool)

        # Default: entire area except borders and obstruction bar
        border = GRAPHICS_CONSTANTS.BORDER_PX
        dims = ImageDimensions(width)
        bar_x_start, _, _, _ = dims.get_obstruction_bar_position()
        mask = np.zeros((height, width), dtype=bool)
        mask[border:height-border, border:bar_x_start] = True

        return mask

    def _enforce_border(self, mask: np.ndarray, height: int, width: int) -> None:
        """
        Enforce 2-pixel border by zeroing out border pixels in mask

        Args:
            mask: Binary mask array to modify in-place
            height: Image height
            width: Image width
        """
        border = GRAPHICS_CONSTANTS.BORDER_PX
        mask[0:border, :] = 0  # Top rows
        mask[height-border:height, :] = 0  # Bottom rows
        mask[:, 0:border] = 0  # Left columns


class WindowRegionEncoder(BaseRegionEncoder):
    """
    Encodes window region parameters

    Window is viewed from top (plan view):
    - Located 12 pixels from right edge (8 pixels from obstruction bar)
    - Appears as vertical line/rectangle
    - Width (horizontal) = wall thickness (~0.3m = 3px)
    - Height (vertical) = window width in 3D space (x2-x1)

    CORRECTED CHANNEL MAPPINGS:
    - Red: sill_height (0-5m input → 0-1 normalized)
    - Green: frame_ratio (1-0 input → 0-1 normalized, REVERSED)
    - Blue: window_height (0.2-5m input → 0.99-0.01 normalized, REVERSED)
    - Alpha: window_frame_reflectance (0-1 input → 0-1 normalized, optional, default=0.8)
    """

    def __init__(self, encoding_scheme: EncodingScheme = EncodingScheme.RGB):
        super().__init__(RegionType.WINDOW, encoding_scheme)  
    
    def _update_parameters(self, params):
        calculated_params = ParameterCalculatorRegistry.calculate_derived_parameters(
            params,
            logger=None  # Strict mode: raise on failure
        )
        params.update(calculated_params)
        return params
    
        
    def _get_area_mask(self, image, parameters, model_type)->np.ndarray[Any, Any]:
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        x_start, y_start, x_end, y_end = self._get_window_bounds(
            image, parameters
        )
        mask[y_start:y_end, x_start:x_end] = True
        return mask

    def _validate_required_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate required window parameters using list comprehension"""
        missing = validate_required_parameters(self._region_type, parameters)

        # Also need window geometry (either window_geometry or individual coords)
        has_geometry = (
            ParameterName.WINDOW_GEOMETRY.value in parameters or
            all(key in parameters for key in [
                ParameterName.X1.value, ParameterName.Y1.value, ParameterName.Z1.value,
                ParameterName.X2.value, ParameterName.Y2.value, ParameterName.Z2.value
            ])
        )

        if not has_geometry:
            missing.append("window geometry (x1,y1,z1,x2,y2,z2 or window_geometry)")

        if missing:
            raise ValueError(f"Missing required window parameters: {', '.join(missing)}")

    def _get_window_bounds(
        self,
        image: np.ndarray,
        parameters: Dict[str, Any]
    ) -> tuple:
        """
        Get window bounds in pixels from geometry

        Args:
            image: Image array
            parameters: Parameters including window geometry

        Returns:
            (x_start, y_start, x_end, y_end) tuple
        """
        height, width = image.shape[:2]

        # Get window geometry (already normalized to WindowGeometry class at entry point)
        if ParameterName.WINDOW_GEOMETRY.value in parameters:
            window_geom: WindowGeometry = parameters[ParameterName.WINDOW_GEOMETRY.value]
        else:
            # Create from individual coordinates
            window_geom = WindowGeometry(
                x1=parameters[ParameterName.X1.value],
                y1=parameters[ParameterName.Y1.value],
                z1=parameters[ParameterName.Z1.value],
                x2=parameters[ParameterName.X2.value],
                y2=parameters[ParameterName.Y2.value],
                z2=parameters[ParameterName.Z2.value],
                direction_angle=parameters.get(ParameterName.DIRECTION_ANGLE.value, 0)
            )

        # Get pixel bounds from geometry
        x_start, y_start, x_end, y_end = window_geom.get_pixel_bounds(image_size=width)

        # Snap window to room's facade edge if available (preserves width, eliminates gap)
        room_facade_right_edge = parameters.get('_room_facade_right_edge')
        if room_facade_right_edge is not None:
            window_width = x_end - x_start
            x_start = room_facade_right_edge + 1
            x_end = x_start + window_width

        # Enforce border (must remain background)
        border = GRAPHICS_CONSTANTS.BORDER_PX
        x_start = max(x_start, border)
        y_start = max(y_start, border)
        x_end = min(x_end, width - border)
        y_end = min(y_end, height - border)

        return (x_start, y_start, x_end, y_end)


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
            param_name = channel_map[channel_type]
            param_value = self._get_parameter_with_default(parameters, param_name)

            if param_value is None:
                raise ValueError(f"Missing parameter '{param_name.value}' for obstruction bar")

            # Check if using default value for HSV override
            is_using_default = param_name.value not in parameters

            # Alpha channel is constant, RGB channels vary per row
            if channel_type == ChannelType.ALPHA:
                # Check for HSV pixel override
                if (is_using_default and
                    self._encoding_scheme == EncodingScheme.HSV and
                    model_type is not None):
                    override_key = (self._region_type, channel_type, model_type)
                    if override_key in HSV_DEFAULT_PIXEL_OVERRIDES:
                        encoded = HSV_DEFAULT_PIXEL_OVERRIDES[override_key]
                        image[bar_y_start:bar_y_end, bar_x_start:bar_x_end, channel_idx] = encoded
                        continue

                encoded = self._encode_parameter(param_name.value, param_value)
                image[bar_y_start:bar_y_end, bar_x_start:bar_x_end, channel_idx] = encoded
            else:
                # Check for HSV pixel override for RGB channels
                if (is_using_default and
                    self._encoding_scheme == EncodingScheme.HSV and
                    model_type is not None):
                    override_key = (self._region_type, channel_type, model_type)
                    if override_key in HSV_DEFAULT_PIXEL_OVERRIDES:
                        px = HSV_DEFAULT_PIXEL_OVERRIDES[override_key]
                        encoded = np.array([px] * actual_bar_height)[:, np.newaxis]
                        image[bar_y_start:bar_y_end, bar_x_start:bar_x_end, channel_idx] = encoded
                        continue

                # Array values that vary per row
                normalized_array = self._normalize_parameter(param_value, bar_height)
                encoded = np.array([
                    self._encode_parameter(param_name.value, val)
                    for val in normalized_array[:actual_bar_height]
                ])[:, np.newaxis]
                # Broadcast each row value across the bar width
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

   
        
    def _get_area_mask(self, image, parameters, model_type)->np.ndarray[Any, Any]:
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=bool)

        x_start, y_start, x_end, y_end = self._get_final_bar_position(image)
        mask[y_start:y_end, x_start:x_end] = True
        return mask
    
    def _get_final_bar_position(self, image)->tuple[int, int, int, int]:
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
        model_type: ModelType = None,
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

    def _encode_channel(
        self,
        parameters: Dict[str, Any],
        channel_map: Dict[ChannelType, ParameterName],
        channel_type: ChannelType,
        model_type: ModelType = None,
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
        param_value = parameters.get(
            param_name.value,
            DEFAULT_PARAMETER_VALUES.get(param_name, parameters.get(param_name.value))
        )

        # Check if this is a default value and HSV encoding with an override
        is_using_default = param_name.value not in parameters

        # Alpha channel is constant, RGB channels vary per row
        if channel_type == ChannelType.ALPHA:
            # Check for HSV pixel override for alpha channel
            if (is_using_default and
                self._encoding_scheme == EncodingScheme.HSV and
                model_type is not None):
                override_key = (self._region_type, channel_type, model_type)
                if override_key in HSV_DEFAULT_PIXEL_OVERRIDES:
                    px = HSV_DEFAULT_PIXEL_OVERRIDES[override_key]
                    encoded = np.array([px] * bar_height)
                    return encoded

            # Constant value across entire bar
            px = self._encode_parameter(param_name.value, param_value)
            encoded = np.array([px] * bar_height)
        else:
            # Check for HSV pixel override for RGB channels (less common, but supported)
            if (is_using_default and
                self._encoding_scheme == EncodingScheme.HSV and
                model_type is not None):
                override_key = (self._region_type, channel_type, model_type)
                if override_key in HSV_DEFAULT_PIXEL_OVERRIDES:
                    px = HSV_DEFAULT_PIXEL_OVERRIDES[override_key]
                    encoded = np.array([px] * actual_bar_height)[:, np.newaxis]
                    return encoded

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


class RegionEncoderFactory:
    """Factory for creating region encoders (Factory Pattern + Singleton per encoding scheme)"""

    _instances: Dict[tuple, IRegionEncoder] = {}

    @classmethod
    def get_encoder(cls, region_type: RegionType, encoding_scheme: EncodingScheme = EncodingScheme.RGB) -> IRegionEncoder:
        """
        Get encoder for a region type and encoding scheme (Singleton pattern per encoding scheme)

        Args:
            region_type: The region type
            encoding_scheme: The encoding scheme (default: HSV)

        Returns:
            Region encoder instance
        """
        cache_key = (region_type, encoding_scheme)

        if cache_key not in cls._instances:
            encoder_map = {
                RegionType.BACKGROUND: BackgroundRegionEncoder,
                RegionType.ROOM: RoomRegionEncoder,
                RegionType.WINDOW: WindowRegionEncoder,
                RegionType.OBSTRUCTION_BAR: ObstructionBarEncoder,
            }

            encoder_class = encoder_map.get(region_type)
            if not encoder_class:
                raise ValueError(f"Unknown region type: {region_type}")

            cls._instances[cache_key] = encoder_class(encoding_scheme=encoding_scheme)

        return cls._instances[cache_key]
