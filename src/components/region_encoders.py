from typing import Dict, Any, List
import numpy as np
import cv2
from src.components.interfaces import IRegionEncoder
from src.components.enums import ParameterName, RegionType, ModelType, ChannelType, ImageDimensions, REQUIRED_PARAMETERS, DEFAULT_PARAMETER_VALUES, REGION_CHANNEL_MAPPING, FACADE_ROTATION_MAP
from src.components.encoders import EncoderFactory
from src.components.geometry import RoomPolygon, WindowPosition, WindowGeometry





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

    def __init__(self, region_type: RegionType):
        self._region_type = region_type
        self._encoder_factory = EncoderFactory()

    def get_region_type(self) -> RegionType:
        """Get the region type"""
        return self._region_type

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


class BackgroundRegionEncoder(BaseRegionEncoder):
    """
    Encodes background region parameters

    Background fills entire image except obstruction bar, window, and room areas.

    CORRECTED CHANNEL MAPPINGS:
    - Red: facade_reflectance (0-1 → 0-1, default=1) [OPTIONAL]
    - Green: floor_height_above_terrain (0-10m → 0.1-1) [REQUIRED]
    - Blue: terrain_reflectance (0-1 → 0-1, default=1) [OPTIONAL]
    - Alpha: window_orientation (0-360° → 0-1, default=0° South) [OPTIONAL]
    """

    def __init__(self):
        super().__init__(RegionType.BACKGROUND)

    def encode_region(
        self,
        image: np.ndarray,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> np.ndarray:
        """
        Encode background region with corrected channel mappings

        Args:
            image: Image array (H, W, 4)
            parameters: Background parameters including:
                - floor_height_above_terrain: Required, 0-10m
                - facade_reflectance: Optional, 0-1, default=1
                - terrain_reflectance: Optional, 0-1, default=1
                - window_orientation: Optional, 0-360°, default=0.8
            model_type: Model type enum

        Returns:
            Updated image array
        """
        

        # Validate required parameters
        self._validate_required_parameters(parameters)

        height, width = image.shape[:2]

        # Get channel mapping for this region
        channel_map = REGION_CHANNEL_MAPPING[self._region_type]

        # Fill image with background, excluding 2-pixel border on top, bottom, and left
        # Right side has no border (obstruction bar extends to edge)
        # Background is drawn first, then room, window, and obstruction bar are drawn on top
        base = np.ones((*image.shape[:2], 1))

        # Encode channels using mapping (Strategy Pattern)
        channels = []
        for channel_type in [ChannelType.RED, ChannelType.GREEN, ChannelType.BLUE, ChannelType.ALPHA]:
            param_name = channel_map[channel_type]
            # Get parameter value with default
            
            param_value = parameters.get(
                param_name.value,
                DEFAULT_PARAMETER_VALUES.get(param_name, parameters.get(param_name.value))
            )
            
            encoded = self._encode_parameter(param_name.value, param_value)
            channels.append(base * encoded)
            
        return np.concatenate(channels, axis=2).astype(float)
    

    def _validate_required_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate required background parameters using list comprehension"""
        missing = validate_required_parameters(self._region_type, parameters)
        if missing:
            raise ValueError(f"Missing required background parameters: {', '.join(missing)}")


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

    def __init__(self):
        super().__init__(RegionType.ROOM)

    def encode_region(
        self,
        image: np.ndarray,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> np.ndarray:
        """
        Encode room region with corrected channel mappings

        Args:
            image: Image array (H, W, 4)
            parameters: Room parameters including:
                - height_roof_over_floor: Required, 0-30m
                - room_polygon: Optional, array of [x,y] coordinates in meters
                - horizontal_reflectance: Optional, 0-1, default=1
                - vertical_reflectance: Optional, 0-1, default=1
                - ceiling_reflectance: Optional, 0.5-1, default=1
            model_type: Model type enum

        Returns:
            Updated image array
        """
        

        # Validate required parameters
        self._validate_required_parameters(parameters)

        height, width = image.shape[:2]

        # Create room mask from polygon if provided
        room_mask = self._create_room_mask(image, parameters, model_type)

        # Get channel mapping for this region
        channel_map = REGION_CHANNEL_MAPPING[self._region_type]

        # Encode channels using mapping (Strategy Pattern)
        channels = []
        for channel_type in [ChannelType.RED, ChannelType.GREEN, ChannelType.BLUE, ChannelType.ALPHA]:
            param_name = channel_map[channel_type]
            # Get parameter value with default
            param_value = parameters.get(
                param_name.value,
                DEFAULT_PARAMETER_VALUES.get(param_name, parameters.get(param_name.value))
            )
            # Encode and append
            encoded = self._encode_parameter(param_name.value, param_value)
            channels.append(encoded)

        image[room_mask] = channels

        return image

    def _validate_required_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate required room parameters using list comprehension"""
        missing = validate_required_parameters(self._region_type, parameters)
        if missing:
            raise ValueError(f"Missing required room parameters: {', '.join(missing)}")

    def _create_room_mask(
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

            # Handle both list of dicts and RoomPolygon instance
            if isinstance(polygon_data, RoomPolygon):
                polygon = polygon_data
            else:
                polygon = RoomPolygon.from_dict(polygon_data)

            # Get window coordinates for positioning
            window_x1 = parameters.get(ParameterName.X1.value)
            window_y1 = parameters.get(ParameterName.Y1.value)
            window_x2 = parameters.get(ParameterName.X2.value)
            window_y2 = parameters.get(ParameterName.Y2.value)

            # Also check window_geometry dict
            if window_x1 is None and ParameterName.WINDOW_GEOMETRY.value in parameters:
                geom = parameters[ParameterName.WINDOW_GEOMETRY.value]
                window_x1 = geom.get(ParameterName.X1.value)
                window_y1 = geom.get(ParameterName.Y1.value)
                window_x2 = geom.get(ParameterName.X2.value)
                window_y2 = geom.get(ParameterName.Y2.value)

            # Note: Rotation is handled at a higher level (in image builder)
            # so polygon and window coordinates here are already rotated if needed

            # Create mask
            mask = np.zeros((height, width), dtype=np.uint8)
            pixel_coords = polygon.to_pixel_array(
                image_size=width,
                window_x1=window_x1,
                window_y1=window_y1,
                window_x2=window_x2,
                window_y2=window_y2
            )
            cv2.fillPoly(mask, pixel_coords, 1)

            # Enforce border
            self._enforce_border(mask, height, width)

            return mask.astype(bool)

        # Default: entire area except borders and obstruction bar
        border = 2
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
        border = 2
        mask[0:border, :] = 0  # Top 2 rows
        mask[height-border:height, :] = 0  # Bottom 2 rows
        mask[:, 0:border] = 0  # Left 2 columns


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

    def __init__(self):
        super().__init__(RegionType.WINDOW)

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
        # Calculate derived parameters (window_sill_height, window_height)
        # Strict mode (no logger) - will raise ValueError if calculation fails
        from src.components.encoding_service import ParameterCalculatorRegistry
        calculated_params = ParameterCalculatorRegistry.calculate_derived_parameters(
            parameters,
            logger=None  # Strict mode: raise on failure
        )
        parameters.update(calculated_params)

        # Validate required parameters
        self._validate_required_parameters(parameters)

        height, width = image.shape[:2]

        # Get window bounds from geometry
        x_start, y_start, x_end, y_end = self._get_window_bounds(
            image, parameters
        )

        # Get channel mapping for this region
        channel_map = REGION_CHANNEL_MAPPING[self._region_type]

        # Encode channels using mapping (Strategy Pattern)
        channels = []
        for channel_type in [ChannelType.RED, ChannelType.GREEN, ChannelType.BLUE, ChannelType.ALPHA]:
            param_name = channel_map[channel_type]
            # Get parameter value with default
            param_value = parameters.get(
                param_name.value,
                DEFAULT_PARAMETER_VALUES.get(param_name, parameters.get(param_name.value))
            )

            # Check if parameter is None (missing or failed calculation)
            if param_value is None:
                raise ValueError(
                    f"Window parameter '{param_name.value}' is missing or could not be calculated. "
                    f"Available parameters: {list(parameters.keys())}. "
                    f"For window_sill_height and window_height, ensure window_geometry "
                    f"(with z1, z2) and floor_height_above_terrain are provided."
                )

            # Encode and append
            encoded = self._encode_parameter(param_name.value, param_value)
            channels.append(encoded)

        image[y_start:y_end, x_start:x_end] = channels

        return image

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

        # Get window geometry
        if ParameterName.WINDOW_GEOMETRY.value in parameters:
            geom_data = parameters[ParameterName.WINDOW_GEOMETRY.value]

            # Handle WindowGeometry instance or dict
            if isinstance(geom_data, WindowGeometry):
                window_geom = geom_data
            else:
                window_geom = WindowGeometry.from_dict(geom_data)
        else:
            # Create from individual coordinates
            window_geom = WindowGeometry(
                x1=parameters[ParameterName.X1.value],
                y1=parameters[ParameterName.Y1.value],
                z1=parameters[ParameterName.Z1.value],
                x2=parameters[ParameterName.X2.value],
                y2=parameters[ParameterName.Y2.value],
                z2=parameters[ParameterName.Z2.value]
            )

        # Get pixel bounds from geometry
        x_start, y_start, x_end, y_end = window_geom.get_pixel_bounds(image_size=width)

        # Enforce 2-pixel border (must remain background)
        border = 2
        x_start = max(x_start, border)
        y_start = max(y_start, border)
        x_end = min(x_end, width - border)
        y_end = min(y_end, height - border)

        return (x_start, y_start, x_end, y_end)


class ObstructionBarEncoder(BaseRegionEncoder):
    """
    Encodes obstruction bar region parameters

    CORRECTED CHANNEL MAPPINGS:
    - Red: obstruction_angle_horizon (0-90° input → 0-1 normalized)
    - Green: context_reflectance (0.1-0.6 input → 0-1 normalized, default=1 if unobstructed)
    - Blue: obstruction_angle_zenith (0-70° input → 0.2-0.8 normalized)
    - Alpha: balcony_reflectance (0-1 input → 0-1 normalized, default=0.8)
    """

    def __init__(self):
        super().__init__(RegionType.OBSTRUCTION_BAR)

    def encode_region(
        self,
        image: np.ndarray,
        parameters: Dict[str, Any],
        model_type: ModelType
    ) -> np.ndarray:
        """
        Encode obstruction bar region

        Bar: 4 pixels wide × 64 pixels tall, centered vertically, at right edge
        Each row represents a different azimuth angle (±72.5° from façade normal)

        Channels (CORRECTED):
        - Red: obstruction_angle_horizon (0-90° → 0-1) [REQUIRED]
        - Green: context_reflectance (0.1-0.6 → 0-1, default=1) [OPTIONAL]
        - Blue: obstruction_angle_zenith (0-70° → 0.2-0.8) [REQUIRED]
        - Alpha: balcony_reflectance (0-1 → 0-1, default=0.8) [OPTIONAL]

        Raises:
            ValueError: If required parameters are missing
        """
        

        # Validate required parameters
        self._validate_required_parameters(parameters)

        height, width = image.shape[:2]

        # Get obstruction bar dimensions based on image size
        dims = ImageDimensions(width)
        bar_x_start, bar_y_start, bar_x_end, bar_y_end = dims.get_obstruction_bar_position()
        
        bar_height = dims.obstruction_bar_height

        # Get channel mapping for this region
        channel_map = REGION_CHANNEL_MAPPING[self._region_type]

        # Enforce 2-pixel border (must remain background)
        border = 2
        bar_y_start = max(bar_y_start, border)
        bar_y_end = min(bar_y_end, height - border)

        # Recalculate actual bar height after border enforcement
        actual_bar_height = bar_y_end - bar_y_start

        # Encode channels using mapping (Strategy Pattern)
        # Channels are ordered: RED, GREEN, BLUE, ALPHA (indices 0, 1, 2, 3)
        channel_order = [ChannelType.RED, ChannelType.GREEN, ChannelType.BLUE, ChannelType.ALPHA]

        for channel_idx, channel_type in enumerate(channel_order):
            param_name = channel_map[channel_type]

            # Get parameter value with default
            param_value = parameters.get(
                param_name.value,
                DEFAULT_PARAMETER_VALUES.get(param_name, parameters.get(param_name.value))
            )

            # Alpha channel is constant, RGB channels vary per row
            if channel_type == ChannelType.ALPHA:
                # Constant value across entire bar
                encoded = self._encode_parameter(param_name.value, param_value)
                image[bar_y_start:bar_y_end, bar_x_start:bar_x_end, channel_idx] = encoded
            else:
                # Array values that vary per row
                normalized_array = self._normalize_parameter(param_value, bar_height)
                encoded_array = np.array([
                    self._encode_parameter(param_name.value, val)
                    for val in normalized_array[:actual_bar_height]
                ])
                # Broadcast each row value across the bar width
                image[bar_y_start:bar_y_end, bar_x_start:bar_x_end, channel_idx] = encoded_array[:, np.newaxis]

        return image

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

    def _normalize_parameter(
        self,
        value: Any,
        expected_length: int
    ) -> list:
        """
        Normalize parameter to list format, distributing values evenly across bar height

        If array has fewer values than expected_length, each value is distributed
        over multiple pixels. For example:
        - 4 values over 64 pixels: each value covers 16 pixels
        - 64 values over 64 pixels: each value covers 1 pixel

        Args:
            value: Single value or list/array
            expected_length: Expected length of the list (bar height in pixels)

        Returns:
            List of values with expected length
        """
        if isinstance(value, (list, np.ndarray)):
            values = list(value)
            if len(values) == expected_length:
                # Perfect match: use as-is
                return values
            elif len(values) < expected_length:
                # Distribute values evenly across bar height
                # Each value covers multiple pixels
                result = []
                pixels_per_value = expected_length / len(values)
                for i in range(expected_length):
                    value_index = int(i / pixels_per_value)
                    result.append(values[value_index])
                return result
            else:
                # More values than pixels: downsample
                # Take evenly spaced values
                indices = np.linspace(0, len(values) - 1, expected_length).astype(int)
                return [values[i] for i in indices]
        else:
            # Single value: replicate for all rows
            return [value] * expected_length


class RegionEncoderFactory:
    """Factory for creating region encoders (Factory Pattern + Singleton)"""

    _instances: Dict[RegionType, IRegionEncoder] = {}

    @classmethod
    def get_encoder(cls, region_type: RegionType) -> IRegionEncoder:
        """
        Get encoder for a region type (Singleton pattern)

        Args:
            region_type: The region type

        Returns:
            Region encoder instance
        """
        if region_type not in cls._instances:
            encoder_map = {
                RegionType.BACKGROUND: BackgroundRegionEncoder,
                RegionType.ROOM: RoomRegionEncoder,
                RegionType.WINDOW: WindowRegionEncoder,
                RegionType.OBSTRUCTION_BAR: ObstructionBarEncoder,
            }

            encoder_class = encoder_map.get(region_type)
            if not encoder_class:
                raise ValueError(f"Unknown region type: {region_type}")

            cls._instances[region_type] = encoder_class()

        return cls._instances[region_type]
