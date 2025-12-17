from enum import Enum


class ModelType(Enum):
    """Model types for daylight prediction"""
    DF_DEFAULT = "df_default"
    DA_DEFAULT = "da_default"
    DF_CUSTOM = "df_custom"
    DA_CUSTOM = "da_custom"


class ChannelType(Enum):
    """RGBA channel types"""
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    ALPHA = "alpha"


class RegionType(Enum):
    """Image region types"""
    BACKGROUND = "background"
    ROOM = "room"
    WINDOW = "window"
    OBSTRUCTION_BAR = "obstruction_bar"


class ImageDimensions:
    """
    Image dimensions configuration (Singleton Pattern)

    All dimensions are calculated proportionally based on image_size
    """
    DEFAULT_IMAGE_SIZE = 128

    def __init__(self, image_size: int = DEFAULT_IMAGE_SIZE):
        """
        Initialize image dimensions

        Args:
            image_size: Size of the square image (width=height) in pixels
        """
        self._image_size = image_size

        # Calculate proportional dimensions
        # Base: 128x128 image
        # Obstruction bar: 4px wide × 64px tall
        scale = image_size / 128.0

        self._obstruction_bar_width = max(1, int(4 * scale))
        self._obstruction_bar_height = max(1, int(64 * scale))

    @property
    def image_size(self) -> int:
        """Get image size (width=height)"""
        return self._image_size

    @property
    def obstruction_bar_width(self) -> int:
        """Get obstruction bar width in pixels"""
        return self._obstruction_bar_width

    @property
    def obstruction_bar_height(self) -> int:
        """Get obstruction bar height in pixels"""
        return self._obstruction_bar_height

    def get_obstruction_bar_position(self) -> tuple:
        """
        Get obstruction bar position (centered on right edge)

        Returns:
            (x_start, y_start, x_end, y_end) tuple
        """
        x_start = self._image_size - self._obstruction_bar_width
        y_start = (self._image_size - self._obstruction_bar_height) // 2
        x_end = self._image_size
        y_end = y_start + self._obstruction_bar_height

        return (x_start, y_start, x_end, y_end)


class ParameterName(Enum):
    """Encoding parameter names"""
    # Background parameters
    WINDOW_ORIENTATION = "window_orientation"
    FACADE_REFLECTANCE = "facade_reflectance"
    FLOOR_HEIGHT_ABOVE_TERRAIN = "floor_height_above_terrain"
    TERRAIN_REFLECTANCE = "terrain_reflectance"

    # Room parameters
    CEILING_REFLECTANCE = "ceiling_reflectance"
    HEIGHT_ROOF_OVER_FLOOR = "height_roof_over_floor"
    FLOOR_REFLECTANCE = "floor_reflectance"
    WALL_REFLECTANCE = "wall_reflectance"

    # Room geometry parameters
    ROOM_POLYGON = "room_polygon"
    MODEL_TYPE = "model_type"
    PARAMETERS = "parameters"

    # Window parameters
    WINDOW_FRAME_REFLECTANCE = "window_frame_reflectance"
    WINDOW_SILL_HEIGHT = "window_sill_height"
    WINDOW_FRAME_RATIO = "window_frame_ratio"
    WINDOW_HEIGHT = "window_height"
    WINDOW_WIDTH = "window_width"

    # Window position parameters
    WINDOW_POSITION_X = "window_position_x"
    WINDOW_POSITION_Y = "window_position_y"
    WINDOW_POSITION_Z = "window_position_z"

    # Obstruction bar parameters
    BALCONY_REFLECTANCE = "balcony_reflectance"
    OBSTRUCTION_ANGLE_HORIZON = "obstruction_angle_horizon"
    CONTEXT_REFLECTANCE = "context_reflectance"
    OBSTRUCTION_ANGLE_ZENITH = "obstruction_angle_zenith"
    
    

    # Structure keys
    WINDOWS = "windows"
    WINDOW_GEOMETRY = "window_geometry"

    # Window coordinate keys
    X1 = "x1"
    Y1 = "y1"
    Z1 = "z1"
    X2 = "x2"
    Y2 = "y2"
    Z2 = "z2"
    DIRECTION_ANGLE = "direction_angle"
    WALL_THICKNESS = "wall_thickness"


# Validation map: RegionType -> List of required ParameterName values (Strategy Pattern)
# Note: window_sill_height and window_height are auto-calculated from window geometry
REQUIRED_PARAMETERS = {
    RegionType.BACKGROUND: [
        ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN
    ],
    RegionType.ROOM: [
        ParameterName.HEIGHT_ROOF_OVER_FLOOR,
        ParameterName.ROOM_POLYGON
    ],
    RegionType.WINDOW: [
        # window_sill_height - auto-calculated from min(z1,z2) - floor_height_above_terrain
        # window_height - auto-calculated from abs(z2 - z1)
        ParameterName.WINDOW_FRAME_RATIO,
    ],
    RegionType.OBSTRUCTION_BAR: [
        ParameterName.OBSTRUCTION_ANGLE_HORIZON,
        ParameterName.OBSTRUCTION_ANGLE_ZENITH
    ]
}

# Parameter categorization map: parameter name -> RegionType (Strategy Pattern)
PARAMETER_REGIONS = {
    # Background parameters
    ParameterName.FACADE_REFLECTANCE.value: RegionType.BACKGROUND,
    "facade_reflectance": RegionType.BACKGROUND,
    ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value: RegionType.BACKGROUND,
    "floor_height_above_terrain": RegionType.BACKGROUND,
    "terrain_reflectance": RegionType.BACKGROUND,
    ParameterName.WINDOW_ORIENTATION.value: RegionType.BACKGROUND,

    # Room parameters
    ParameterName.CEILING_REFLECTANCE.value: RegionType.ROOM,
    ParameterName.HEIGHT_ROOF_OVER_FLOOR.value: RegionType.ROOM,
    "height_roof_over_floor": RegionType.ROOM,
    ParameterName.FLOOR_REFLECTANCE.value: RegionType.ROOM,
    "horizontal_reflectance": RegionType.ROOM,
    ParameterName.WALL_REFLECTANCE.value: RegionType.ROOM,
    "vertical_reflectance": RegionType.ROOM,
    ParameterName.ROOM_POLYGON.value: RegionType.ROOM,

    # Window parameters
    ParameterName.WINDOW_FRAME_REFLECTANCE.value: RegionType.WINDOW,
    ParameterName.WINDOW_SILL_HEIGHT.value: RegionType.WINDOW,
    ParameterName.WINDOW_FRAME_RATIO.value: RegionType.WINDOW,
    ParameterName.WINDOW_HEIGHT.value: RegionType.WINDOW,
    ParameterName.WINDOW_GEOMETRY.value: RegionType.WINDOW,
    ParameterName.X1.value: RegionType.WINDOW,
    ParameterName.Y1.value: RegionType.WINDOW,
    ParameterName.Z1.value: RegionType.WINDOW,
    ParameterName.X2.value: RegionType.WINDOW,
    ParameterName.Y2.value: RegionType.WINDOW,
    ParameterName.Z2.value: RegionType.WINDOW,
    ParameterName.DIRECTION_ANGLE.value: RegionType.WINDOW,
    ParameterName.WALL_THICKNESS.value: RegionType.WINDOW,

    # Obstruction bar parameters
    ParameterName.BALCONY_REFLECTANCE.value: RegionType.OBSTRUCTION_BAR,
    ParameterName.BALCONY_REFLECTANCE.value: RegionType.OBSTRUCTION_BAR,
    ParameterName.OBSTRUCTION_ANGLE_HORIZON.value: RegionType.OBSTRUCTION_BAR,
   
    ParameterName.CONTEXT_REFLECTANCE.value: RegionType.OBSTRUCTION_BAR,
    "context_reflectance": RegionType.OBSTRUCTION_BAR,
    ParameterName.OBSTRUCTION_ANGLE_ZENITH.value: RegionType.OBSTRUCTION_BAR
}


class EncoderType(str, Enum):
    """Encoder type enumeration (Enumerator Pattern)"""
    LINEAR = "linear"
    ANGLE = "angle"
    REFLECTANCE = "reflectance"


class GeometryType(str, Enum):
    """Shapely geometry type enumeration (Enumerator Pattern)"""
    POLYGON = "Polygon"
    MULTI_POLYGON = "MultiPolygon"
    GEOMETRY_COLLECTION = "GeometryCollection"
    POINT = "Point"
    LINE_STRING = "LineString"
    MULTI_POINT = "MultiPoint"
    MULTI_LINE_STRING = "MultiLineString"


class EncodingScheme(str, Enum):
    """Encoding scheme enumeration for different parameter-to-channel mappings"""
    RGB = "rgb"
    HSV = "hsv"


# Default parameter values map (Strategy Pattern)
# These are the actual parameter values to use when not provided by the user
DEFAULT_PARAMETER_VALUES = {
    # Background defaults
    ParameterName.FACADE_REFLECTANCE: 1.0,
    ParameterName.TERRAIN_REFLECTANCE: 1.0,
    ParameterName.WINDOW_ORIENTATION: 288,  # 0° = South

    # Room defaults
    ParameterName.FLOOR_REFLECTANCE: 1.0,
    ParameterName.WALL_REFLECTANCE: 1.0,
    ParameterName.CEILING_REFLECTANCE: 1.0,

    # Window defaults
    ParameterName.WINDOW_FRAME_REFLECTANCE: 0.8,

    # Obstruction bar defaults
    ParameterName.CONTEXT_REFLECTANCE: 0.6,
    ParameterName.BALCONY_REFLECTANCE: 0.8,
}


# Default pixel values for HSV encoding scheme (Strategy Pattern)
# These override the encoded values for specific parameters when using default materials
# Format: {(RegionType, ChannelType, ModelType): pixel_value}
# Only specified combinations are overridden; others use normal encoding of DEFAULT_PARAMETER_VALUES
HSV_DEFAULT_PIXEL_OVERRIDES = {
    # Background region defaults (alpha=windowOrientation, hue=facadeReflectance, sat=floorHeight, val=terrainReflectance)
    (RegionType.BACKGROUND, ChannelType.ALPHA, ModelType.DF_DEFAULT): 190,  # window_orientation
    (RegionType.BACKGROUND, ChannelType.ALPHA, ModelType.DF_CUSTOM): 190,   # window_orientation
    (RegionType.BACKGROUND, ChannelType.RED, ModelType.DF_DEFAULT): 190,    # facade_reflectance (hue)
    (RegionType.BACKGROUND, ChannelType.RED, ModelType.DA_DEFAULT): 200,    # facade_reflectance (hue)
    (RegionType.BACKGROUND, ChannelType.BLUE, ModelType.DF_DEFAULT): 190,   # terrain_reflectance (value)
    (RegionType.BACKGROUND, ChannelType.BLUE, ModelType.DA_DEFAULT): 200,   # terrain_reflectance (value)

    # Obstruction bar defaults (alpha=balconyReflectance, hue=obstructionAngleHorizon, sat=contextReflectance, val=obstructionAngleZenith)
    (RegionType.OBSTRUCTION_BAR, ChannelType.ALPHA, ModelType.DF_DEFAULT): 210,   # balcony_reflectance
    (RegionType.OBSTRUCTION_BAR, ChannelType.ALPHA, ModelType.DA_DEFAULT): 210,   # balcony_reflectance
    (RegionType.OBSTRUCTION_BAR, ChannelType.ALPHA, ModelType.DF_CUSTOM): 210,    # balcony_reflectance
    (RegionType.OBSTRUCTION_BAR, ChannelType.ALPHA, ModelType.DA_CUSTOM): 210,    # balcony_reflectance
    (RegionType.OBSTRUCTION_BAR, ChannelType.GREEN, ModelType.DF_DEFAULT): 210,   # context_reflectance (saturation)
    (RegionType.OBSTRUCTION_BAR, ChannelType.GREEN, ModelType.DA_DEFAULT): 210,   # context_reflectance (saturation)
    (RegionType.OBSTRUCTION_BAR, ChannelType.GREEN, ModelType.DF_CUSTOM): 210,    # context_reflectance (saturation)
    (RegionType.OBSTRUCTION_BAR, ChannelType.GREEN, ModelType.DA_CUSTOM): 210,    # context_reflectance (saturation)

    # Room defaults (alpha=ceilingReflectance, hue=heightRoofOverFloor, sat=floorReflectance, val=wallReflectance)
    (RegionType.ROOM, ChannelType.ALPHA, ModelType.DF_DEFAULT): 220,    # ceiling_reflectance
    (RegionType.ROOM, ChannelType.ALPHA, ModelType.DA_DEFAULT): 220,    # ceiling_reflectance
    (RegionType.ROOM, ChannelType.ALPHA, ModelType.DF_CUSTOM): 220,     # ceiling_reflectance
    (RegionType.ROOM, ChannelType.ALPHA, ModelType.DA_CUSTOM): 220,     # ceiling_reflectance
    (RegionType.ROOM, ChannelType.GREEN, ModelType.DF_DEFAULT): 220,    # floor_reflectance (saturation)
    (RegionType.ROOM, ChannelType.GREEN, ModelType.DA_DEFAULT): 220,    # floor_reflectance (saturation)
    (RegionType.ROOM, ChannelType.GREEN, ModelType.DF_CUSTOM): 220,     # floor_reflectance (saturation)
    (RegionType.ROOM, ChannelType.GREEN, ModelType.DA_CUSTOM): 220,     # floor_reflectance (saturation)
    (RegionType.ROOM, ChannelType.BLUE, ModelType.DF_DEFAULT): 220,     # wall_reflectance (value)
    (RegionType.ROOM, ChannelType.BLUE, ModelType.DA_DEFAULT): 220,     # wall_reflectance (value)
    (RegionType.ROOM, ChannelType.BLUE, ModelType.DF_CUSTOM): 220,      # wall_reflectance (value)
    (RegionType.ROOM, ChannelType.BLUE, ModelType.DA_CUSTOM): 220,      # wall_reflectance (value)

    # Window defaults (alpha=frameReflectance, hue=sillHeight, sat=frameRatio, val=windowHeight)
    (RegionType.WINDOW, ChannelType.ALPHA, ModelType.DF_DEFAULT): 230,  # window_frame_reflectance
    (RegionType.WINDOW, ChannelType.ALPHA, ModelType.DA_DEFAULT): 230,  # window_frame_reflectance
    (RegionType.WINDOW, ChannelType.ALPHA, ModelType.DF_CUSTOM): 230,   # window_frame_reflectance
    (RegionType.WINDOW, ChannelType.ALPHA, ModelType.DA_CUSTOM): 230,   # window_frame_reflectance
}


# Channel mapping: defines which parameter goes into which channel for each region (Strategy Pattern)
# RGB Color Space Mapping (Legacy)
REGION_CHANNEL_MAPPING_RGB = {
    RegionType.BACKGROUND: {
        ChannelType.RED: ParameterName.FACADE_REFLECTANCE,
        ChannelType.GREEN: ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN,
        ChannelType.BLUE: ParameterName.TERRAIN_REFLECTANCE,
        ChannelType.ALPHA: ParameterName.WINDOW_ORIENTATION,
    },
    RegionType.ROOM: {
        ChannelType.RED: ParameterName.HEIGHT_ROOF_OVER_FLOOR,
        ChannelType.GREEN: ParameterName.FLOOR_REFLECTANCE,
        ChannelType.BLUE: ParameterName.WALL_REFLECTANCE,
        ChannelType.ALPHA: ParameterName.CEILING_REFLECTANCE,
    },
    RegionType.WINDOW: {
        ChannelType.RED: ParameterName.WINDOW_SILL_HEIGHT,
        ChannelType.GREEN: ParameterName.WINDOW_FRAME_RATIO,
        ChannelType.BLUE: ParameterName.WINDOW_HEIGHT,
        ChannelType.ALPHA: ParameterName.WINDOW_FRAME_REFLECTANCE,
    },
    RegionType.OBSTRUCTION_BAR: {
        ChannelType.RED: ParameterName.OBSTRUCTION_ANGLE_HORIZON,
        ChannelType.GREEN: ParameterName.CONTEXT_REFLECTANCE,
        ChannelType.BLUE: ParameterName.OBSTRUCTION_ANGLE_ZENITH,
        ChannelType.ALPHA: ParameterName.BALCONY_REFLECTANCE,
    },
}

# HSV Encoding Scheme Mapping (Default)
# Note: "HSV" refers to Hue/Saturation/Value parameter assignment, not color space conversion
# All channels remain RGBA in the actual image
REGION_CHANNEL_MAPPING_HSV = {
    RegionType.BACKGROUND: {
        ChannelType.ALPHA: ParameterName.WINDOW_ORIENTATION,  # alpha channel
        ChannelType.RED: ParameterName.FACADE_REFLECTANCE,  # hue → red channel
        ChannelType.GREEN: ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN,  # saturation → green channel
        ChannelType.BLUE: ParameterName.TERRAIN_REFLECTANCE,  # value → blue channel
    },
    RegionType.ROOM: {
        ChannelType.ALPHA: ParameterName.CEILING_REFLECTANCE,  # alpha channel
        ChannelType.RED: ParameterName.HEIGHT_ROOF_OVER_FLOOR,  # hue → red channel
        ChannelType.GREEN: ParameterName.FLOOR_REFLECTANCE,  # saturation → green channel
        ChannelType.BLUE: ParameterName.WALL_REFLECTANCE,  # value → blue channel
    },
    RegionType.WINDOW: {
        ChannelType.ALPHA: ParameterName.WINDOW_FRAME_REFLECTANCE,  # alpha channel
        ChannelType.RED: ParameterName.WINDOW_SILL_HEIGHT,  # hue → red channel
        ChannelType.GREEN: ParameterName.WINDOW_FRAME_RATIO,  # saturation → green channel
        ChannelType.BLUE: ParameterName.WINDOW_HEIGHT,  # value → blue channel
    },
    RegionType.OBSTRUCTION_BAR: {
        ChannelType.ALPHA: ParameterName.BALCONY_REFLECTANCE,  # alpha channel
        ChannelType.RED: ParameterName.OBSTRUCTION_ANGLE_HORIZON,  # hue → red channel
        ChannelType.GREEN: ParameterName.CONTEXT_REFLECTANCE,  # saturation → green channel
        ChannelType.BLUE: ParameterName.OBSTRUCTION_ANGLE_ZENITH,  # value → blue channel
    },
}

# Default mapping (backward compatibility - will be overridden)
REGION_CHANNEL_MAPPING = REGION_CHANNEL_MAPPING_RGB


# Encoding scheme mapping selector (Strategy Pattern)
ENCODING_SCHEME_MAPPINGS = {
    EncodingScheme.RGB: REGION_CHANNEL_MAPPING_RGB,
    EncodingScheme.HSV: REGION_CHANNEL_MAPPING_HSV,
}


def get_channel_mapping(encoding_scheme: EncodingScheme = EncodingScheme.RGB):
    """
    Get channel mapping for specified encoding scheme

    Args:
        encoding_scheme: Encoding scheme enum (default: HSV)

    Returns:
        Channel mapping dictionary for the encoding scheme
    """
    return ENCODING_SCHEME_MAPPINGS.get(encoding_scheme, REGION_CHANNEL_MAPPING_HSV)



