from src.core import RegionType, EncodingScheme
from src.components.region_encoders.base_region_encoder import BaseRegionEncoder


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
