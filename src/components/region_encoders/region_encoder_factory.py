from typing import Dict

from src.core import RegionType, EncodingScheme
from src.components.region_encoders.background_region_encoder import BackgroundRegionEncoder
from src.components.region_encoders.base_region_encoder import BaseRegionEncoder
from src.components.region_encoders.room_region_encoder import RoomRegionEncoder
from src.components.region_encoders.window_region_encoder import WindowRegionEncoder
from src.components.region_encoders.obstruction_bar_encoder import ObstructionBarEncoder


class RegionEncoderFactory:
    """Factory for creating region encoders (Factory Pattern + Singleton per encoding scheme)"""

    _instances: Dict[tuple, BaseRegionEncoder] = {}

    @classmethod
    def get_encoder(cls, region_type: RegionType, encoding_scheme: EncodingScheme = EncodingScheme.RGB) -> BaseRegionEncoder:
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
