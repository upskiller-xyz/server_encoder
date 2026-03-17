from typing import Dict

from src.core import RegionType, EncodingScheme
from src.components.region_encoders.background_region_encoder import BackgroundRegionEncoder
from src.components.region_encoders.base_region_encoder import BaseRegionEncoder
from src.components.region_encoders.room_region_encoder import RoomRegionEncoder
from src.components.region_encoders.window_region_encoder import WindowRegionEncoder
from src.components.region_encoders.obstruction_bar_encoder import ObstructionBarEncoder

# V3/V4 share V2's channel mappings for non-obstruction regions
_V2_EQUIVALENT_SCHEMES = frozenset({EncodingScheme.V3, EncodingScheme.V4})


class RegionEncoderFactory:
    """Factory for creating region encoders (Factory Pattern + Singleton per encoding scheme)"""

    _instances: Dict[tuple, BaseRegionEncoder] = {}

    @classmethod
    def get_encoder(cls, region_type: RegionType, encoding_scheme: EncodingScheme = EncodingScheme.V2) -> BaseRegionEncoder:
        """
        Get encoder for a region type and encoding scheme (Singleton pattern per encoding scheme).

        V3 and V4 reuse V2 encoders for all non-obstruction regions; their obstruction
        behaviour is handled separately by ObstructionEncodingStrategy subclasses.

        Args:
            region_type: The region type
            encoding_scheme: The encoding scheme (default: V2)

        Returns:
            Region encoder instance
        """
        # V3/V4 use V2 encoders for background/room/window regions
        effective_scheme = (
            EncodingScheme.V2 if encoding_scheme in _V2_EQUIVALENT_SCHEMES else encoding_scheme
        )
        cache_key = (region_type, effective_scheme)

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

            cls._instances[cache_key] = encoder_class(encoding_scheme=effective_scheme)

        return cls._instances[cache_key]
