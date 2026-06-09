from src.core import RegionType, EncodingScheme
from src.components.region_encoders.background_region_encoder import BackgroundRegionEncoder
from src.components.region_encoders.base_region_encoder import BaseRegionEncoder
from src.components.region_encoders.room_region_encoder import RoomRegionEncoder
from src.components.region_encoders.window_region_encoder import WindowRegionEncoder
from src.components.region_encoders.obstruction_bar_encoder import ObstructionBarEncoder, V11ObstructionBarEncoder

# V3/V4 share V2's channel mappings for non-obstruction regions
_V2_EQUIVALENT_SCHEMES = frozenset({EncodingScheme.V3, EncodingScheme.V4})


class RegionEncoderFactory:
    """Factory for creating region encoders (Factory Pattern).

    A fresh encoder is created on every call. Region encoders carry mutable
    per-request state (BaseRegionEncoder._last_mask, set during encode_region
    and read back via get_last_mask), so they must NOT be cached/shared across
    concurrent requests - otherwise one request's room mask leaks into another's.
    Construction is cheap.
    """

    @classmethod
    def get_encoder(cls, region_type: RegionType, encoding_scheme: EncodingScheme = EncodingScheme.V2) -> BaseRegionEncoder:
        """
        Create a fresh region encoder for a region type and encoding scheme.

        V3 and V4 reuse V2 encoders for all non-obstruction regions; their obstruction
        behaviour is handled separately by ObstructionEncodingStrategy subclasses.

        Args:
            region_type: The region type
            encoding_scheme: The encoding scheme (default: V2)

        Returns:
            A new region encoder instance
        """
        # V3/V4 use V2 encoders for background/room/window regions
        effective_scheme = (
            EncodingScheme.V2 if encoding_scheme in _V2_EQUIVALENT_SCHEMES else encoding_scheme
        )

        obstruction_encoder = (
            V11ObstructionBarEncoder if effective_scheme == EncodingScheme.V11 else ObstructionBarEncoder
        )
        encoder_map = {
            RegionType.BACKGROUND: BackgroundRegionEncoder,
            RegionType.ROOM: RoomRegionEncoder,
            RegionType.WINDOW: WindowRegionEncoder,
            RegionType.OBSTRUCTION_BAR: obstruction_encoder,
        }

        encoder_class = encoder_map.get(region_type)
        if not encoder_class:
            raise ValueError(f"Unknown region type: {region_type}")

        return encoder_class(encoding_scheme=effective_scheme)
