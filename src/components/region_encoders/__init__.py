"""
Region encoders module for encoding different regions of window images.

This module provides encoders for different regions including background,
room, window, and obstruction bar regions.
"""

from src.components.region_encoders.validation_helpers import validate_required_parameters
from src.components.region_encoders.base_region_encoder import BaseRegionEncoder
from src.components.region_encoders.background_region_encoder import BackgroundRegionEncoder
from src.components.region_encoders.room_region_encoder import RoomRegionEncoder
from src.components.region_encoders.window_region_encoder import WindowRegionEncoder
from src.components.region_encoders.obstruction_bar_encoder import ObstructionBarEncoder
from src.components.region_encoders.region_encoder_factory import RegionEncoderFactory

__all__ = [
    'validate_required_parameters',
    'BaseRegionEncoder',
    'BackgroundRegionEncoder',
    'RoomRegionEncoder',
    'WindowRegionEncoder',
    'ObstructionBarEncoder',
    'RegionEncoderFactory',
]
