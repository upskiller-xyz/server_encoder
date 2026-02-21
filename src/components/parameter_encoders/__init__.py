"""
Parameter encoders module for encoding parameter values to pixel intensities.

This module provides encoder classes for different parameter types including
linear, angular, and reflectance values.
"""

from src.components.parameter_encoders.linear_channel_encoder import LinearChannelEncoder
from src.components.parameter_encoders.angle_channel_encoder import AngleChannelEncoder
from src.components.parameter_encoders.reflectance_channel_encoder import ReflectanceChannelEncoder
from src.components.parameter_encoders.encoder_factory import EncoderFactory

__all__ = [
    'LinearChannelEncoder',
    'AngleChannelEncoder',
    'ReflectanceChannelEncoder',
    'EncoderFactory',
]
