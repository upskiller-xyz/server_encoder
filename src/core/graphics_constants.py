"""
Graphics Constants for Image Encoding

Centralized location for all magic numbers related to image rendering.
Following the Single Responsibility Principle and avoiding magic numbers in code.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageGraphicsConstants:
    """
    Immutable constants for image graphics rendering (Immutable Object Pattern)

    All measurements are in their natural units:
    - Pixels for image-space measurements
    - Meters for world-space measurements
    """

    # Window positioning
    WINDOW_OFFSET_PX: int = 12  # pixels from right edge of image
    WALL_THICKNESS_M: float = 0.3  # meters, outer wall thickness

    # Image borders
    BORDER_PX: int = 2  # pixel border that must remain background

    # Obstruction bar spacing
    OBSTRUCTION_BAR_GAP_PX: int = 3  # pixels gap between room and obstruction bar

    # Resolution and scaling
    BASE_RESOLUTION_M_PER_PX: float = 0.1  # 10cm per pixel at 128x128
    BASE_IMAGE_SIZE_PX: int = 128

    # Room positioning
    ROOM_FACADE_OFFSET_PX: int = 1  # pixels offset from window for C-frame adjacency

    # Window placement tolerance (meters) for edge matching
    WINDOW_PLACEMENT_TOLERANCE: float = 0.05
    WINDOW_HEIGHT_TOLERANCE:float =  1e-6

    @classmethod
    def get_pixel_value(cls, value:float, 
                        image_size:int=BASE_IMAGE_SIZE_PX):
        
        return round(value / cls.get_resolution(image_size))

    @classmethod
    def get_resolution(cls, image_size: int) -> float:
        """
        Calculate resolution (meters per pixel) for given image size

        Args:
            image_size: Image dimension in pixels

        Returns:
            Resolution in meters per pixel
        """
        scale = image_size / cls.BASE_IMAGE_SIZE_PX
        return cls.BASE_RESOLUTION_M_PER_PX / scale

    @classmethod
    def get_scale_factor(cls, image_size: int) -> float:
        """
        Calculate scale factor relative to base image size

        Args:
            image_size: Image dimension in pixels

        Returns:
            Scale factor (1.0 for 128x128, 2.0 for 256x256, etc.)
        """
        return image_size / cls.BASE_IMAGE_SIZE_PX


# Singleton instance for easy access
GRAPHICS_CONSTANTS = ImageGraphicsConstants()
