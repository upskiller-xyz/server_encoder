from typing import Dict, Any
import numpy as np
import cv2
from src.core import RegionType, ModelType, ParameterName, ImageDimensions, EncodingScheme
from src.components.region_encoders.base_region_encoder import BaseRegionEncoder
from src.components.geometry import RoomPolygon, WindowGeometry
from src.core import GRAPHICS_CONSTANTS


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

            cv2.fillPoly(mask, pixel_coords, 1) # type: ignore

            # Enforce border
            mask = self._enforce_border(mask, height, width)

            return mask.astype(bool)

        # Default: entire area except borders and obstruction bar
        border = GRAPHICS_CONSTANTS.BORDER_PX
        dims = ImageDimensions(width)
        bar_x_start, _, _, _ = dims.get_obstruction_bar_position()
        mask = np.zeros((height, width), dtype=bool)
        mask[border:height-border, border:bar_x_start] = True

        return mask

    def _enforce_border(self, mask: np.ndarray, height: int, width: int) -> np.ndarray:
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
        return mask
