"""
V5 image director: single-channel float32 geometric mask.

Instead of encoding room parameters into RGBA channels, V5 produces a
(H, W, 1) float32 image where each pixel is assigned a fixed value based
on which region it belongs to:

    Background → 0.0
    Room       → 1.0
    Window     → 0.6

No obstruction bar, no parameter encoding.
"""
from typing import Dict, Any, Optional, Tuple
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from src.core import ModelType, ParameterName, RegionType, V5_MASK_VALUES
from src.core.graphics_constants import GRAPHICS_CONSTANTS
from src.components.geometry import RoomPolygon, WindowGeometry
from src.components.image_builder.parameter_normalizer import ParameterNormalizer
from src.components.image_builder.geometry_rotator import GeometryRotator
from src.components.region_encoders.window_region_encoder import WindowRegionEncoder
from src.core.enums import EncodingScheme
from src.models import EncodingParameters, EncodingResult
from src.core.enums import PARAMETER_REGIONS


class V5ImageDirector:
    """
    Director for V5 encoding: single-channel float32 geometric mask.

    Produces a (H, W, 1) float32 image. Room polygon area is filled with 1.0,
    window area with 0.6, background remains 0.0. No obstruction bar.
    """

    IMAGE_SIZE = 128

    # Reuse a single WindowRegionEncoder instance for pixel-bound calculation
    _window_encoder: WindowRegionEncoder = WindowRegionEncoder(EncodingScheme.V2)

    def construct_full_image(
        self,
        model_type: ModelType,
        all_parameters: EncodingParameters,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Build the V5 mask image.

        Args:
            model_type: Unused for V5 (no parameter encoding), kept for interface parity.
            all_parameters: Grouped encoding parameters.

        Returns:
            Tuple of (image, room_mask):
            - image: (H, W, 1) float32 array in range [0, 1]
            - room_mask: (H, W) uint8 binary mask (1 = room, 0 = outside)
        """
        image = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE, 1), dtype=np.float32)

        all_parameters = self._rotate_geometry_if_needed(all_parameters)

        room_mask = self._draw_room(image, all_parameters)
        self._draw_window(image, all_parameters)

        return image, room_mask

    # ------------------------------------------------------------------
    # Region drawing helpers
    # ------------------------------------------------------------------

    def _draw_room(
        self, image: np.ndarray, all_parameters: EncodingParameters
    ) -> Optional[np.ndarray]:
        """Fill the room polygon area with V5_MASK_VALUES[ROOM] (1.0)."""
        room_params = all_parameters.room.parameters
        if not room_params or ParameterName.ROOM_POLYGON.value not in room_params:
            return None

        h, w = image.shape[:2]
        polygon_data = room_params[ParameterName.ROOM_POLYGON.value]
        polygon = (
            polygon_data
            if isinstance(polygon_data, RoomPolygon)
            else RoomPolygon.from_dict(polygon_data)
        )

        window_x1 = room_params.get(ParameterName.X1.value)
        window_y1 = room_params.get(ParameterName.Y1.value)
        window_x2 = room_params.get(ParameterName.X2.value)
        window_y2 = room_params.get(ParameterName.Y2.value)
        direction_angle = room_params.get(ParameterName.DIRECTION_ANGLE.value)

        if window_x1 is None and ParameterName.WINDOW_GEOMETRY.value in room_params:
            geom: WindowGeometry = room_params[ParameterName.WINDOW_GEOMETRY.value]
            window_x1, window_y1 = geom.x1, geom.y1
            window_x2, window_y2 = geom.x2, geom.y2
            if direction_angle is None:
                direction_angle = geom.calculate_direction_from_polygon(polygon)

        pixel_coords = polygon.to_pixel_array(
            window_x1=window_x1,
            window_y1=window_y1,
            window_x2=window_x2,
            window_y2=window_y2,
            image_size=w,
            direction_angle=direction_angle,
        )

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, pixel_coords, 1)  # type: ignore

        # Enforce border (all four edges must remain background)
        border = GRAPHICS_CONSTANTS.BORDER_PX
        mask[:border, :] = 0
        mask[h - border :, :] = 0
        mask[:, :border] = 0
        mask[:, w - border :] = 0

        image[mask.astype(bool)] = V5_MASK_VALUES[RegionType.ROOM]
        return mask

    def _draw_window(self, image: np.ndarray, all_parameters: EncodingParameters) -> None:
        """Fill the window area with V5_MASK_VALUES[WINDOW] (0.6)."""
        window_params = all_parameters.window.parameters
        if not window_params:
            return

        h, w = image.shape[:2]
        dummy_rgba = np.zeros((h, w, 4), dtype=np.uint8)

        try:
            updated = self._window_encoder._update_parameters(dict(window_params))
            x_start, y_start, x_end, y_end = self._window_encoder._get_window_bounds(
                dummy_rgba, updated
            )
            image[y_start:y_end, x_start:x_end] = V5_MASK_VALUES[RegionType.WINDOW]
        except (KeyError, ValueError) as exc:
            logger.warning("V5 window drawing skipped: %s", exc)

    # ------------------------------------------------------------------
    # Geometry rotation (mirrors RoomImageDirector logic)
    # ------------------------------------------------------------------

    def _rotate_geometry_if_needed(
        self, all_parameters: EncodingParameters
    ) -> EncodingParameters:
        """Rotate room polygon and window coordinates when window is not on south facade."""
        if not all_parameters.window.parameters:
            return all_parameters

        window_geom = ParameterNormalizer.normalize_window_geometry(
            all_parameters.window.parameters
        )
        if window_geom is None:
            return all_parameters

        room_polygon = (
            ParameterNormalizer.normalize_room_polygon(all_parameters.room.parameters)
            if all_parameters.room.parameters
            else None
        )

        if window_geom.direction_angle is None and room_polygon is not None:
            try:
                calculated_angle = window_geom.calculate_direction_from_polygon(room_polygon)
                window_geom = WindowGeometry(
                    x1=window_geom.x1, y1=window_geom.y1, z1=window_geom.z1,
                    x2=window_geom.x2, y2=window_geom.y2, z2=window_geom.z2,
                    direction_angle=calculated_angle,
                )
            except ValueError:
                pass

        direction_angle = window_geom.direction_angle
        all_parameters.set_global(ParameterName.DIRECTION_ANGLE.value, direction_angle)
        all_parameters.window[ParameterName.DIRECTION_ANGLE.value] = direction_angle
        if all_parameters.room.parameters:
            all_parameters.room[ParameterName.DIRECTION_ANGLE.value] = direction_angle

        return GeometryRotator.rotate_if_needed(all_parameters, window_geom, room_polygon)

    # ------------------------------------------------------------------
    # Flat-parameter and multi-window entry points (same contract as RoomImageDirector)
    # ------------------------------------------------------------------

    def construct_from_flat_parameters(
        self,
        model_type: ModelType,
        parameters: Dict[str, Any],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Construct image from a flat parameter dictionary."""
        windows_key = ParameterName.WINDOWS.value
        if windows_key in parameters and isinstance(parameters[windows_key], dict):
            result = self.construct_multi_window_images(model_type, parameters)
            return result.get_first_image(), result.get_first_mask()

        grouped = self._group_parameters(parameters)
        return self.construct_full_image(model_type, grouped)

    def construct_multi_window_images(
        self,
        model_type: ModelType,
        parameters: Dict[str, Any],
    ) -> EncodingResult:
        """Construct one V5 image per window."""
        result = EncodingResult()
        windows_key = ParameterName.WINDOWS.value

        if windows_key not in parameters:
            image, mask = self.construct_from_flat_parameters(model_type, parameters)
            result.add_window("window_1", image, mask)
            return result

        windows_config = parameters[windows_key]
        if not isinstance(windows_config, dict):
            raise ValueError("'windows' parameter must be a dictionary")

        shared_params = {k: v for k, v in parameters.items() if k != windows_key}
        for window_id, window_params in windows_config.items():
            image, mask = self.construct_from_flat_parameters(
                model_type, {**shared_params, **window_params}
            )
            result.add_window(window_id, image, mask)

        return result

    @staticmethod
    def _group_parameters(parameters: Dict[str, Any]) -> EncodingParameters:
        """Group flat parameters by region and normalise geometry (same as RoomImageDirector)."""
        encoding_params = EncodingParameters()

        for key, value in parameters.items():
            region = PARAMETER_REGIONS.get(key)
            if region:
                encoding_params.get_region(region)[key] = value

        window_params = encoding_params.window.parameters
        if window_params:
            window_geom = ParameterNormalizer.normalize_window_geometry(window_params)
            if window_geom:
                window_params[ParameterName.WINDOW_GEOMETRY.value] = window_geom
                window_params[ParameterName.X1.value] = window_geom.x1
                window_params[ParameterName.Y1.value] = window_geom.y1
                window_params[ParameterName.Z1.value] = window_geom.z1
                window_params[ParameterName.X2.value] = window_geom.x2
                window_params[ParameterName.Y2.value] = window_geom.y2
                window_params[ParameterName.Z2.value] = window_geom.z2
                if window_geom.direction_angle is not None:
                    window_params[ParameterName.DIRECTION_ANGLE.value] = window_geom.direction_angle

        room_params = encoding_params.room.parameters
        if room_params:
            room_polygon = ParameterNormalizer.normalize_room_polygon(room_params)
            if room_polygon:
                room_params[ParameterName.ROOM_POLYGON.value] = room_polygon

        coord_keys = [
            ParameterName.X1.value, ParameterName.Y1.value,
            ParameterName.X2.value, ParameterName.Y2.value,
            ParameterName.WINDOW_GEOMETRY.value, ParameterName.DIRECTION_ANGLE.value,
            ParameterName.WALL_THICKNESS.value,
        ]
        if room_params and window_params:
            for k in coord_keys:
                if k in window_params:
                    room_params[k] = window_params[k]

        background_params = encoding_params.background.parameters
        if (background_params and window_params and
                ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value in background_params):
            window_params[ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value] = (
                background_params[ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value]
            )

        return encoding_params
