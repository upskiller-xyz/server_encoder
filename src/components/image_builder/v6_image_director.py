"""
V6 image director: single-channel float32 geometric mask with bounding-box obstruction
and a companion static-parameter vector.

V6 combines:
- V5 image encoding: (H, W, 1) float32 mask (background=0, room=1, window=0.6)
- V4 obstruction approach: the obstruction data is applied to the room-polygon
  bounding box via the V6BoundingBoxObstructionStrategy (single-channel variant)
- Static vector output: scalar parameters (sill_height, roof_over_floor, etc.) are
  NOT encoded into the image; instead they are normalised and returned as a separate
  1-D float32 array whose order is defined by V6_STATIC_PARAMS.
"""
from typing import Any, Dict, Optional, Tuple
import logging

import cv2
import numpy as np

from src.core import ModelType, ParameterName, RegionType
from src.core.enums import PARAMETER_REGIONS, V5_MASK_VALUES, V6_STATIC_PARAMS
from src.core.graphics_constants import GRAPHICS_CONSTANTS
from src.components.geometry import RoomPolygon, WindowGeometry
from src.components.image_builder.parameter_normalizer import ParameterNormalizer
from src.components.image_builder.geometry_rotator import GeometryRotator
from src.components.parameter_encoders.encoder_factory import EncoderFactory
from src.components.calculators.parameter_calculator_registry import ParameterCalculatorRegistry
from src.components.region_encoders.obstruction_strategies import V6BoundingBoxObstructionStrategy
from src.models import EncodingParameters, EncodingResult

logger = logging.getLogger(__name__)


class V6ImageDirector:
    """
    Director for V6 encoding.

    Produces:
    - image:         (H, W, 1) float32 geometric mask with obstruction applied to bbox
    - room_mask:     (H, W) uint8 binary mask
    - static_vector: 1-D float32 array of normalised scalar parameters (V6_STATIC_PARAMS order)
    """

    IMAGE_SIZE = 128

    def __init__(self) -> None:
        self._obstruction_strategy = V6BoundingBoxObstructionStrategy()
        self._encoder_factory = EncoderFactory()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def construct_full_image(
        self,
        model_type: ModelType,
        all_parameters: EncodingParameters,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Build the V6 image, room mask, and static vector.

        Args:
            model_type: Used for obstruction-vector HSV-override lookup.
            all_parameters: Grouped encoding parameters.

        Returns:
            Tuple of (image, room_mask, static_vector):
            - image:         (H, W, 1) float32 in [0, 1]
            - room_mask:     (H, W) uint8 binary mask, or None
            - static_vector: 1-D float32 array (length = len(V6_STATIC_PARAMS))
        """
        image = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE, 1), dtype=np.float32)

        all_parameters = self._rotate_geometry_if_needed(all_parameters)

        room_mask = self._draw_room(image, all_parameters)
        self._draw_window(image, all_parameters)

        # Apply bounding-box obstruction (V4-style, single-channel)
        obstruction_params = all_parameters.obstruction_bar.parameters
        if obstruction_params and room_mask is not None:
            image = self._obstruction_strategy.apply(
                image, room_mask, obstruction_params, model_type
            )

        static_vector = self._build_static_vector(all_parameters)

        return image, room_mask, static_vector

    # ------------------------------------------------------------------
    # Region drawing (mirrored from V5ImageDirector)
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

        border = GRAPHICS_CONSTANTS.BORDER_PX
        mask[:border, :] = 0
        mask[h - border:, :] = 0
        mask[:, :border] = 0
        mask[:, w - border:] = 0

        image[mask.astype(bool)] = V5_MASK_VALUES[RegionType.ROOM]
        return mask

    def _draw_window(self, image: np.ndarray, all_parameters: EncodingParameters) -> None:
        """Fill the window area with V5_MASK_VALUES[WINDOW] (0.6)."""
        window_params = all_parameters.window.parameters
        if not window_params:
            return

        h, w = image.shape[:2]

        try:
            window_geom = WindowGeometry(
                x1=float(window_params[ParameterName.X1.value]),
                y1=float(window_params[ParameterName.Y1.value]),
                z1=float(window_params[ParameterName.Z1.value]),
                x2=float(window_params[ParameterName.X2.value]),
                y2=float(window_params[ParameterName.Y2.value]),
                z2=float(window_params[ParameterName.Z2.value]),
            )
            x_start, y_start, x_end, y_end = window_geom.get_pixel_bounds(image_size=w)

            border = GRAPHICS_CONSTANTS.BORDER_PX
            x_start = max(x_start, border)
            y_start = max(y_start, border)
            x_end = min(x_end, w - border)
            y_end = min(y_end, h - border)

            image[y_start:y_end, x_start:x_end] = V5_MASK_VALUES[RegionType.WINDOW]
        except (KeyError, ValueError) as exc:
            logger.warning("V6 window drawing skipped: %s", exc)

    # ------------------------------------------------------------------
    # Static vector
    # ------------------------------------------------------------------

    def _build_static_vector(self, all_parameters: EncodingParameters) -> np.ndarray:
        """
        Build the static parameter vector.

        For each parameter in V6_STATIC_PARAMS, look it up across all region
        parameter dicts, encode it with the corresponding EncoderFactory encoder,
        and normalise to [0, 1] by dividing by 255.  Missing parameters are encoded
        as 0.0.

        Returns:
            1-D float32 array of length len(V6_STATIC_PARAMS)
        """
        # Merge all region parameters into one flat dict for easy lookup
        flat: Dict[str, Any] = {}
        for region in [
            all_parameters.background,
            all_parameters.room,
            all_parameters.window,
            all_parameters.obstruction_bar,
        ]:
            if region.parameters:
                flat.update(region.parameters)
        flat.update(all_parameters.global_params)

        # Auto-calculate derived parameters (window_sill_height, window_height)
        derived = ParameterCalculatorRegistry.calculate_derived_parameters(flat)
        flat.update(derived)

        vector = np.zeros(len(V6_STATIC_PARAMS), dtype=np.float32)
        for idx, param_name in enumerate(V6_STATIC_PARAMS):
            key = param_name.value
            if key not in flat:
                logger.debug("V6 static vector: parameter '%s' not found, using 0.0", key)
                continue
            try:
                encoder = self._encoder_factory.create_encoder(key)
                pixel = encoder.encode(float(flat[key]))
                vector[idx] = float(pixel) / 255.0
            except (ValueError, TypeError) as exc:
                logger.warning("V6 static vector: could not encode '%s': %s", key, exc)

        return vector

    # ------------------------------------------------------------------
    # Geometry rotation (mirrors V5ImageDirector)
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
    # Flat-parameter and multi-window entry points
    # ------------------------------------------------------------------

    def construct_from_flat_parameters(
        self,
        model_type: ModelType,
        parameters: Dict[str, Any],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Construct image, mask, and static vector from a flat parameter dictionary."""
        windows_key = ParameterName.WINDOWS.value
        if windows_key in parameters and isinstance(parameters[windows_key], dict):
            result = self.construct_multi_window_images(model_type, parameters)
            return result.get_first_image(), result.get_first_mask(), result.get_first_static_vector()

        grouped = self._group_parameters(parameters)
        return self.construct_full_image(model_type, grouped)

    def construct_multi_window_images(
        self,
        model_type: ModelType,
        parameters: Dict[str, Any],
    ) -> "V6EncodingResult":
        """Construct one V6 image (+ static vector) per window."""
        result = V6EncodingResult()
        windows_key = ParameterName.WINDOWS.value

        if windows_key not in parameters:
            image, mask, static_vector = self.construct_from_flat_parameters(model_type, parameters)
            result.add_window("window_1", image, mask, static_vector)
            return result

        windows_config = parameters[windows_key]
        if not isinstance(windows_config, dict):
            raise ValueError("'windows' parameter must be a dictionary")

        shared_params = {k: v for k, v in parameters.items() if k != windows_key}
        for window_id, window_params in windows_config.items():
            image, mask, static_vector = self.construct_from_flat_parameters(
                model_type, {**shared_params, **window_params}
            )
            result.add_window(window_id, image, mask, static_vector)

        return result

    @staticmethod
    def _group_parameters(parameters: Dict[str, Any]) -> EncodingParameters:
        """Group flat parameters by region and normalise geometry."""
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


class V6EncodingResult(EncodingResult):
    """
    Extends EncodingResult with per-window static vectors for V6 encoding.
    """

    def __init__(self) -> None:
        super().__init__()
        self.static_vectors: Dict[str, np.ndarray] = {}

    def add_window(  # type: ignore[override]
        self,
        window_id: str,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        static_vector: Optional[np.ndarray] = None,
    ) -> None:
        super().add_window(window_id, image, mask)
        self.static_vectors[window_id] = (
            static_vector if static_vector is not None
            else np.zeros(len(V6_STATIC_PARAMS), dtype=np.float32)
        )

    def get_static_vector(self, window_id: str = "window_1") -> Optional[np.ndarray]:
        """Get static vector for a specific window."""
        return self.static_vectors.get(window_id)

    def get_first_static_vector(self) -> np.ndarray:
        """Get the first static vector (single-window convenience)."""
        return next(iter(self.static_vectors.values()))
