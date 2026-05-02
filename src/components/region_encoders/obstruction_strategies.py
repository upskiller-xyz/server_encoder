"""
Obstruction encoding strategies (Strategy Pattern).

Each strategy defines how obstruction data (horizon, zenith, context/balcony reflectance)
is applied to the image for a given encoding scheme:

  V1 / V2           : ObstructionBarStrategy             – small bar on the right edge of the image
  V3                : NoObstructionStrategy               – obstruction data is omitted entirely
  V4 / V7 / V8      : BoundingBoxObstructionStrategy     – obstruction vector multiplied element-wise
                                                            into the floor-plan bounding box region
  V6                : V6BoundingBoxObstructionStrategy   – like V4 but for single-channel float32 images
  V11               : V11BoundingBoxObstructionStrategy  – like V10 but uses gap/midpoint encoder
  V12 / V13         : WindowProjectionObstructionStrategy – rectangle protruding into the room from the
                                                            window, filled uniformly with obstruction values;
                                                            width = window_height, height = window_width_3d
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from src.components.geometry import WindowGeometry
from src.core.enums import (
    EncodingScheme,
    ModelType,
    RegionType,
)
from src.core.graphics_constants import GRAPHICS_CONSTANTS


class ObstructionEncodingStrategy(ABC):
    """Abstract base class for obstruction encoding strategies (Strategy Pattern)"""

    @abstractmethod
    def apply(
        self,
        image: np.ndarray,
        room_mask: Optional[np.ndarray],
        parameters: Dict[str, Any],
        model_type: ModelType,
        window_geometry: Optional[WindowGeometry] = None,
        window_height: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply obstruction encoding to the image.

        Args:
            image: RGBA image array to encode into
            room_mask: Binary room mask (1 = room area) used by bounding-box strategy
            parameters: Obstruction bar parameters (horizon, zenith, …)
            model_type: Model type for HSV pixel-override lookup
            window_geometry: Typed window geometry; required by WindowProjectionObstructionStrategy
            window_height: Window height in metres; required by WindowProjectionObstructionStrategy

        Returns:
            Modified image array
        """


class ObstructionBarStrategy(ObstructionEncodingStrategy):
    """
    Renders obstruction data as a small vertical bar on the right edge of the image.
    Used by V1 and V2 encoding schemes.
    """

    def __init__(self, encoding_scheme: EncodingScheme) -> None:
        from src.components.region_encoders.obstruction_bar_encoder import ObstructionBarEncoder
        self._encoder = ObstructionBarEncoder(encoding_scheme=encoding_scheme)

    def apply(
        self,
        image: np.ndarray,
        room_mask: Optional[np.ndarray],
        parameters: Dict[str, Any],
        model_type: ModelType,
        window_geometry: Optional[WindowGeometry] = None,
        window_height: Optional[float] = None,
    ) -> np.ndarray:
        return self._encoder.encode_region(image, parameters, model_type)


class NoObstructionStrategy(ObstructionEncodingStrategy):
    """
    No-op strategy for V3 and V5 encoding: obstruction data is not encoded anywhere.
    The obstruction bar is removed and the bounding box is left unmodified.
    """

    def apply(
        self,
        image: np.ndarray,
        room_mask: Optional[np.ndarray],
        parameters: Dict[str, Any],
        model_type: ModelType,
        window_geometry: Optional[WindowGeometry] = None,
        window_height: Optional[float] = None,
    ) -> np.ndarray:
        return image


class BoundingBoxObstructionStrategy(ObstructionEncodingStrategy):
    """
    V4 obstruction strategy.

    Instead of a separate bar, the obstruction vector is placed vertically
    (one value per row) and expanded horizontally across the full width of the
    floor-plan bounding box [miny:maxy, minx:maxx].  Each pixel in that region
    is multiplied element-wise by the normalised obstruction factor (0–1),
    effectively modulating the room encoding with the obstruction data.

    No separate obstruction bar is rendered.
    """

    def __init__(self) -> None:
        from src.components.region_encoders.obstruction_bar_encoder import ObstructionBarEncoder
        # Use V2 channel mapping for the obstruction vector computation
        self._encoder = ObstructionBarEncoder(encoding_scheme=EncodingScheme.V2)

    def apply(
        self,
        image: np.ndarray,
        room_mask: Optional[np.ndarray],
        parameters: Dict[str, Any],
        model_type: ModelType,
        window_geometry: Optional[WindowGeometry] = None,
        window_height: Optional[float] = None,
    ) -> np.ndarray:
        if room_mask is None or not np.any(room_mask):
            return image

        # Derive bounding box from the room mask
        rows = np.any(room_mask, axis=1)
        cols = np.any(room_mask, axis=0)
        if not rows.any() or not cols.any():
            return image

        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        min_y = int(row_indices[0])
        max_y = int(row_indices[-1]) + 1
        min_x = int(col_indices[0])
        max_x = int(col_indices[-1]) + 1

        bbox_height = max_y - min_y
        if bbox_height <= 0:
            return image

        # Build the 4-channel obstruction vector for the bounding-box height
        obstruction_vector = self._encoder.compute_obstruction_vector(
            parameters, model_type, bbox_height
        )

        # Expand the vector horizontally: (H, 1, 4) broadcasts over (H, W, 4)
        obs_expanded = obstruction_vector[:, np.newaxis, :]  # (bbox_height, 1, 4)

        # Element-wise multiplication: scale each row by the obstruction factor [0, 1]
        region = image[min_y:max_y, min_x:max_x, :].astype(np.float64)
        region = region * obs_expanded / 255.0
        image[min_y:max_y, min_x:max_x, :] = np.clip(region, 0, 255).astype(np.uint8)

        return image


class V6BoundingBoxObstructionStrategy(ObstructionEncodingStrategy):
    """
    V6 obstruction strategy.

    Identical bounding-box derivation as V4, but the image is a single-channel
    float32 mask instead of a 4-channel uint8 image.  The 4-channel obstruction
    vector is collapsed to a per-row scalar factor (mean of normalised channels)
    and multiplied element-wise into the bounding-box region of the float32 image.
    """

    def __init__(self) -> None:
        from src.components.region_encoders.obstruction_bar_encoder import ObstructionBarEncoder
        self._encoder = ObstructionBarEncoder(encoding_scheme=EncodingScheme.V2)

    def apply(
        self,
        image: np.ndarray,
        room_mask: Optional[np.ndarray],
        parameters: Dict[str, Any],
        model_type: ModelType,
        window_geometry: Optional[WindowGeometry] = None,
        window_height: Optional[float] = None,
    ) -> np.ndarray:
        if room_mask is None or not np.any(room_mask):
            return image

        rows = np.any(room_mask, axis=1)
        cols = np.any(room_mask, axis=0)
        if not rows.any() or not cols.any():
            return image

        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        min_y = int(row_indices[0])
        max_y = int(row_indices[-1]) + 1
        min_x = int(col_indices[0])
        max_x = int(col_indices[-1]) + 1

        bbox_height = max_y - min_y
        if bbox_height <= 0:
            return image

        # Compute 4-channel obstruction vector and collapse to a per-row scalar
        obs_vector = self._encoder.compute_obstruction_vector(
            parameters, model_type, bbox_height
        )  # (bbox_height, 4) float64, values in [0, 255]
        obs_factor = (obs_vector.mean(axis=1) / 255.0).astype(np.float32)  # (bbox_height,)

        # Broadcast: (bbox_height, 1, 1) over (bbox_height, W, 1)
        obs_factor_3d = obs_factor[:, np.newaxis, np.newaxis]

        region = image[min_y:max_y, min_x:max_x, :]
        image[min_y:max_y, min_x:max_x, :] = np.clip(region * obs_factor_3d, 0.0, 1.0)

        return image


class V11BoundingBoxObstructionStrategy(BoundingBoxObstructionStrategy):
    """
    V11 obstruction strategy.

    Same bounding-box multiplication as V4/V7/V8/V9/V10, but uses
    V11ObstructionBarEncoder so the obstruction vector is computed from
    the visible-sky gap and midpoint rather than raw zenith/horizon values.
    """

    def __init__(self) -> None:
        from src.components.region_encoders.obstruction_bar_encoder import V11ObstructionBarEncoder
        self._encoder = V11ObstructionBarEncoder(encoding_scheme=EncodingScheme.V11)


class WindowProjectionObstructionStrategy(ObstructionEncodingStrategy):
    """
    V12 / V13 obstruction strategy.

    Renders a rectangle that protrudes leftward from the window stripe into the room,
    filled uniformly with the obstruction channel values (V8 mapping: R=zenith,
    G=context_reflectance, B=horizon, A=balcony_reflectance).

    Rectangle dimensions (image coordinates):
    - Vertical extent  : same as the window stripe — determined by window_width_3d
    - Horizontal extent: window_height metres converted to pixels (protrudes left)

    A single-row obstruction vector is computed so the fill is uniform across the
    rectangle; the rectangle area itself encodes window_height spatially.
    No separate obstruction bar is rendered.
    """

    def __init__(self) -> None:
        from src.components.region_encoders.obstruction_bar_encoder import ObstructionBarEncoder
        self._encoder = ObstructionBarEncoder(encoding_scheme=EncodingScheme.V8)

    def apply(
        self,
        image: np.ndarray,
        room_mask: Optional[np.ndarray],
        parameters: Dict[str, Any],
        model_type: ModelType,
        window_geometry: Optional[WindowGeometry] = None,
        window_height: Optional[float] = None,
    ) -> np.ndarray:
        if window_geometry is None or window_height is None or window_height <= 0:
            return image

        _height, width = image.shape[:2]
        border = GRAPHICS_CONSTANTS.BORDER_PX

        # Vertical extent matches the window stripe (based on window_width_3d)
        x_stripe_start, y_start, _x_stripe_end, y_end = window_geometry.get_pixel_bounds(
            image_size=width
        )

        # Rectangle extends leftward from the window stripe's left edge
        window_height_px = GRAPHICS_CONSTANTS.get_pixel_value(window_height, width)
        rect_x_end = x_stripe_start
        rect_x_start = max(border, rect_x_end - window_height_px)

        if rect_x_start >= rect_x_end or y_start >= y_end:
            return image

        # Single-row obstruction vector → uniform fill across the rectangle
        obs_vector = self._encoder.compute_obstruction_vector(parameters, model_type, height=1)
        obs_fill = obs_vector[0].astype(np.uint8)  # shape (4,)

        # Clip to room mask so the rectangle never bleeds into background pixels
        if room_mask is not None:
            mask_slice = room_mask[y_start:y_end, rect_x_start:rect_x_end]
            image[y_start:y_end, rect_x_start:rect_x_end, :] = np.where(
                mask_slice[:, :, np.newaxis] > 0, obs_fill, image[y_start:y_end, rect_x_start:rect_x_end, :]
            )
        else:
            image[y_start:y_end, rect_x_start:rect_x_end, :] = obs_fill

        return image


# Factory map: EncodingScheme -> strategy constructor (Strategy Pattern)
_STRATEGY_MAP = {
    EncodingScheme.V1: lambda: ObstructionBarStrategy(EncodingScheme.V1),
    EncodingScheme.V2: lambda: ObstructionBarStrategy(EncodingScheme.V2),
    EncodingScheme.V3: lambda: NoObstructionStrategy(),
    EncodingScheme.V4: lambda: BoundingBoxObstructionStrategy(),
    EncodingScheme.V5: lambda: NoObstructionStrategy(),
    EncodingScheme.V6: lambda: V6BoundingBoxObstructionStrategy(),
    EncodingScheme.V7: lambda: BoundingBoxObstructionStrategy(),  # Same as V4
    EncodingScheme.V8: lambda: BoundingBoxObstructionStrategy(),  # Same as V4
    EncodingScheme.V9: lambda: BoundingBoxObstructionStrategy(),  # Same as V7; alpha dropped in service
    EncodingScheme.V10: lambda: BoundingBoxObstructionStrategy(),  # Same as V8; alpha dropped in service
    EncodingScheme.V11: lambda: V11BoundingBoxObstructionStrategy(),  # Like V10; gap/midpoint encoder
    EncodingScheme.V12: lambda: WindowProjectionObstructionStrategy(),
    EncodingScheme.V13: lambda: WindowProjectionObstructionStrategy(),
}


class ObstructionStrategyFactory:
    """Factory for creating obstruction encoding strategies (Factory Pattern)"""

    @staticmethod
    def create(encoding_scheme: EncodingScheme) -> ObstructionEncodingStrategy:
        """
        Create the appropriate obstruction strategy for the given encoding scheme.

        Args:
            encoding_scheme: One of V1, V2, V3, V4

        Returns:
            Corresponding ObstructionEncodingStrategy instance

        Raises:
            ValueError: If the encoding scheme is not recognised
        """
        factory_fn = _STRATEGY_MAP.get(encoding_scheme)
        if factory_fn is None:
            raise ValueError(f"Unknown encoding scheme for obstruction strategy: {encoding_scheme}")
        return factory_fn()
