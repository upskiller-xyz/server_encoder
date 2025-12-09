from typing import Dict, Any, Optional, Tuple
import copy
import math
import numpy as np
from src.components.interfaces import IImageBuilder
from src.components.enums import ModelType, RegionType, ParameterName, PARAMETER_REGIONS, EncodingScheme
from src.components.region_encoders import RegionEncoderFactory
from src.components.geometry import WindowGeometry, RoomPolygon, Point2D

class RoomImageBuilder(IImageBuilder):
    """
    Builder for creating encoded room images (Builder Pattern)

    Constructs 128×128 RGBA images with encoded room parameters
    """

    def __init__(self, encoding_scheme: EncodingScheme = EncodingScheme.RGB):
        self._image: Optional[np.ndarray] = None
        self._model_type: Optional[ModelType] = None
        self._encoding_scheme = encoding_scheme
        self._region_encoder_factory = RegionEncoderFactory()
        self.reset()

    def reset(self) -> 'RoomImageBuilder':
        """Reset builder to initial state"""
        # Create 128×128 RGBA image initialized to zeros
        self._image = np.zeros((128, 128, 4), dtype=np.uint8)
        self._model_type = None
        return self

    def set_model_type(self, model_type: ModelType) -> 'RoomImageBuilder':
        """Set the model type for encoding"""
        if not isinstance(model_type, ModelType):
            raise ValueError(f"Invalid model type: {model_type}")
        self._model_type = model_type
        return self

    def encode_region(self, region_type: RegionType, parameters: Dict[str, Any]) -> 'RoomImageBuilder':
        """
        Encode a region using its encoder (Single Responsibility Principle)

        Args:
            region_type: Type of region to encode
            parameters: Region-specific parameters

        Returns:
            Self for chaining
        """
        self._validate_state()
        encoder = self._region_encoder_factory.get_encoder(region_type, self._encoding_scheme)
        self._image = encoder.encode_region(self._image, parameters, self._model_type)
        return self

    def build(self) -> np.ndarray:
        """
        Build and return the final encoded image

        Returns:
            128×128 RGBA numpy array

        Raises:
            RuntimeError: If model type not set
        """
        self._validate_state()
        if self._image is None:
            raise RuntimeError("Image not initialized")
        return self._image.copy()

    def _validate_state(self) -> None:
        """Validate builder state before operations"""
        if self._model_type is None:
            raise RuntimeError("Model type must be set before encoding")
        if self._image is None:
            raise RuntimeError("Image not initialized. Call reset() first.")


class WindowParameterExtractor:
    """
    Extracts window parameters from parameter dictionaries (Single Responsibility)

    Handles both direct parameters and WindowGeometry objects
    """

    @staticmethod
    def extract_window_coordinates(
        window_params: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Extract window coordinates and direction angle from parameters

        Args:
            window_params: Window parameter dictionary

        Returns:
            Tuple of (x1, y1, x2, y2, direction_angle)
        """
        window_x1 = window_params.get(ParameterName.X1.value)
        window_y1 = window_params.get(ParameterName.Y1.value)
        window_x2 = window_params.get(ParameterName.X2.value)
        window_y2 = window_params.get(ParameterName.Y2.value)
        direction_angle = window_params.get(ParameterName.DIRECTION_ANGLE.value)

        # Check window_geometry dict if individual coords not found
        if window_x1 is None and ParameterName.WINDOW_GEOMETRY.value in window_params:
            geom = window_params[ParameterName.WINDOW_GEOMETRY.value]
            if isinstance(geom, WindowGeometry):
                window_x1 = geom.x1
                window_y1 = geom.y1
                window_x2 = geom.x2
                window_y2 = geom.y2
                if direction_angle is None:
                    direction_angle = geom.direction_angle
            else:
                window_x1 = geom.get(ParameterName.X1.value)
                window_y1 = geom.get(ParameterName.Y1.value)
                window_x2 = geom.get(ParameterName.X2.value)
                window_y2 = geom.get(ParameterName.Y2.value)
                if direction_angle is None:
                    direction_angle = geom.get(ParameterName.DIRECTION_ANGLE.value)

        return window_x1, window_y1, window_x2, window_y2, direction_angle


class DirectionAngleCalculator:
    """
    Calculates direction angle from room polygon (Single Responsibility)
    """

    @staticmethod
    def calculate_from_polygon(
        room_params: Dict[str, Any],
        window_x1: float,
        window_y1: float,
        window_x2: float,
        window_y2: float,
        current_direction_angle: Optional[float]
    ) -> Optional[float]:
        """
        Calculate direction angle from room polygon if available

        Args:
            room_params: Room parameter dictionary
            window_x1, window_y1, window_x2, window_y2: Window coordinates
            current_direction_angle: Current direction angle (may be None)

        Returns:
            Calculated or current direction angle
        """
        if ParameterName.ROOM_POLYGON.value not in room_params:
            return current_direction_angle

        polygon_data = room_params[ParameterName.ROOM_POLYGON.value]
        polygon = polygon_data if isinstance(polygon_data, RoomPolygon) else RoomPolygon.from_dict(polygon_data)

        try:
            temp_window_geom = WindowGeometry(
                window_x1, window_y1, 0,
                window_x2, window_y2, 0,
                direction_angle=current_direction_angle
            )
            calculated_direction = temp_window_geom.calculate_direction_from_polygon(polygon)
            print(f"[ROTATION] Calculated direction angle: {calculated_direction:.4f} rad ({calculated_direction * 180 / math.pi:.2f}°)")
            print(f"[ROTATION] Current direction angle: {current_direction_angle}")

            if current_direction_angle is None:
                print(f"[ROTATION] Using calculated direction angle")
                return calculated_direction
        except ValueError:
            pass

        return current_direction_angle


class GeometryRotator:
    """
    Rotates window and room geometry (Single Responsibility)
    """

    @staticmethod
    def rotate_if_needed(
        all_parameters: Dict[str, Any],
        direction_angle: float,
        window_x1: float,
        window_y1: float,
        window_x2: float,
        window_y2: float
    ) -> Dict[str, Any]:
        """
        Rotate geometry if window is not pointing right (0 degrees)

        Args:
            all_parameters: All parameters grouped by region
            direction_angle: Current direction angle in radians
            window_x1, window_y1, window_x2, window_y2: Window coordinates

        Returns:
            Parameters with rotated geometry
        """
        rotation_angle = direction_angle * 180 / math.pi  # Convert to degrees
        print(f"[ROTATION] Direction angle in radians: {direction_angle:.4f}, degrees: {rotation_angle:.2f}°")

        # If already pointing right (within tolerance), no rotation needed
        if abs(rotation_angle) < 0.01:
            print(f"[ROTATION] No rotation needed (already pointing right)")
            return all_parameters

        print(f"[ROTATION] Rotating geometry by {rotation_angle:.2f}° to align window to point right")

        origin = Point2D(0, 0)
        temp_window_geom = WindowGeometry(window_x1, window_y1, 0, window_x2, window_y2, 0, direction_angle)

        # Calculate wall thickness BEFORE rotation (it's invariant under rotation)
        wall_thickness_m = temp_window_geom.wall_thickness

        # Make a deep copy of parameters to avoid modifying original
        rotated_params = copy.deepcopy(all_parameters)

        # Rotate window geometry
        print(f"[ROTATION] Before rotation: window ({window_x1:.2f}, {window_y1:.2f}) to ({window_x2:.2f}, {window_y2:.2f})")
        rotated_window = temp_window_geom.rotate(rotation_angle, origin)
        print(f"[ROTATION] After rotation by {rotation_angle:.2f}°: window ({rotated_window.x1:.2f}, {rotated_window.y1:.2f}) to ({rotated_window.x2:.2f}, {rotated_window.y2:.2f})")
        window_params_copy = rotated_params[RegionType.WINDOW.value]

        GeometryRotator._update_window_coords(window_params_copy, rotated_window, wall_thickness_m)

        # Rotate room polygon if present
        room_params = rotated_params.get(RegionType.ROOM.value, {})
        print(f"[ROTATION] Room params present: {room_params is not None and len(room_params) > 0}")
        if ParameterName.ROOM_POLYGON.value in room_params:
            polygon_data = room_params[ParameterName.ROOM_POLYGON.value]
            polygon = polygon_data if isinstance(polygon_data, RoomPolygon) else RoomPolygon.from_dict(polygon_data)

            print(f"[ROTATION] Room polygon before rotation: {len(polygon.vertices)} vertices")
            print(f"[ROTATION] First 3 vertices: {[(v.x, v.y) for v in polygon.vertices[:3]]}")

            rotated_polygon = polygon.rotate(rotation_angle, origin)
            print(f"[ROTATION] Room polygon after rotation by {rotation_angle:.2f}°: {len(rotated_polygon.vertices)} vertices")
            print(f"[ROTATION] First 3 rotated vertices: {[(v.x, v.y) for v in rotated_polygon.vertices[:3]]}")

            room_params[ParameterName.ROOM_POLYGON.value] = rotated_polygon

            GeometryRotator._update_window_coords(room_params, rotated_window, wall_thickness_m)

            # Set direction_angle to 0 after rotation (window now points right)
            room_params[ParameterName.DIRECTION_ANGLE.value] = 0.0
            print(f"[ROTATION] Set direction_angle to 0.0 after rotation")
        else:
            print(f"[ROTATION] No room polygon found in room_params, keys: {list(room_params.keys())}")

        # Set direction_angle to 0 after rotation (window now points right)
        if RegionType.WINDOW.value in rotated_params:
            rotated_params[RegionType.WINDOW.value][ParameterName.DIRECTION_ANGLE.value] = 0.0

        return rotated_params

    @staticmethod
    def _update_window_coords(param_dict: dict, window: WindowGeometry, thickness: float) -> dict:
        """Update parameter dict with window coordinates and thickness"""
        param_dict[ParameterName.X1.value] = window.x1
        param_dict[ParameterName.Y1.value] = window.y1
        param_dict[ParameterName.X2.value] = window.x2
        param_dict[ParameterName.Y2.value] = window.y2
        param_dict[ParameterName.WALL_THICKNESS.value] = thickness
        return param_dict


class RoomImageDirector:
    """
    Director class for orchestrating image building (Director Pattern)

    Provides high-level interface for building complete encoded images
    """

    def __init__(self, builder: IImageBuilder):
        self._builder = builder

    def construct_full_image(
        self,
        model_type: ModelType,
        all_parameters: Dict[str, Any]
    ) -> np.ndarray:
        """
        Construct a complete encoded image with all regions

        Args:
            model_type: The model type to use
            all_parameters: All parameters grouped by region

        Returns:
            Complete encoded image
        """
        # Reset and configure builder
        self._builder.reset().set_model_type(model_type)

        # Rotate geometry if window is not on south facade
        all_parameters = self._rotate_geometry_if_needed(all_parameters)

        # Define region encoding order (list-based iteration)
        region_order = [
            RegionType.BACKGROUND,
            RegionType.ROOM,
            RegionType.WINDOW,
            RegionType.OBSTRUCTION_BAR
        ]

        # Encode regions in order using list comprehension
        [self._builder.encode_region(region, all_parameters[region.value])
         for region in region_order
         if region.value in all_parameters]

        # Build final image
        return self._builder.build()

    def _rotate_geometry_if_needed(self, all_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rotate room polygon and window coordinates if window is not on south facade

        Uses helper classes to maintain Single Responsibility Principle

        Args:
            all_parameters: Parameters grouped by region

        Returns:
            Parameters with rotated geometry
        """
        # Get window parameters
        window_params = all_parameters.get(RegionType.WINDOW.value, {})
        if not window_params:
            return all_parameters

        # Extract window coordinates using WindowParameterExtractor
        window_x1, window_y1, window_x2, window_y2, direction_angle = \
            WindowParameterExtractor.extract_window_coordinates(window_params)

        if window_x1 is None:
            return all_parameters

        # Calculate direction_angle from room polygon if available
        room_params = all_parameters.get(RegionType.ROOM.value, {})
        print(f"[ROTATION] all_parameters keys: {list(all_parameters.keys())}")
        print(f"[ROTATION] room_params keys: {list(room_params.keys()) if room_params else 'None'}")

        direction_angle = DirectionAngleCalculator.calculate_from_polygon(
            room_params, window_x1, window_y1, window_x2, window_y2, direction_angle
        )

        # Rotate geometry using GeometryRotator
        print(f"[ROTATION] Calling GeometryRotator.rotate_if_needed with direction_angle: {direction_angle}")
        return GeometryRotator.rotate_if_needed(
            all_parameters, direction_angle, window_x1, window_y1, window_x2, window_y2
        )

    def construct_from_flat_parameters(
        self,
        model_type: ModelType,
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """
        Construct image from flat parameter dictionary

        Automatically groups parameters by region.
        Handles both legacy flat structure and unified windows structure.

        Args:
            model_type: The model type to use
            parameters: Dictionary of all parameters (flat or with windows)

        Returns:
            Complete encoded image
        """
        # Check if using unified structure with windows
        windows_key = ParameterName.WINDOWS.value
        if windows_key in parameters and isinstance(parameters[windows_key], dict):
            # Use multi-window construction which handles merging
            images_dict = self.construct_multi_window_images(model_type, parameters)
            # Return the first (and possibly only) image
            return next(iter(images_dict.values()))

        # Legacy flat structure - group and construct directly
        grouped = self._group_parameters(parameters)
        return self.construct_full_image(model_type, grouped)

    def construct_multi_window_images(
        self,
        model_type: ModelType,
        parameters: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Construct multiple images for multiple windows in the same room

        Each window gets its own image with the room positioned relative to that window.
        Shared parameters (room, background) are reused across all images.

        Args:
            model_type: The model type to use
            parameters: Parameters including 'windows' dict with per-window configs

        Returns:
            Dictionary mapping window_id to encoded image
            Example: {"window_1": image1, "window_2": image2}
        """
        windows_key = ParameterName.WINDOWS.value
        if windows_key not in parameters:
            # Single window case - fallback to normal construction
            return {"window_1": self.construct_from_flat_parameters(model_type, parameters)}

        windows_config = parameters[windows_key]
        if not isinstance(windows_config, dict):
            raise ValueError("'windows' parameter must be a dictionary")

        # Extract shared parameters (everything except windows)
        shared_params = {k: v for k, v in parameters.items() if k != windows_key}

        # Build images using dict comprehension
        return {
            window_id: self.construct_from_flat_parameters(
                model_type,
                {**shared_params, **window_params}
            )
            for window_id, window_params in windows_config.items()
        }

    @staticmethod
    def _group_parameters(parameters: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Group flat parameters by region using map (Strategy Pattern)

        Args:
            parameters: Flat parameter dictionary

        Returns:
            Parameters grouped by region
        """
        # Initialize grouped parameters
        grouped = {region.value: {} for region in RegionType}

        # Group parameters using map
        for key, value in parameters.items():
            region = PARAMETER_REGIONS.get(key)
            if region:
                grouped[region.value][key] = value

        # Room positioning depends on window coordinates - copy using list comprehension
        room_key = RegionType.ROOM.value
        window_key = RegionType.WINDOW.value
        background_key = RegionType.BACKGROUND.value

        # Copy window geometry to room
        coord_keys = [ParameterName.X1.value, ParameterName.Y1.value,
                      ParameterName.X2.value, ParameterName.Y2.value,
                      ParameterName.WINDOW_GEOMETRY.value, ParameterName.DIRECTION_ANGLE.value,
                      ParameterName.WALL_THICKNESS.value]

        [grouped[room_key].update({k: grouped[window_key][k]})
         for k in coord_keys if k in grouped[window_key]]

        # Window sill height calculation depends on floor_height_above_terrain from background
        # Copy it to window region for calculator access
        if "floor_height_above_terrain" in grouped[background_key]:
            grouped[window_key]["floor_height_above_terrain"] = grouped[background_key]["floor_height_above_terrain"]

        return grouped
    